import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import torch
import random

INPUT_DIR = "traindata/processed_videos"
OUTPUT_CSV = "testdata/back_features.csv"

POSE_MODEL = YOLO("yolov11s-pose.pt")
SEG_MODEL  = YOLO("yolov11s-seg.pt")
DEVICE = "cuda"
# ============================== Geometry helpers ==============================

def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.cross(b - a, c - a))

def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> bool:
    if abs(_orient(a, b, p)) > eps:
        return False
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)

def segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray, eps: float = 1e-9) -> bool:
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)
    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True
    if abs(o1) <= eps and _on_segment(p1, p2, q1, eps): return True
    if abs(o2) <= eps and _on_segment(p1, p2, q2, eps): return True
    if abs(o3) <= eps and _on_segment(q1, q2, p1, eps): return True
    if abs(o4) <= eps and _on_segment(q1, q2, p2, eps): return True
    return False

# ============================== GPU IoU 計算 ==============================
def compute_iou_gpu(mask_tensor, box, H, W):
    if mask_tensor is None or mask_tensor.shape[0] == 0:
        return -1, 0.0
    device = mask_tensor.device
    mask_box = torch.zeros((H, W), dtype=torch.bool, device=device)
    x1, y1, x2, y2 = box.int()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return -1, 0.0
    mask_box[y1:y2, x1:x2] = True
    masks_resized = torch.nn.functional.interpolate(
        mask_tensor.unsqueeze(1).float(),
        size=(H, W),
        mode="nearest"
    ).squeeze(1).bool()
    inter = (masks_resized & mask_box).sum(dim=(1, 2)).float()
    union = (masks_resized | mask_box).sum(dim=(1, 2)).float()
    ious = torch.where(union > 0, inter / union, torch.zeros_like(union))
    max_iou, max_idx = torch.max(ious, dim=0)
    return max_idx.item(), max_iou.item()

# ============================== Shoulder / Hip 選擇 ==============================

def pick_shoulder_hip_from_keypoints(kp_xy, kp_conf):
    if kp_conf[6] > kp_conf[5] and kp_conf[12] > kp_conf[11]:
        return kp_xy[6], kp_xy[12]
    elif kp_conf[5] >= kp_conf[6] and kp_conf[11] >= kp_conf[12]:
        return kp_xy[5], kp_xy[11]
    else:
        return None, None
# ============================== 視角計算 ==============================
def compute_view_angle(kp_xy, kp_conf, trunk_length):
    # left shoulder = 5, right shoulder = 6 (YOLO COCO)
    if kp_conf[5] > 0.3 and kp_conf[6] > 0.3 and trunk_length > 0:
        ls = kp_xy[5]
        rs = kp_xy[6]
        shoulder_width = np.linalg.norm(ls - rs)
        return shoulder_width / trunk_length
    return np.nan


# ========================== Back Curve Core ==========================
def calculate_back_curve(mask_tensor: torch.Tensor,
                         shoulder_xy: np.ndarray,
                         hip_xy: np.ndarray,
                         image_rgb: np.ndarray,
                         nose_xy=None,
                         corridor_height_ratio: float = 1.5,
                         back_offset_ratio: float = 1.2,
                         eps: float = 1e-6):

    # --- mask 與輪廓 ---
    mask_bin_np = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
    if mask_bin_np.shape[:2] != image_rgb.shape[:2]:
        mask_resized = cv2.resize(mask_bin_np, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask_bin_np

    # === 幾何向量 ===
    a = np.array(shoulder_xy, dtype=float)
    c = np.array(hip_xy, dtype=float)
    trunk_vec = c - a
    trunk_norm = np.linalg.norm(trunk_vec)
    if trunk_norm <= eps:
        return None, None, image_rgb, "No Trunk", None

    u_trunk = trunk_vec / trunk_norm
    u_perp  = np.array([-u_trunk[1], u_trunk[0]])
    mid_ac  = (a + c) / 2.0

    # === 判斷背部方向 (鼻子反向) ===
    if nose_xy is not None:
        side = np.sign(np.dot(nose_xy - mid_ac, u_perp))
    else:
        side = 1.0
    back_dir = -side * u_perp

    # === 生成背部矩形 ===
    back_width  = trunk_norm * back_offset_ratio
    half_height = (trunk_norm * corridor_height_ratio) / 2.0
    rect_back = np.array([
        mid_ac - u_trunk * half_height,
        mid_ac - u_trunk * half_height + back_dir * back_width,
        mid_ac + u_trunk * half_height + back_dir * back_width,
        mid_ac + u_trunk * half_height
    ], dtype=np.int32)

    # === 建立背部矩形 mask ===
    mask_back = np.zeros_like(mask_resized, dtype=np.uint8)
    cv2.fillPoly(mask_back, [rect_back], 1)
    mask_back_final = mask_resized.copy()
    mask_back_final[mask_back == 0] = 0

    # --- 找輪廓 ---
    contours, _ = cv2.findContours(mask_back_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, image_rgb, "No Person", None
    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim != 2 or contour.shape[0] < 2:
        return None, None, image_rgb, "No Person", None

    # === 定義 LA/LC ===
    def line_points_through(p, u_perp, length=2000):
        return p - u_perp * length, p + u_perp * length
    LA1, LA2 = line_points_through(a, u_perp)
    LC1, LC2 = line_points_through(c, u_perp)

    # === 找交點候選 ===
    def cross2d(x, y): return x[...,0]*y[...,1] - x[...,1]*y[...,0]
    def farthest_intersection(base, contour, p1, p2, eps: float = 1e-9):
        p1 = np.asarray(p1, dtype=float); p2 = np.asarray(p2, dtype=float)
        contour = np.asarray(contour, dtype=float)
        r = p2 - p1
        q1 = contour
        q2 = np.roll(contour, -1, axis=0)
        s  = q2 - q1
        r_b = np.broadcast_to(r, s.shape)
        rxs = cross2d(r_b, s)
        q_p = q1 - p1
        denom = np.where(np.abs(rxs) < eps, np.inf, rxs)
        t = cross2d(q_p, s) / denom
        u = cross2d(q_p, r_b) / denom
        valid = (np.abs(rxs) > eps) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        if not np.any(valid): return None
        inter_pts = p1 + t[valid, None] * r
        dists = np.linalg.norm(inter_pts - base, axis=1)
        return inter_pts[np.argmax(dists)]

    P = farthest_intersection(a, contour, LA1, LA2)
    Q = farthest_intersection(c, contour, LC1, LC2)
    if P is None or Q is None:
        return None, None, image_rgb, "No PQ", None

    # === 從輪廓切出 PQ 曲線 (兩條，取短的) ===
    idx_p = np.argmin(np.linalg.norm(contour - P, axis=1))
    idx_q = np.argmin(np.linalg.norm(contour - Q, axis=1))
    if idx_p < idx_q:
        curve1 = contour[idx_p:idx_q+1]
        curve2 = np.concatenate([contour[idx_q:], contour[:idx_p+1]], axis=0)
    else:
        curve1 = contour[idx_q:idx_p+1]
        curve2 = np.concatenate([contour[idx_p:], contour[:idx_q+1]], axis=0)

    def curve_length(curve):
        diffs = np.diff(curve, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
    len1, len2 = curve_length(curve1), curve_length(curve2)
    back_curve = curve1 if len1 < len2 else curve2
    curve_len  = min(len1, len2)
    proportion = curve_len / max(trunk_norm, eps)

    # === 取 P, R, Q
    r = 0.57
    b = a + r * (c - a)
    AP = back_curve[np.argmin(np.linalg.norm(back_curve - a, axis=1))]  # P
    BR = back_curve[np.argmin(np.linalg.norm(back_curve - b, axis=1))]  # R
    CQ = back_curve[np.argmin(np.linalg.norm(back_curve - c, axis=1))]  # Q

    # === 三點圓擬合 ===
    def circle_from_3pts(p1, p2, p3, eps=1e-9):
        temp = p2[0]**2 + p2[1]**2
        bc = (p1[0]**2 + p1[1]**2 - temp)/2.0
        cd = (temp - p3[0]**2 - p3[1]**2)/2.0
        det = (p1[0]-p2[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p2[1])
        if abs(det) < eps: return None
        cx = (bc*(p2[1]-p3[1]) - cd*(p1[1]-p2[1])) / det
        cy = ((p1[0]-p2[0])*cd - (p2[0]-p3[0])*bc) / det
        R  = np.sqrt((p1[0]-cx)**2 + (p1[1]-cy)**2)
        return np.array([cx, cy]), R

    circle_result = circle_from_3pts(AP, BR, CQ)
    if circle_result is None:
        return None, None, image_rgb, "Collinear", None
    center, radius = circle_result

    # === 曲率正規化（以 PQ 弦長） ===
    PQ_len = np.linalg.norm(CQ - AP)
    k = (1.0 / max(radius, eps)) * PQ_len

    # === 分類規則 ===
    posture_type = "neutral"
    orientation = None

    if nose_xy is not None:
        v_center = center - mid_ac
        relation = np.dot(v_center, u_perp) * side
        
        if relation < 0:
            orientation = "back"
        else:
            orientation = "nose"

        if k > 0.25 and relation > 0:
            posture_type = "rounded_back"
        else:
            posture_type = "neutral"


    # 一律回傳 5 個值
    return curve_len, proportion, posture_type, k, orientation



# ================= Frame sampler =================

def sample_frames(video_path, n_samples=20, ratio=1):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = int(total * ratio)

    idxs = np.linspace(0, limit-1, n_samples).astype(int)
    frames = []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames

# ================= Main process =================
videos = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".mp4", ".avi", ".mov"))
]
print(f"Total videos found: {len(videos)}")

rows = []

for vid in tqdm(videos, desc="Processing videos"):
    video_path = os.path.join(INPUT_DIR, vid)

    frames = sample_frames(video_path)

    ks = []
    os_ = []
    vs = []   # 👈 新增 view angles

    for frame in frames:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = POSE_MODEL(rgb, verbose=False)[0]
        seg_res  = SEG_MODEL(rgb, task="segment", verbose=False)[0]

        # if pose_res.keypoints is None:
        #     ks.append(np.nan); os_.append("none"); vs.append(np.nan)
        #     continue
        if (
            pose_res.keypoints is None or
            pose_res.keypoints.xy is None or
            pose_res.keypoints.xy.shape[0] == 0
        ):
            ks.append(np.nan)
            os_.append("none")
            vs.append(np.nan)
            continue

        kp_xy = pose_res.keypoints.xy[0].cpu().numpy()
        kp_conf = pose_res.keypoints.conf[0].cpu().numpy()

        shoulder, hip = pick_shoulder_hip_from_keypoints(kp_xy, kp_conf)
        if shoulder is None:
            ks.append(np.nan); os_.append("none"); vs.append(np.nan)
            continue

        trunk_length = np.linalg.norm(shoulder - hip)

        # ===== view angle =====
        view_angle = compute_view_angle(kp_xy, kp_conf, trunk_length)
        vs.append(view_angle)

        box = pose_res.boxes.xyxy[0]
        H, W = frame.shape[:2]

        if seg_res.masks is None:
            ks.append(np.nan); os_.append("none")
            continue

        idx, _ = compute_iou_gpu(seg_res.masks.data, box, H, W)
        mask = seg_res.masks.data[idx]

        nose = kp_xy[0] if kp_conf[0] > 0.3 else None

        _, _, _, k, orient = calculate_back_curve(
            mask, shoulder, hip, rgb, nose
        )

        ks.append(k if k is not None else np.nan)
        os_.append(orient if orient else "none")

    # ===== ground truth label from filename =====
    label_char = vid[3].lower()
    gt_label = "rounded_back" if label_char == "r" else "neutral"

    rows.append([vid] + ks + os_ + vs + [gt_label])

# ================= Save CSV =================
columns = (
    ["filename"] +
    [f"k{i+1}" for i in range(20)] +
    [f"o{i+1}" for i in range(20)] +
    [f"v{i+1}" for i in range(20)] +
    ["true_label"]
)


df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("✅ Feature CSV generated with ground truth:", OUTPUT_CSV)

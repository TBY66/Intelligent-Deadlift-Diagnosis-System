import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

NUM_FRAMES = 20
# ------------------------------------------------------------
# compute angle
# ------------------------------------------------------------
def compute_angle(A, B, C):
    """
    A, B, C: (..., 2)
    angle at point B (radians)
    """
    BA = A - B
    BC = C - B

    dot = np.sum(BA * BC, axis=-1)
    norm = np.linalg.norm(BA, axis=-1) * np.linalg.norm(BC, axis=-1) + 1e-6
    cos = np.clip(dot / norm, -1.0, 1.0)
    return np.arccos(cos)

# ------------------------------------------------------------
# angle clipping helper
# ------------------------------------------------------------
SHOULDER_MIN_RAD = np.deg2rad(0)
SHOULDER_MAX_RAD = np.deg2rad(170)

HIP_MIN_RAD  = np.deg2rad(45)
HIP_MAX_RAD  = np.deg2rad(180)

KNEE_MIN_RAD = np.deg2rad(40)
KNEE_MAX_RAD = np.deg2rad(180)

def clip_joint_angles(r_sh, r_hip, r_knee, l_sh, l_hip, l_knee):
    r_sh = np.clip(r_sh, SHOULDER_MIN_RAD, SHOULDER_MAX_RAD) / np.pi
    l_sh = np.clip(l_sh, SHOULDER_MIN_RAD, SHOULDER_MAX_RAD) / np.pi

    r_hip = np.clip(r_hip, HIP_MIN_RAD, HIP_MAX_RAD) / np.pi
    l_hip = np.clip(l_hip, HIP_MIN_RAD, HIP_MAX_RAD) / np.pi

    r_knee = np.clip(r_knee, KNEE_MIN_RAD, KNEE_MAX_RAD) / np.pi
    l_knee = np.clip(l_knee, KNEE_MIN_RAD, KNEE_MAX_RAD) / np.pi

    return r_sh, r_hip, r_knee, l_sh, l_hip, l_knee

# ------------------------------------------------------------
# uniform sampling
# ------------------------------------------------------------
def sample_indices(n, k=NUM_FRAMES):
    if n <= k:
        return np.linspace(0, n-1, n, dtype=int)
    return np.linspace(0, n-1, k, dtype=int)

# ------------------------------------------------------------
# target joints (YOLO naming)
# ------------------------------------------------------------
J_TARGET = [
    "right_shoulder","right_elbow","right_wrist",
    "right_hip","right_knee","right_ankle",
    "left_shoulder","left_elbow","left_wrist",
    "left_hip","left_knee","left_ankle"
]

# YOLOv8 keypoint names in order
YOLO_KEYS = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

# mapping target → yolo index
KP_INDEX = {name: YOLO_KEYS.index(name) for name in J_TARGET}

# ------------------------------------------------------------
# extract 12 markers × 2 = 24 dims
# ------------------------------------------------------------
def extract_markers_from_yolo(video_path, model):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    idxs = sample_indices(total_frames, NUM_FRAMES)
    frame_features = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if not ret:
            frame_features.append(np.full((30,), np.nan))
            continue

        results = model(frame, verbose=False)
        if results[0].keypoints is None:
            frame_features.append(np.full((30,), np.nan))
            continue

        kpts_all = results[0].keypoints.xy.cpu().numpy()

        if len(kpts_all) > 1 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            kpts = kpts_all[np.argmax(areas)]
        else:
            kpts = kpts_all[0]

        # ---------- Normalization ----------
        h, w = frame.shape[:2]
        kpts[:, 0] /= w
        kpts[:, 1] /= h

        lhip = kpts[YOLO_KEYS.index("left_hip")]
        rhip = kpts[YOLO_KEYS.index("right_hip")]
        center = (lhip + rhip) / 2
        kpts = kpts - center

        ls = kpts[YOLO_KEYS.index("left_shoulder")]
        rs = kpts[YOLO_KEYS.index("right_shoulder")]
        lh = kpts[YOLO_KEYS.index("left_hip")]
        rh = kpts[YOLO_KEYS.index("right_hip")]

        torso = np.linalg.norm(((ls + rs) / 2) - ((lh + rh) / 2))
        if torso > 1e-6:
            kpts /= torso

        kp = {}
        for name in J_TARGET:
            kp[name] = kpts[KP_INDEX[name]]

        # ---------- 24 coords ----------
        coords = []
        for j in J_TARGET:
            coords.extend(kp[j])

        coords = np.array(coords)  # (24,)

        # ---------- 6 ANGLES ----------
        ang_r_sh = compute_angle(kp["right_elbow"], kp["right_shoulder"], kp["right_hip"])
        ang_r_hip = compute_angle(kp["right_shoulder"], kp["right_hip"], kp["right_knee"])
        ang_r_knee = compute_angle(kp["right_hip"], kp["right_knee"], kp["right_ankle"])

        ang_l_sh = compute_angle(kp["left_elbow"], kp["left_shoulder"], kp["left_hip"])
        ang_l_hip = compute_angle(kp["left_shoulder"], kp["left_hip"], kp["left_knee"])
        ang_l_knee = compute_angle(kp["left_hip"], kp["left_knee"], kp["left_ankle"])

        ang_r_sh, ang_r_hip, ang_r_knee, ang_l_sh, ang_l_hip, ang_l_knee = clip_joint_angles(
            ang_r_sh, ang_r_hip, ang_r_knee, ang_l_sh, ang_l_hip, ang_l_knee
        )

        angles = np.array([
            ang_r_sh, ang_r_hip, ang_r_knee,
            ang_l_sh, ang_l_hip, ang_l_knee
        ])

        feature_vec = np.concatenate([coords, angles])
        frame_features.append(feature_vec)

    cap.release()
    return np.array(frame_features)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    video_dir = "traindata/processed_videos"
    out_dir = "testdata/yolomarkers"
    os.makedirs(out_dir, exist_ok=True)

    model = YOLO("yolov8s-pose.pt")

    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
    print(f"Total videos: {len(videos)}")

    for i, f in enumerate(videos, start=1):
        base = os.path.splitext(f)[0]
        out_path = os.path.join(out_dir, f"{base}.csv")

        if os.path.exists(out_path):
            print(f"[{i}/{len(videos)}] ⏭ 已存在，跳過 {base}.csv")
            continue

        print(f"[{i}/{len(videos)}] Processing {f}")

        video_path = os.path.join(video_dir, f)

        try:
            f1 = extract_markers_from_yolo(video_path, model)
            pd.DataFrame(f1).to_csv(out_path, index=False, header=False)
        except Exception as e:
            print(f"Error: {f}: {e}")

    print("Finished extracting YOLO markers.")


if __name__ == "__main__":
    main()

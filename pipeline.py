"""
pipeline.py – Intelligent Deadlift Diagnosis System processing backend.

Every YOLO model runs exactly once per frame (no redundant passes).
Pipeline:
  1. Scan video → nose Y + equipment Y for rep detection
  2. Per rep, sample 20 frames → run pose + seg once each → f1 (30-dim) + back features (3-dim)
  3. Classify with fold-5 TF1 / TF2 / XGBoost ensemble
  4. Generate rule-based feedback
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Auto-select best available device: MPS (Apple Silicon) > CUDA > CPU
def _best_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _odd_window(n, cap=21, floor=5):
    w = min(cap, max(floor, n if n % 2 == 1 else n - 1))
    if w < 3:
        w = 3
    if w % 2 == 0:
        w -= 1
    return max(3, w)


def _interp_nans(y):
    out = y.astype(float).copy()
    valid = ~np.isnan(out)
    if valid.sum() == 0:
        return out
    if valid.sum() == 1:
        out[:] = out[valid][0]
        return out
    x = np.arange(len(out))
    out[~valid] = np.interp(x[~valid], x[valid], out[valid])
    return out

DEVICE = _best_device()

# Equipment class indices from data.yaml: Barbell=5, Dumbbell=9, Gymball=10, Kettlebell=12
EQUIP_CLASSES = {5, 9, 10, 12}
NUM_FRAMES    = 20

REP_NOSE_SMOOTH_WIN_CAP = 21
REP_NOSE_PROM_FRAC = 0.01
REP_MIN_DISTANCE_SEC = 0.3
REP_GRACE_SEC = 3.0
REP_MIN_REP_SEC = 0.7
REP_MAX_REP_SEC = 8.0

# ── YOLO-COCO keypoint layout ────────────────────────────────────────────────
YOLO_KEYS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
J_TARGET = [
    "right_shoulder", "right_elbow", "right_wrist",
    "right_hip",      "right_knee",  "right_ankle",
    "left_shoulder",  "left_elbow",  "left_wrist",
    "left_hip",       "left_knee",   "left_ankle",
]
KP_IDX = {n: YOLO_KEYS.index(n) for n in J_TARGET}

# Angle normalisation bounds (radians)
_SMIN, _SMAX = np.deg2rad(0),  np.deg2rad(170)
_HMIN, _HMAX = np.deg2rad(45), np.deg2rad(180)
_KMIN, _KMAX = np.deg2rad(40), np.deg2rad(180)


# ════════════════════════════════════════════════════════════════════════════
# Neural-network definitions (identical to train_model.py)
# ════════════════════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe  = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# TransformerEncoder matches train_model.py attribute names exactly
class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        n = self.norm1(x)
        x = x + self.attn(n, n, n)[0]
        return x + self.mlp(self.norm2(x))


class TF1(nn.Module):
    def __init__(self, input_dim=30, embed_dim=64, depth=3, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pe         = PositionalEncoding(embed_dim)
        self.blocks     = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.pose_head = nn.Linear(embed_dim, embed_dim)
        self.view_head = nn.Linear(embed_dim, embed_dim)
        self.fc_out    = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 66)
        )

    def forward(self, x):
        x = self.pe(self.input_proj(x))
        for b in self.blocks:
            x = b(x)
        g  = x.mean(1)
        zp = self.pose_head(g)
        zv = self.view_head(g)
        zv_exp = zv.unsqueeze(1).expand(-1, x.size(1), -1)
        return self.fc_out(torch.cat([x, zv_exp], -1)), zp, zv


class TF2(nn.Module):
    def __init__(self, f1_dim=30, f3_dim=66, embed_dim=128, depth=3, num_heads=4, dropout=0.3):
        super().__init__()
        def _proj(d):
            return nn.Sequential(
                nn.Linear(d, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout)
            )
        self.f1_proj            = _proj(f1_dim)
        self.f3_proj            = _proj(f3_dim)
        self.f1_pe              = PositionalEncoding(embed_dim)
        self.f3_pe              = PositionalEncoding(embed_dim)
        self.f1_to_f3_attn      = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_gate         = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim), nn.Sigmoid()
        )
        self.cross_attn_dropout = nn.Dropout(dropout)
        self.cross_attn_norm    = nn.LayerNorm(embed_dim)
        self.blocks             = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(depth)
        ])
        self.temporal_pool = nn.AvgPool1d(3, 2, 1)
        self.cls_head      = nn.Sequential(
            nn.Linear(embed_dim * 2, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 2)
        )

    def forward(self, f1, f3):
        f1 = self.f1_pe(self.f1_proj(f1))
        f3 = self.f3_pe(self.f3_proj(f3))
        a, _ = self.f1_to_f3_attn(f1, f3, f3, need_weights=False)
        g = self.cross_gate(torch.cat([f1, a], -1))
        x = self.cross_attn_norm(f1 + self.cross_attn_dropout(g * a))
        for b in self.blocks:
            x = b(x)
        x = self.temporal_pool(x.transpose(1, 2)).transpose(1, 2)
        return self.cls_head(torch.cat([x.mean(1), x.max(1)[0]], 1))


# ════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ════════════════════════════════════════════════════════════════════════════
def _joint_angle(a, b, c):
    """Angle at vertex b."""
    ba = a - b;  bc = c - b
    cos = np.clip(
        np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8), -1, 1
    )
    return float(np.arccos(cos))


def _cross2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _farthest_intersect(base, contour, p1, p2, eps=1e-9):
    """Return the intersection of line p1-p2 with contour, farthest from base."""
    p1 = np.asarray(p1, float);  p2 = np.asarray(p2, float)
    r  = p2 - p1
    q1 = np.asarray(contour, float)
    q2 = np.roll(q1, -1, axis=0)
    s  = q2 - q1
    rb = np.broadcast_to(r, s.shape)
    rxs = _cross2d(rb, s)
    qp  = q1 - p1
    d   = np.where(np.abs(rxs) < eps, np.inf, rxs)
    t   = _cross2d(qp, s)  / d
    u   = _cross2d(qp, rb) / d
    ok  = (np.abs(rxs) > eps) & (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1)
    if not ok.any():
        return None
    pts = p1 + t[ok, None] * r
    return pts[np.argmax(np.linalg.norm(pts - base, axis=1))]


def _back_curve_kv(mask_np, shoulder, hip, nose=None, eps=1e-6):
    """
    Compute back-curve curvature k and orientation string.
    Returns (k: float|None, orient: 'nose'|'back'|None)
    """
    H, W = mask_np.shape[:2]
    a = np.array(shoulder, float)
    c = np.array(hip,      float)
    tv = c - a
    tn = np.linalg.norm(tv)
    if tn < eps:
        return None, None

    ut   = tv / tn
    up   = np.array([-ut[1], ut[0]])
    mid  = (a + c) / 2.0
    side = 1.0
    if nose is not None:
        s = float(np.sign(np.dot(np.array(nose, float) - mid, up)))
        side = s if s != 0 else 1.0
    back_dir = -side * up

    bw = tn * 1.2
    hh = tn * 1.5 / 2.0
    rect = np.array([
        mid - ut * hh,
        mid - ut * hh + back_dir * bw,
        mid + ut * hh + back_dir * bw,
        mid + ut * hh,
    ], dtype=np.int32)

    mb = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mb, [rect], 1)
    masked = (mask_np > 0).astype(np.uint8) & mb

    cnts, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None
    cnt = max(cnts, key=cv2.contourArea).squeeze()
    if cnt.ndim != 2 or len(cnt) < 3:
        return None, None

    def _lpts(p):
        return p - up * 2000, p + up * 2000

    P = _farthest_intersect(a, cnt, *_lpts(a))
    Q = _farthest_intersect(c, cnt, *_lpts(c))
    if P is None or Q is None:
        return None, None

    ip = int(np.argmin(np.linalg.norm(cnt - P, axis=1)))
    iq = int(np.argmin(np.linalg.norm(cnt - Q, axis=1)))
    if ip < iq:
        c1 = cnt[ip:iq + 1];  c2 = np.concatenate([cnt[iq:], cnt[:ip + 1]])
    else:
        c1 = cnt[iq:ip + 1];  c2 = np.concatenate([cnt[ip:], cnt[:iq + 1]])

    l1 = float(np.sum(np.linalg.norm(np.diff(c1, axis=0), axis=1)))
    l2 = float(np.sum(np.linalg.norm(np.diff(c2, axis=0), axis=1)))
    bc = c1 if l1 < l2 else c2

    b_pt  = a + 0.57 * (c - a)
    AP    = bc[int(np.argmin(np.linalg.norm(bc - a,    axis=1)))]
    BR    = bc[int(np.argmin(np.linalg.norm(bc - b_pt, axis=1)))]
    CQ    = bc[int(np.argmin(np.linalg.norm(bc - c,    axis=1)))]

    # 3-point circle fit
    p1, p2, p3 = AP, BR, CQ
    tmp = p2[0]**2 + p2[1]**2
    bc_ = (p1[0]**2 + p1[1]**2 - tmp) / 2.0
    cd  = (tmp - p3[0]**2 - p3[1]**2)  / 2.0
    det = (p1[0]-p2[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p2[1])
    if abs(det) < eps:
        return None, None
    cx = (bc_ * (p2[1]-p3[1]) - cd * (p1[1]-p2[1])) / det
    cy = ((p1[0]-p2[0]) * cd - (p2[0]-p3[0]) * bc_) / det
    R  = float(np.linalg.norm(AP - np.array([cx, cy])))

    PQ = float(np.linalg.norm(CQ - AP))
    k  = (1.0 / max(R, eps)) * PQ

    orient = None
    if nose is not None:
        cv = np.array([cx, cy]) - mid
        rel = float(np.dot(cv, up)) * side
        orient = "nose" if rel > 0 else "back"

    return k, orient


# ════════════════════════════════════════════════════════════════════════════
# DiagnosisEngine
# ════════════════════════════════════════════════════════════════════════════
class DiagnosisEngine:
    """
    Load once; call process_video() for each video.
    progress_cb(message: str, percent: int)
    """

    def __init__(self):
        self._loaded = False
        self._last_nose_trace = None  # (sample_frames: np.ndarray, nose_y: np.ndarray) | None

    # ── model loading ────────────────────────────────────────────────────────
    def load_models(self, progress_cb=None):
        def _p(msg, pct):
            if progress_cb:
                progress_cb(msg, pct)

        _p(f"Loading YOLO pose model… (device: {DEVICE})", 5)
        self.pose  = YOLO(os.path.join(BASE_DIR, "yolo_models", "yolov11s-pose.pt"))

        _p("Loading YOLO segmentation model…", 12)
        self.seg   = YOLO(os.path.join(BASE_DIR, "yolo_models", "yolov11s-seg.pt"))

        _p("Loading YOLO equipment model…", 19)
        self.equip = YOLO(os.path.join(BASE_DIR, "yolo_models", "yolov11s-gymequipment.pt"))

        _p("Loading TF1 (fold 5)…", 25)
        self.tf1 = TF1(embed_dim=32, depth=1).to(DEVICE)
        self.tf1.load_state_dict(
            torch.load(
                os.path.join(BASE_DIR, "inference_models", "tf1_best_fold1.pt"),
                map_location=DEVICE,
            )
        )
        self.tf1.eval()

        _p("Loading TF2 (fold 5)…", 33)
        ckpt2 = torch.load(
            os.path.join(BASE_DIR, "inference_models", "tf2_best_fold1.pt"),
            map_location=DEVICE,
        )
        self.tf2 = TF2(embed_dim=96, depth=1).to(DEVICE)
        self.tf2.load_state_dict(ckpt2["model"])
        self.tf2_thr = ckpt2["threshold"]   # [hip_thr, knee_thr]
        self.tf2.eval()

        _p("Loading XGBoost (fold 5)…", 40)
        self.xgb = XGBClassifier()
        self.xgb.load_model(
            os.path.join(BASE_DIR, "inference_models", "xgb_rounded_fold5.json")
        )
        thr_path = os.path.join(BASE_DIR, "inference_models", "xgb_rounded_fold5_thr.txt")
        with open(thr_path) as f:
            self.xgb_thr = float(f.read().strip())

        self._loaded = True
        _p("All models loaded.", 45)

    # ── public API ───────────────────────────────────────────────────────────
    def process_video(self, video_path, progress_cb=None):
        """
        Returns
        -------
        reps    : list[(start_frame, end_frame)]
        results : list[np.ndarray([c, h, k, r])]  – one per rep
        """
        if not self._loaded:
            self.load_models(progress_cb)

        def _p(msg, pct):
            if progress_cb:
                progress_cb(msg, pct)

        _p("Scanning video for repetitions…", 48)
        reps = self._detect_reps(video_path)

        if not reps:
            _p("No repetitions detected.", 100)
            return [], []

        _p(f"Detected {len(reps)} rep(s). Extracting features…", 55)
        results = []
        for i, (s, e) in enumerate(reps):
            pct = 55 + int(40 * (i + 1) / max(len(reps), 1))
            _p(f"Classifying rep {i + 1}/{len(reps)}…", pct)
            feat = self._extract_features(video_path, s, e)
            results.append(self._classify(feat))

        _p("Analysis complete.", 100)
        return reps, results

    # ── rep detection ────────────────────────────────────────────────────────
    def _scan_rep_signals(self, video_path, imgsz):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            cap.release()
            return total, fps, [], np.array([], dtype=float), np.array([], dtype=float)

        # Target ~10 sampled frames/second — deadlift motion is slow.
        scan_step = max(1, round(fps / 10))
        sample_frames = list(range(0, total, scan_step))
        sample_count = len(sample_frames)

        nose_y = np.full(sample_count, np.nan, dtype=float)
        equip_y = np.full(sample_count, np.nan, dtype=float)
        wrist_xy = [None] * sample_count

        for si, fi in enumerate(sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            try:
                frame = np.ascontiguousarray(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception:
                continue

            pr = self.pose(rgb, verbose=False, device=DEVICE, imgsz=imgsz)[0]
            if (pr.keypoints is not None
                    and pr.keypoints.xy is not None
                    and pr.keypoints.xy.shape[0] > 0):
                kpts_all = pr.keypoints.xy.cpu().numpy()
                if pr.boxes is not None and len(pr.boxes) > 1:
                    boxes = pr.boxes.xyxy.cpu().numpy()
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    pi = int(np.argmax(areas))
                else:
                    pi = 0
                kp = kpts_all[pi]
                nose_y[si] = kp[0, 1]
                lw = kp[YOLO_KEYS.index("left_wrist")]
                rw = kp[YOLO_KEYS.index("right_wrist")]
                wrist_xy[si] = (lw + rw) / 2.0

            er = self.equip(rgb, verbose=False, device=DEVICE, imgsz=imgsz)[0]
            if er.boxes is not None and len(er.boxes) > 0:
                boxes_e = er.boxes.xyxy.cpu().numpy()
                cls_e = er.boxes.cls.cpu().numpy().astype(int)
                valid = [i for i, c in enumerate(cls_e) if c in EQUIP_CLASSES]
                if valid:
                    if wrist_xy[si] is not None:
                        wx, wy = wrist_xy[si]

                        def _wdist(i):
                            b = boxes_e[i]
                            return (0.5 * (b[0] + b[2]) - wx) ** 2 + (0.5 * (b[1] + b[3]) - wy) ** 2

                        best = min(valid, key=_wdist)
                    else:
                        best = valid[0]
                    b = boxes_e[best]
                    equip_y[si] = 0.5 * (b[1] + b[3])

        cap.release()
        return total, fps, sample_frames, nose_y, equip_y

    def _build_rep_ranges(self, sample_frames, nose_y, equip_y, fps, polarity):
        sample_rate = fps / max(1, sample_frames[1] - sample_frames[0]) if len(sample_frames) > 1 else fps

        equip_start_idx = 0
        ev = ~np.isnan(equip_y)
        if ev.sum() >= 5:
            ey = _interp_nans(equip_y)
            ey_win = _odd_window(len(ey), cap=31, floor=7)
            ey_s = savgol_filter(ey, ey_win, 3) if len(ey) >= ey_win else ey.copy()
            dy = np.abs(np.gradient(ey_s))
            positive = dy[dy > 0]
            thr = np.percentile(positive, 80) if len(positive) else 0.0
            moving = np.where(dy >= thr)[0]
            if len(moving):
                equip_start_idx = int(moving[0])

        ny_win = _odd_window(
            len(nose_y),
            cap=REP_NOSE_SMOOTH_WIN_CAP,
            floor=min(11, REP_NOSE_SMOOTH_WIN_CAP),
        )
        ny_s = savgol_filter(nose_y, ny_win, 3) if len(nose_y) >= ny_win else nose_y.copy()
        signal_range = float(np.percentile(ny_s, 95) - np.percentile(ny_s, 5))
        min_prom = max(1.0, signal_range * REP_NOSE_PROM_FRAC)
        min_distance = max(1, int(round(sample_rate * REP_MIN_DISTANCE_SEC)))
        mean_ny = float(np.mean(ny_s))
        loose_mean_delta = 0.03 * signal_range

        if polarity == "max":
            raw_peaks, _ = find_peaks(ny_s, distance=min_distance, prominence=min_prom)
            candidates = [int(i) for i in raw_peaks if ny_s[i] > (mean_ny - loose_mean_delta)]
        else:
            raw_peaks, _ = find_peaks(-ny_s, distance=min_distance, prominence=min_prom)
            candidates = [int(i) for i in raw_peaks if ny_s[i] < (mean_ny + loose_mean_delta)]

        if len(candidates) < 2:
            return []

        grace_idx = int(round(sample_rate * REP_GRACE_SEC))
        filtered = [v for v in candidates if v >= max(0, equip_start_idx - grace_idx)]
        if len(filtered) < 2:
            filtered = candidates

        min_f = max(1, int(round(sample_rate * REP_MIN_REP_SEC)))
        max_f = max(min_f + 1, int(round(sample_rate * REP_MAX_REP_SEC)))
        reps = []
        for i in range(len(filtered) - 1):
            s_idx, e_idx = filtered[i], filtered[i + 1]
            if min_f <= (e_idx - s_idx) <= max_f:
                reps.append((sample_frames[s_idx], sample_frames[e_idx]))

        return reps

    def _detect_reps(self, video_path):
        self._last_nose_trace = None
        total, fps, sample_frames, nose_y, equip_y = self._scan_rep_signals(video_path, imgsz=320)
        if total < 2:
            return []

        valid = ~np.isnan(nose_y)
        if valid.sum() < 5:
            total, fps, sample_frames, nose_y, equip_y = self._scan_rep_signals(video_path, imgsz=640)
            valid = ~np.isnan(nose_y)
            if valid.sum() < 5:
                return []

        nose_y = _interp_nans(nose_y)
        self._last_nose_trace = (np.array(sample_frames, dtype=int), nose_y.copy())

        # Most clips segment cleanly using screen-space maxima (lowest head position).
        # Fall back to minima for videos whose camera angle or cropping inverts the trend.
        reps = self._build_rep_ranges(sample_frames, nose_y, equip_y, fps, polarity="max")
        if not reps:
            reps = self._build_rep_ranges(sample_frames, nose_y, equip_y, fps, polarity="min")

        return reps

    # ── feature extraction ───────────────────────────────────────────────────
    def _extract_features(self, video_path, start_f, end_f):
        """Returns (20, 33) float array."""
        cap  = cv2.VideoCapture(video_path)
        idxs = np.linspace(start_f, end_f, NUM_FRAMES, dtype=int)

        f1_rows   = []   # 30-dim
        fback_rows = []  # [k, o_enc, v]

        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if not ret or frame is None:
                f1_rows.append(np.full(30, np.nan))
                fback_rows.append([np.nan, 0, np.nan])
                continue
            try:
                frame = np.ascontiguousarray(frame)
                H, W = frame.shape[:2]
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception:
                f1_rows.append(np.full(30, np.nan))
                fback_rows.append([np.nan, 0, np.nan])
                continue

            # ── Pose (once per frame) ────────────────────────────────────────
            pr      = self.pose(rgb, verbose=False, device=DEVICE)[0]
            kp      = None
            kp_conf = None
            box     = None

            if (pr.keypoints is not None
                    and pr.keypoints.xy is not None
                    and pr.keypoints.xy.shape[0] > 0):
                kpts_all = pr.keypoints.xy.cpu().numpy()
                if pr.boxes is not None and len(pr.boxes) > 1:
                    boxes = pr.boxes.xyxy.cpu().numpy()
                    areas = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
                    pi = int(np.argmax(areas))
                else:
                    pi = 0
                kp      = kpts_all[pi]                                        # (17,2)
                kp_conf = (pr.keypoints.conf[pi].cpu().numpy()
                            if pr.keypoints.conf is not None
                            else np.ones(17))
                if pr.boxes is not None and len(pr.boxes) > pi:
                    box = pr.boxes.xyxy[pi].cpu().numpy()

            # ── f1: 24 normalised coords + 6 angles = 30 dims ────────────────
            if kp is not None:
                kn = kp.copy().astype(float)
                kn[:, 0] /= W;  kn[:, 1] /= H
                lhip = kn[YOLO_KEYS.index("left_hip")]
                rhip = kn[YOLO_KEYS.index("right_hip")]
                ctr  = (lhip + rhip) / 2.0
                kn  -= ctr
                ls   = kn[YOLO_KEYS.index("left_shoulder")]
                rs   = kn[YOLO_KEYS.index("right_shoulder")]
                lh   = kn[YOLO_KEYS.index("left_hip")]
                rh   = kn[YOLO_KEYS.index("right_hip")]
                torso = np.linalg.norm(((ls + rs) / 2) - ((lh + rh) / 2))
                if torso > 1e-6:
                    kn /= torso
                jkp    = {n: kn[KP_IDX[n]] for n in J_TARGET}
                coords = np.concatenate([jkp[j] for j in J_TARGET])     # 24

                def _a(a_, b_, c_):
                    return _joint_angle(jkp[a_], jkp[b_], jkp[c_])

                angles = np.array([
                    np.clip(_a("right_elbow", "right_shoulder", "right_hip"), _SMIN, _SMAX) / np.pi,
                    np.clip(_a("right_shoulder", "right_hip",   "right_knee"), _HMIN, _HMAX) / np.pi,
                    np.clip(_a("right_hip",    "right_knee",   "right_ankle"), _KMIN, _KMAX) / np.pi,
                    np.clip(_a("left_elbow",   "left_shoulder", "left_hip"),   _SMIN, _SMAX) / np.pi,
                    np.clip(_a("left_shoulder","left_hip",     "left_knee"),   _HMIN, _HMAX) / np.pi,
                    np.clip(_a("left_hip",     "left_knee",    "left_ankle"),  _KMIN, _KMAX) / np.pi,
                ])
                f1_rows.append(np.concatenate([coords, angles]))
            else:
                f1_rows.append(np.full(30, np.nan))

            # ── Back features: k (curvature), o (orientation), v (view) ──────
            k_val, o_val, v_val = np.nan, 0, np.nan

            if kp is not None and kp_conf is not None:
                # choose the side with higher confidence
                if kp_conf[6] > kp_conf[5] and kp_conf[12] > kp_conf[11]:
                    shoulder, hip = kp[6], kp[12]
                elif kp_conf[5] >= kp_conf[6] and kp_conf[11] >= kp_conf[12]:
                    shoulder, hip = kp[5], kp[11]
                else:
                    shoulder, hip = None, None

                if shoulder is not None:
                    trunk_len = float(np.linalg.norm(shoulder - hip))
                    # view angle = shoulder width / trunk length
                    if kp_conf[5] > 0.3 and kp_conf[6] > 0.3 and trunk_len > 0:
                        v_val = float(np.linalg.norm(kp[5] - kp[6]) / trunk_len)

                    # segmentation (once per frame, alongside pose)
                    sr = self.seg(rgb, verbose=False, device=DEVICE)[0]
                    if sr.masks is not None and box is not None:
                        masks_np = sr.masks.data.cpu().numpy()   # (N, Hm, Wm)
                        x1, y1, x2, y2 = map(int, box[:4])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W, x2), min(H, y2)
                        bx_mask = np.zeros((H, W), bool)
                        if x2 > x1 and y2 > y1:
                            bx_mask[y1:y2, x1:x2] = True

                        best_iou, best_mi = 0.0, -1
                        for mi, m in enumerate(masks_np):
                            mr = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
                            i  = (mr & bx_mask).sum()
                            u  = (mr | bx_mask).sum()
                            iou = i / (u + 1e-6)
                            if iou > best_iou:
                                best_iou, best_mi = iou, mi

                        if best_mi >= 0:
                            m_res = (
                                cv2.resize(masks_np[best_mi], (W, H), interpolation=cv2.INTER_NEAREST)
                            )
                            nose_pt = kp[0] if kp_conf[0] > 0.3 else None
                            kv, orient = _back_curve_kv(m_res, shoulder, hip, nose_pt)
                            if kv is not None:
                                k_val = kv
                                o_val = 1 if orient == "nose" else 0

            fback_rows.append([k_val, o_val, v_val])

        cap.release()

        # ── assemble & fill NaNs ─────────────────────────────────────────────
        f1 = np.array(f1_rows,   dtype=float)   # (20, 30)
        fb = np.array(fback_rows, dtype=float)  # (20, 3)

        # k: fill NaN → 0.5
        fb[:, 0] = np.where(np.isnan(fb[:, 0]), 0.5, fb[:, 0])
        # v: fill NaN → column mean or 0
        vm   = fb[:, 2]
        vmean = float(np.nanmean(vm)) if not np.all(np.isnan(vm)) else 0.0
        fb[:, 2] = np.where(np.isnan(vm), vmean, vm)

        # f1: fill NaN → column mean or 0
        col_means = np.where(np.isnan(np.nanmean(f1, 0)), 0.0, np.nanmean(f1, 0))
        nan_mask  = np.isnan(f1)
        f1[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        return np.concatenate([f1, fb], axis=1).astype(np.float32)   # (20, 33)

    # ── inference ────────────────────────────────────────────────────────────
    def _classify(self, f5):
        """Returns np.array([correct, hip, knee, rounded], dtype=int)."""
        f1_30 = torch.tensor(f5[:, :30]).unsqueeze(0).to(DEVICE)   # (1, 20, 30)

        with torch.no_grad():
            f3, _, _ = self.tf1(f1_30)                              # (1, 20, 66)
            L = min(f1_30.shape[1], f3.shape[1])
            logits = self.tf2(f1_30[:, :L], f3[:, :L])
            prob   = torch.sigmoid(logits).cpu().numpy()[0]   # [hip_p, knee_p]

        thr      = self.tf2_thr
        hip_pred  = int(prob[0] >= thr[0])
        knee_pred = int(prob[1] >= thr[1])

        k = f5[:, 30];  o = f5[:, 31];  v = f5[:, 32]
        kov60   = np.concatenate([k, o, v]).reshape(1, -1)
        p_round = float(self.xgb.predict_proba(kov60)[0, 1])
        round_pred = int(p_round >= self.xgb_thr)

        if hip_pred + knee_pred + round_pred == 0:
            return np.array([1, 0, 0, 0], int)
        return np.array([0, hip_pred, knee_pred, round_pred], int)

    # ── feedback ─────────────────────────────────────────────────────────────
    @staticmethod
    def generate_feedback(results):
        """
        results : list of np.array([c, h, k, r])
        Returns a feedback string following the proposal rules.
        """
        if not results:
            return "No repetitions were detected."

        MSGS = {
            "h": ("Your hips rose too early. "
                  "Focus on driving through your legs and raise your chest simultaneously."),
            "k": ("You push your knees too forward and your trunk too upright. "
                  "Work on hinging your hips."),
            "r": "Brace the core and tighten your lower back.",
        }

        n      = len(results)
        errs   = [
            [key for key in ("h", "k", "r") if results[i][{"h":1,"k":2,"r":3}[key]]]
            for i in range(n)
        ]
        ok     = [not bool(e) for e in errs]

        def _etxt(keys):
            return "  ".join(MSGS[k] for k in keys)

        # ── 1 rep ────────────────────────────────────────────────────────────
        if n == 1:
            return "You did it well!" if ok[0] else _etxt(errs[0])

        # ── 2 reps ───────────────────────────────────────────────────────────
        if n == 2:
            if ok[0] and ok[1]:
                return "You did both reps well!"
            if ok[0]:
                return f"You did rep 1 well, but the other rep — {_etxt(errs[1])}"
            if ok[1]:
                return f"You did rep 2 well, but the other rep — {_etxt(errs[0])}"
            return (f"For the first rep: {_etxt(errs[0])}\n"
                    f"For the second rep: {_etxt(errs[1])}")

        # ── ≥3 reps: split into thirds ───────────────────────────────────────
        sz  = n // 3
        parts = {
            "beginning": list(range(0, sz)),
            "middle":    list(range(sz, 2 * sz)),
            "end":       list(range(2 * sz, n)),
        }

        def _vote(idx_list):
            arr  = np.array([results[i] for i in idx_list])
            vote = (arr.sum(0) >= len(idx_list) / 2).astype(int)
            if vote[1] + vote[2] + vote[3] == 0:
                vote[0] = 1
            return vote

        part_res  = {p: _vote(idx) for p, idx in parts.items()}
        part_errs = {
            p: [key for key in ("h", "k", "r") if part_res[p][{"h":1,"k":2,"r":3}[key]]]
            for p in parts
        }
        part_ok = {p: not bool(part_errs[p]) for p in parts}

        if all(part_ok.values()):
            return "You did well on all reps!"

        good_parts = [p for p in parts if part_ok[p]]
        bad_parts  = [p for p in parts if not part_ok[p]]

        if good_parts:
            good_str = " and the ".join(good_parts)
            lines = [f"You did well in the {good_str}."]
            for p in bad_parts:
                lines.append(f"In the {p}: {_etxt(part_errs[p])}")
            return "\n".join(lines)

        # all parts wrong
        err_sets = [frozenset(part_errs[p]) for p in parts]
        if err_sets[0] == err_sets[1] == err_sets[2]:
            return f"For the whole set: {_etxt(list(err_sets[0]))}"

        lines = []
        for p in parts:
            if part_errs[p]:
                lines.append(f"In the {p}: {_etxt(part_errs[p])}")
        return "\n".join(lines) if lines else "No clear pattern detected."

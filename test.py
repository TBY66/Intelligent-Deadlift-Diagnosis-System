"""
Standalone rep-count test script.

Purpose:
  Count deadlift repetitions only, using a more conservative version of the
  rep-detection logic from pipeline.py.

Main differences from pipeline.py:
  - work in sampled-time domain instead of filling a full-frame array
  - interpolate missing nose/equipment values on sampled frames
  - use stronger smoothing + prominence filtering
  - require larger valley separation to avoid over-splitting one rep
  - gate valleys using equipment movement start

Example:
  python3 test.py --video sample/0016a_1.mp4
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: opencv-python\n"
        "Install with:\n"
        "  python3 -m pip install opencv-python"
    ) from exc

import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: torch\n"
        "Install with:\n"
        "  python3 -m pip install torch torchvision torchaudio\n"
        "\n"
        "If you also need the full script dependencies, run:\n"
        "  python3 -m pip install torch torchvision torchaudio ultralytics scipy tqdm opencv-python"
    ) from exc

from scipy.signal import find_peaks, savgol_filter
from tqdm.auto import tqdm

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: ultralytics\n"
        "Install with:\n"
        "  python3 -m pip install ultralytics"
    ) from exc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL = os.path.join(BASE_DIR, "yolo_models", "yolov11s-pose.pt")
EQUIP_MODEL = os.path.join(BASE_DIR, "yolo_models", "yolov11s-gymequipment.pt")
YOLO_KEYS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
EQUIP_CLASSES = {5, 9, 10, 12}


def best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def odd_window(n: int, cap: int = 21, floor: int = 5) -> int:
    w = min(cap, max(floor, n if n % 2 == 1 else n - 1))
    if w < 3:
        w = 3
    if w % 2 == 0:
        w -= 1
    return max(3, w)


def interp_nans(y: np.ndarray) -> np.ndarray:
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


@dataclass
class RepDetectionResult:
    rep_ranges: list[tuple[int, int]]
    valley_frames: list[int]
    equip_start_frame: int
    total_frames: int
    fps: float


NOSE_SMOOTH_WIN_CAP = 21
NOSE_PROM_FRAC = 0.01
MIN_DISTANCE_SEC = 0.3
GRACE_SEC = 3.0
MIN_REP_SEC = 0.7
MAX_REP_SEC = 8.0


class RepCounter:
    def __init__(self, device: str):
        self.device = device
        self.pose = YOLO(POSE_MODEL)
        self.equip = YOLO(EQUIP_MODEL)

    def count_reps(self, video_path: str) -> RepDetectionResult:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            cap.release()
            return RepDetectionResult([], [], 0, total, fps)

        scan_step = max(1, round(fps / 10))
        sample_frames = list(range(0, total, scan_step))
        sample_count = len(sample_frames)

        nose_y = np.full(sample_count, np.nan, dtype=float)
        equip_y = np.full(sample_count, np.nan, dtype=float)
        wrist_xy: list[np.ndarray | None] = [None] * sample_count

        for si, fi in enumerate(tqdm(sample_frames, desc="Scanning video", unit="frame")):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pr = self.pose(rgb, verbose=False, device=self.device, imgsz=320)[0]
            if (
                pr.keypoints is not None
                and pr.keypoints.xy is not None
                and pr.keypoints.xy.shape[0] > 0
            ):
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

            er = self.equip(rgb, verbose=False, device=self.device, imgsz=320)[0]
            if er.boxes is not None and len(er.boxes) > 0:
                boxes_e = er.boxes.xyxy.cpu().numpy()
                cls_e = er.boxes.cls.cpu().numpy().astype(int)
                valid = [i for i, c in enumerate(cls_e) if c in EQUIP_CLASSES]
                if valid:
                    if wrist_xy[si] is not None:
                        wx, wy = wrist_xy[si]

                        def wrist_dist(i: int) -> float:
                            b = boxes_e[i]
                            cx = 0.5 * (b[0] + b[2])
                            cy = 0.5 * (b[1] + b[3])
                            return (cx - wx) ** 2 + (cy - wy) ** 2

                        best = min(valid, key=wrist_dist)
                    else:
                        best = valid[0]
                    b = boxes_e[best]
                    equip_y[si] = 0.5 * (b[1] + b[3])

        cap.release()

        valid_nose = ~np.isnan(nose_y)
        if valid_nose.sum() < 5:
            return RepDetectionResult([], [], 0, total, fps)

        nose_y = interp_nans(nose_y)
        equip_has_signal = (~np.isnan(equip_y)).sum() >= 5
        if equip_has_signal:
            equip_y = interp_nans(equip_y)

        sample_rate = fps / scan_step

        equip_start_idx = 0
        if equip_has_signal:
            ey_win = odd_window(len(equip_y), cap=31, floor=7)
            ey_s = savgol_filter(equip_y, ey_win, 3) if len(equip_y) >= ey_win else equip_y.copy()
            dy = np.abs(np.gradient(ey_s))
            positive = dy[dy > 0]
            if len(positive):
                thr = np.percentile(positive, 80)
                moving = np.where(dy >= thr)[0]
                if len(moving):
                    equip_start_idx = int(moving[0])

        ny_win = odd_window(len(nose_y), cap=NOSE_SMOOTH_WIN_CAP, floor=min(11, NOSE_SMOOTH_WIN_CAP))
        ny_s = savgol_filter(nose_y, ny_win, 3) if len(nose_y) >= ny_win else nose_y.copy()

        signal_range = float(np.percentile(ny_s, 95) - np.percentile(ny_s, 5))
        min_prom = max(1.0, signal_range * NOSE_PROM_FRAC)
        min_distance = max(1, int(round(sample_rate * MIN_DISTANCE_SEC)))
        raw_min, _ = find_peaks(-ny_s, distance=min_distance, prominence=min_prom)

        mean_ny = float(np.mean(ny_s))
        loose_mean = mean_ny + 0.03 * signal_range
        minima = [int(i) for i in raw_min if ny_s[i] < loose_mean]
        if len(minima) < 2:
            return RepDetectionResult([], [], sample_frames[equip_start_idx] if sample_frames else 0, total, fps)

        grace_idx = int(round(sample_rate * GRACE_SEC))
        filtered = [i for i in minima if i >= max(0, equip_start_idx - grace_idx)]
        if len(filtered) < 2:
            filtered = minima

        min_rep_idx = max(1, int(round(sample_rate * MIN_REP_SEC)))
        max_rep_idx = max(min_rep_idx + 1, int(round(sample_rate * MAX_REP_SEC)))

        rep_ranges: list[tuple[int, int]] = []
        valley_frames: list[int] = [sample_frames[i] for i in filtered]
        for a, b in zip(filtered[:-1], filtered[1:]):
            if min_rep_idx <= (b - a) <= max_rep_idx:
                rep_ranges.append((sample_frames[a], sample_frames[b]))

        return RepDetectionResult(
            rep_ranges=rep_ranges,
            valley_frames=valley_frames,
            equip_start_frame=sample_frames[equip_start_idx] if sample_frames else 0,
            total_frames=total,
            fps=fps,
        )


def main():
    parser = argparse.ArgumentParser(description="Count reps only with the adjusted detection logic.")
    parser.add_argument("--video", required=True, help="Path to input video")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = best_device()
    print(f"Using device: {device}")
    print(f"Video: {video_path}")

    counter = RepCounter(device)
    result = counter.count_reps(video_path)

    print()
    print(f"Total frames: {result.total_frames}")
    print(f"FPS: {result.fps:.2f}")
    print(f"Equipment movement starts around frame: {result.equip_start_frame}")
    print(f"Valley frames: {result.valley_frames}")
    print(f"Rep count: {len(result.rep_ranges)}")
    for idx, (start_f, end_f) in enumerate(result.rep_ranges, start=1):
        start_t = start_f / result.fps
        end_t = end_f / result.fps
        print(
            f"Rep {idx:02d}: frames {start_f:5d} -> {end_f:5d} "
            f"({start_t:6.2f}s -> {end_t:6.2f}s)"
        )


if __name__ == "__main__":
    main()

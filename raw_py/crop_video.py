import cv2
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN
import os
import warnings
warnings.filterwarnings('ignore')

###########################
# 可調參數
###########################
TRC_MARKER_NAME = "Neck"   # ← marker名稱
FPS = 60                    # ← 依你的資料是 60 Hz

# 關節角度計算所需的 markers
HIP_MARKERS = ["RShoulder", "RHip", "RKnee"]  # 用於計算髖關節角度
KNEE_MARKERS = ["RHip", "RKnee", "RAnkle"]    # 用於計算膝關節角度


###########################
# 讀取 TRC
###########################
def read_trc(filename):
    df = pd.read_csv(filename, sep="\t", header=3)
    df = df.dropna(how="all")
    df.columns = df.columns.str.strip()
    df.rename(columns={"Frame#":"Frame"}, inplace=True)
    return df

###########################
# 找出 marker 的 X,Y,Z 欄位名稱
###########################
def trc_marker_columns(df, marker_name):
    cols = df.columns.tolist()
    # index of marker (skip Frame & Time ⇒ starting from col=2)
    marker_index = cols.index(marker_name) - 2
    X = cols[2 + marker_index*3 + 0]
    Y = cols[2 + marker_index*3 + 1]
    Z = cols[2 + marker_index*3 + 2]
    return X, Y, Z

###########################
# 取某 marker 的 Y 座標
###########################
def get_marker_Y(df, marker_name):
    _, Ycol, _ = trc_marker_columns(df, marker_name)
    y = pd.to_numeric(df[Ycol], errors='coerce')
    y = y.fillna(method='ffill').fillna(method='bfill')
    return y.values.astype(float)

###########################
# 取某 marker 的 3D 座標
###########################
def get_marker_3D(df, marker_name):
    """獲取某個 marker 的 X, Y, Z 座標"""
    try:
        Xcol, Ycol, Zcol = trc_marker_columns(df, marker_name)
        x = pd.to_numeric(df[Xcol], errors='coerce').fillna(method='ffill').fillna(method='bfill')
        y = pd.to_numeric(df[Ycol], errors='coerce').fillna(method='ffill').fillna(method='bfill')
        z = pd.to_numeric(df[Zcol], errors='coerce').fillna(method='ffill').fillna(method='bfill')
        return np.column_stack([x.values, y.values, z.values])
    except:
        return None

###########################
# 計算三點形成的角度
###########################
def calculate_angle(p1, p2, p3):
    """
    計算由三點形成的角度 (p2 是頂點)
    p1, p2, p3: (N, 3) arrays
    返回: (N,) array of angles in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # 計算每個時間點的角度
    angles = []
    for i in range(len(v1)):
        cos_angle = np.dot(v1[i], v2[i]) / (np.linalg.norm(v1[i]) * np.linalg.norm(v2[i]))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        angles.append(np.degrees(angle))
    
    return np.array(angles)

###########################
# 計算關節角度軌跡
###########################
def get_joint_angles(df):
    """計算髖關節和膝關節角度"""
    angles = {}
    
    # Hip angle
    try:
        shoulder = get_marker_3D(df, HIP_MARKERS[0])
        hip = get_marker_3D(df, HIP_MARKERS[1])
        knee = get_marker_3D(df, HIP_MARKERS[2])
        if shoulder is not None and hip is not None and knee is not None:
            angles['hip'] = calculate_angle(shoulder, hip, knee)
        else:
            angles['hip'] = None
    except:
        angles['hip'] = None
    
    # Knee angle
    try:
        hip = get_marker_3D(df, KNEE_MARKERS[0])
        knee = get_marker_3D(df, KNEE_MARKERS[1])
        ankle = get_marker_3D(df, KNEE_MARKERS[2])
        if hip is not None and knee is not None and ankle is not None:
            angles['knee'] = calculate_angle(hip, knee, ankle)
        else:
            angles['knee'] = None
    except:
        angles['knee'] = None
    
    return angles

###########################
# 偵測 rep 循環
###########################
def detect_reps(y, hip_angle=None, knee_angle=None, fps=60):
    """
    Deadlift segmentation by valleys only (NO validation):
    - Detect minima of Neck Y
    - DBSCAN decluster → keep deepest valley
    - Adjacent valleys form reps
    """

    y = np.array(y, dtype=float)
    N = len(y)

    # ---------------------------------------
    # smoothing
    # ---------------------------------------
    win = min(21, max(5, N // 8))
    if win % 2 == 0:
        win += 1
    y_s = savgol_filter(y, win, 3) if N > win else y.copy()

    # ---------------------------------------
    # local minima below mean
    # ---------------------------------------
    y_mean = np.mean(y_s)
    raw_minima, _ = find_peaks(-y_s, distance=int(0.10 * fps))
    minima = [i for i in raw_minima if y_s[i] < y_mean]

    if len(minima) < 2:
        print("[valley] insufficient minima")
        return []

    # ---------------------------------------
    # decluster: DBSCAN keep deepest
    # ---------------------------------------
    eps_val = int(0.20 * fps)
    minima_arr = np.array(minima).reshape(-1, 1)

    db = DBSCAN(eps=eps_val, min_samples=1).fit(minima_arr)
    labels = db.labels_

    valleys = []
    for lbl in np.unique(labels):
        cluster = minima_arr[labels == lbl].flatten()
        best = cluster[np.argmin(y_s[cluster])]
        valleys.append(best)

    valleys = sorted(valleys)

    if len(valleys) < 2:
        print("[valley] insufficient valleys after DBSCAN")
        return []

    # ---------------------------------------
    # adjacent valley pairing → reps
    # ---------------------------------------
    reps = []
    for i in range(len(valleys) - 1):
        s = valleys[i]
        e = valleys[i + 1]
        reps.append((s, e))

    # ---------------------------------------
    # duration filtering
    # ---------------------------------------
    min_frames = int(0.30 * fps)
    max_frames = int(10.0 * fps)
    reps = [(s, e) for (s, e) in reps if min_frames <= (e - s) <= max_frames]

    print(f"[valley seg NO-VALID] raw_minima={len(minima)}, valleys={len(valleys)}, reps={len(reps)}")
    return reps

###########################
# 裁切影片
###########################
def cut_video(src, dst, start_frame, end_frame):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(dst, fourcc, fps, (w, h))

    for f in range(start_frame, end_frame+1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    cap.release()

###########################
# 裁切 TRC
###########################
def cut_trc(src, dst, start_frame, end_frame):
    with open(src, "r") as f:
        lines = f.readlines()

    header = lines[:5]         # 前 5 行保留
    data_lines = lines[5:]     # 後面是資料行

    # Parse data lines to dataframe
    from io import StringIO
    df = pd.read_csv(StringIO("".join(data_lines)), sep="\t", header=None)

    # frame column is ALWAYS col=0
    df_cut = df[(df[0] >= start_frame) & (df[0] <= end_frame)].copy()

    # reset frame
    df_cut[0] = range(1, len(df_cut)+1)

    # 修改 NumFrames (第 3 行)
    parts = header[2].split()
    parts[2] = str(len(df_cut))
    header[2] = "\t".join(parts) + "\n"

    # save file
    with open(dst, "w") as f:
        f.writelines(header)
        df_cut.to_csv(f, sep="\t", index=False, header=False)

###########################
# 處理單一 pair
###########################
def process_pair(video_path, trc_path, out_vid_dir, out_trc_dir):
    df = read_trc(trc_path)
    
    # 獲取 wrist Y 座標
    y = get_marker_Y(df, TRC_MARKER_NAME)
    
    # 獲取關節角度
    angles = get_joint_angles(df)
    hip_angle = angles.get('hip')
    knee_angle = angles.get('knee')
    
    # 顯示數據統計幫助調參
    print(f"Data stats - Y range: {np.min(y):.2f} to {np.max(y):.2f}")
    if hip_angle is not None:
        print(f"Hip angle range: {np.min(hip_angle):.2f} to {np.max(hip_angle):.2f}")
    if knee_angle is not None:
        print(f"Knee angle range: {np.min(knee_angle):.2f} to {np.max(knee_angle):.2f}")
    
    # 偵測動作循環
    reps = detect_reps(y, hip_angle, knee_angle, FPS)
    print(f"  Detected reps: {len(reps)}")
    
    base = os.path.splitext(os.path.basename(video_path))[0]
    results = []

    # if no rep found: export full video
    if len(reps) == 0:
        print("  No reps detected, exporting full video...")
        out_mp4 = os.path.join(out_vid_dir, f"{base}_full.mp4")
        out_trc = os.path.join(out_trc_dir, f"{base}_full.trc")
        cut_video(video_path, out_mp4, 0, len(y)-1)
        cut_trc(trc_path, out_trc, 0, len(y)-1)
        return [(out_mp4, out_trc)]

    for idx, (s,e) in enumerate(reps, start=1):
        out_mp4 = os.path.join(out_vid_dir, f"{base}_{s}_{e}.mp4")
        out_trc = os.path.join(out_trc_dir, f"{base}_{s}_{e}.trc")

        cut_video(video_path, out_mp4, s, e)
        cut_trc(trc_path, out_trc, s, e)

        print(f"    Saved rep {idx}: frames {s}-{e} ({(e-s)/FPS:.2f}s)")
        results.append((out_mp4, out_trc))

    return results

###########################
# 主流程 – 全資料夾 run
###########################
def process_all():
    raw_vid_dir = "TrainData/raw_videos"
    raw_trc_dir = "TrainData/raw_markers"
    out_vid_dir = "TrainData/processed_videos"
    out_trc_dir = "TrainData/processed_markers"

    os.makedirs(out_vid_dir, exist_ok=True)
    os.makedirs(out_trc_dir, exist_ok=True)

    # 以 TRC 為主進行掃描
    trc_files = [f for f in os.listdir(raw_trc_dir) if f.lower().endswith(".trc")]

    total = len(trc_files)
    print(f"Total TRC files to process: {total}\n")

    for idx, fname in enumerate(trc_files, start=1):
        base = os.path.splitext(fname)[0]
        trc_path = os.path.join(raw_trc_dir, fname)

        # 找出所有前綴相符的影片，例如 0063k_1.mp4, 0063k_2.mp4
        all_videos = sorted([
            os.path.join(raw_vid_dir, v)
            for v in os.listdir(raw_vid_dir)
            if v.lower().startswith(base.lower()) and v.lower().endswith(".mp4")
        ])

        print(f"[{idx}/{total}] Processing {fname} ... ({(idx/total)*100:.2f}%)")

        # -----------------------------------------------------
        # Step 1. 用 TRC 偵測 reps (只做一次)
        # -----------------------------------------------------
        df = read_trc(trc_path)
        y = get_marker_Y(df, TRC_MARKER_NAME)
        angles = get_joint_angles(df)
        hip_angle = angles.get('hip')
        knee_angle = angles.get('knee')
        reps = detect_reps(y, hip_angle, knee_angle, FPS)

        if len(reps) == 0:
            print("  No reps detected, exporting full TRC & videos...")

            # 1️輸出完整 TRC
            out_trc = os.path.join(out_trc_dir, f"{base}_full.trc")
            cut_trc(trc_path, out_trc, 0, len(y)-1)

            # 2️對應的所有影片全剪
            for vpath in all_videos:
                out_vid = os.path.join(out_vid_dir, os.path.basename(vpath).replace(".mp4", "_full.mp4"))
                cut_video(vpath, out_vid, 0, len(y)-1)

        else:
            # -----------------------------------------------------
            # Step 2. 根據 reps 裁切所有影片與 TRC
            # -----------------------------------------------------
            print(f"  Detected {len(reps)} reps → cutting {len(all_videos)} videos")

            for idx_r, (s, e) in enumerate(reps, start=1):
                # TRC 只剪一次
                out_trc = os.path.join(out_trc_dir, f"{base}_{s}_{e}.trc")
                cut_trc(trc_path, out_trc, s, e)

                # 每支影片都依同樣時間段剪
                for vpath in all_videos:
                    vid_name = os.path.basename(vpath).replace(".mp4", f"_{s}_{e}.mp4")
                    out_vid = os.path.join(out_vid_dir, vid_name)
                    cut_video(vpath, out_vid, s, e)

                print(f"    Saved rep {idx_r}: frames {s}-{e} ({(e-s)/FPS:.2f}s)")

    print("\nAll processing complete!")



###########################
# 執行入口
###########################
if __name__ == "__main__":
    process_all()
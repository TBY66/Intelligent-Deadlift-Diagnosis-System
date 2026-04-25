import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===================== CONFIG =====================
INPUT_CSV_DIR = "traindata/2dprojection"
INPUT_TRC_DIR = "traindata/processed_markers"
INPUT_STO_DIR = "traindata/JCF"

OUT_FEATURE1_DIR = "traindata/features/feature1"
OUT_FEATURE2_DIR = "traindata/features/feature2"
OUT_FEATURE3_DIR = "traindata/features/feature3"

NUM_FRAMES = 20
# ===== force scaling constants =====
HIP_SCALE   = 2.5e5
KNEE_SCALE  = 7e4
ANKLE_SCALE = 4e3

os.makedirs(OUT_FEATURE1_DIR, exist_ok=True)
os.makedirs(OUT_FEATURE2_DIR, exist_ok=True)
os.makedirs(OUT_FEATURE3_DIR, exist_ok=True)

# ===================== JOINTS =====================
J_TARGET = [
    "RShoulder","RElbow","RWrist",
    "RHip","RKnee","RAnkle",
    "LShoulder","LElbow","LWrist",
    "LHip","LKnee","LAnkle"
]

ALIAS = {
    "RShoulder": ["RShoulder", "RightShoulder"],
    "RElbow": ["RElbow", "RightElbow"],
    "RWrist": ["RWrist", "RightWrist"],
    "RHip": ["RHip", "RightHip"],
    "RKnee": ["RKnee", "RightKnee"],
    "RAnkle": ["RAnkle", "RightAnkle"],
    "LShoulder": ["LShoulder", "LeftShoulder"],
    "LElbow": ["LElbow", "LeftElbow"],
    "LWrist": ["LWrist", "LeftWrist"],
    "LHip": ["LHip", "LeftHip"],
    "LKnee": ["LKnee", "LeftKnee"],
    "LAnkle": ["LAnkle", "LeftAnkle"]
}

SHOULDER_MIN_RAD = np.deg2rad(0)
SHOULDER_MAX_RAD = np.deg2rad(170)

HIP_MIN_RAD  = np.deg2rad(45)
HIP_MAX_RAD  = np.deg2rad(180)

KNEE_MIN_RAD = np.deg2rad(40)
KNEE_MAX_RAD = np.deg2rad(180)

# ===============================================================
# compute angle function
# ===============================================================
def compute_angle(A, B, C):
    """
    A, B, C: (..., 2)
    angle at B
    """
    BA = A - B
    BC = C - B

    dot = np.sum(BA * BC, axis=-1)
    norm = np.linalg.norm(BA, axis=-1) * np.linalg.norm(BC, axis=-1) + 1e-6
    cos = np.clip(dot / norm, -1.0, 1.0)
    return np.arccos(cos)  # radians

def compute_angle_3d(A, B, C):
    """
    A, B, C: (..., 3)
    angle at B (3D)
    """
    BA = A - B
    BC = C - B
    dot = np.sum(BA * BC, axis=-1)
    norm = np.linalg.norm(BA, axis=-1) * np.linalg.norm(BC, axis=-1) + 1e-6
    cos = np.clip(dot / norm, -1.0, 1.0)
    return np.arccos(cos)  # radians

def clip_joint_angles(r_sh, r_hip, r_knee, l_sh, l_hip, l_knee):
    r_sh = np.clip(r_sh, SHOULDER_MIN_RAD, SHOULDER_MAX_RAD) / np.pi
    l_sh = np.clip(l_sh, SHOULDER_MIN_RAD, SHOULDER_MAX_RAD) / np.pi

    r_hip = np.clip(r_hip, HIP_MIN_RAD, HIP_MAX_RAD) / np.pi
    l_hip = np.clip(l_hip, HIP_MIN_RAD, HIP_MAX_RAD) / np.pi

    r_knee = np.clip(r_knee, KNEE_MIN_RAD, KNEE_MAX_RAD) / np.pi
    l_knee = np.clip(l_knee, KNEE_MIN_RAD, KNEE_MAX_RAD) / np.pi

    return r_sh, r_hip, r_knee, l_sh, l_hip, l_knee
# ===============================================================
# unified sampling function
# ===============================================================
def sample_indices(n, k=20):
    if n <= 1:
        return np.zeros(k, dtype=int)
    return np.linspace(0, n-1, k, dtype=int)


# ===============================================================
# 1. feature1 — extract & normalize 2D CSV keypoints
# ===============================================================
def extract_markers_from_csv(path):
    df = pd.read_csv(path)

    # ---------- midHip center ----------
    RHx, RHy = df["RHip_x"].values, df["RHip_y"].values
    LHx, LHy = df["LHip_x"].values, df["LHip_y"].values
    mid_x = (RHx + LHx) / 2
    mid_y = (RHy + LHy) / 2

    # ---------- femur normalization ----------
    RHip = np.stack([df["RHip_x"], df["RHip_y"]], axis=1)
    RKnee = np.stack([df["RKnee_x"], df["RKnee_y"]], axis=1)
    LHip = np.stack([df["LHip_x"], df["LHip_y"]], axis=1)
    LKnee = np.stack([df["LKnee_x"], df["LKnee_y"]], axis=1)

    rf = np.linalg.norm(RHip - RKnee, axis=1)
    lf = np.linalg.norm(LHip - LKnee, axis=1)
    femur = (rf + lf) / 2
    femur = np.where(femur == 0, 1e-6, femur)

    # ================== NORMALIZED KEYPOINTS ==================
    kp = {}
    for j in J_TARGET:
        X = (df[f"{j}_x"].values - mid_x) / femur
        Y = (df[f"{j}_y"].values - mid_y) / femur
        kp[j] = np.stack([X, Y], axis=1)

    # ================== ANGLE COMPUTATION ==================
    angles = []

    # Right side
    ang_r_sh = compute_angle(kp["RElbow"], kp["RShoulder"], kp["RHip"])
    ang_r_hip = compute_angle(kp["RShoulder"], kp["RHip"], kp["RKnee"])
    ang_r_knee = compute_angle(kp["RHip"], kp["RKnee"], kp["RAnkle"])

    # Left side
    ang_l_sh = compute_angle(kp["LElbow"], kp["LShoulder"], kp["LHip"])
    ang_l_hip = compute_angle(kp["LShoulder"], kp["LHip"], kp["LKnee"])
    ang_l_knee = compute_angle(kp["LHip"], kp["LKnee"], kp["LAnkle"])

    ang_r_sh, ang_r_hip, ang_r_knee, ang_l_sh, ang_l_hip, ang_l_knee = clip_joint_angles(
        ang_r_sh, ang_r_hip, ang_r_knee, ang_l_sh, ang_l_hip, ang_l_knee
    )

    angles = np.stack(
        [ang_r_sh, ang_r_hip, ang_r_knee, ang_l_sh, ang_l_hip, ang_l_knee],
        axis=1
    )

    angles = np.stack(
        [ang_r_sh, ang_r_hip, ang_r_knee, ang_l_sh, ang_l_hip, ang_l_knee],
        axis=1
    )  # (T, 6)

    # ================== CONCAT FEATURE ==================
    coords = []
    for j in J_TARGET:
        coords += [kp[j][:,0], kp[j][:,1]]
    coords = np.stack(coords, axis=1)  # (T, 24)

    out = np.concatenate([coords, angles], axis=1)  # (T, 30)

    idxs = sample_indices(len(out), NUM_FRAMES)
    return out[idxs]



# ===============================================================
# 2. feature2 — extract raw TRC 3D markers
# ===============================================================
def extract_markers_from_trc(path):
    with open(path, "r") as f:
        lines = f.readlines()

    raw_markers = lines[3].strip().split("\t")[2:]
    markers = [m for m in raw_markers if m.strip() != ""]

    df = pd.read_csv(path, sep="\t", header=4)

    marker2cols = {}
    col_index = 1
    for name in markers:
        marker2cols[name] = (f"X{col_index}", f"Y{col_index}", f"Z{col_index}")
        col_index += 1

    def get(marker):
        if marker in ALIAS:
            for m in ALIAS[marker]:
                if m in marker2cols:
                    X, Y, Z = marker2cols[m]
                    return df[X].values, df[Y].values, df[Z].values
        if marker in marker2cols:
            X, Y, Z = marker2cols[marker]
            return df[X].values, df[Y].values, df[Z].values
        raise KeyError(f"Marker {marker} not found in TRC")

    # ========= 取得所有 joint =========
    joints = {}
    for j in J_TARGET:
        x, y, z = get(j)
        joints[j] = np.stack([x, y, z], axis=1)  # (T,3)

    T = joints["RHip"].shape[0]

    # ========= 1️⃣ midHip center =========
    mid = (joints["RHip"] + joints["LHip"]) / 2.0  # (T,3)

    # ========= 2️⃣ femur normalization =========
    rf = np.linalg.norm(joints["RHip"] - joints["RKnee"], axis=1)
    lf = np.linalg.norm(joints["LHip"] - joints["LKnee"], axis=1)
    femur = (rf + lf) / 2.0
    femur = np.where(femur == 0, 1e-6, femur)

    # ========= 3️⃣ normalize =========
    data = []
    for j in J_TARGET:
        centered = joints[j] - mid          # center
        normalized = centered / femur[:,None]  # scale
        data.append(normalized)

    data = np.stack(data, axis=1)  # (T, 12, 3)
    
    # ========= 3D angles (T,6) =========
    # 角度定義：右/左 shoulder, hip, knee
    # Right side
    ang_r_sh = compute_angle_3d(
        data[:, J_TARGET.index("RElbow"), :],
        data[:, J_TARGET.index("RShoulder"), :],
        data[:, J_TARGET.index("RHip"), :]
    )
    ang_r_hip = compute_angle_3d(
        data[:, J_TARGET.index("RShoulder"), :],
        data[:, J_TARGET.index("RHip"), :],
        data[:, J_TARGET.index("RKnee"), :]
    )
    ang_r_knee = compute_angle_3d(
        data[:, J_TARGET.index("RHip"), :],
        data[:, J_TARGET.index("RKnee"), :],
        data[:, J_TARGET.index("RAnkle"), :]
    )

    # Left side
    ang_l_sh = compute_angle_3d(
        data[:, J_TARGET.index("LElbow"), :],
        data[:, J_TARGET.index("LShoulder"), :],
        data[:, J_TARGET.index("LHip"), :]
    )
    ang_l_hip = compute_angle_3d(
        data[:, J_TARGET.index("LShoulder"), :],
        data[:, J_TARGET.index("LHip"), :],
        data[:, J_TARGET.index("LKnee"), :]
    )
    ang_l_knee = compute_angle_3d(
        data[:, J_TARGET.index("LHip"), :],
        data[:, J_TARGET.index("LKnee"), :],
        data[:, J_TARGET.index("LAnkle"), :]
    )

    # clip hip / knee angles
    ang_r_sh, ang_r_hip, ang_r_knee, ang_l_sh, ang_l_hip, ang_l_knee = clip_joint_angles(
    ang_r_sh, ang_r_hip, ang_r_knee, ang_l_sh, ang_l_hip, ang_l_knee
    )

    angles = np.stack(
        [ang_r_sh, ang_r_hip, ang_r_knee, ang_l_sh, ang_l_hip, ang_l_knee],
        axis=1
    )  # (T,6)

    # 把座標攤平 + angles 接在後面
    data_flat = data.reshape(T, -1)                       # (T,36)
    data_feat = np.concatenate([data_flat, angles], axis=1)  # (T,42)

    return data_feat, T

# ===============================================================
# 3. feature3 — extract force data from JCF STO
# ===============================================================
def load_sto(path):
    with open(path, "r") as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            start = i + 1
            break

    if start is None:
        raise ValueError("Cannot find endheader")

    return pd.read_csv(path, delim_whitespace=True, skiprows=start, engine="python")

EPS = 1e-8

# ===== TRC: 讀取並算整段平均 yaw（circular mean）=====
def read_trc_df_and_marker2cols(trc_path):
    with open(trc_path, "r") as f:
        lines = f.readlines()

    raw_markers = lines[3].strip().split("\t")[2:]
    markers = [m for m in raw_markers if m.strip() != ""]
    df = pd.read_csv(trc_path, sep="\t", header=4)

    marker2cols = {}
    col_index = 1
    for name in markers:
        marker2cols[name] = (f"X{col_index}", f"Y{col_index}", f"Z{col_index}")
        col_index += 1
    return df, marker2cols

def get_trc_marker_xyz(df, m2c, name):
    X, Y, Z = m2c[name]
    return np.stack([df[X].values, df[Y].values, df[Z].values], axis=1)  # (T,3)

def circular_mean_deg(angles_deg):
    a = np.deg2rad(np.asarray(angles_deg, dtype=float))
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return np.nan
    C = np.mean(np.cos(a))
    S = np.mean(np.sin(a))
    return float((np.degrees(np.arctan2(S, C)) % 360.0))

def yaw_mean_from_trc(trc_path):
    """
    用 pelvis->shoulder_mid 在 xz 平面算每幀 yaw，再取 circular mean
    回傳 yaw_mean_deg (0~360)
    """
    df, m2c = read_trc_df_and_marker2cols(trc_path)
    for need in ["RHip", "LHip", "RShoulder", "LShoulder"]:
        if need not in m2c:
            return np.nan

    RHip = get_trc_marker_xyz(df, m2c, "RHip")
    LHip = get_trc_marker_xyz(df, m2c, "LHip")
    pelvis = 0.5 * (RHip + LHip)

    RSh = get_trc_marker_xyz(df, m2c, "RShoulder")
    LSh = get_trc_marker_xyz(df, m2c, "LShoulder")
    sh_mid = 0.5 * (RSh + LSh)

    f = sh_mid - pelvis                     # (T,3)
    fx = f[:, 0]
    fz = f[:, 2]
    # 避免超小向量造成噪聲（可選）
    mag = np.sqrt(fx*fx + fz*fz)
    valid = mag > 1e-6
    yaw_t = (np.degrees(np.arctan2(fz[valid], fx[valid])) % 360.0)

    return circular_mean_deg(yaw_t)

# ===== JCF: 把 (fx,fz) 旋轉 -yaw（canonical）=====
def rotate_xz_by_minus_yaw(fx, fz, yaw_deg):
    th = np.deg2rad(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    fx2 = c * fx + s * fz
    fz2 = -s * fx + c * fz
    return fx2, fz2


# ===================== 你要改的 extract_feature3 =====================
def extract_feature3(sto_path, trc_path, k=20):
    df = load_sto(sto_path)

    yaw_mean = yaw_mean_from_trc(trc_path)
    if np.isnan(yaw_mean):
        # 沒肩/髖 marker 就退回不做 canonical（或你也可以直接 raise）
        yaw_mean = 0.0

    joints = [
        "hip_r_on_femur_r_in_ground",
        "walker_knee_r_on_tibia_r_in_ground",
        "ankle_r_on_talus_r_in_ground",
        "hip_l_on_femur_l_in_ground",
        "walker_knee_l_on_tibia_l_in_ground",
        "ankle_l_on_talus_l_in_ground",
    ]

    out = []
    for j in joints:
        fx_raw = df[f"{j}_fx"].values.astype(float)
        fy_raw = df[f"{j}_fy"].values.astype(float)
        fz_raw = df[f"{j}_fz"].values.astype(float)

        # canonical：只旋轉水平面 xz，fy 不動
        fx_can, fz_can = rotate_xz_by_minus_yaw(fx_raw, fz_raw, yaw_mean)

        fres_can = np.sqrt(fx_can**2 + fy_raw**2 + fz_can**2)

        # ===== choose scaling constant =====
        if "hip" in j:
            scale = HIP_SCALE
        elif "knee" in j:
            scale = KNEE_SCALE
        else:
            scale = ANKLE_SCALE

        out.extend([
            fx_can / scale,
            fy_raw / scale,
            fz_can / scale,
            fres_can / scale
        ])

    data = np.array(out).T  # (T, 6 joints * 4 = 24)
    idx = sample_indices(len(data), k)
    return data[idx]


# ===============================================================
# MAIN PIPELINE
# ===============================================================
all_csv_files = [f for f in os.listdir(INPUT_CSV_DIR) if f.endswith(".csv")]
print(f"共找到 {len(all_csv_files)} 個 input CSV 檔案")


for csv_file in tqdm(all_csv_files, desc="擷取特徵中", unit="file"):

    # --------- 用前四碼當唯一識別 ---------
    prefix = csv_file[:4]
    base = csv_file.replace(".csv", "")
    parts = base.split("_")

    if len(parts) < 4:
        print(f"⚠️ 跳過無效檔名: {csv_file}")
        continue

    filename = "_".join(parts[:-3])   
    viewname = parts[-3]
    startframe, endframe = parts[-2], parts[-1]

    # ================= FEATURE 1 =================
    f1_name = f"{filename}_{viewname}_{startframe}_{endframe}.csv"
    f1_path = os.path.join(OUT_FEATURE1_DIR, f1_name)

    if not os.path.exists(f1_path):
        csv_path = os.path.join(INPUT_CSV_DIR, csv_file)
        f1 = extract_markers_from_csv(csv_path)
        pd.DataFrame(f1).to_csv(f1_path, index=False, header=False)

    # ================= FEATURE 2 =================
    f2_name = f"{filename}_{startframe}_{endframe}.csv"
    f2_path = os.path.join(OUT_FEATURE2_DIR, f2_name)

    trc_name = f"{filename}_{startframe}_{endframe}.trc"
    trc_path = os.path.join(INPUT_TRC_DIR, trc_name)

    if not os.path.exists(f2_path) and os.path.exists(trc_path):
        f2_raw, T = extract_markers_from_trc(trc_path)
        idxs = sample_indices(T, NUM_FRAMES)
        f2 = f2_raw[idxs].reshape(NUM_FRAMES, -1)
        pd.DataFrame(f2).to_csv(f2_path, index=False, header=False)

    # ================= FEATURE 3 =================
    f3_name = f"{filename}_{startframe}_{endframe}.csv"
    f3_path = os.path.join(OUT_FEATURE3_DIR, f3_name)

    sto_path = os.path.join(INPUT_STO_DIR, f"{filename}_{startframe}_{endframe}.sto")

    if not os.path.exists(f3_path) and os.path.exists(sto_path):
        f3 = extract_feature3(sto_path, trc_path, k=20)
        header_cols = list(range(f3.shape[1]))
        pd.DataFrame(f3).to_csv(f3_path, index=False, header=False)

print("✅ feature1 / feature2 / feature3 生成完成！")

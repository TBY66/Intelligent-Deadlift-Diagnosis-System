import os
import csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")

# ===================== REPRODUCIBILITY =====================
SEED = 42
N_WORKERS = 0

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def make_torch_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

seed_everything(SEED)

# ===================== CONFIG =====================
EPOCHS = 50
BATCH = 128
LR = 1e-4
WEIGHTDECAY = 3e-4
PATIENCE = 10

device = "cpu" if torch.cuda.is_available() else "cpu"
os.makedirs("traindata/models5", exist_ok=True)
os.makedirs("traindata/plots5", exist_ok=True)
os.makedirs("traindata/f3_pred5", exist_ok=True)

# ===================== LABELS =====================
def load_labels(path):
    labels = {}
    with open(path, "r") as f:
        for line in f:
            key, label = line.strip().split(",")
            labels[key.strip()] = label.strip()
    return labels

CLASS_MAP = {"correct": 0, "hip_first": 1, "knee_first": 2, "rounded_back": 3}
CLS_REV = {v: k for k, v in CLASS_MAP.items()}

# ===================== DATASETS =====================
class DeadliftDataset(torch.utils.data.Dataset):
    """
    for TF1:
    feature5: <filename>_<viewangle>_<start>_<end>.csv (33 dims, we only use first 30 dims)
    feature3_dir: <filename>_<start>_<end>.csv
    """
    def __init__(self, f1_dir, f3_dir, file_list=None, cache_in_memory=False):
        self.f1_dir = f1_dir
        self.f3_dir = f3_dir

        if file_list is None:
            self.files = sorted([f.replace(".csv", "") for f in os.listdir(f1_dir) if f.endswith(".csv")])
        else:
            self.files = sorted(list(file_list))

        self.cache_in_memory = cache_in_memory
        self.cache = None

        if self.cache_in_memory:
            self.cache = [self._load_item(sid) for sid in self.files]

    def __len__(self):
        return len(self.files)

    def _load_item(self, sid):
        parts = sid.split("_")
        if len(parts) < 4:
            raise ValueError(f"Invalid filename format: {sid}")

        # 取掉第二段視角碼
        base_no_view = "_".join([parts[0]] + parts[2:])
        f1_path = os.path.join(self.f1_dir, sid + ".csv")
        f3_path = os.path.join(self.f3_dir, base_no_view + ".csv")

        if not os.path.exists(f3_path):
            raise FileNotFoundError(f"Cannot find matching feature3 for {sid}: {f3_path}")

        f1 = pd.read_csv(f1_path, header=None).values[:, :30]
        f3 = pd.read_csv(f3_path, header=None).values

        f1 = torch.tensor(f1, dtype=torch.float32)
        f3 = torch.tensor(f3, dtype=torch.float32)
        return f1, f3, sid

    def __getitem__(self, idx):
        if self.cache is not None:
            return self.cache[idx]
        sid = self.files[idx]
        return self._load_item(sid)


class DeadliftDatasetCls(torch.utils.data.Dataset):
    """
    Read ground_truth:
    parts[2:4] -> [hip_first, knee_first]  (2 labels)
    parts[4]   -> rounded_back (0/1)       (for XGBoost)
    """
    def __init__(self, f1_dir, gt_path, file_list=None, cache_in_memory=False):
        self.f1_dir = f1_dir

        if file_list is None:
            self.files = sorted([f.replace(".csv", "") for f in os.listdir(f1_dir) if f.endswith(".csv")])
        else:
            self.files = sorted(list(file_list))

        self.labels = {}
        with open(gt_path, "r") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                sid = parts[0]

                cls2 = list(map(int, parts[2:4]))
                rounded = int(parts[4])

                self.labels[sid] = (cls2, rounded)

        self.cache_in_memory = cache_in_memory
        self.cache = None

        if self.cache_in_memory:
            self.cache = [self._load_item(sid) for sid in self.files]

    def __len__(self):
        return len(self.files)

    def _load_item(self, sid):
        f1 = np.loadtxt(os.path.join(self.f1_dir, sid + ".csv"), delimiter=",")
        f1 = f1[:, :30]
        f1 = torch.tensor(f1, dtype=torch.float32)

        y_cls2, y_round = self.labels[sid]
        y_cls2 = torch.tensor(y_cls2, dtype=torch.float32)
        y_round = torch.tensor(y_round, dtype=torch.long)

        return f1, y_cls2, y_round, sid

    def __getitem__(self, idx):
        if self.cache is not None:
            return self.cache[idx]
        sid = self.files[idx]
        return self._load_item(sid)


def load_f3_pred_cache(file_list, pred_dir):
    cache = {}
    for sid in sorted(set(file_list)):
        pred = np.load(os.path.join(pred_dir, f"{sid}.npy"))
        cache[sid] = torch.tensor(pred, dtype=torch.float32)
    return cache

def get_f3_batch_from_cache(sid_batch, f3_pred_cache):
    return torch.stack([f3_pred_cache[s] for s in sid_batch], dim=0)

def temporal_shift(f1, max_shift=2):
    """
    Circular temporal shift
    f1: (B, T, C)
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    return torch.roll(f1, shifts=shift, dims=1)


def augment_f1(
    f1,
    p=0.9,
    noise_sigma=0.3,
    scale_jitter=0.3,
    rot_deg=15,
    dropout_prob=0.05,
    temporal_shift_range=1,
    angle_noise=0.02
):
    """
    f1 shape: (B, T, 30)
    24 coords + 6 angles
    angle features are assumed to be scaled roughly within [-1, 1]
    """
    if random.random() > p:
        return f1

    B, T, C = f1.shape
    assert C >= 30, "Expected 30 dims (24 coords + 6 angles)"

    coords = f1[:, :, :24]     # (B,T,24)
    angles = f1[:, :, 24:30]   # (B,T,6)

    pts = coords.view(B, T, -1, 2)
    pts_aug = pts.clone()

    for b in range(B):
        shoulder = pts[b, :, 0:2].mean(0)
        hip      = pts[b, :, 6:8].mean(0)
        body_scale = torch.norm(shoulder - hip) + 1e-6

        # 1) Gaussian noise
        std = noise_sigma * body_scale
        pts_aug[b] += torch.randn_like(pts_aug[b]) * std

        # 2) Scale jitter
        if scale_jitter > 0:
            s = 1.0 + torch.randn(1).item() * scale_jitter
            center = pts_aug[b].mean(dim=1, keepdim=True)
            pts_aug[b] = (pts_aug[b] - center) * s + center

        # 3) Rotation jitter
        if rot_deg > 0:
            ang = np.radians(np.random.uniform(-rot_deg, rot_deg))
            c, s_ = np.cos(ang), np.sin(ang)
            R = torch.tensor([[c, -s_], [s_, c]], dtype=torch.float32, device=f1.device)
            pts_aug[b] = torch.matmul(pts_aug[b], R)

        # 4) Keypoint dropout
        if dropout_prob > 0:
            mask = (torch.rand(T, pts_aug.shape[2], 1, device=f1.device) > dropout_prob).float()
            pts_aug[b] *= mask

    coords_aug = pts_aug.view(B, T, 24)

    # angle augmentation for features scaled to about [0, 1]
    angles_aug = angles + torch.randn_like(angles) * angle_noise

    # clamp angle features after augmentation
    angles_aug = torch.clamp(angles_aug, 0.0, 1.0)

    f1_aug = torch.cat([coords_aug, angles_aug], dim=-1)

    # temporal shift
    if temporal_shift_range > 0:
        shift = np.random.randint(-temporal_shift_range, temporal_shift_range + 1)
        f1_aug = torch.roll(f1_aug, shifts=shift, dims=1)

    return f1_aug

def get_subject_id(filename):
    return int(filename[:4])

# ===================== Positional Encoding =====================
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape = (1, max_len, dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T]


# ===================== Transformer Encoder =====================
class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x2 = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + x2

        x2 = self.mlp(self.norm2(x))
        x = x + x2

        return x

# ===================== TF1 (Transformer version of TF1) =====================
class TF1(nn.Module):
    def __init__(self, input_dim=30, embed_dim=64, depth=3, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pe = PositionalEncoding(embed_dim)

        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])

        # pose_head 負責學「動作」，這部分會被 Contrastive Loss 強制對齊
        self.pose_head = nn.Linear(embed_dim, embed_dim)
        # view_head 負責學「視角」，這部分不參與對齊，用來存放每個視角獨特的資訊
        self.view_head = nn.Linear(embed_dim, embed_dim)

        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), 
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 66),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pe(x)

        for blk in self.blocks:
            x = blk(x)

        # 1. 提取 Transformer 的全局特徵
        latent_base = x.mean(dim=1) 
        
        # 2. 分流：將特徵拆解為 Pose 和 View
        z_pose = self.pose_head(latent_base) 
        z_view = self.view_head(latent_base) 

        # 3. 合併：將 View 資訊「貼」回每一幀的特徵後面，幫助重建 3D 坐標
        # 我們把 z_view 複製成跟時間長度一樣，然後拼接到 x 後面
        z_view_expanded = z_view.unsqueeze(1).expand(-1, x.size(1), -1)
        x_combined = torch.cat([x, z_view_expanded], dim=-1) # 維度從 64 變成 128

        pred = self.fc_out(x_combined)
        
        # 回傳 z_pose 用於計算 Contrastive Loss
        return pred, z_pose, z_view


# ===================== TF2 (Transformer version of TF2) =====================
class TF2(nn.Module):
    def __init__(self, f1_dim=30, f3_dim=66, embed_dim=128, depth=3, num_heads=4, dropout=0.3):
        super().__init__()

        self.f1_proj = nn.Sequential(
            nn.Linear(f1_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.f3_proj = nn.Sequential(
            nn.Linear(f3_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.f1_pe = PositionalEncoding(embed_dim)
        self.f3_pe = PositionalEncoding(embed_dim)

        # feature1 作為主幹，feature3_2 提供補充資訊
        self.f1_to_f3_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.cross_attn_dropout = nn.Dropout(dropout)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])

        self.temporal_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)   # multi-label logits
        )

    def forward(self, f1, f3):
        f1 = self.f1_proj(f1)
        f3 = self.f3_proj(f3)

        f1 = self.f1_pe(f1)
        f3 = self.f3_pe(f3)

        attn_out, _ = self.f1_to_f3_attn(
            query=f1,
            key=f3,
            value=f3,
            need_weights=False
        )
        gate = self.cross_gate(torch.cat([f1, attn_out], dim=-1))
        x = self.cross_attn_norm(
            f1 + self.cross_attn_dropout(gate * attn_out)
        )

        for blk in self.blocks:
            x = blk(x)

        x = x.transpose(1, 2)
        x = self.temporal_pool(x)
        x = x.transpose(1, 2)

        mean_feat = x.mean(dim=1)
        max_feat, _ = x.max(dim=1)
        feat = torch.cat([mean_feat, max_feat], dim=1)

        cls_logits = self.cls_head(feat)
        return cls_logits


# ===================== Positive weighted group loss =====================
class PositiveWeightedGroupLoss(nn.Module):
    def __init__(self, init_weights=(2.0, 0.3, 0.2, 0.3, 0.1), eps=1e-6):
        super().__init__()
        init_weights = torch.tensor(init_weights, dtype=torch.float32)
        raw_init = torch.log(torch.expm1(init_weights.clamp(min=eps)))

        self.raw_recon = nn.Parameter(raw_init[0].clone())
        self.raw_represent = nn.Parameter(raw_init[1].clone())
        self.raw_bone = nn.Parameter(raw_init[2].clone())
        self.raw_angle = nn.Parameter(raw_init[3].clone())
        self.raw_temp = nn.Parameter(raw_init[4].clone())
        self.eps = eps

    def get_weights(self):
        return {
            "recon": F.softplus(self.raw_recon) + self.eps,
            "represent": F.softplus(self.raw_represent) + self.eps,
            "bone": F.softplus(self.raw_bone) + self.eps,
            "angle": F.softplus(self.raw_angle) + self.eps,
            "temp": F.softplus(self.raw_temp) + self.eps,
        }

    def get_weight_vector(self, device=None, dtype=None):
        weights = self.get_weights()
        vec = torch.stack([
            weights["recon"],
            weights["represent"],
            weights["bone"],
            weights["angle"],
            weights["temp"],
        ])
        if device is not None or dtype is not None:
            vec = vec.to(device=device, dtype=dtype)
        return vec

    def forward(self, l_recon, l_represent, l_bone, l_angle, l_temp):
        weights = self.get_weights()
        total = (
            weights["recon"] * l_recon +
            weights["represent"] * l_represent +
            weights["bone"] * l_bone +
            weights["angle"] * l_angle +
            weights["temp"] * l_temp
        )
        return total

# ========== Contrastive Latent Loss Functions ==========

def contrastive_latent_loss(z_group):
    """
    z_group: (K, latent_dim)
    拉近同一影片多視角 latent（positive contrastive）
    """
    if len(z_group) < 2:
        return torch.tensor(0.0, device=z_group.device)

    # normalize
    z = nn.functional.normalize(z_group, dim=1)

    loss = 0.0
    count = 0
    for i in range(len(z)):
        for j in range(i+1, len(z)):
            loss += (1 - torch.cosine_similarity(z[i], z[j], dim=0))
            count += 1

    return loss / max(count, 1)

def parse_filename(fname):
    name = fname.replace(".csv", "")
    # subject = 前3碼
    subject = int(name[:4])
    # class = 第4碼 (單字母)
    cls = name[4]
    # 剩下部分: _view_start_end
    parts = name[6:].split("_")
    if len(parts) != 3:
        raise ValueError(f"Filename format error: {fname}")

    view = int(parts[0])
    start = int(parts[1])
    end = int(parts[2])

    return subject, cls, view, start, end

def get_video_group_id(fname):
    subject, cls, view, start, end = parse_filename(fname)
    return f"{subject}{cls}_{start}_{end}"

# ===================== XGBOOST FEATURE BUILDERS =====================
def flatten_kov_60_from_feature5(csv_path, target_T=20):
    """
    Robust KOV flattener:
    - support csv with / without header
    - support T != 20 (pad or truncate)
    - output fixed 60-dim feature
    """
    # ---------- read csv (auto header handling) ----------
    try:
        df = pd.read_csv(csv_path, header=None)
        arr = df.values
    except Exception:
        arr = np.loadtxt(csv_path, delimiter=",")

    if arr.ndim != 2 or arr.shape[1] < 33:
        raise ValueError(f"{csv_path}: invalid csv shape {arr.shape}")

    # ---------- temporal length handling ----------
    T = arr.shape[0]

    if T > target_T:
        arr = arr[:target_T]
    elif T < target_T:
        # padding: repeat last frame
        pad = np.repeat(arr[-1:, :], target_T - T, axis=0)
        arr = np.concatenate([arr, pad], axis=0)

    # ---------- extract k, o, v ----------
    kov = arr[:, 30:33]     # (20,3)
    k = kov[:, 0]
    o = kov[:, 1]
    v = kov[:, 2]

    feat = np.concatenate([k, o, v], axis=0).astype(np.float32)

    assert feat.shape[0] == 60, f"{csv_path}: output dim != 60"
    return feat

def build_xgb_xy(
    subject_list,
    feature5_dir,
    labels_dict
):
    """
    subject_list: list[int] or list[str]
        e.g. [13, 15, 16] or ["0013", "0015"]
    feature5_dir: "testdata/feature5"
    labels_dict: dict[sid] -> rounded_back (0/1)

    Returns:
        X: (N_trials, 60)
        y: (N_trials,)
    """

    # 將 subject 轉成 4-digit string
    subj_prefixes = [f"{int(s):04d}" for s in subject_list]

    X_list, y_list = [], []

    for fname in sorted(os.listdir(feature5_dir)):
        if not fname.endswith(".csv"):
            continue

        sid = fname.replace(".csv", "")

        # subject filter
        if not any(sid.startswith(pref) for pref in subj_prefixes):
            continue

        # label 必須存在
        if sid not in labels_dict:
            continue

        # feature
        csv_path = os.path.join(feature5_dir, fname)
        kov = flatten_kov_60_from_feature5(csv_path)

        X_list.append(kov)
        y_list.append(int(labels_dict[sid]))

    if len(X_list) == 0:
        raise ValueError("No XGB samples found for given subject list.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=int)

    return X, y

def youden_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    return float(thresholds[best_idx]), float(youden[best_idx])

import torch
import torch.nn.functional as F

def compute_bone_length_loss(pred_3d):
    """
    pred_3d: (B, T, 36) -> 12 joints (x,y,z)
    return:
        bone_stability_loss
        bone_symmetry_loss
    """

    B, T, _ = pred_3d.shape
    coords = pred_3d[:, :, :36]
    joints = coords.view(B, T, 12, 3)

    # -------- bones --------
    bone_pairs = [
        (0,1),(1,2),   # right arm
        (6,7),(7,8),   # left arm
        (3,4),(4,5),   # right leg
        (9,10),(10,11),# left leg
        (0,6),(3,9),(0,3),(6,9) # torso
    ]

    # ----------------------
    # 1. temporal stability
    # ----------------------

    bone_var_total = 0.0

    for i, j in bone_pairs:
        diff = joints[:,:,i,:] - joints[:,:,j,:]
        lengths = torch.norm(diff, dim=-1)
        lengths = torch.clamp(lengths, min=1e-4)

        bone_var_total += torch.var(lengths, unbiased=False)

    bone_stability_loss = bone_var_total / len(bone_pairs)

    # ----------------------
    # 2. symmetry constraint
    # ----------------------

    symmetry_pairs = [
        ((0,1),(6,7)),   # upper arm
        ((1,2),(7,8)),   # forearm
        ((3,4),(9,10)),  # thigh
        ((4,5),(10,11))  # shank
    ]

    symmetry_loss = 0.0

    for (r1,r2),(l1,l2) in symmetry_pairs:

        right_len = torch.norm(
            joints[:,:,r1,:] - joints[:,:,r2,:],
            dim=-1
        )

        left_len = torch.norm(
            joints[:,:,l1,:] - joints[:,:,l2,:],
            dim=-1
        )

        symmetry_loss += torch.mean((right_len - left_len)**2)

    symmetry_loss = symmetry_loss / len(symmetry_pairs)

    return bone_stability_loss, symmetry_loss

def signed_angle(v1, v2, ref_axis):
    cross = torch.cross(v1, v2, dim=-1)
    sin = (cross * ref_axis).sum(-1)
    cos = (v1 * v2).sum(-1)
    cos = torch.clamp(cos, -1.0, 1.0)
    angle = torch.atan2(sin, cos)
    return angle / torch.pi

def compute_body_axes(joints):
    """
    joints: (B,T,12,3)
    return body axes
    """

    RShoulder = joints[:,:,0]
    RHip      = joints[:,:,3]
    LShoulder = joints[:,:,6]
    LHip      = joints[:,:,9]

    mid_shoulder = (RShoulder + LShoulder) / 2
    mid_hip      = (RHip + LHip) / 2

    x_axis = F.normalize(RShoulder - LShoulder,  dim=-1, eps=1e-6)
    y_axis = F.normalize(mid_shoulder - mid_hip, dim=-1, eps=1e-6)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    z_axis = F.normalize(z_axis, dim=-1, eps=1e-6)

    return x_axis, y_axis, z_axis

def angle_range_loss(angle, lower, upper):

    low_penalty = torch.relu(lower - angle)
    high_penalty = torch.relu(angle - upper)

    return (low_penalty + high_penalty).mean()

def compute_biomechanical_angle_loss(pred):

    B,T,_ = pred.shape
    coords = pred[:,:,:36]
    joints = coords.view(B,T,12,3)

    x_axis, y_axis, z_axis = compute_body_axes(joints)

    # -------- Knee --------
    knee_r = signed_angle(
        F.normalize(joints[:,:,3] - joints[:,:,4], dim=-1, eps=1e-6),
        F.normalize(joints[:,:,5] - joints[:,:,4], dim=-1, eps=1e-6),
        x_axis
    )
    knee_l = signed_angle(
        F.normalize(joints[:,:,9]  - joints[:,:,10], dim=-1, eps=1e-6),
        F.normalize(joints[:,:,11] - joints[:,:,10], dim=-1, eps=1e-6),
        x_axis
    )

    # -------- Hip --------
    hip_r = signed_angle(
        F.normalize(joints[:,:,0] - joints[:,:,3], dim=-1, eps=1e-6),
        F.normalize(joints[:,:,4] - joints[:,:,3], dim=-1, eps=1e-6),
        x_axis
    )
    hip_l = signed_angle(
        F.normalize(joints[:,:,6]  - joints[:,:,9], dim=-1, eps=1e-6),
        F.normalize(joints[:,:,10] - joints[:,:,9], dim=-1, eps=1e-6),
        x_axis
    )

    # -------- Shoulder --------
    shoulder_r = signed_angle(
        F.normalize(joints[:,:,1] - joints[:,:,0], dim=-1, eps=1e-6),
        F.normalize(joints[:,:,3] - joints[:,:,0], dim=-1, eps=1e-6),
        x_axis
    )
    shoulder_l = signed_angle(
        F.normalize(joints[:,:,7] - joints[:,:,6], dim=-1, eps=1e-6),
        F.normalize(joints[:,:,9] - joints[:,:,6], dim=-1, eps=1e-6),
        x_axis
    )

    KNEE_MIN = 40 / 180
    KNEE_MAX = 1.0
    HIP_MIN = 45 / 180
    HIP_MAX = 1.0
    SHOULDER_MIN = 0.0
    SHOULDER_MAX = 170 / 180

    loss_knee = (
        angle_range_loss(knee_r, KNEE_MIN, KNEE_MAX) +
        angle_range_loss(knee_l, KNEE_MIN, KNEE_MAX)
    )
    loss_hip = (
        angle_range_loss(hip_r, HIP_MIN, HIP_MAX) +
        angle_range_loss(hip_l, HIP_MIN, HIP_MAX)
    )
    loss_shoulder = (
        angle_range_loss(shoulder_r, SHOULDER_MIN, SHOULDER_MAX) +
        angle_range_loss(shoulder_l, SHOULDER_MIN, SHOULDER_MAX)
    )

    angles = torch.stack([shoulder_r, shoulder_l, hip_r, hip_l, knee_r, knee_l], dim=-1)

    return angles, loss_shoulder, loss_hip, loss_knee

def compute_angle_consistency_loss(pred, f1):

    """
    pred: (B,T,60)
    f1:   (B,T,30)

    f1 last 6 dims:
    [r_shoulder,r_hip,r_knee,l_shoulder,l_hip,l_knee]
    """

    B,T,_ = pred.shape

    # ---------- input angles ----------
    angle_in = f1[:,:,24:30]

    shoulder_avg = (angle_in[:,:,0] + angle_in[:,:,3]) / 2
    hip_avg      = (angle_in[:,:,1] + angle_in[:,:,4]) / 2
    knee_avg     = (angle_in[:,:,2] + angle_in[:,:,5]) / 2

    avg_input_angles = torch.stack([
        shoulder_avg,
        hip_avg,
        knee_avg
    ], dim=-1)   # (B,T,3)

    # ---------- compute angles from pred 3D ----------
    coords = pred[:,:,:36]
    joints = coords.view(B,T,12,3)

    x_axis, y_axis, z_axis = compute_body_axes(joints)

    knee_r = signed_angle(
        F.normalize(joints[:,:,3]-joints[:,:,4], dim=-1, eps=1e-6),
        F.normalize(joints[:,:,5]-joints[:,:,4], dim=-1, eps=1e-6),
        x_axis
    )

    knee_l = signed_angle(
        F.normalize(joints[:,:,9]-joints[:,:,10], dim=-1, eps=1e-6),
        F.normalize(joints[:,:,11]-joints[:,:,10], dim=-1, eps=1e-6),
        x_axis
    )

    hip_r = signed_angle(
        F.normalize(joints[:,:,0]-joints[:,:,3], dim=-1, eps=1e-6),
        F.normalize(joints[:,:,4]-joints[:,:,3], dim=-1, eps=1e-6),
        x_axis
    )

    hip_l = signed_angle(
        F.normalize(joints[:,:,6]-joints[:,:,9], dim=-1, eps=1e-6),
        F.normalize(joints[:,:,10]-joints[:,:,9], dim=-1, eps=1e-6),
        x_axis
    )

    shoulder_r = signed_angle(
        F.normalize(joints[:,:,1]-joints[:,:,0], dim=-1, eps=1e-6),
        F.normalize(joints[:,:,3]-joints[:,:,0], dim=-1, eps=1e-6),
        x_axis
    )

    shoulder_l = signed_angle(
        F.normalize(joints[:,:,7]-joints[:,:,6], dim=-1, eps=1e-6),
        F.normalize(joints[:,:,9]-joints[:,:,6], dim=-1, eps=1e-6),
        x_axis
    )

    shoulder_avg_pred = (shoulder_r + shoulder_l) / 2
    hip_avg_pred      = (hip_r + hip_l) / 2
    knee_avg_pred     = (knee_r + knee_l) / 2

    avg_pred_angles = torch.stack([
        shoulder_avg_pred,
        hip_avg_pred,
        knee_avg_pred
    ], dim=-1)

    # ---------- consistency loss ----------
    loss = torch.mean((avg_pred_angles - avg_input_angles)**2)

    return loss

def compute_temporal_loss(pred, gt):
    pred = pred[:, :, :36]
    gt = gt[:, :, :36]
    
    pred_var = torch.var(pred, dim=1, unbiased=False)
    gt_var = torch.var(gt, dim=1, unbiased=False)
    pred_var = torch.clamp(pred_var, min=1e-4)
    gt_var = torch.clamp(gt_var, min=1e-4)
    ratio = (gt_var + 1e-4) / (pred_var + 1e-4)
    ratio = torch.clamp(ratio, 0.1, 10.0)
    loss_var = torch.mean(torch.log(ratio) ** 2)
    
    vel_pred = pred[:, 1:, :] - pred[:, :-1, :]
    vel_gt   = gt[:, 1:, :]   - gt[:, :-1, :]
    
    accel_pred = vel_pred[:, 1:, :] - vel_pred[:, :-1, :]
    accel_gt   = vel_gt[:, 1:, :]   - vel_gt[:, :-1, :]
    loss_accel = torch.mean((accel_pred - accel_gt) ** 2)

    vel_pred_norm = vel_pred / (vel_pred.norm(dim=-1, keepdim=True).clamp(min=1e-6))
    vel_gt_norm   = vel_gt   / (vel_gt.norm(dim=-1, keepdim=True).clamp(min=1e-6))
    cos_sim = (vel_pred_norm * vel_gt_norm).sum(dim=-1)
    cos_sim = torch.clamp(cos_sim, -0.999, 0.999)
    loss_dir = torch.mean(1 - cos_sim)
    
    return loss_var, loss_accel, loss_dir

# ===================== LOSS MEDIANS (for normalization) =====================
LOSS_MEDIANS = {
    "coord_recon": 0.392437,
    "angle_recon": 0.136637,
    "force_recon": 0.30,
    "contrast": 0.01,
    "ortho": 0.007659,
    "bone_stab": 0.048608,
    "bone_sym": 0.055016,
    "shoulder_angle": 0.12320,
    "hip_angle": 1.21044,
    "knee_angle": 0.81744,
    "angle_consistency": 1.55535,
    "var_temp": 1.708845,
    "accel_temp": 0.132899,
    "dir_temp": 0.927269
}

# ===================== SPLIT SUBJECTS BY VIEW PAIR =====================
count = 0
while True:
    count+=1
    trial_seed = SEED + count
    print(f"Trial{count}")   
    TRA_DIR = "traindata/features/feature5"
    TST_DIR = "testdata/feature5"

    def list_base_names(folder):
        return sorted([
            f.replace(".csv", "")
            for f in os.listdir(folder)
            if f.endswith(".csv")
        ])

    train_all = list_base_names(TRA_DIR)
    test_all  = list_base_names(TST_DIR)

    # --------------------------------
    # build subject -> file mapping
    # --------------------------------
    subject_dict = {}

    for src, files in [("tra", train_all), ("tst", test_all)]:
        for f in files:
            sid, cls, view, start, end = parse_filename(f)
            subject_dict.setdefault(sid, []).append((src, f))

    all_subjects = sorted(subject_dict.keys())

    # --------------------------------
    # read viewpair mapping
    # --------------------------------
    viewpair_map = {}

    with open("traindata/viewpair.txt", "r") as f:
        for line in f:

            line = line.strip()
            if not line:
                continue

            line = line.replace(",", " ")
            p = line.split()

            sid  = int(p[0])
            pair = p[1].upper()

            viewpair_map[sid] = pair

    # check missing subjects
    missing_subjects = [sid for sid in all_subjects if sid not in viewpair_map]

    if len(missing_subjects) > 0:
        raise ValueError(f"Missing subject ids in viewpair.txt: {missing_subjects}")

    # --------------------------------
    # group subjects by pair
    # --------------------------------
    pair_subjects = {"A": [], "B": [], "C": []}

    for sid in all_subjects:
        pair_subjects[viewpair_map[sid]].append(sid)

    for k in pair_subjects:
        pair_subjects[k] = sorted(pair_subjects[k])
    # --------------------------------
    # 5-fold split (balanced A/B/C)
    # --------------------------------
    def split_balanced(pair_dict, n_fold=5, seed=SEED):
        rng = np.random.default_rng(seed)
        groups = [[] for _ in range(n_fold)]

        for k in pair_dict:
            arr = pair_dict[k].copy()
            rng.shuffle(arr)

            for i, sid in enumerate(arr):
                groups[i % n_fold].append(sid)

        return groups

    groups = split_balanced(pair_subjects, 5, seed=SEED)

    folds = []

    for i in range(5):

        test_subj = groups[i]
        val_subj = groups[(i+1) % 5]

        train_subj = []
        for j in range(5):
            if j not in [i, (i+1)%5]:
                train_subj.extend(groups[j])

        folds.append((train_subj, val_subj, test_subj))

    seed_everything(SEED)
    all_fold_results = []
    all_fold_rmse = []
    all_fold_micro = []
    all_fold_macro = []
    all_fold_c = []
    all_fold_h = []
    all_fold_k = []
    all_fold_r = []
    all_fold_auc_h, all_fold_auc_k, all_fold_auc_r = [], [], [] 
    all_fold_auc_micro, all_fold_auc_macro = [], []
    detail_csv_path = "traindata/plots5/test_detail.csv"
    with open(detail_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold_id",
            "filename",
            "gt_c", "gt_h", "gt_k", "gt_r",
            "pred_c", "pred_h", "pred_k", "pred_r"
        ])

    for fold_id, (train_subj, val_subj, test_subj) in enumerate(folds, 1):
        print(f"Fold {fold_id}")

        train_pairs, val_pairs, test_pairs = [], [], []

        for sid in sorted(subject_dict.keys()):
            files = subject_dict[sid]

            if sid in train_subj:
                train_pairs.extend(files)

            elif sid in val_subj:
                val_pairs.extend(files)

            elif sid in test_subj:
                test_pairs.extend(files)

        def split_and_sample(pairs, ratio=0.1, seed=SEED):
            rng = random.Random(seed)
            tra = sorted([f for (src, f) in pairs if src == "tra"])
            tst = sorted([f for (src, f) in pairs if src == "tst"])

            # tra 抽；tst 全留
            if len(tra) > 0:
                k = max(1, int(len(tra) * ratio))
                tra = sorted(rng.sample(tra, k))
            return tra, tst

        train_tra, train_tst = split_and_sample(train_pairs, 0.01, seed=SEED + fold_id * 10 + 1)
        val_tra,   val_tst   = split_and_sample(val_pairs,   0.01, seed=SEED + fold_id * 10 + 2)
        test_tra,  test_tst  = split_and_sample(test_pairs,  0.01, seed=SEED + fold_id * 10 + 3)

        train_files = train_tra + train_tst
        val_files   = val_tra   + val_tst
        test_files  = test_tra  + test_tst

        seed_everything(trial_seed + fold_id)

        # ===================== TF1 TRAIN =====================
        tf1 = TF1().to(device)
        loss_balancer = PositiveWeightedGroupLoss(init_weights=(1.0, 0.1, 0.1, 0.1, 0.1)).to(device)
        opt1 = torch.optim.Adam(
            list(tf1.parameters()) + list(loss_balancer.parameters()),
            lr=LR,
            weight_decay=WEIGHTDECAY
        )
        loss_fn = nn.MSELoss()

        train_dataset = ConcatDataset([
            DeadliftDataset("traindata/features/feature5", "traindata/features/feature3_2", file_list=train_tra, cache_in_memory=True),
            DeadliftDataset("testdata/feature5", "traindata/features/feature3_2", file_list=train_tst, cache_in_memory=True),
        ])
        val_dataset = ConcatDataset([
            DeadliftDataset("traindata/features/feature5", "traindata/features/feature3_2", file_list=val_tra, cache_in_memory=True),
            DeadliftDataset("testdata/feature5", "traindata/features/feature3_2", file_list=val_tst, cache_in_memory=True),
        ])
        test_dataset = ConcatDataset([
            DeadliftDataset("traindata/features/feature5", "traindata/features/feature3_2", file_list=test_tra, cache_in_memory=True),
            DeadliftDataset("testdata/feature5", "traindata/features/feature3_2", file_list=test_tst, cache_in_memory=True),
        ])

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH,
            shuffle=True,
            num_workers=N_WORKERS,
            worker_init_fn=seed_worker if N_WORKERS > 0 else None,
            generator=make_torch_generator(SEED + fold_id * 100 + 1)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH,
            shuffle=False,
            num_workers=N_WORKERS,
            worker_init_fn=seed_worker if N_WORKERS > 0 else None,
            generator=make_torch_generator(SEED + fold_id * 100 + 2)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH,
            shuffle=False,
            num_workers=N_WORKERS,
            worker_init_fn=seed_worker if N_WORKERS > 0 else None,
            generator=make_torch_generator(SEED + fold_id * 100 + 3)
        )
        # ===================== Training setup =====================
        best_rmse, wait = 1e9, 0
        history = {"train_loss": [], "val_loss": [], "train_rmse": [], "val_rmse": []}
        GROUP_NAMES = ["recon", "represent", "bone", "angle", "temp"]
        GROUP_COLORS = {
            "recon": "#4C78A8",      # muted blue
            "represent": "#F58518",  # amber
            "bone": "#54A24B",       # green
            "angle": "#E45756",      # muted red
            "temp": "#B279A2",       # dusty purple
        }

        weight_history = {
            "raw": {k: [] for k in GROUP_NAMES},
            "softplus": {k: [] for k in GROUP_NAMES},
            "norm": {k: [] for k in GROUP_NAMES},
        }

        def plot_loss_weight():
            epochs = np.arange(1, len(weight_history["raw"]["recon"]) + 1)

            plot_specs = [
                ("raw", "Raw Weight", f"traindata/plots5/loss_weight_raw_fold{fold_id}.png", "Raw Weight over Epochs"),
                ("softplus", "Softplused Weight", f"traindata/plots5/loss_weight_softplus_fold{fold_id}.png", "Softplused Weight over Epochs"),
                ("norm", "Normalized Weight (sum=1)", f"traindata/plots5/loss_weight_norm_fold{fold_id}.png", "Normalized Weight over Epochs"),
            ]

            for key, ylabel, save_path, title in plot_specs:
                fig, ax = plt.subplots(figsize=(6, 4))

                for g in GROUP_NAMES:
                    ax.plot(
                        epochs,
                        weight_history[key][g],
                        label=g,
                        color=GROUP_COLORS[g],
                        linewidth=2
                    )

                ax.set_xlabel("Epoch")
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.legend(
                    loc="upper right",
                    bbox_to_anchor=(1, 1),
                    edgecolor="#666666",
                    fontsize=9
                )
                fig.tight_layout()
                fig.savefig(save_path, dpi=200)
                plt.close(fig)

        def plot_metrics():
            fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
            ax_loss.plot(history["train_loss"], label="Train Loss", color="tab:blue")
            ax_loss.plot(history["val_loss"], label="Val Loss", color="tab:orange")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title("TF1-Loss")
            ax_loss.legend()
            fig_loss.tight_layout()
            fig_loss.savefig(f"traindata/plots5/tf1_loss_fold{fold_id}.png")
            plt.close(fig_loss)

            fig_rmse, ax_rmse = plt.subplots(figsize=(6, 4))
            ax_rmse.plot(history["train_rmse"], label="Train RMSE", color="tab:blue")
            ax_rmse.plot(history["val_rmse"], label="Val RMSE", color="tab:orange")
            ax_rmse.set_xlabel("Epoch")
            ax_rmse.set_ylabel("RMSE")
            ax_rmse.set_title("TF1-RMSE")
            ax_rmse.legend()
            fig_rmse.tight_layout()
            fig_rmse.savefig(f"traindata/plots5/tf1_rmse_fold{fold_id}.png")
            plt.close(fig_rmse)

        # ===================== TRAIN LOOP =====================
        for epoch in range(1, EPOCHS + 1):
            tf1.train()
            losses, rmses = [], []
            epoch_coord_recon_losses = []
            epoch_angle_recon_losses = []
            epoch_force_recon_losses = []
            epoch_con_losses = []
            epoch_ortho_losses = []
            epoch_bone_stab_losses = []
            epoch_bone_sym_losses = []
            epoch_shoulder_angle_losses = []
            epoch_hip_angle_losses = []
            epoch_knee_angle_losses = []
            epoch_angle_consistency_losses = []
            epoch_var_temp_losses = []
            epoch_accel_temp_losses = []
            epoch_dir_temp_losses = []
    
            progress_bar = tqdm(train_loader, desc=f"[Epoch {epoch:03d}] Training", unit="batch", leave=False)
            for batch_idx, (f1, f3, sid) in enumerate(progress_bar):
                f1, f3 = f1.to(device), f3.to(device)
                f1 = augment_f1(f1)
                pred, z_pose, z_view = tf1(f1)
                # ----- Reconstruction -----
                min_len = min(pred.shape[1], f3.shape[1])
                pred = pred[:, :min_len]
                f3   = f3[:, :min_len]

                # split coords / angles / forces
                pred_coord  = pred[:, :, :36]
                pred_angle  = pred[:, :, 36:42]
                pred_force  = pred[:, :, 42:66]

                gt_coord    = f3[:, :, :36]
                gt_angle    = f3[:, :, 36:42]
                gt_force    = f3[:, :, 42:66]

                # individual losses
                loss_coord_recon = loss_fn(pred_coord, gt_coord)
                loss_angle_recon = loss_fn(pred_angle, gt_angle)
                loss_force_recon = loss_fn(pred_force, gt_force)
                # ----- Contrastive Latent Loss -----
                video_ids = [get_video_group_id(s) for s in sid]
                unique_videos = sorted(set(video_ids))

                con_losses = []

                for vid in unique_videos:
                    idx = [i for i, v in enumerate(video_ids) if v == vid]
                    if len(idx) > 1:
                        # 這裡確保傳入的是當前 batch 的 z_pose
                        con_losses.append(contrastive_latent_loss(z_pose[idx]))

                if len(con_losses) > 0:
                    # 使用 torch.stack 將所有 loss 結合後取平均，這樣計算圖只會被建立一次
                    con_loss_total = torch.stack(con_losses).mean()
                else:
                    con_loss_total = torch.tensor(0.0, device=device)
                # ----- Orthogonality -----
                loss_ortho = torch.abs(
                    nn.functional.cosine_similarity(z_pose, z_view)
                ).mean()

                # ----- Bone Loss -----
                loss_bone_stab, loss_bone_sym = compute_bone_length_loss(pred)

                # ----- Angle Loss -----
                angles, loss_shoulder, loss_hip, loss_knee = compute_biomechanical_angle_loss(pred)
                loss_angle_consistency = compute_angle_consistency_loss(pred, f1)

                # ----- Temporal Loss -----
                loss_var_temp, loss_accel_temp, loss_dir_temp = compute_temporal_loss(pred, f3)

                # ===================== Normalize each loss by median =====================
                loss_coord_recon_n = loss_coord_recon / LOSS_MEDIANS["coord_recon"]
                loss_angle_recon_n = loss_angle_recon / LOSS_MEDIANS["angle_recon"]
                loss_force_recon_n = loss_force_recon / LOSS_MEDIANS["force_recon"]
                con_loss_total_n   = con_loss_total   / LOSS_MEDIANS["contrast"]
                loss_ortho_n       = loss_ortho       / LOSS_MEDIANS["ortho"]
                loss_bone_stab_n   = loss_bone_stab   / LOSS_MEDIANS["bone_stab"]
                loss_bone_sym_n    = loss_bone_sym    / LOSS_MEDIANS["bone_sym"]
                loss_shoulder_n    = loss_shoulder    / LOSS_MEDIANS["shoulder_angle"]
                loss_hip_n         = loss_hip         / LOSS_MEDIANS["hip_angle"]
                loss_knee_n        = loss_knee        / LOSS_MEDIANS["knee_angle"]
                loss_angle_consistency_n = loss_angle_consistency / LOSS_MEDIANS["angle_consistency"]
                loss_var_temp_n    = loss_var_temp    / LOSS_MEDIANS["var_temp"]
                loss_accel_temp_n  = loss_accel_temp  / LOSS_MEDIANS["accel_temp"]
                loss_dir_temp_n    = loss_dir_temp    / LOSS_MEDIANS["dir_temp"]

                # ===================== Group normalized losses =====================
                # group_recon_n = (
                #     loss_coord_recon_n + loss_angle_recon_n + loss_force_recon_n
                # ) / 3.0

                # group_represent_n = (
                #     con_loss_total_n + loss_ortho_n
                # ) / 2.0

                # group_bone_n = (
                #     loss_bone_stab_n + loss_bone_sym_n
                # ) / 2.0

                # group_angle_n = (
                #     loss_shoulder_n + loss_hip_n + loss_knee_n + loss_angle_consistency_n
                # ) / 4.0

                # group_temp_n = (
                #     loss_var_temp_n + loss_accel_temp_n + loss_dir_temp_n
                # ) / 3.0

                group_recon_n = (
                    0.6 * loss_coord_recon_n + 0.2 * loss_angle_recon_n + 0.2 * loss_force_recon_n
                )

                group_represent_n = (
                    0.8 * con_loss_total_n + 0.2 * loss_ortho_n
                )

                group_bone_n = (
                    0.8 * loss_bone_stab_n + 0.2 * loss_bone_sym_n
                )

                group_angle_n = (
                    0.1 * loss_shoulder_n + 0.15 * loss_hip_n + 0.15 * loss_knee_n + 0.6 * loss_angle_consistency_n
                )

                group_temp_n = (
                    0.6 * loss_var_temp_n + 0.2 * loss_accel_temp_n + 0.2 * loss_dir_temp_n
                )


                group_vec = torch.stack([
                    group_recon_n,
                    group_represent_n,
                    group_bone_n,
                    group_angle_n,
                    group_temp_n
                ]).clamp(max=10.0)

                # ===================== positive learnable group weights =====================
                loss = loss_balancer(*group_vec.unbind())

                opt1.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(tf1.parameters()) + list(loss_balancer.parameters()), 
                    max_norm=1.0
                )
                opt1.step()

                mse = mean_squared_error(
                    f3.detach().cpu().numpy().flatten(),
                    pred.detach().cpu().numpy().flatten()
                )
                rmses.append(np.sqrt(mse))
                losses.append(loss.item())
                epoch_coord_recon_losses.append(loss_coord_recon.item())
                epoch_angle_recon_losses.append(loss_angle_recon.item())
                epoch_force_recon_losses.append(loss_force_recon.item())
                epoch_con_losses.append(con_loss_total.item())
                epoch_ortho_losses.append(loss_ortho.item())
                epoch_bone_stab_losses.append(loss_bone_stab.item())
                epoch_bone_sym_losses.append(loss_bone_sym.item())
                epoch_shoulder_angle_losses.append(loss_shoulder.item())
                epoch_hip_angle_losses.append(loss_hip.item())
                epoch_knee_angle_losses.append(loss_knee.item())
                epoch_angle_consistency_losses.append(loss_angle_consistency.item())
                epoch_var_temp_losses.append(loss_var_temp.item())
                epoch_accel_temp_losses.append(loss_accel_temp.item())
                epoch_dir_temp_losses.append(loss_dir_temp.item())

                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "rmse": f"{np.sqrt(mse):.4f}"
                })

            train_loss = np.mean(losses)
            train_rmse = np.mean(rmses)
            avg_coord_recon = np.mean(epoch_coord_recon_losses)
            avg_angle_recon = np.mean(epoch_angle_recon_losses)
            avg_force_recon = np.mean(epoch_force_recon_losses)
            avg_con = np.mean(epoch_con_losses)
            avg_ortho = np.mean(epoch_ortho_losses)
            avg_bone_stab = np.mean(epoch_bone_stab_losses)
            avg_bone_sym = np.mean(epoch_bone_sym_losses)
            avg_shoulder_angle = np.mean(epoch_shoulder_angle_losses)
            avg_hip_angle = np.mean(epoch_hip_angle_losses)
            avg_knee_angle = np.mean(epoch_knee_angle_losses)
            avg_angle_consistency = np.mean(epoch_angle_consistency_losses)
            avg_var_temp = np.mean(epoch_var_temp_losses)
            avg_accel_temp = np.mean(epoch_accel_temp_losses)
            avg_dir_temp = np.mean(epoch_dir_temp_losses)
            # ===================== Compute average group losses =====================
            # group_recon = (avg_coord_recon + avg_angle_recon + avg_force_recon) / 3.0

            # group_represent = (avg_con + avg_ortho) / 2.0

            # group_bone = (avg_bone_stab + avg_bone_sym) / 2.0

            # group_angle = (
            #     avg_shoulder_angle +
            #     avg_hip_angle +
            #     avg_knee_angle +
            #     avg_angle_consistency
            # ) / 4.0

            # group_temp = (
            #     avg_var_temp +
            #     avg_accel_temp +
            #     avg_dir_temp
            # ) / 3.0

            group_recon = (
                0.6 * avg_coord_recon +
                0.2 * avg_angle_recon +
                0.2 * avg_force_recon
            )

            group_represent = (
                0.8 * avg_con +
                0.2 * avg_ortho
            )

            group_bone = (
                0.8 * avg_bone_stab +
                0.2 * avg_bone_sym
            )

            group_angle = (
                0.1 * avg_shoulder_angle +
                0.15 * avg_hip_angle +
                0.15 * avg_knee_angle +
                0.6 * avg_angle_consistency
            )

            group_temp = (
                0.6 * avg_var_temp +
                0.2 * avg_accel_temp +
                0.2 * avg_dir_temp
            )
            # ---------- VALIDATION ----------
            tf1.eval()
            vlosses, vrmses = [], []
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"[Epoch {epoch:03d}] Validation", unit="batch", leave=False)
                for f1, f3, sid in val_bar:
                    f1, f3 = f1.to(device), f3.to(device)
                    pred, z_pose, z_view = tf1(f1)

                    min_len = min(pred.shape[1], f3.shape[1])
                    pred = pred[:, :min_len]
                    f3   = f3[:, :min_len]

                    # split coords / angles / forces
                    pred_coord  = pred[:, :, :36]
                    pred_angle  = pred[:, :, 36:42]
                    pred_force  = pred[:, :, 42:66]

                    gt_coord    = f3[:, :, :36]
                    gt_angle    = f3[:, :, 36:42]
                    gt_force    = f3[:, :, 42:66]

                    # ---------- coord loss ----------
                    coord_sq = (pred_coord - gt_coord) ** 2
                    coord_frame_loss = coord_sq.mean(dim=2)
                    loss_coord_recon = loss_coord_recon.mean()

                    # ---------- angle / force ----------
                    loss_angle_recon = loss_fn(pred_angle, gt_angle)
                    loss_force_recon = loss_fn(pred_force, gt_force)

                    # ---------- contrast ----------
                    video_ids = [get_video_group_id(s) for s in sid]
                    unique_videos = sorted(set(video_ids))

                    con_losses = []

                    for vid in unique_videos:
                        idx = [i for i, v in enumerate(video_ids) if v == vid]
                        if len(idx) > 1:
                            con_losses.append(contrastive_latent_loss(z_pose[idx]))

                    if len(con_losses) > 0:
                        con_loss_total = torch.stack(con_losses).mean()
                    else:
                        con_loss_total = torch.tensor(0.0, device=device)

                    # ---------- ortho ----------
                    loss_ortho = torch.abs(
                        nn.functional.cosine_similarity(z_pose, z_view)
                    ).mean()

                    # ---------- bone ----------
                    loss_bone_stab, loss_bone_sym = compute_bone_length_loss(pred)

                    # ---------- angle ----------
                    angles, loss_shoulder, loss_hip, loss_knee = compute_biomechanical_angle_loss(pred)
                    loss_angle_consistency = compute_angle_consistency_loss(pred, f1)

                    # ---------- temporal ----------
                    loss_var_temp, loss_accel_temp, loss_dir_temp = compute_temporal_loss(pred, f3)

                    # ---------- normalize ----------
                    loss_coord_recon_n = loss_coord_recon / LOSS_MEDIANS["coord_recon"]
                    loss_angle_recon_n = loss_angle_recon / LOSS_MEDIANS["angle_recon"]
                    loss_force_recon_n = loss_force_recon / LOSS_MEDIANS["force_recon"]

                    con_loss_total_n = con_loss_total / LOSS_MEDIANS["contrast"]
                    loss_ortho_n = loss_ortho / LOSS_MEDIANS["ortho"]

                    loss_bone_stab_n = loss_bone_stab / LOSS_MEDIANS["bone_stab"]
                    loss_bone_sym_n = loss_bone_sym / LOSS_MEDIANS["bone_sym"]

                    loss_shoulder_n = loss_shoulder / LOSS_MEDIANS["shoulder_angle"]
                    loss_hip_n = loss_hip / LOSS_MEDIANS["hip_angle"]
                    loss_knee_n = loss_knee / LOSS_MEDIANS["knee_angle"]
                    loss_angle_consistency_n = loss_angle_consistency / LOSS_MEDIANS["angle_consistency"]

                    loss_var_temp_n = loss_var_temp / LOSS_MEDIANS["var_temp"]
                    loss_accel_temp_n = loss_accel_temp / LOSS_MEDIANS["accel_temp"]
                    loss_dir_temp_n = loss_dir_temp / LOSS_MEDIANS["dir_temp"]

                    # ---------- group ----------
                    # group_recon = (loss_coord_recon_n + loss_angle_recon_n + loss_force_recon_n) / 3
                    # group_rep = (con_loss_total_n + loss_ortho_n) / 2
                    # group_bone = (loss_bone_stab_n + loss_bone_sym_n) / 2
                    # group_angle = (loss_shoulder_n + loss_hip_n + loss_knee_n + loss_angle_consistency_n) / 4
                    # group_temp = (loss_var_temp_n + loss_accel_temp_n + loss_dir_temp_n) / 3

                    group_recon = (
                        0.6 * loss_coord_recon_n +
                        0.2 * loss_angle_recon_n +
                        0.2 * loss_force_recon_n
                    )

                    group_rep = (
                        0.8 * con_loss_total_n +
                        0.2 * loss_ortho_n
                    )

                    group_bone = (
                        0.8 * loss_bone_stab_n +
                        0.2 * loss_bone_sym_n
                    )

                    group_angle = (
                        0.1 * loss_shoulder_n +
                        0.15 * loss_hip_n +
                        0.15 * loss_knee_n +
                        0.6 * loss_angle_consistency_n
                    )

                    group_temp = (
                        0.6 * loss_var_temp_n +
                        0.2 * loss_accel_temp_n +
                        0.2 * loss_dir_temp_n
                    )
                    
                    group_vec = torch.stack([
                        group_recon,
                        group_rep,
                        group_bone,
                        group_angle,
                        group_temp
                    ])

                    # ===================== positive learnable group weights =====================
                    loss = loss_balancer(*group_vec.unbind())

                    vlosses.append(loss.item())

                    mse = mean_squared_error(
                        f3.cpu().numpy().flatten(),
                        pred.cpu().numpy().flatten()
                    )

                    vrmses.append(np.sqrt(mse))
                    val_bar.set_postfix({
                        "val_loss": f"{loss:.4f}",
                        "val_rmse": f"{np.sqrt(mse):.4f}"
                    })

            val_loss = np.mean(vlosses)
            val_rmse = np.mean(vrmses)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_rmse"].append(train_rmse)
            history["val_rmse"].append(val_rmse)

            with torch.no_grad():
                raw_w = np.array([
                    loss_balancer.raw_recon.detach().cpu().item(),
                    loss_balancer.raw_represent.detach().cpu().item(),
                    loss_balancer.raw_bone.detach().cpu().item(),
                    loss_balancer.raw_angle.detach().cpu().item(),
                    loss_balancer.raw_temp.detach().cpu().item(),
                ], dtype=np.float32)

                softplus_w = np.array([
                    (F.softplus(loss_balancer.raw_recon) + loss_balancer.eps).detach().cpu().item(),
                    (F.softplus(loss_balancer.raw_represent) + loss_balancer.eps).detach().cpu().item(),
                    (F.softplus(loss_balancer.raw_bone) + loss_balancer.eps).detach().cpu().item(),
                    (F.softplus(loss_balancer.raw_angle) + loss_balancer.eps).detach().cpu().item(),
                    (F.softplus(loss_balancer.raw_temp) + loss_balancer.eps).detach().cpu().item(),
                ], dtype=np.float32)

                norm_w = softplus_w / softplus_w.sum()

            for g, rv, sv, nv in zip(GROUP_NAMES, raw_w, softplus_w, norm_w):
                weight_history["raw"][g].append(rv)
                weight_history["softplus"][g].append(sv)
                weight_history["norm"][g].append(nv)
            
            plot_metrics()
            plot_loss_weight()

            w_np = loss_balancer.get_weight_vector().detach().cpu().numpy()
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(tf1.state_dict(), f"traindata/models5/tf1_best_fold{fold_id}.pt")
                wait = 0
            else:
                wait += 1

            if wait >= PATIENCE:
                break

        # ===================== TF1 TEST + SAVE PRED (only used files) =====================
        tf1.load_state_dict(torch.load(f"traindata/models5/tf1_best_fold{fold_id}.pt"))
        tf1.eval()

        # 建立輸出資料夾
        os.makedirs("traindata/f3_pred5", exist_ok=True)

        test_rmses = []

        with torch.no_grad():
            for f1, f3, sid in tqdm(test_loader, desc=f"TF1 Test RMSE Fold {fold_id}", unit="batch", leave=False):
                f1 = f1.to(device)
                f3 = f3.to(device)

                pred, _, _ = tf1(f1)

                min_len = min(pred.shape[1], f3.shape[1])
                pred = pred[:, :min_len]
                f3   = f3[:, :min_len]

                mse = mean_squared_error(
                    f3.cpu().numpy().flatten(),
                    pred.cpu().numpy().flatten()
                )
                test_rmses.append(np.sqrt(mse))

        test_rmse = np.mean(test_rmses)
        all_fold_rmse.append(float(test_rmse))

        # ===== combine file lists =====

        # 收集所有 TF2 會用到的 files
        used_files = sorted(list(set(
            train_files + val_files + test_files
        )))


        for sid in tqdm(used_files, desc="Saving TF1 predicted f3"):
            # 讀取當前 sid 的 feature1
            tra_path = f"traindata/features/feature5/{sid}.csv"
            tst_path = f"testdata/feature5/{sid}.csv"

            if os.path.exists(tra_path):
                csv_path = tra_path
            else:
                csv_path = tst_path

            f1 = pd.read_csv(csv_path, header=None).values
            f1 = f1[:, :30]
            f1 = torch.tensor(f1, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred, _, _= tf1(f1)

            # 存成 npy
            np.save(
                f"traindata/f3_pred5/{sid}.npy",
                pred.squeeze(0).cpu().numpy()
            )

        f3_pred_cache = load_f3_pred_cache(used_files, "traindata/f3_pred5")

        seed_everything(SEED + 1000 + fold_id)

        # ===================== TF2 TRAIN =====================
        tf2 = TF2().to(device)
        opt2 = torch.optim.Adam(tf2.parameters(), lr=LR, weight_decay=WEIGHTDECAY)
        loss_cls = nn.BCEWithLogitsLoss()

        ds2_train = ConcatDataset([
            DeadliftDatasetCls("traindata/features/feature5", "traindata/ground_truth.txt", file_list=train_tra, cache_in_memory=True),
            DeadliftDatasetCls("testdata/feature5", "testdata/ground_truth.txt", file_list=train_tst, cache_in_memory=True),
        ])
        ds2_val = ConcatDataset([
            DeadliftDatasetCls("traindata/features/feature5", "traindata/ground_truth.txt", file_list=val_tra, cache_in_memory=True),
            DeadliftDatasetCls("testdata/feature5", "testdata/ground_truth.txt", file_list=val_tst, cache_in_memory=True),
        ])
        ds2_test = ConcatDataset([
            DeadliftDatasetCls("traindata/features/feature5", "traindata/ground_truth.txt", file_list=test_tra, cache_in_memory=True),
            DeadliftDatasetCls("testdata/feature5", "testdata/ground_truth.txt", file_list=test_tst, cache_in_memory=True),
        ])

        train_loader = DataLoader(
            ds2_train,
            batch_size=BATCH,
            shuffle=True,
            num_workers=N_WORKERS,
            worker_init_fn=seed_worker if N_WORKERS > 0 else None,
            generator=make_torch_generator(SEED + 2000 + fold_id * 100 + 1)
        )
        val_loader = DataLoader(
            ds2_val,
            batch_size=BATCH,
            shuffle=False,
            num_workers=N_WORKERS,
            worker_init_fn=seed_worker if N_WORKERS > 0 else None,
            generator=make_torch_generator(SEED + 2000 + fold_id * 100 + 2)
        )
        test_loader = DataLoader(
            ds2_test,
            batch_size=BATCH,
            shuffle=False,
            num_workers=N_WORKERS,
            worker_init_fn=seed_worker if N_WORKERS > 0 else None,
            generator=make_torch_generator(SEED + 2000 + fold_id * 100 + 3)
        )

        # ---------- Training setup ----------
        best_score, wait = -1e9, 0
        history = {"train_loss": [], "val_loss": [], "train_micro_f1": [], "val_micro_f1": []}

        def plot_metrics():
            # ---------- Loss ----------
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(history["train_loss"], label="Train Loss", color="tab:blue")
            ax.plot(history["val_loss"], label="Val Loss", color="tab:orange")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("TF2-Loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"traindata/plots5/tf2_loss_fold{fold_id}.png")
            plt.close(fig)

            # ---------- micro-F1 ----------
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(history["train_micro_f1"], label="Train micro-F1", color="tab:blue")
            ax.plot(history["val_micro_f1"], label="Val micro-F1", color="tab:orange")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Micro-F1")
            ax.set_title("TF2-Micro-F1")
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"traindata/plots5/tf2_microf1_fold{fold_id}.png")
            plt.close(fig)

        # ---------- Training loop ----------
        for epoch in range(1, EPOCHS + 1):
            tf2.train()

            losses = []
            all_y_cls, all_p_cls = [], []
            all_y_rpe, all_p_rpe = [], []

            # ---------- Train ----------
            for f1, y_cls, y_round, sid in tqdm(
                train_loader,
                desc=f"[Epoch {epoch:03d}] Training",
                unit="batch",
                leave=False
            ):
                f1 = f1.to(device)
                y_cls = y_cls.to(device)

                f1 = augment_f1(f1)

                f3 = get_f3_batch_from_cache(sid, f3_pred_cache).to(device)

                min_len = min(f1.shape[1], f3.shape[1])
                f1_tf2 = f1[:, :min_len]
                f3_tf2 = f3[:, :min_len]

                logits = tf2(f1_tf2, f3_tf2)

                loss = loss_cls(logits, y_cls)

                opt2.zero_grad()
                loss.backward()
                opt2.step()

                losses.append(loss.item())

                prob = torch.sigmoid(logits)
                pred_cls = (prob >= 0.5).int()

                all_y_cls.append(y_cls.cpu().numpy())
                all_p_cls.append(pred_cls.cpu().numpy())

            train_loss = np.mean(losses)

            train_micro_f1 = f1_score(
                np.vstack(all_y_cls),
                np.vstack(all_p_cls),
                average="micro"
            )

            # ---------- Validation ----------
            tf2.eval()
            vloss = []

            all_y_cls = []
            all_prob_cls = []

            for f1, y_cls, y_round, sid in tqdm(
                val_loader,
                desc=f"[Epoch {epoch:03d}] Validation",
                unit="batch",
                leave=False
            ):
                with torch.no_grad():
                    f1 = f1.to(device)
                    y_cls = y_cls.to(device)

                    f3 = get_f3_batch_from_cache(sid, f3_pred_cache).to(device)

                    L = min(f1.shape[1], f3.shape[1])
                    f1_tf2 = f1[:, :L]
                    f3_tf2 = f3[:, :L]

                    logits = tf2(f1_tf2, f3_tf2)

                    loss_c = loss_cls(logits, y_cls)
                    vloss.append(loss_c.item())

                    prob = torch.sigmoid(logits)

                    all_y_cls.append(y_cls.cpu().numpy())
                    all_prob_cls.append(prob.cpu().numpy())

            val_loss = np.mean(vloss)

            Y = np.vstack(all_y_cls)
            P = np.vstack(all_prob_cls)

            # ===== AUC (用來選模型) =====
            try:
                auc_hip = roc_auc_score(Y[:,0], P[:,0])
            except:
                auc_hip = 0.0

            try:
                auc_knee = roc_auc_score(Y[:,1], P[:,1])
            except:
                auc_knee = 0.0

            val_macro_auc = (auc_hip + auc_knee) / 2.0

            # ===== Youden threshold (各自) =====
            if len(np.unique(Y[:, 0])) > 1:
                best_t_hip, _ = youden_threshold(Y[:, 0], P[:, 0])
            else:
                best_t_hip = 0.5

            if len(np.unique(Y[:, 1])) > 1:
                best_t_knee, _ = youden_threshold(Y[:, 1], P[:, 1])
            else:
                best_t_knee = 0.5

            # 用這兩個 threshold 算 val micro-F1（只做紀錄）
            pred_val = np.zeros_like(P)
            pred_val[:,0] = (P[:,0] >= best_t_hip).astype(int)
            pred_val[:,1] = (P[:,1] >= best_t_knee).astype(int)

            val_micro_f1 = f1_score(Y, pred_val, average="micro")

            val_score = val_macro_auc

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_micro_f1"].append(train_micro_f1)
            history["val_micro_f1"].append(val_micro_f1)
            plot_metrics()

            if val_score > best_score:
                best_score = val_score
                best_threshold = [best_t_hip, best_t_knee]
                torch.save(
                    {"model": tf2.state_dict(), "threshold": best_threshold},
                    f"traindata/models5/tf2_best_fold{fold_id}.pt"
                )
                wait = 0
            else:
                wait += 1

            if wait >= PATIENCE:
                break

        # ===================== XGBOOST TRAIN (rounded_back) =====================
        round_labels = {}
        with open("testdata/ground_truth.txt", "r") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                sid = parts[0]
                round_labels[sid] = int(parts[4])

        xgb_train_X, xgb_train_y = build_xgb_xy(train_subj, "testdata/feature5", round_labels)
        xgb_val_X, xgb_val_y     = build_xgb_xy(val_subj,   "testdata/feature5", round_labels)

        xgb = XGBClassifier(
            n_estimators=1500,
            max_depth=10,
            learning_rate=0.008,
            subsample=0.1,
            colsample_bytree=0.55,
            reg_lambda=0.75,
            random_state=SEED,
            tree_method="hist",
            eval_metric="auc"
        )

        xgb.fit(
            xgb_train_X, xgb_train_y,
            eval_set=[(xgb_val_X, xgb_val_y)],
            verbose=False
        )


        p_val = xgb.predict_proba(xgb_val_X)[:,1]

        try:
            val_auc = roc_auc_score(xgb_val_y, p_val)
        except:
            val_auc = 0.0

        try:
            xgb_best_t, _ = youden_threshold(xgb_val_y, p_val)
        except:
            xgb_best_t = 0.5


        xgb.save_model(f"traindata/models5/xgb_rounded_fold{fold_id}.json")
        with open(f"traindata/models5/xgb_rounded_fold{fold_id}_thr.txt", "w") as f:
            f.write(str(xgb_best_t))


        # ===================== TEST EVALUATION (TF2 for hip/knee + XGB for rounded) =====================
        # Load test labels (4-hot output reference)
        test_labels = {}
        with open("testdata/ground_truth.txt", "r") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                sid = parts[0]
                y_cls4 = list(map(int, parts[1:5]))
                test_labels[sid] = y_cls4

        # Load TF2 ckpt
        ckpt = torch.load(f"traindata/models5/tf2_best_fold{fold_id}.pt")
        tf2.load_state_dict(ckpt["model"])
        best_threshold = ckpt["threshold"]
        tf2.eval()

        # Collect test files by subject filtering
        all_test_files = [
            f.replace(".csv", "")
            for f in os.listdir("testdata/feature5")
            if f.endswith(".csv")
        ]
        test_files_real = [f for f in all_test_files if get_subject_id(f) in test_subj]
        test_files_real = sorted(test_files_real)

        all_y_cls, all_p_cls = [], []
        with open(detail_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            with torch.no_grad():
                for sid in tqdm(test_files_real, desc="Testing", unit="file"):
                    arr = np.loadtxt(f"testdata/feature5/{sid}.csv", delimiter=",")  # (T,33)
                    f1_30 = arr[:, :30]
                    f1_30_t = torch.tensor(f1_30, dtype=torch.float32).unsqueeze(0).to(device)

                    # TF1 --> f3
                    f3, _, _= tf1(f1_30_t)  # (1,T,60)

                    L = min(f1_30_t.shape[1], f3.shape[1])
                    f1_tf2 = f1_30_t[:, :L]
                    f3_tf2 = f3[:, :L]

                    logits = tf2(f1_tf2, f3_tf2)

                    prob2 = torch.sigmoid(logits).cpu().numpy()[0]          # (2,)
                    pred_err2 = (prob2 >= best_threshold).astype(int)       # (2,) => [hip, knee] in your gt order

                    # XGB for rounded_back using k,o,v flattened
                    kov60 = flatten_kov_60_from_feature5(f"testdata/feature5/{sid}.csv").reshape(1, -1)
                    p_rounded = xgb.predict_proba(kov60)[0, 1]
                    rounded_pred = int(p_rounded >= xgb_best_t)

                    # Build final 4-hot: [correct, hip_first, knee_first, rounded_back]
                    hip_pred = int(pred_err2[0])
                    knee_pred = int(pred_err2[1])
    
                    if (hip_pred + knee_pred + rounded_pred) == 0:
                        pred_cls = np.array([1, 0, 0, 0], dtype=int)
                    else:
                        pred_cls = np.array([0, hip_pred, knee_pred, rounded_pred], dtype=int)

                    y_cls4 = test_labels[sid]
                    
                    # ===== 寫入 CSV =====
                    gt_c, gt_h, gt_k, gt_r = map(int, y_cls4)
                    pred_c, pred_h, pred_k, pred_r = map(int, pred_cls)

                    writer.writerow([
                        fold_id,
                        sid,
                        gt_c, gt_h, gt_k, gt_r,
                        pred_c, pred_h, pred_k, pred_r
                    ])

                    all_y_cls.append(np.array(y_cls4, dtype=int))
                    all_p_cls.append(pred_cls)

                    # 取得機率（ROC 需要連續 score）
                    p_hip, p_knee = prob2
                    p_rounded = xgb.predict_proba(kov60)[0, 1]


        all_y_cls = np.array(all_y_cls)
        all_p_cls = np.array(all_p_cls)

        micro_f1 = f1_score(all_y_cls, all_p_cls, average="micro")
        macro_f1 = f1_score(all_y_cls, all_p_cls, average="macro")

        # KEEP THIS PHASE
        all_fold_micro.append(micro_f1)
        all_fold_macro.append(macro_f1)
        c_f1, h_f1, k_f1, r_f1 = f1_score(all_y_cls, all_p_cls, average=None, zero_division=0)
        all_fold_c.append(float(c_f1))
        all_fold_h.append(float(h_f1))
        all_fold_k.append(float(k_f1))
        all_fold_r.append(float(r_f1))

        # ===================== AUC (hip + knee + rounded) =====================
        try:
            y_hip, s_hip = [], []
            y_knee, s_knee = [], []
            y_round, s_round = [], []

            with torch.no_grad():
                for sid in test_files_real:
                    arr = np.loadtxt(f"testdata/feature5/{sid}.csv", delimiter=",")
                    f1_30 = arr[:, :30]
                    f1_30_t = torch.tensor(f1_30, dtype=torch.float32).unsqueeze(0).to(device)

                    # ---------- TF1 -> f3 ----------
                    f3, _, _ = tf1(f1_30_t)

                    # 對齊長度
                    L = min(f1_30_t.shape[1], f3.shape[1])
                    f1_in = f1_30_t[:, :L]
                    f3_in = f3[:, :L]

                    # ---------- TF2 prob ----------
                    logits = tf2(f1_in, f3_in)
                    prob2 = torch.sigmoid(logits).cpu().numpy()[0]   # (2,)

                    gt = test_labels[sid]

                    y_hip.append(gt[1])
                    y_knee.append(gt[2])
                    s_hip.append(prob2[0])
                    s_knee.append(prob2[1])

                    # ---------- XGB prob ----------
                    if xgb is None:
                        p_round = 0.0
                    else:
                        kov60 = flatten_kov_60_from_feature5(
                            f"testdata/feature5/{sid}.csv"
                        ).reshape(1, -1)
                        p_round = float(xgb.predict_proba(kov60)[0, 1])

                    y_round.append(gt[3])
                    s_round.append(p_round)

            y_hip = np.array(y_hip)
            s_hip = np.array(s_hip)
            y_knee = np.array(y_knee)
            s_knee = np.array(s_knee)
            y_round = np.array(y_round)
            s_round = np.array(s_round)

            auc_hip = np.nan
            auc_knee = np.nan
            auc_round = np.nan
            micro_auc = np.nan
            macro_auc = np.nan

            # ---------- HIP AUC ----------
            if len(np.unique(y_hip)) > 1:
                auc_hip = roc_auc_score(y_hip, s_hip)

            # ---------- KNEE AUC ----------
            if len(np.unique(y_knee)) > 1:
                auc_knee = roc_auc_score(y_knee, s_knee)

            # ---------- ROUNDED AUC ----------
            if len(np.unique(y_round)) > 1:
                auc_round = roc_auc_score(y_round, s_round)

            # ---------- MICRO / MACRO ----------
            y_all = np.column_stack([y_hip, y_knee, y_round])   # (N,3)
            s_all = np.column_stack([s_hip, s_knee, s_round])   # (N,3)

            if len(np.unique(y_all.ravel())) > 1:
                micro_auc = roc_auc_score(y_all.ravel(), s_all.ravel())

            macro_auc = np.nanmean([auc_hip, auc_knee, auc_round])

            all_fold_auc_h.append(auc_hip)
            all_fold_auc_k.append(auc_knee)
            all_fold_auc_r.append(auc_round)
            all_fold_auc_micro.append(micro_auc)
            all_fold_auc_macro.append(macro_auc)

            plt.figure(figsize=(5,5))

            for y, s, name in [
                (y_hip, s_hip, "Hip"),
                (y_knee, s_knee, "Knee"),
                (y_round, s_round, "Rounded")
            ]:
                if len(set(y)) > 1:
                    fpr, tpr, _ = roc_curve(y, s)
                    aucv = roc_auc_score(y, s)
                    plt.plot(fpr, tpr, label=f"{name} AUC={aucv:.3f}")

            plt.plot([0,1],[0,1],"--",color="gray")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC Fold {fold_id}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"traindata/plots5/roc_auc_fold{fold_id}.png")
            plt.close()

        except Exception as e:
            print("AUC failed:", e)

    rows = []
    for i in range(len(all_fold_micro)):
        rows.append([
            f"fold{i+1}",
            all_fold_c[i], all_fold_h[i], all_fold_k[i], all_fold_r[i],
            all_fold_micro[i], all_fold_macro[i],
            all_fold_auc_h[i], all_fold_auc_k[i], all_fold_auc_r[i],
            all_fold_auc_micro[i], all_fold_auc_macro[i],
        ])

    rows.append([
        "mean",
        float(np.mean(all_fold_c)), float(np.mean(all_fold_h)), float(np.mean(all_fold_k)), float(np.mean(all_fold_r)),
        float(np.mean(all_fold_micro)), float(np.mean(all_fold_macro)),
        float(np.mean(all_fold_auc_h)), float(np.mean(all_fold_auc_k)), float(np.mean(all_fold_auc_r)),
        float(np.mean(all_fold_auc_micro)), float(np.mean(all_fold_auc_macro)),
    ])

    rows.append([
        "std",
        float(np.std(all_fold_c)), float(np.std(all_fold_h)), float(np.std(all_fold_k)), float(np.std(all_fold_r)),
        float(np.std(all_fold_micro)), float(np.std(all_fold_macro)),
        float(np.std(all_fold_auc_h)), float(np.std(all_fold_auc_k)), float(np.std(all_fold_auc_r)),
        float(np.std(all_fold_auc_micro)), float(np.std(all_fold_auc_macro)),
    ])

    df = pd.DataFrame(rows, columns=["", "c_f1", "h_f1", "k_f1", "r_f1", "micro_f1", "macro_f1",
                                     "h_auc", "k_auc", "r_auc", "micro_auc", "macro_auc"])

    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    os.makedirs("traindata/plots5", exist_ok=True)
    summary_path = f"traindata/plots5/{count}_cv_summary.csv"
    df.to_csv(summary_path, index=False, float_format="%.4f")

    if count == 1:
        break


import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

F1_DIR = "testdata/yolomarkers"
FEATURE5_DIR = "testdata/feature5"
BACK_FEATURES_FILE = "testdata/back_features.csv"

os.makedirs(FEATURE5_DIR, exist_ok=True)

# back_features 欄位
K_COLS = [f"k{i}" for i in range(1, 21)]
O_COLS = [f"o{i}" for i in range(1, 21)]
V_COLS = [f"v{i}" for i in range(1, 21)]


def build_fback_sequence(row):
    """將 back_features 單列轉為 (20,3) = [k, o_encoded, v]"""

    ks = pd.to_numeric(row[K_COLS], errors="coerce").values.astype(float)
    ks = np.where(np.isnan(ks), 0.5, ks)

    os_raw = row[O_COLS].astype(str).values
    os_encoded = np.array([1 if o == "nose" else 0 for o in os_raw])

    vs = pd.to_numeric(row[V_COLS], errors="coerce").values.astype(float)
    vs = np.where(np.isnan(vs),
                  np.nanmean(vs) if not np.all(np.isnan(vs)) else 0,
                  vs)

    return np.stack([ks, os_encoded, vs], axis=1)


def parse_yolo_filename(fname):
    """解析 001A_1_120_260.csv"""
    base = os.path.basename(fname)
    m = re.match(r"(\d{4}[A-Za-z])_(\d)_(\d+)_(\d+)\.csv", base)
    if not m:
        return None
    return m.groups()


print("📥 讀取 back_features.csv ...")
back_df = pd.read_csv(BACK_FEATURES_FILE)

files = [f for f in os.listdir(F1_DIR) if f.endswith(".csv")]
print(f"📂 總檔案數: {len(files)}")

for file in tqdm(files, desc="Building feature5", unit="file"):

    parsed = parse_yolo_filename(file)
    if parsed is None:
        continue

    subject_class, viewangle, start, end = parsed

    f1_path = os.path.join(F1_DIR, file)

    # === ⚠ 讀取無 header CSV ===
    f1_df = pd.read_csv(f1_path, header=None)
    f1 = f1_df.values

    # shape 必須為 (20, 30)
    if f1.shape != (20, 30):
        print(f"⚠ f1 shape錯誤: {file} -> {f1.shape}")
        continue

    # 找 back_features 對應資料
    target_video_name = f"{subject_class}_{viewangle}_{start}_{end}.mp4"

    row = back_df[back_df["filename"] == target_video_name]

    if row.empty:
        print(f"❌ 找不到 back_features 對應: {target_video_name}")
        continue

    row = row.iloc[0]

    fback = build_fback_sequence(row)

    if fback.shape != (20, 3):
        print(f"⚠ fback shape錯誤: {target_video_name}")
        continue

    # 合併成 (20 × 33)
    f5 = np.concatenate([f1, fback], axis=1)

    # 輸出
    out_path = os.path.join(FEATURE5_DIR, file)
    pd.DataFrame(f5).to_csv(out_path, index=False, header=False)

print("✅ feature5 全部建構完成！")

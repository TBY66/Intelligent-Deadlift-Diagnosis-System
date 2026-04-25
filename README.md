# Intelligent Deadlift Diagnosis System

A macOS desktop application that automatically analyzes deadlift videos — detecting repetitions, classifying movement faults per rep, and generating coaching feedback.

---

## Overview

Upload a deadlift video and the system will:

1. **Detect repetitions** — scans nose Y-coordinate and equipment motion to segment each rep
2. **Classify movement** — runs pose estimation + segmentation + ML models on 20 sampled frames per rep
3. **Generate feedback** — rule-based coaching text based on detected fault patterns

Each rep is classified into a 4-class vector `[correct, hip_first, knee_dominant, rounded_back]`.

---

## Features

- Drag-and-drop or browse video upload (MP4, MOV, AVI, MKV)
- Frame-accurate video player with scrubbing
- **Nose Trajectory chart** — nose Y-coordinate over time, synced with video playback; dashed lines mark rep boundaries
- Per-rep result cards with duration and fault badges
- Rule-based feedback with beginning / middle / end voting for sets ≥ 3 reps
- Upload history with rename, delete, and re-analyze
- Double-clickable `.app` launcher for macOS

---

## Tech Stack

| Layer | Library |
|---|---|
| GUI | PyQt5 |
| Pose estimation | YOLOv11s-pose (`ultralytics`) |
| Segmentation | YOLOv11s-seg |
| Equipment detection | YOLOv11s-gymequipment |
| Joint classifier | Custom Transformer (TF1 + TF2, PyTorch) |
| Back curvature | XGBoost |
| Video I/O | OpenCV |
| Numerics | NumPy, SciPy |

Inference runs on **MPS** (Apple Silicon) → CUDA → CPU, auto-detected at startup.

---

## Project Structure

```
.
├── appv1.2.py          # Main GUI application (current version)
├── appv1.1.py          # Previous version
├── appv1.0.py          # Initial version
├── pipeline.py         # DiagnosisEngine — all model inference logic
├── build_app.sh        # Builds "Deadlift Diagnosis.app" macOS bundle
├── run.sh              # Quick terminal launcher
├── requirements.txt    # Python dependencies
├── materials/          # Icons and layout references
├── raw_py/             # Training & data-preparation scripts
│   ├── train_model.py
│   ├── extract_keypoint.py
│   ├── extract_backfeatures.py
│   ├── build_inputdata.py
│   └── ...
├── yolo_models/        # ⚠ Not in repo — download separately
└── inference_models/   # ⚠ Not in repo — download separately
```

---

## Setup

### Requirements

- macOS (Apple Silicon recommended for MPS acceleration)
- [Anaconda](https://www.anaconda.com/) / conda

### Install dependencies

```bash
conda create -n deadlift python=3.12
conda activate deadlift
pip install -r requirements.txt
```

### Model files

The YOLO and inference model files are not included in this repository due to size.  
Place them as follows before running:

```
yolo_models/
  yolov11s-pose.pt
  yolov11s-seg.pt
  yolov11s-gymequipment.pt

inference_models/
  tf1_best_fold1.pt
  tf2_best_fold1.pt
  xgb_rounded_fold5.json
  xgb_rounded_fold5_thr.txt
```

---

## Running

### Terminal

```bash
./run.sh
# or
/opt/anaconda3/bin/python appv1.2.py
```

### macOS App Bundle

Build a double-clickable `.app` (runs in-place, no bundling of models):

```bash
./build_app.sh
```

Then double-click `Deadlift Diagnosis.app`.  
First launch: right-click → **Open** to bypass Gatekeeper.

Crash logs: `~/Library/Logs/DeadliftDiagnosis.log`

---

## Versions

| Version | Description |
|---|---|
| v1.0 | Initial working GUI |
| v1.1 | UI redesign — dark OLED theme, financial dashboard palette |
| v1.2 | Nose Trajectory chart synced with video playback; macOS .app packaging |
| v2.0 *(planned)* | Windows `.exe` |
| v3.0 *(planned)* | iPhone Safari web app |
| v4.0 *(planned)* | Cross-browser (Chrome, Android) |

---

## Classification Model

Each rep is sampled at 20 evenly-spaced frames. Features extracted:

- **f1** (30-dim): 12 normalised joint coordinates + 6 joint angles
- **f3** (66-dim): output of TF1 Transformer encoder
- **back** (3-dim): curvature `k`, orientation `o`, view angle `v` from segmentation mask

**TF1 + TF2** (cross-attention Transformer) → hip / knee fault  
**XGBoost** → rounded back fault

Models were trained with 5-fold cross-validation; fold-5 checkpoint is used for inference.

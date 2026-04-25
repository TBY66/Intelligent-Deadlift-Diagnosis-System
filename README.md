# Intelligent Deadlift Diagnosis System

A macOS desktop application that analyzes deadlift videos using computer vision and machine learning — automatically detecting repetitions, classifying movement faults, and delivering coaching feedback.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange)
![Platform](https://img.shields.io/badge/Platform-macOS-lightgrey)

---

## Demo

| Upload & Analyze | Results |
|---|---|
| Drag-and-drop a video and click Analyze | Per-rep fault badges, nose trajectory chart, and coaching feedback |

---

## How It Works

1. **Scan** — YOLOv11s-pose tracks the nose Y-coordinate across the entire video; YOLOv11s-gymequipment tracks the barbell/kettlebell. Peak detection on the nose signal segments the video into individual reps.
2. **Extract** — 20 frames are sampled per rep. Pose keypoints (30-dim) and back-curvature features (3-dim) are extracted via pose estimation and segmentation.
3. **Classify** — A cross-attention Transformer (TF1 → TF2) classifies hip and knee faults; XGBoost classifies rounded back. Each rep receives a 4-class label `[correct, hip_first, knee_dominant, rounded_back]`.
4. **Feedback** — Rule-based coaching text is generated, with beginning / middle / end voting for sets of 3+ reps.

---

## Features

- Drag-and-drop or browse video upload (MP4, MOV, AVI, MKV)
- Frame-accurate video player with scrubbing
- **Nose Trajectory chart** — nose Y-coordinate over time, synced with playback; amber dashed lines mark rep boundaries
- Per-rep result cards showing duration and fault badges
- Coaching feedback panel
- Upload history with rename, delete, and re-analyze
- Double-clickable macOS `.app` launcher

---

## Requirements

- macOS (Apple Silicon recommended — MPS acceleration)
- Python 3.12 via [Anaconda](https://www.anaconda.com/)

```bash
pip install -r requirements.txt
```

### Model files (not included)

Place the following in the project root before running:

```
yolo_models/
├── yolov11s-pose.pt
├── yolov11s-seg.pt
└── yolov11s-gymequipment.pt

inference_models/
├── tf1_best_fold1.pt
├── tf2_best_fold1.pt
├── xgb_rounded_fold5.json
└── xgb_rounded_fold5_thr.txt
```

---

## Usage

**Terminal**
```bash
./run.sh
```

**macOS App**
```bash
./build_app.sh          # creates "Deadlift Diagnosis.app"
open "Deadlift Diagnosis.app"
```

First launch: right-click → **Open** to bypass Gatekeeper.  
Logs: `~/Library/Logs/DeadliftDiagnosis.log`

---

## Project Structure

```
├── appv1.2.py          # GUI application
├── pipeline.py         # Inference backend (DiagnosisEngine)
├── build_app.sh        # macOS .app bundle builder
├── run.sh              # Terminal launcher
├── requirements.txt
└── materials/          # App icons
```

---

## Roadmap

| Version | Target | Status |
|---|---|---|
| v1 | macOS desktop app | ✅ Done |
| v2 | Windows `.exe` | Planned |
| v3 | iPhone Safari web app | Planned |
| v4 | Cross-browser (Chrome, Android) | Planned |

# Intelligent Deadlift Diagnosis System

A desktop application for deadlift video analysis that detects repetitions, classifies lifting faults, and generates coaching feedback from computer vision and machine learning models.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange)
![Platform](https://img.shields.io/badge/Platform-macOS-lightgrey)

---

## Overview

This project combines a PyQt5 desktop UI with a multi-stage inference pipeline for deadlift assessment:

1. Repetitions are detected from pose and motion signals across the full video.
2. Per-rep features are extracted from sampled frames.
3. Faults are classified into four labels:
   - `correct`
   - `hip_first`
   - `knee_dominant`
   - `rounded_back`
4. Styled coaching feedback is generated in either `professional` or `friendly` mode.

The current app entrypoint is `appv1.3.py`, which uses `pipelinev1.3.py`.

---

## Current Features

- Drag-and-drop or file-picker video upload
- Per-video history with rename, delete, and re-analysis
- Repetition segmentation and per-rep diagnosis
- Fault badges and rep timing summaries
- Nose trajectory visualization
- Two feedback styles: `professional` and `friendly`
- Optional local GGUF feedback model with rule-based fallback
- Standard and fast analysis modes

---

## Tech Stack

- Python 3.12
- PyQt5 for the desktop interface
- OpenCV for video processing
- PyTorch for model inference
- Ultralytics YOLO for pose, segmentation, and equipment detection
- XGBoost for rounded-back classification
- Optional `llama-cpp-python` for local styled feedback generation

---

## Repository Layout

```text
.
├── appv1.3.py                         # Desktop GUI
├── pipelinev1.3.py                    # Inference pipeline
├── profile_pipeline_v13.py            # Pipeline profiling helper
├── requirements.txt                   # Full dependencies
├── requirements_app_only.txt          # Lighter app-focused dependencies
├── inference_models/                  # Transformer + XGBoost inference weights
├── yolo_models/                       # YOLO and MoveNet model files
├── materials/                         # UI assets
├── sample/                            # Example videos
├── deadlift_hybrid_feedback_colab.ipynb
└── Fitness Equipment Recognition/     # Training data/assets for equipment detection
```

---

## Setup

### 1. Create an environment

```bash
conda create -n deadlift-diagnosis python=3.12
conda activate deadlift-diagnosis
```

### 2. Install dependencies

For the full feature set, including optional local LLM feedback support:

```bash
pip install -r requirements.txt
```

For the desktop app without GGUF feedback dependencies:

```bash
pip install -r requirements_app_only.txt
```

---

## Model Files

The app expects model assets under these folders:

```text
yolo_models/
inference_models/
```

`pipelinev1.3.py` currently loads these primary inference files:

```text
yolo_models/yolov11s-pose.pt
yolo_models/yolov11n-pose.pt
yolo_models/yolov11s-seg.pt
yolo_models/yolov11n-seg.pt
yolo_models/yolov11s-gymequipment.pt
inference_models/tf1_best_fold1.pt
inference_models/tf2_best_fold1.pt
inference_models/xgb_rounded_fold5.json
inference_models/xgb_rounded_fold5_thr.txt
```

Optional feedback model:

```text
feedback_model/smollm2_deadlift_hybrid_q4_k_m.gguf
```

If the GGUF file is missing, the app automatically falls back to built-in rule-based feedback.

---

## Running the App

Launch the desktop app with:

```bash
python appv1.3.py
```

Then:

1. Upload a deadlift video.
2. Choose analysis mode if needed.
3. Run analysis.
4. Review rep-level fault predictions and feedback.
5. Switch between `professional` and `friendly` feedback styles.

---

## How the Pipeline Works

1. `process_video()` scans the video and detects rep boundaries.
2. The pipeline samples 20 frames per repetition.
3. Pose and segmentation features are extracted for each rep.
4. Transformer models classify hip-first and knee-dominant faults.
5. XGBoost classifies rounded-back faults.
6. Feedback is generated from either:
   - the optional local GGUF model, or
   - the built-in fallback cue system

---

## Notes

- The current project is built around a desktop workflow, with macOS being the primary target environment.
- `requirements.txt` includes `llama-cpp-python` and `tflite-runtime`; these are not strictly required for the basic app path if you use the fallback behavior.

---

## Roadmap

- Improve packaging for easier desktop distribution
- Add clearer setup steps for model asset download and placement
- Expand platform support beyond the current macOS-focused workflow
- Continue refining styled feedback generation

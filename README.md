# Second Vision AI

**Dataset curation, model training, and export pipeline for the Second Vision assistive navigation system.**

Second Vision is an IoT-based smart glass for visually impaired users. This repository contains everything needed to produce a deployment-ready YOLOv8s object detection model — from raw dataset curation through ONNX export and Hailo compilation preparation.

> **Note:** This repository does **not** contain embedded production code. Raspberry Pi runtime, threading, UART communication, ESP32 integration, and application logic are maintained in the separate [second-vision](https://github.com/KRMeeag/second-vision) production repository.

---

## Objective

Produce a YOLOv8s model that:

1. Performs well on the target assistive navigation use case
2. Exports successfully to ONNX
3. Remains compatible with the Hailo Dataflow Compiler (DFC)
4. Compiles through the Hailo Model Zoo (HMZ)
5. Deploys as a HEF model on Raspberry Pi 5 + Hailo-8 (26 TOPS)

Every decision in this repository is evaluated against the **full deployment pipeline**, not just training accuracy.

---

## End-to-End Pipeline

```
Dataset Acquisition (Open Images V7, CrowdHuman, ExDark, Roboflow, Dataset Ninja)
       ↓
Conversion → Box Audit (+ Roboflow near-exhaustive review) → Per-Class Capping
       ↓
Model-Assisted Curation (FiftyOne mistakenness — DEC-019)
       ↓
Merge + Dedup → Final Curation Gate (DEC-020) → Source-Stratified Split
       ↓
YOLOv8s Training (Ultralytics)
       ↓
ONNX Export
       ↓
Hailo Dataflow Compiler (DFC) + Hailo Model Zoo (HMZ)
       ↓
HEF Generation
       ↓
Deployment via hailo-apps on Raspberry Pi 5
```

---

## Target Classes

The following **16 classes** (15 confirmed + 1 possible) define the canonical label schema:

| ID | Class | Rationale |
|----|-------|-----------| 
| 0 | Person | Dynamic collision hazard — crowds and moving pedestrians |
| 1 | Vehicle | High-mass kinetic threat during outdoor transit |
| 2 | Motorcycle | Fast-moving, high-speed hazard common in Philippine streets |
| 3 | Pole | Utility poles as common street-level collision obstacles |
| 4 | Animals | Unpredictable low-level living obstacles (dogs, cats) |
| 5 | Stairs | Elevation change requiring rhythmic stepping preparation |
| 6 | Shelf | Common mall/retail/grocery obstacle (aisle end-caps, protruding shelving) — replaces Escalator, DEC-083 |
| 7 | Doors | Primary navigational anchor for room/building transitions |
| 8 | Chairs | Targeted waypoint — finding empty seating |
| 9 | Tables | Major routing obstacle and transaction counter locator |
| 10 | Tricycle | Common Philippine three-wheeler (status: possible) |
| 11 | Potholes | Ground anomalies that depth sensors struggle with |
| 12 | Trash Bins | Frequently relocated static barriers on memorized routes |
| 13 | Elevator | Multi-floor transit point with specific interaction needs |
| 14 | Pedestrian Lane | Painted crossings invisible to depth — guides safe crossing |
| 15 | Bicycle | Slow, silent hazard — easy to miss without visual/audio cues |

These classes were selected based on user research and are the authoritative class list unless explicitly changed. See DEC-002, DEC-015, DEC-021.

---

## Dataset Sources

The final training dataset combines multiple public sources:

| Source | Directory | Role |
|--------|-----------|------|
| Open Images V7 | `dataset/raw/open_images/` | Primary for 7 classes |
| CrowdHuman | `dataset/raw/crowdhuman/` | Secondary for Person |
| ExDark | `dataset/raw/exdark/` | Low-light augmentation for 6 classes |
| Roboflow Universe (14 projects) | `dataset/raw/roboflow_<project>/` | Niche classes + secondary |
| Dataset Ninja (pothole-detection) | `dataset/raw/dataset_ninja_pothole_detection/` | Primary for Potholes |
| Dataset Ninja (road-damage-detector) | `dataset/raw/dataset_ninja_road_damage_detector/` | Secondary for Potholes |

All sources are mapped to the single canonical class list above. Roboflow sources are pulled locally via pinned SDK versions (DEC-018r).

---

## Repository Structure

```
second-vision-ai/
│
├── README.md                  # This file
├── AGENTS.md                  # AI agent behavior rules
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
│
├── config/
│   ├── classes.yaml           # Canonical class list + per-class source config
│   ├── datasets.yaml          # Per-source connection/license/version info
│   └── training.yaml          # YOLOv8s training hyperparameters
│
├── dataset/
│   ├── raw/                   # Per-source raw downloads (gitignored)
│   │   ├── open_images/
│   │   ├── crowdhuman/
│   │   ├── exdark/
│   │   ├── dataset_ninja_*/
│   │   └── roboflow_<project>/
│   ├── processed/             # Intermediate COCO-style, per (source, class)
│   ├── curated/               # Post-mistakenness correction pools (DEC-019)
│   ├── merged/                # Post-cap, post-merge, pre-split
│   ├── final/                 # Train/val/test splits ready for training
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── reports/               # Auto-generated logs and audit reports
│
├── scripts/
│   ├── acquire/               # Dataset download scripts
│   ├── convert/               # Format converters → intermediate schema
│   ├── preprocess/            # Capping, dedup, box audit
│   ├── curate/                # FiftyOne mistakenness + correction reimport
│   ├── build/                 # Merge, split, YAML generation
│   └── utils/                 # Shared utilities
│
├── notebooks/
│   ├── train_yolov8.ipynb     # Training notebook
│   ├── hailo_compile.ipynb    # Hailo compilation guide
│   ├── inspect_dataset.ipynb  # Dataset exploration
│   ├── error_analysis.ipynb   # Post-training error analysis
│   └── inference_test.ipynb   # Inference verification
│
├── models/
│   ├── weights/               # Trained PyTorch weights (gitignored)
│   ├── onnx/                  # Exported ONNX models (gitignored)
│   ├── hef/                   # Compiled Hailo HEF models (gitignored)
│   ├── archived/              # Previous experiment weights (gitignored)
│   └── current/               # Pointer/README for the production-candidate model (tracked)
│
└── docs/
    ├── PROJECT.md             # Project context and architecture overview
    ├── PLAN.md                # Phased implementation roadmap
    ├── DECISIONS.md           # Decision log with rationale
    ├── experiments.md         # Experiment log
    ├── preprocessing.md       # Preprocessing pipeline docs
    └── training.md            # Training configuration docs
```

---

## Hailo Deployment Constraints

These constraints affect every decision in this repository:

| Constraint | Detail |
|------------|--------|
| **Calibration dataset** | Hailo DFC uses the `val` split for calibration — validation must remain clean and representative |
| **Background class injection** | Hailo inserts class `0: Background` at runtime, shifting all class IDs by +1 |
| **Custom labels** | Production uses custom label JSON, not default COCO labels |
| **GPU compilation** | GPU-accelerated compilation is required for production-quality optimization |
| **Architecture compatibility** | Only standard YOLOv8 architectures that export cleanly to ONNX are used |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/KRMeeag/second-vision-ai.git
cd second-vision-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Development Philosophy

| Priority | Principle |
|----------|-----------| 
| 1 | **Correctness** — Accurate annotations and reproducible results |
| 2 | **Reproducibility** — Every experiment can be replicated |
| 3 | **Deployment compatibility** — Models must survive the full Hailo pipeline |
| 4 | **Dataset quality** — Data improvement before technique complexity |
| 5 | **Maintainability** — Clean scripts, clear documentation |
| 6 | **Practicality** — Engineering pragmatism over research novelty |

---

## License

This project is part of an academic thesis. See the production repository for licensing details.

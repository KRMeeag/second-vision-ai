# Second Vision AI

**Dataset curation, model training, and export pipeline for the Second Vision assistive navigation system.**

Second Vision is an IoT-based smart glass for visually impaired users. This repository contains everything needed to produce a deployment-ready YOLOv8 object detection model — from raw dataset curation through ONNX export and Hailo compilation preparation.

> **Note:** This repository does **not** contain embedded production code. Raspberry Pi runtime, threading, UART communication, ESP32 integration, and application logic are maintained in the separate [second-vision](https://github.com/KRMeeag/second-vision) production repository.

---

## Objective

Produce a YOLOv8 model that:

1. Performs well on the target assistive navigation use case
2. Exports successfully to ONNX
3. Remains compatible with the Hailo Dataflow Compiler (DFC)
4. Compiles through the Hailo Model Zoo (HMZ)
5. Deploys as a HEF model on Raspberry Pi 5 + Hailo-8 (26 TOPS)

Every decision in this repository is evaluated against the **full deployment pipeline**, not just training accuracy.

---

## End-to-End Pipeline

```
Dataset Collection
       ↓
Dataset Curation (class mapping, annotation normalization)
       ↓
Dataset Validation (integrity checks, duplicate detection)
       ↓
YOLOv8 Training (Ultralytics)
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

The following **15 classes** define the canonical label schema for this project:

| ID | Class | Rationale |
|----|-------|-----------|
| 0 | Person | Dynamic collision hazard — crowds and moving pedestrians |
| 1 | Vehicle | High-mass kinetic threat during outdoor transit |
| 2 | Two Wheeler | Fast-moving hazard common in Philippine streets |
| 3 | Cart | Erratically moving obstacle in malls and streets |
| 4 | Animals | Unpredictable low-level living obstacles (dogs, cats) |
| 5 | Stairs | Elevation change requiring rhythmic stepping preparation |
| 6 | Escalator | Moving tread requiring timing and handrail location |
| 7 | Doors | Primary navigational anchor for room/building transitions |
| 8 | Chairs | Targeted waypoint — finding empty seating |
| 9 | Tables | Major routing obstacle and transaction counter locator |
| 10 | Wet Floor Sign | Visual proxy for slip hazards invisible to depth sensors |
| 11 | Potholes | Ground anomalies that depth sensors struggle with |
| 12 | Trash Bins | Frequently relocated static barriers on memorized routes |
| 13 | Elevator | Multi-floor transit point with specific interaction needs |
| 14 | Pedestrian Lane | Painted crossings invisible to depth — guides safe crossing |

These classes were selected based on user research and remain the authoritative class list unless explicitly changed.

---

## Dataset Sources

The final training dataset combines multiple public sources:

| Source | Directory |
|--------|-----------|
| MS COCO 2017 | `dataset/raw/coco/` |
| Google Open Images V7 | `dataset/raw/openimages/` |
| Mapillary Vistas | `dataset/raw/mapillary/` |
| Roboflow datasets | `dataset/raw/roboflow/` |
| Custom-collected images | `dataset/raw/custom/` |

All sources are mapped to the single canonical class list above.

---

## Repository Structure

```
second-vision-ai/
│
├── README.md                  # This file
├── AGENTS.md                  # AI agent behavior rules
├── PROJECT.md                 # Project context and architecture overview
├── PLAN.md                    # Phased implementation roadmap
├── DECISIONS.md               # Decision log with rationale
├── TASKS.md                   # Current task tracking
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
│
├── config/
│   ├── classes.yaml           # Canonical class list (15 classes)
│   ├── datasets.yaml          # Dataset source configurations
│   └── training.yaml          # YOLOv8 training hyperparameters
│
├── dataset/
│   ├── raw/                   # Original downloaded datasets (gitignored)
│   ├── processed/             # Per-source converted to YOLO format
│   ├── merged/                # Combined dataset before splitting
│   ├── final/                 # Train/val/test splits ready for training
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── reports/               # Auto-generated dataset statistics
│
├── scripts/
│   ├── convert/               # Format converters (COCO→YOLO, etc.)
│   ├── preprocess/            # Filtering, dedup, validation
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
│   ├── weights/               # Trained PyTorch weights
│   ├── onnx/                  # Exported ONNX models
│   ├── hef/                   # Compiled Hailo HEF models
│   └── archived/              # Previous experiment weights
│
└── docs/
    ├── dataset.md             # Dataset documentation
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
git clone https://github.com/your-org/second-vision-ai.git
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

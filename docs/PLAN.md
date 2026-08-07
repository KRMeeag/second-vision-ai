# PLAN.md — Second Vision AI Implementation Roadmap

> **Last updated:** 2026-08-06
>
> Pipeline architecture: HANDOFF v3 + D-018r, DEC-024

This document outlines the phased implementation plan for the Second Vision AI model pipeline. The dataset pipeline follows a 9-stage architecture (Stages 5.1–5.9) defined in HANDOFF v3 with revisions, progressing from acquisition through final YAML generation. Stage 5.0 (Fork & Fix) was eliminated by D-018r — Roboflow sources are pulled locally and curated through the same pipeline as all other sources.

---

## Phase 0: Repository Initialization ✅

**Goal:** Establish the project structure, documentation, and configuration foundation.

| Task | Status |
|------|--------|
| Create folder structure per spec | ✅ Done |
| Write README.md | ✅ Done |
| Write AGENTS.md | ✅ Done |
| Write PROJECT.md | ✅ Done |
| Write PLAN.md | ✅ Done |
| Write DECISIONS.md | ✅ Done |
| Write TASKS.md | ✅ Done |
| Create `.gitignore` | ✅ Done |
| Create `config/classes.yaml` (v3 — OI primary, local pull) | ✅ Done |
| Create `config/datasets.yaml` (per-source config) | ✅ Done |
| Create `.gitkeep` files for empty directories | ✅ Done |
| Scaffold directory structure per HANDOFF v3 | ✅ Done |
| Align docs with HANDOFF v3 + D-018r + DEC-024 | ✅ Done |
| Create `requirements.txt` | ⬜ Todo |

---

## Phase 1: Configuration & Tooling

**Goal:** Set up tooling dependencies, shared utilities, and validate configuration files.

> Scope narrowed by DEC-025 (2026-08-06) — see docs/DECISIONS.md.

| Task | Status |
|------|--------|
| Create `requirements.txt` with pinned dependencies | ✅ Done |
| Install and verify FiftyOne | ✅ Done |
| Install and verify Roboflow SDK | ✅ Done |
| Verify CVAT/Label Studio integration via FiftyOne | ⬜ Todo |
| Build `scripts/utils/file_utils.py` — path helpers, safe copy/move | ✅ Done |
| Build `scripts/utils/bbox_utils.py` — ExDark + CrowdHuman parsing, validate/clip guard (DEC-025) | ✅ Done |
| ~~Build `scripts/utils/image_utils.py`~~ — dropped, see DEC-025 | — |
| Build `scripts/utils/config_loader.py` — YAML config reader | ✅ Done |
| Create `config/training.yaml` with YOLOv8s baseline hyperparameters | ✅ Done |

### Key Dependencies

| Tool | Purpose |
|------|---------|
| FiftyOne | Dataset acquisition (OI Zoo), curation, mistakenness, dedup, CVAT/LS integration |
| Roboflow SDK | Roboflow pinned-version dataset download |
| Ultralytics | YOLOv8s training target + pretrained model for mistakenness |
| supervision | Dataset Ninja format handling |

---

## Phase 2: Dataset Pipeline (Stages 5.1–5.9)

**Goal:** Execute the full dataset acquisition, conversion, curation, merging, and splitting pipeline.

This phase follows a 9-stage pipeline (Stage 5.0 eliminated by D-018r). Each stage produces outputs consumed by the next stage, with audit gates enforced at key checkpoints.

### Stage 5.1: Acquisition

| Task | Status |
|------|--------|
| Verify Open Images native class name strings (Blocker #5 scope) | ⬜ Todo |
| Build `scripts/acquire/acquire_openimages.py` (FiftyOne Zoo) | ⬜ Todo |
| Build `scripts/acquire/acquire_crowdhuman.py` | ⬜ Todo |
| Build `scripts/acquire/acquire_exdark.py` | ⬜ Todo |
| Build `scripts/acquire/acquire_datasetninja.py` | ⬜ Todo |
| Build `scripts/acquire/acquire_roboflow.py` (pinned SDK versions) | ⬜ Todo |
| Pin versions for all Roboflow projects in `datasets.yaml` | ⬜ Todo |

### Stage 5.2: Conversion

| Task | Status |
|------|--------|
| Build `scripts/convert/openimages_to_intermediate.py` | ⬜ Todo |
| Build `scripts/convert/crowdhuman_odgt_to_intermediate.py` | ⬜ Todo |
| Build `scripts/convert/exdark_to_intermediate.py` | ⬜ Todo |
| Build `scripts/convert/voc_to_intermediate.py` | ⬜ Todo |
| Build `scripts/convert/yolo_to_intermediate.py` (Roboflow YOLO format) | ⬜ Todo |

### Stage 5.3: Box Audit

| Task | Status |
|------|--------|
| Build `scripts/preprocess/box_audit.py` | ⬜ Todo |
| Run box audit on all `native_unspecified` sources | ⬜ Todo |
| Run box audit on all `project_dependent` sources | ⬜ Todo |
| Resolve audit: `elevator-status-s4lrk` (detection vs classification) | ⬜ Todo |
| Resolve audit: `traffico-y1` (full class list, cross-class overlap) | ⬜ Todo |
| Resolve audit: `pedestrian-and-animal-crossing` (class names) | ⬜ Todo |
| Near-exhaustive review for smaller Roboflow pools (D-018r) | ⬜ Todo |
| Review and approve audit results | ⬜ Todo |

### Stage 5.4: Per-Class Capping

| Task | Status |
|------|--------|
| Build `scripts/preprocess/cap_per_class.py` (3-tier fill logic) | ⬜ Todo |
| Run capping — verify ExDark guaranteed floor applied | ⬜ Todo |
| Review `dataset/reports/cap_report.json` | ⬜ Todo |

### Stage 5.5: Model-Assisted Curation — DEC-019

| Task | Status |
|------|--------|
| Build `scripts/curate/run_mistakenness.py` (FiftyOne) | ⬜ Todo |
| Build `scripts/curate/reimport_corrections.py` (CVAT/LS) | ⬜ Todo |
| Run mistakenness on capped pools (all sources, including Roboflow) | ⬜ Todo |
| Review and correct flagged samples in CVAT/Label Studio | ⬜ Todo |
| Re-import corrections into `dataset/curated/` | ⬜ Todo |

### Stage 5.6: Merge + Dedup

| Task | Status |
|------|--------|
| Build `scripts/build/merge.py` | ⬜ Todo |
| Build `scripts/preprocess/dedup.py` | ⬜ Todo |
| Run merge with source-prefixed filenames | ⬜ Todo |
| Run cross-source dedup (highest risk pairs flagged) | ⬜ Todo |

### Stage 5.7: Final Pre-Split Curation Gate — DEC-020

| Task | Status |
|------|--------|
| Build `scripts/curate/final_merge_curation.py` | ⬜ Todo |
| Run final mistakenness pass on merged pool | ⬜ Todo |
| Fix flagged samples in place | ⬜ Todo |

### Stage 5.8: Split

| Task | Status |
|------|--------|
| Build `scripts/build/split.py` (source-stratified) | ⬜ Todo |
| Verify no cross-split data leakage | ⬜ Todo |
| Verify class distribution across splits | ⬜ Todo |
| Verify source representation across splits | ⬜ Todo |

### Stage 5.9: YAML Generation

| Task | Status |
|------|--------|
| Build `scripts/build/generate_yaml.py` | ⬜ Todo |
| Generate final `data.yaml` for YOLOv8s | ⬜ Todo |
| Verify generated YAML matches `config/classes.yaml` schema | ⬜ Todo |

### Split Strategy

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | ~70-80% | Model training |
| Val | ~10-15% | Evaluation + Hailo calibration dataset |
| Test | ~10-15% | Final holdout evaluation |

> **Critical:** The validation split doubles as the Hailo calibration dataset. It must be representative of the deployment domain and retain proportional source representation per class.

---

## Phase 3: Baseline Training

**Goal:** Train an initial YOLOv8s model and establish baseline metrics.

| Task | Status |
|------|--------|
| Create `notebooks/train_yolov8.ipynb` | ⬜ Todo |
| Train YOLOv8s baseline with default augmentation | ⬜ Todo |
| Log baseline metrics (mAP, precision, recall per class) | ⬜ Todo |
| Generate confusion matrix | ⬜ Todo |
| Document experiment in `docs/experiments.md` | ⬜ Todo |
| Identify weak classes for targeted improvement | ⬜ Todo |

### Baseline Configuration

```yaml
model: yolov8s.pt        # Small — upgraded from nano per DEC-022
imgsz: 640               # Standard, Hailo-compatible
epochs: 100              # With early stopping patience
batch: 16                # Adjust based on GPU memory
```

---

## Phase 4: ONNX Export & Validation

**Goal:** Export the trained model to ONNX and verify export integrity.

| Task | Status |
|------|--------|
| Export best weights to ONNX | ⬜ Todo |
| Verify ONNX model loads and runs inference | ⬜ Todo |
| Compare ONNX output against PyTorch output | ⬜ Todo |
| Document export parameters | ⬜ Todo |

---

## Phase 5: Hailo Compilation Preparation

**Goal:** Prepare the ONNX model for Hailo DFC compilation.

| Task | Status |
|------|--------|
| Create `notebooks/hailo_compile.ipynb` with compilation guide | ⬜ Todo |
| Prepare calibration dataset from val split | ⬜ Todo |
| Generate custom label JSON for hailo-apps | ⬜ Todo |
| Document Hailo-specific configurations | ⬜ Todo |
| Verify HEF compilation succeeds | ⬜ Todo |

---

## Phase 6: Iteration & Optimization

**Goal:** Improve model performance based on evaluation results.

| Task | Status |
|------|--------|
| Analyze per-class performance gaps | ⬜ Todo |
| Identify dataset improvements needed | ⬜ Todo |
| Add targeted data for weak classes | ⬜ Todo |
| Experiment with augmentation strategies | ⬜ Todo |
| Re-train and compare against baseline | ⬜ Todo |
| Repeat export → compile → verify cycle | ⬜ Todo |

---

## Phase 7: Final Model Production

**Goal:** Produce the final deployment-ready model.

| Task | Status |
|------|--------|
| Select best experiment based on balanced metrics | ⬜ Todo |
| Final ONNX export | ⬜ Todo |
| Final Hailo HEF compilation | ⬜ Todo |
| Verify HEF on Raspberry Pi 5 + Hailo-8 | ⬜ Todo |
| Archive final model artifacts | ⬜ Todo |
| Document final configuration and results | ⬜ Todo |

---

## Decision Points

Key decisions that will arise during execution:

| Decision | Phase | Notes |
|----------|-------|-------|
| Open Images native class name verification | 2 (5.1) | Verify before writing acquire script — replaces old Blocker #5 |
| traffico-y1 canonical class contributions | 2 (5.3) | Resolve during box audit |
| elevator-status-s4lrk taxonomy | 2 (5.3) | Verify during box audit |
| Stairs source narrowing (4 → fewer?) | 2 (5.3) | After box audit quality comparison |
| Duplicate detection threshold | 2 (5.6) | Perceptual hash distance cutoff |
| Train/val/test split ratios | 2 (5.8) | Balance between training data and evaluation quality |
| Image size (640 vs 320) | 3 | Tradeoff between accuracy and inference speed |
| Augmentation strategy | 6 | Mosaic, mixup, color jitter configurations |
| Class-specific thresholds | 6 | Safety-critical classes may need lower confidence thresholds |

These decisions will be documented in [DECISIONS.md](file:///Users/luna/Projects/Thesis/second-vision-ai/docs/DECISIONS.md) as they are made.

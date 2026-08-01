# PLAN.md — Second Vision AI Implementation Roadmap

> **Last updated:** 2026-07-31

This document outlines the phased implementation plan for the Second Vision AI model pipeline. Each phase builds on the previous one, progressing from dataset foundation through to deployment-ready model production.

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
| Create `config/classes.yaml` | ✅ Done |
| Create `.gitkeep` files for empty directories | ⬜ Todo |
| Create `requirements.txt` | ⬜ Todo |

---

## Phase 1: Configuration & Tooling

**Goal:** Set up the canonical class list, dataset configuration, and shared utility scripts.

| Task | Status |
|------|--------|
| Finalize `config/classes.yaml` with 15-class schema | ⬜ Todo |
| Create `config/datasets.yaml` with source configurations | ⬜ Todo |
| Create `config/training.yaml` with baseline hyperparameters | ⬜ Todo |
| Build `scripts/utils/file_utils.py` | ⬜ Todo |
| Build `scripts/utils/bbox_utils.py` | ⬜ Todo |
| Build `scripts/utils/image_utils.py` | ⬜ Todo |
| Set up `requirements.txt` with pinned dependencies | ⬜ Todo |

---

## Phase 2: Dataset Acquisition & Conversion

**Goal:** Download source datasets and convert each to YOLO format with canonical class mapping.

| Task | Status |
|------|--------|
| Document download instructions for each source | ⬜ Todo |
| Build `scripts/convert/coco_to_yolo.py` | ⬜ Todo |
| Build `scripts/convert/openimages_to_yolo.py` | ⬜ Todo |
| Build `scripts/convert/mapillary_to_yolo.py` | ⬜ Todo |
| Build `scripts/convert/roboflow_to_yolo.py` | ⬜ Todo |
| Verify each converter outputs valid YOLO labels | ⬜ Todo |
| Generate per-source statistics reports | ⬜ Todo |

### Class Mapping Strategy

Each converter must map source-specific class names to the canonical 15-class IDs:

```
Source Class Name → Canonical ID
─────────────────────────────────
"person", "pedestrian", "man", "woman" → 0 (Person)
"car", "truck", "bus", "van"          → 1 (Vehicle)
"bicycle", "motorcycle"               → 2 (Two Wheeler)
...
```

Unmapped classes are **discarded**, not force-mapped.

---

## Phase 3: Dataset Preprocessing & Validation

**Goal:** Clean, deduplicate, and validate the converted datasets before merging.

| Task | Status |
|------|--------|
| Build `scripts/preprocess/filter_classes.py` | ⬜ Todo |
| Build `scripts/preprocess/rename_files.py` | ⬜ Todo |
| Build `scripts/preprocess/remove_duplicates.py` | ⬜ Todo |
| Build `scripts/preprocess/validate_labels.py` | ⬜ Todo |
| Build `scripts/preprocess/dataset_statistics.py` | ⬜ Todo |
| Run validation on all processed sources | ⬜ Todo |
| Review and resolve flagged issues | ⬜ Todo |

### Validation Checks

- [ ] No missing label files for images
- [ ] No corrupt or unreadable images
- [ ] All bounding boxes within `[0, 1]` normalized range
- [ ] All class IDs in `[0, 14]` range
- [ ] No duplicate images within each source
- [ ] No empty label files (images with no annotations flagged for review)

---

## Phase 4: Dataset Merging & Splitting

**Goal:** Combine all sources into a single unified dataset and create train/val/test splits.

| Task | Status |
|------|--------|
| Build `scripts/build/merge_datasets.py` | ⬜ Todo |
| Build `scripts/build/split_dataset.py` | ⬜ Todo |
| Build `scripts/build/create_data_yaml.py` | ⬜ Todo |
| Build `scripts/build/build_dataset.py` (orchestrator) | ⬜ Todo |
| Generate merged dataset statistics | ⬜ Todo |
| Verify no cross-split data leakage | ⬜ Todo |
| Verify class distribution across splits | ⬜ Todo |

### Split Strategy

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | ~70-80% | Model training |
| Val | ~10-15% | Evaluation + Hailo calibration dataset |
| Test | ~10-15% | Final holdout evaluation |

> **Critical:** The validation split doubles as the Hailo calibration dataset. It must be representative of the deployment domain.

---

## Phase 5: Baseline Training

**Goal:** Train an initial YOLOv8 model and establish baseline metrics.

| Task | Status |
|------|--------|
| Create `notebooks/train_yolov8.ipynb` | ⬜ Todo |
| Train YOLOv8n baseline with default augmentation | ⬜ Todo |
| Log baseline metrics (mAP, precision, recall per class) | ⬜ Todo |
| Generate confusion matrix | ⬜ Todo |
| Document experiment in `docs/experiments.md` | ⬜ Todo |
| Identify weak classes for targeted improvement | ⬜ Todo |

### Baseline Configuration

```yaml
model: yolov8n.pt        # Nano — fastest, most deployment-friendly
imgsz: 640               # Standard, Hailo-compatible
epochs: 100              # With early stopping patience
batch: 16                # Adjust based on GPU memory
```

---

## Phase 6: ONNX Export & Validation

**Goal:** Export the trained model to ONNX and verify export integrity.

| Task | Status |
|------|--------|
| Export best weights to ONNX | ⬜ Todo |
| Verify ONNX model loads and runs inference | ⬜ Todo |
| Compare ONNX output against PyTorch output | ⬜ Todo |
| Document export parameters | ⬜ Todo |

---

## Phase 7: Hailo Compilation Preparation

**Goal:** Prepare the ONNX model for Hailo DFC compilation.

| Task | Status |
|------|--------|
| Create `notebooks/hailo_compile.ipynb` with compilation guide | ⬜ Todo |
| Prepare calibration dataset from val split | ⬜ Todo |
| Generate custom label JSON for hailo-apps | ⬜ Todo |
| Document Hailo-specific configurations | ⬜ Todo |
| Verify HEF compilation succeeds | ⬜ Todo |

---

## Phase 8: Iteration & Optimization

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

## Phase 9: Final Model Production

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
| Which COCO classes to include | 2 | Not all 80 COCO classes are needed |
| Duplicate detection threshold | 3 | Perceptual hash distance cutoff |
| Train/val/test split ratios | 4 | Balance between training data and evaluation quality |
| YOLOv8 variant (n/s/m) | 5 | Nano is baseline; larger variants may improve accuracy |
| Image size (640 vs 320) | 5 | Tradeoff between accuracy and inference speed |
| Augmentation strategy | 8 | Mosaic, mixup, color jitter configurations |
| Class-specific thresholds | 8 | Safety-critical classes may need lower confidence thresholds |

These decisions will be documented in [DECISIONS.md](file:///Users/luna/Projects/Thesis/second-vision-ai/DECISIONS.md) as they are made.

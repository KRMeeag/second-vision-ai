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
| Verify Open Images native class name strings (Blocker #5 scope) | ✅ Done (DEC-029) |
| Build `scripts/acquire/acquire_openimages.py` (FiftyOne Zoo) | ✅ Done (DEC-040, DEC-042-044) — full 8-class run complete (DEC-051) |
| Build `scripts/acquire/acquire_crowdhuman.py` | ✅ Done (DEC-047, DEC-048) — full real pull complete (DEC-056) |
| Build `scripts/acquire/acquire_exdark.py` | ✅ Done (DEC-046) — full real conversion run, output on disk |
| Build `scripts/acquire/acquire_datasetninja.py` | ✅ Done (DEC-049) — full real conversion run, output on disk |
| Build `scripts/acquire/acquire_roboflow.py` (pinned SDK versions) | ✅ Done (DEC-045) — all 11 eligible projects pulled (DEC-051) |
| Pin versions for all Roboflow projects in `datasets.yaml` | ✅ Done — all except `traffico_y1` (blocked, zero generated versions) |

**Stage 5.1 status: ✅ 100% complete** (2026-08-13).

### Stage 5.2: Conversion

| Task | Status |
|------|--------|
| Build `scripts/convert/openimages_to_intermediate.py` | ✅ Done (DEC-052) — real run complete, output on disk |
| Build `scripts/convert/yolo_to_intermediate.py` (Roboflow YOLO format) | ✅ Done (DEC-053, DEC-055) — all 11 projects converted, output on disk |
| ~~Build `scripts/convert/crowdhuman_odgt_to_intermediate.py`~~ | Not needed — folded into `acquire_crowdhuman.py` (DEC-041, DEC-048) |
| ~~Build `scripts/convert/exdark_to_intermediate.py`~~ | Not needed — folded into `acquire_exdark.py` (DEC-046) |
| ~~Build `scripts/convert/voc_to_intermediate.py`~~ | Not needed — Dataset Ninja sources are Supervisely, not VOC; folded into `acquire_datasetninja.py` (DEC-049) |

**Stage 5.2 status: ✅ 100% complete** (2026-08-13, DEC-056) — every source now sits in `dataset/processed/<source>/` in DEC-046's intermediate schema, ready for Stage 5.3.

### Stage 5.3: Box Audit

| Task | Status |
|------|--------|
| Build `scripts/preprocess/box_audit.py` | ✅ Done (DEC-057) — source-agnostic, one script covers every processed source regardless of bbox_mode |
| Run box audit on all `native_unspecified` sources | ✅ Done (DEC-057) — see note above, one run covers both this row and the next |
| Run box audit on all `project_dependent` sources | ✅ Done (DEC-057) |
| Resolve audit: `elevator-status-s4lrk` (detection vs classification) | 🟡 Heuristic built + flagged list produced (844 boxes, DEC-057) — student visual review still needed |
| Resolve audit: `traffico-y1` (full class list, cross-class overlap) | ⬜ N/A for now — no data on disk (never pulled, blocked on Roboflow's side) |
| Resolve audit: `pedestrian-and-animal-crossing` (class names) | ✅ Already resolved DEC-053/055 |
| Near-exhaustive review for smaller Roboflow pools (D-018r) | ⬜ Todo — student's visual-review time via `notebooks/fiftyone_review_processed.ipynb`, not a script gap |
| Review and approve audit results | ⬜ Todo — student review pending (`dataset/reports/box_audit_report.json`, `elevator_status_s4lrk_flagged.json`) |

### Stage 5.4: Per-Class Capping

| Task | Status |
|------|--------|
| Build `scripts/preprocess/cap_per_class.py` (3-tier fill logic) | ✅ Done (DEC-058) — decision/report only, not wired into merge.py (see DEC-058) |
| Run capping — verify ExDark guaranteed floor applied | ✅ Done (DEC-058) — verified, dense classes (Person, Chairs) correctly hit instance_target before image cap |
| Review `dataset/reports/cap_report.json` | ⬜ Todo — student review pending; ratio invariant not met (4.08 vs 3:1), cap-recompute trigger fired but not applied, both tied to `docs/OPEN_QUESTIONS.md` #1 |

### Stage 5.5: Model-Assisted Curation — DEC-019

| Task | Status |
|------|--------|
| Build `scripts/curate/run_mistakenness.py` (FiftyOne) | ✅ Done (DEC-060) — COCO-pretrained proxy model, 7/16 classes with a COCO analog |
| Build `scripts/curate/reimport_corrections.py` (CVAT/LS) | ⬜ **Skipped** — genuinely blocked on CVAT-vs-Label-Studio choice, `docs/OPEN_QUESTIONS.md` #5 |
| Run mistakenness on capped pools (all sources, including Roboflow) | ✅ Done (DEC-060) — 22,846 images scored and ranked |
| Review and correct flagged samples in CVAT/Label Studio | ⬜ Todo — student review pending |
| Re-import corrections into `dataset/curated/` | ⬜ Todo — blocked on above |

### Stage 5.6: Merge + Dedup

| Task | Status |
|------|--------|
| Build `scripts/build/merge.py` | ✅ Done (DEC-059) — merges Stage 5.4's capped selection (corrects DEC-058's original assumption; see DEC-059) |
| Build `scripts/preprocess/dedup.py` | ✅ Done (DEC-062) |
| Run merge with source-prefixed filenames | ✅ Done (DEC-061) — 51,529 images merged (post RNG fix), 0 missing, verified on disk |
| Run cross-source dedup (highest risk pairs flagged) | ✅ Done (DEC-062) — exact: 1,893 groups/2,143 files (full pool). Near-dup: 520 groups/818 files (6,000-image stratified sample, threshold=0.2). Student review pending, `docs/OPEN_QUESTIONS.md` #7 |

### Stage 5.7: Final Pre-Split Curation Gate — DEC-020

| Task | Status |
|------|--------|
| Build `scripts/curate/final_merge_curation.py` | ✅ Done (DEC-064) — reuses DEC-060's crosswalk directly |
| Run final mistakenness pass on merged pool | ✅ Done (DEC-064) — 22,824 images scored, `dataset/reports/final_merge_curation_report.json` |
| Fix flagged samples in place | ⬜ Todo — student review pending, blocked on `docs/OPEN_QUESTIONS.md` #5 |

### Stage 5.8: Split

| Task | Status |
|------|--------|
| Build `scripts/build/split.py` (source-stratified) | ✅ Done (DEC-063) — found + fixed a real union-find leakage bug along the way |
| Verify no cross-split data leakage | ✅ Done (DEC-063) — 0 leakage after the union-find fix (56 groups initially straddled splits, now 0) |
| Verify class distribution across splits | ✅ Done — all 16 classes proportionally represented, `dataset/reports/split_report.json` |
| Verify source representation across splits | ✅ Done — source-stratified by construction, verified in `split_report.json`'s `per_split_source_counts` |

### Stage 5.9: YAML Generation

| Task | Status |
|------|--------|
| Build `scripts/build/generate_yaml.py` | ✅ Done (DEC-065) |
| Generate final `data.yaml` for YOLOv8s | ✅ Done (DEC-065) — `dataset/final/data.yaml`, points at real populated splits |
| Verify generated YAML matches `config/classes.yaml` schema | ✅ Done (DEC-065) — verified against the authoritative `names:` field specifically, not the separately-ordered `classes:` metadata block |

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

> Runs on RunPod (cloud GPU), not the local machine — see DEC-026.

| Task | Status |
|------|--------|
| Package `dataset/final/` + `data.yaml` and upload to RunPod | ⬜ Todo |
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

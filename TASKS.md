# TASKS.md — Second Vision AI Active Tasks

> **Last updated:** 2026-07-31
>
> This document tracks the current working tasks. For the full phased roadmap, see [PLAN.md](file:///Users/luna/Projects/Thesis/second-vision-ai/PLAN.md).

---

## Current Phase: Phase 0 → Phase 1 Transition

Repository documentation is initialized. Next priority is setting up configuration files and utility scripts.

---

## Active Tasks

### 🟢 Phase 0: Repository Initialization (In Progress)

- [x] Create folder structure
- [x] Write `README.md`
- [x] Write `AGENTS.md`
- [x] Write `PROJECT.md`
- [x] Write `PLAN.md`
- [x] Write `DECISIONS.md`
- [x] Write `TASKS.md`
- [x] Create `.gitignore`
- [x] Create `config/classes.yaml`
- [ ] Create `.gitkeep` files for empty directories
- [ ] Create `requirements.txt` with pinned dependencies
- [ ] Initial commit and push

---

### ⬜ Phase 1: Configuration & Tooling (Up Next)

- [ ] Finalize `config/classes.yaml` — review class names and IDs
- [ ] Create `config/datasets.yaml` — source paths, URLs, class mappings
- [ ] Create `config/training.yaml` — baseline YOLOv8n hyperparameters
- [ ] Build `scripts/utils/file_utils.py` — path helpers, safe copy/move
- [ ] Build `scripts/utils/bbox_utils.py` — bbox validation, conversion, clipping
- [ ] Build `scripts/utils/image_utils.py` — image reading, validation, hashing
- [ ] Finalize `requirements.txt`

---

### ⬜ Phase 2: Dataset Acquisition & Conversion (Blocked by Phase 1)

- [ ] Document download instructions per source dataset
- [ ] Build COCO → YOLO converter
- [ ] Build Open Images → YOLO converter
- [ ] Build Mapillary → YOLO converter
- [ ] Build Roboflow → YOLO converter
- [ ] Verify all converters produce valid YOLO format labels

---

## Backlog

See [PLAN.md](file:///Users/luna/Projects/Thesis/second-vision-ai/PLAN.md) Phases 3–9 for upcoming work.

---

## Blocked Items

| Item | Blocked By | Notes |
|------|-----------|-------|
| Dataset conversion scripts | Phase 1 utilities | Converters depend on shared utility functions |
| Training notebook | Phase 4 dataset | Need final merged dataset before training |
| Hailo compilation | Phase 6 ONNX export | Need exported ONNX model first |

---

## Completed

| Task | Date | Notes |
|------|------|-------|
| Repository structure created | 2026-07-31 | Folders per spec, `.gitignore` configured |
| Documentation initialized | 2026-07-31 | README, AGENTS, PROJECT, PLAN, DECISIONS, TASKS |
| `config/classes.yaml` created | 2026-07-31 | 15-class canonical schema |

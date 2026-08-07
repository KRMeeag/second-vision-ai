# TASKS.md — Second Vision AI Active Tasks

> **Last updated:** 2026-08-06
>
> This document tracks the current working tasks. For the full phased roadmap, see [PLAN.md](file:///Users/luna/Projects/Thesis/second-vision-ai/docs/PLAN.md).

---

## Current Phase: Phase 0 (complete) → Phase 1 Transition

Repository documentation aligned with HANDOFF v3 + D-018r + DEC-024. Configuration files updated. Next priority: tooling setup and utility scripts.

---

## Active Tasks

### ✅ Phase 0: Repository Initialization (Complete)

- [x] Create folder structure
- [x] Write `README.md`
- [x] Write `AGENTS.md`
- [x] Write `PROJECT.md`
- [x] Write `PLAN.md`
- [x] Write `DECISIONS.md` (DEC-001–005, DEC-012–024)
- [x] Write `TASKS.md`
- [x] Create `.gitignore`
- [x] Create `config/classes.yaml` (v3 — OI primary, local pull)
- [x] Create `config/datasets.yaml` (per-source config, pinned versions)
- [x] Scaffold directory structure with `.gitkeep` files
- [x] Align all documentation with HANDOFF v3 + D-018r + DEC-024
- [ ] Create `requirements.txt` with pinned dependencies
- [ ] Initial commit and push

---

### ⬜ Phase 1: Configuration & Tooling (Up Next)

> Scope narrowed by DEC-025 (2026-08-06) — see docs/DECISIONS.md.

- [x] Create `requirements.txt` — FiftyOne, Roboflow SDK, Ultralytics, supervision, etc.
- [x] Install and verify FiftyOne
- [x] Install and verify Roboflow SDK
- [ ] Verify CVAT/Label Studio integration via FiftyOne
- [x] Build `scripts/utils/config_loader.py` — YAML config reader for classes.yaml + datasets.yaml
- [x] Build `scripts/utils/file_utils.py` — path helpers, safe copy/move
- [x] Build `scripts/utils/bbox_utils.py` — ExDark + CrowdHuman raw box parsing, validate/clip guard (DEC-025: narrowed — VOC/COCO-style/YOLO sources use FiftyOne's native importers instead of custom conversion code)
- [ ] ~~Build `scripts/utils/image_utils.py`~~ — dropped per DEC-025: corrupt-image checks via `compute_metadata()`, dedup via FiftyOne Brain, no standalone module
- [x] Create `config/training.yaml` — YOLOv8s baseline hyperparameters

---

### ⬜ Phase 2: Dataset Pipeline (Blocked by Phase 1)

#### Stage 5.1: Acquisition
- [ ] Verify Open Images native class name strings (Blocker #5 scope)
- [ ] Build `scripts/acquire/acquire_openimages.py` (FiftyOne Zoo, 7 classes)
- [ ] Build `scripts/acquire/acquire_crowdhuman.py`
- [ ] Build `scripts/acquire/acquire_exdark.py`
- [ ] Build `scripts/acquire/acquire_datasetninja.py`
- [ ] Build `scripts/acquire/acquire_roboflow.py` (pinned SDK versions)
- [ ] Pin versions for all Roboflow projects in `datasets.yaml`

#### Stage 5.2: Conversion
- [ ] Build 5 converter scripts (OI, CrowdHuman, ExDark, VOC, YOLO → intermediate)

#### Stage 5.3: Box Audit (includes Roboflow audit — D-018r)
- [ ] Build `scripts/preprocess/box_audit.py`
- [ ] Resolve audit: `elevator-status-s4lrk` (detection vs classification)
- [ ] Resolve audit: `traffico-y1` (full class list dump)
- [ ] Resolve audit: `pedestrian-and-animal-crossing` (class names)
- [ ] Near-exhaustive review for smaller Roboflow pools

#### Stages 5.4–5.9: Cap → Split → YAML
- [ ] Build `cap_per_class.py`, `dedup.py`
- [ ] Build `run_mistakenness.py`, `reimport_corrections.py`, `final_merge_curation.py`
- [ ] Build `merge.py`, `split.py`, `generate_yaml.py`

---

## Backlog

See [PLAN.md](file:///Users/luna/Projects/Thesis/second-vision-ai/docs/PLAN.md) Phases 3–7 for training through deployment.

---

## Blocked Items

| Item | Blocked By | Notes |
|------|-----------|-------|
| Acquisition scripts | Phase 1 utilities | Converters depend on shared config loader + utilities |
| Roboflow acquisition | Version pinning | Must identify and pin version numbers first |
| Audit-flagged projects | Stage 5.3 box audit | `traffico-y1`, `elevator-status-s4lrk` resolved at box audit |
| Cap/merge pipeline | Stage 5.3 box audit | Audit must pass before capping |
| Training notebook | Phase 2 Stage 5.9 | Need final merged dataset before training |
| Hailo compilation | Phase 4 ONNX export | Need exported ONNX model first |

---

## Open Blockers (from HANDOFF v3 Section 7, updated)

| # | Blocker | Status |
|---|---------|--------|
| 1 | Trash Bins secondary Roboflow project — TBD | ⬜ Unresolved |
| 2 | traffico-y1 full class audit | ⬜ Resolve during Stage 5.3 |
| 3 | elevator-status-s4lrk audit | ⬜ Resolve during Stage 5.3 |
| 4 | pedestrian-and-animal-crossing class names | ⬜ Resolve during Stage 5.3 |
| 5 | Open Images native class name verification | ⬜ Verify before acquire script |
| 6 | Stairs — 4 sources quality comparison | ⬜ After box audit |

---

## Completed

| Task | Date | Notes |
|------|------|-------|
| Repository structure created | 2026-07-31 | Folders per spec, `.gitignore` configured |
| Documentation initialized | 2026-07-31 | README, AGENTS, PROJECT, PLAN, DECISIONS, TASKS |
| `config/classes.yaml` v1 created | 2026-07-31 | 15-class canonical schema (original) |
| HANDOFF v3 alignment | 2026-08-04 | Full doc overhaul — DEC-012–023, classes.yaml v2, datasets.yaml |
| D-018r + DEC-024 alignment | 2026-08-06 | Fork→local pull, Objects365→Open Images, pipeline to 9 stages |
| Repo hygiene pass + AGENTS.md mentor-mode update | 2026-08-06 | Removed stale `curate.py`, fixed README clone URL, added `models/current/`, DEC-002 status note, Documentation & Recording Policy + student-mentor collaboration rules added to AGENTS.md |
| `file_utils.py` + `bbox_utils.py` built (DEC-025 scope) | 2026-08-07 | Path helpers/safe copy-move; ExDark+CrowdHuman box conversion + validate/clip guard. Both smoke-tested clean |
| FiftyOne + Roboflow SDK verified, `config/training.yaml` created | 2026-08-07 | fiftyone 1.20.0 (pre-installed) + roboflow 1.4.0 (installed) both import cleanly together (roboflow pinned numpy/opencv-python-headless down slightly — no conflict). Baseline training config parses clean |

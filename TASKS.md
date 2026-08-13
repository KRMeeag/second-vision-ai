# TASKS.md — Second Vision AI Active Tasks

> **Last updated:** 2026-08-13
>
> This document tracks the current working tasks. For the full phased roadmap, see [PLAN.md](file:///Users/luna/Projects/Thesis/second-vision-ai/docs/PLAN.md).

---

## Current Phase: Phase 2, Stage 5.3 (Box Audit) — up next

Phases 0-1 complete. **Stage 5.1 (Acquisition) and Stage 5.2 (Conversion) are both 100% complete** as of 2026-08-13 — all 5 sources (Open Images, Roboflow, ExDark, Dataset Ninja ×2, CrowdHuman) pulled and converted to DEC-046's intermediate schema, real data on disk, verified for real at every step. Next up: Stage 5.3 Box Audit — `scripts/preprocess/box_audit.py`, plus the specific per-project reviews already queued in `docs/PLAN.md`.

Student is time-constrained — planning an autonomous session to build as much of Stage 5.3–5.9 as possible unattended. See `docs/OPEN_QUESTIONS.md` (new, 2026-08-13) for the standing list of decisions that block specific scripts; check it before assuming something is a fresh question.

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
- [x] Create `requirements.txt` with pinned dependencies — exists on disk (2026-08-04); stale checkbox, actually done before this session started
- [ ] Initial commit and push — commits exist locally, but `main` has no upstream/remote configured yet, so nothing has actually been pushed

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
- [x] Verify Open Images native class name strings (Blocker #5 scope) — resolved 2026-08-08, DEC-029
- [x] Build `scripts/acquire/acquire_openimages.py` (FiftyOne Zoo, 8 classes — Motorcycle/Bicycle split per DEC-038) — built + smoke-tested 2026-08-10, DEC-040
- [x] Run full `acquire_openimages.py` for real — all 8 classes pulled 2026-08-13, DEC-051. 0 errors, 9.7GB total. Trash Bins came in low (1,106, ~18% of request) — flags DEC-039's revisit trigger, not acted on yet
- [x] Build `scripts/acquire/acquire_crowdhuman.py` — scripted download+parse (DEC-048, reverses DEC-041's manual-only stance for this source now that it's HF-hosted, public/ungated, real SDK)
- [x] Run full `acquire_crowdhuman.py` for real — 2026-08-13, DEC-056. 0 errors, 21GB raw/11GB processed. 19,370 images, 439,046 boxes, 18,917 clipped — exact match to the earlier annotation-only dry-run prediction. The one previously-unverified assumption (`Images/<ID>.jpg` zip layout) confirmed correct against real bytes
- [x] Build `scripts/acquire/acquire_exdark.py` — parse-only (DEC-041); built + run for real 2026-08-13, DEC-046. 6,042 images / 18,366 boxes converted to `dataset/processed/exdark/`, canonical class ids applied, 41 boxes clipped, non-canonical classes (Boat/Bottle/Bus/Cup) correctly dropped
- [x] Build `scripts/acquire/acquire_datasetninja.py` — parse-only (DEC-041); built + run for real 2026-08-13, DEC-049. Both sources placed and converted: pothole_detection 665/665 images, road_damage_detector 1331/3321 (rest had no pothole-class object). Format turned out genuine Supervisely (corner-pair boxes), not Pascal VOC as previously assumed — parsed directly, no SDK needed
- [x] Build `scripts/acquire/acquire_roboflow.py` (pinned SDK versions) — built + smoke-tested 2026-08-13, DEC-045. Eligibility rule (`pending` + pinned version) pulls 11/16 projects, skips 5 with reasons.
- [x] Run full `acquire_roboflow.py` for real — all 11 eligible projects pulled 2026-08-13, 0 errors, 2.6GB total, all counts verified against `datasets.yaml`'s documented estimates
- [x] Pin versions for all Roboflow projects in `datasets.yaml` — resolved 2026-08-08 via `list_roboflow_versions.py` + DEC-036 (splits discarded regardless, so version choice = latest by default). Only remaining exception: `traffico_y1` (zero generated versions, needs checking on Roboflow's side). `me5_u6rvg` reactivated and pinned (v1, DEC-037); `trash_bins` secondary benched, not pursued (DEC-039)

#### Stage 5.2: Conversion
- [x] Build `scripts/convert/openimages_to_intermediate.py` — built + run for real 2026-08-13, DEC-052. Filters each class-folder to its own native classes (raw exports weren't pre-filtered — 213 categories present in the "animals" export, not just Dog/Cat), merges boxes for images pulled under >1 class-folder (2,758 of them) instead of dropping on collision. 26,715 unique images, 67,760 boxes, 81 clipped, 0 invalid. Final per-class image counts audited (non-exclusive)
- [x] Build `scripts/convert/yolo_to_intermediate.py` — built + run for real 2026-08-13, DEC-053/DEC-055. Real per-project format audit found 4 mismatches vs. `datasets.yaml`'s assumptions (escalator_stairs, cv_project_hovyc, me5_u6rvg placeholder class names recovered via API, pedestrian_and_animal_crossing's needed class missing from its only export). Forked + regenerated the last one on Roboflow to pull in the real class (retried once after a genuine dropped-connection failure). **All 11/11 projects converted for real: 38,426 images, 52,304 boxes total.** Fixed a real report-overwrite bug found along the way (`--projects` subset runs now merge into the existing report instead of clobbering it)
- ~~CrowdHuman/ExDark/Dataset Ninja converters~~ — not needed, folded into their acquire scripts (DEC-041, DEC-046, DEC-049)

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
| 1 | Trash Bins secondary Roboflow project — TBD | 🔵 Benched 2026-08-08 (DEC-039) — Open Images primary judged sufficient; revisit only if evaluation shows it's not |
| 2 | traffico-y1 full class audit | ✅ Resolved 2026-08-08 — DEC-032 |
| 3 | elevator-status-s4lrk audit | 🟡 Partially resolved 2026-08-08 (DEC-031) — usable subset confirmed, full isolation still needs Stage 5.3 |
| 4 | pedestrian-and-animal-crossing class names | ⬜ Resolve during Stage 5.3 |
| 5 | Open Images native class name verification | ✅ Resolved 2026-08-08 — DEC-029 |
| 6 | Stairs — 4 sources quality comparison | 🟡 Partial finding 2026-08-08 (DEC-031) — 2 of 4 sources show box-shape issue, marked `failed`; full comparison still after box audit |
| 7 | Pothole source: Dataset Ninja vs. new Roboflow candidates | ✅ Resolved 2026-08-08 — DEC-034: combine both, Dataset Ninja currently blocked (Dropbox rate limit), retry later |

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
| Source re-audit round: Stairs/Elevator box-shape findings, traffico-y1 resolved, Jeepney consolidated, Potholes sourcing resolved, Pole/Potholes dual-source added | 2026-08-08 | DEC-031 through DEC-035. `docs/preprocessing.md` created for findings log + model-assisted labeling procedure. Dataset Ninja `pothole-detection` confirmed blocked (Dropbox rate limit) via live attempt, not just guessed |
| Jeepney secondary reworked, `Two Wheeler` split into `Motorcycle` + `Bicycle` | 2026-08-08 | DEC-037 (jeep_hozhs benched, me5_u6rvg reactivated), DEC-038 (schema: nc 15→16, ID 2 renamed Two Wheeler→Motorcycle, new ID 15 Bicycle). Propagated across classes.yaml, datasets.yaml, config_loader.py, AGENTS.md, README.md, PROJECT.md, DECISIONS.md |
| Trash Bins secondary search benched | 2026-08-08 | DEC-039 — Blocker #1 closed as benched, not resolved. Open Images `"Waste container"` primary judged sufficient for now |
| `scripts/acquire/acquire_openimages.py` built | 2026-08-10 | DEC-040 — pulls all 8 open_images-primary classes via FiftyOne Zoo, bounded by `max_samples` (DEC-027) computed from each class's `classes.yaml` cap × 1.35 buffer factor (DEC-030). Exports per-class COCO-style `images/ + labels.json` to `dataset/raw/open_images/<class>/`. Smoke-tested end-to-end (real 3-image pull + COCO export verified, then cleaned up) |
| Per-class image cap set to 4,500 (floor 1,500, 3:1 ratio) | 2026-08-11 | DEC-042 — literature-grounded replacement for the old ad-hoc 3500/5000 numbers. `config/classes.yaml`'s `cap` field updated for all 16 classes (uniformly 4500, no more `null`). `acquire_openimages.py` needed no code change — `max_samples` recalculates automatically via `compute_max_samples()` |
| `acquire_openimages.py` filters `IsDepiction`/`IsGroupOf` detections | 2026-08-11 | DEC-043 — student-written edit (guided), verified for real: filtered a 30-image Chair pull against an unfiltered baseline, confirmed exactly 31 detections removed (16 depiction + 15 group-of, no overlap) and 1 fully-emptied image correctly dropped |
| Two exploration notebooks built: `fiftyone_explore.ipynb`, `fiftyone_preview.ipynb` | 2026-08-11 | Post-export browser (loads an already-acquired class folder back into FiftyOne, no network) vs. pre-acquisition tuner (pulls a class at real buffered volume directly from the Zoo to preview filters before running the real script). Both tested end-to-end for real |
| `acquire_openimages.py` corrected to pull all 3 Open Images splits | 2026-08-13 | DEC-044 — was only pulling `train`, contradicting DEC-036's own consequence that acquire scripts should pull every split. Fixed: buffered total now split evenly across train/validation/test. Verified for real (5+5+5=15) |
| `scripts/acquire/acquire_roboflow.py` built | 2026-08-13 | DEC-045 — pulls 11/16 eligible Roboflow projects (`pending` + pinned version), skips the other 5 with reasons. Real pull verified (`cv_project_hovyc`, 1,341 images, correct yolov8 folder structure) |
| Stage 5.2 intermediate schema defined; `scripts/acquire/acquire_exdark.py` built | 2026-08-13 | DEC-046 — flat images/+labels/ pool, canonical class ids, no split. Reused `bbox_utils.py` (built DEC-025, unused until now) for clamping — a gap the student's found reference script (wraphex/ExDark2Yolo) has. Full real run: 6,042 images, 18,366 boxes, 41 clipped, output verified on disk |
| CrowdHuman mirror switched to HuggingFace, Kaggle pseudo-label alternative rejected | 2026-08-13 | DEC-047 — original Baidu/Google Drive mirrors dead. HF mirror confirmed genuine (real vbox/fbox/hbox, matches official stats) via API before switching; a Kaggle version was rejected for using YOLOv8-model-generated pseudo-labels, which would undermine DEC-028's reason for using CrowdHuman at all |
| `scripts/acquire/acquire_crowdhuman.py` built (scripted, via `huggingface_hub`) | 2026-08-13 | DEC-048 — reverses DEC-041's manual-only stance for this one source (HF mirror is public/ungated, real SDK). Verified for real: both annotation files downloaded + parsed (19,370 images, 439,046 boxes kept), filter logic confirmed against real tag/extra.ignore data, box-conversion math verified against a real annotation entry. Image zips (~14GB) left for a deliberate full run |
| `scripts/acquire/acquire_datasetninja.py` built — Stage 5.1 acquisition scripts now all built | 2026-08-13 | DEC-049 — real format turned out Supervisely (corner-pair boxes), not Pascal VOC as `datasets.yaml` assumed; parsed directly, `supervisely`/`dataset_tools` SDK rejected as unnecessary. `bbox_utils.xyxy_to_yolo` promoted from private. Full real run both sources: 665/665 + 1331/3321 images, all output verified on disk, all labels correctly class id 11 (Potholes) |
| `acquire_openimages.py` reruns now clean stale exports; `fiftyone_preview.ipynb` regression fixed | 2026-08-13 | DEC-050 — verified `max_samples` is a hard slice at the FiftyOne source level; found + fixed a real bug where reruns left orphaned image files (labels.json correctly updated, image files didn't); fixed the preview notebook's broken `ZOO_SPLIT` import (stale since DEC-044's rename). All three verified for real |
| Full `acquire_roboflow.py` run — all 11 eligible projects | 2026-08-13 | DEC-051 — run in parallel with the full `acquire_openimages.py` run (independent processes, no shared state, student explicitly OK'd running concurrently). 0 errors, 2.6GB, all 11 counts cross-checked against `datasets.yaml`'s documented per-project estimates |
| CrowdHuman/ExDark/Dataset Ninja scoped to manual download | 2026-08-10 | DEC-041 — checked each source's actual distribution mechanism (CrowdHuman: Baidu/Google Drive mirrors, no API; ExDark: single static archive; Dataset Ninja: SDK is a thin wrapper around a Dropbox host that already failed live). All three rescoped to parse-only scripts; student downloads manually into `dataset/raw/<source>/` first. `datasets.yaml` updated with exact files needed and target paths |
| `scripts/convert/openimages_to_intermediate.py` built — Stage 5.2 begun | 2026-08-13 | DEC-052 — found the per-class raw exports weren't actually pre-filtered to their own class (213 categories present in the "animals" folder alone); found 2,758 images independently pulled under >1 class-folder (chairs∩tables = 1,578) and merged their boxes rather than dropping on collision, per student's explicit call. Real run: 26,715 unique images, 67,760 boxes, 81 clipped. Added a non-exclusive final per-class image + instance (box) count audit per student's request. `docs/PLAN.md`'s stale Stage 5.2 table (still listing 3 already-folded-in converters as Todo) corrected in the same pass |
| `scripts/convert/yolo_to_intermediate.py` built — Stage 5.2 nearly done | 2026-08-13 | DEC-053 — real per-project format audit found 4 mismatches vs. `datasets.yaml` (escalator_stairs filter keys, cv_project_hovyc's undeclared Exit-signage classes, me5_u6rvg's placeholder class names recovered via Roboflow API, pedestrian_and_animal_crossing's needed class missing from its only export). Forked + regenerated that last project on Roboflow to pull in the real class. 10/11 projects converted for real (36,268 images, 49,694 boxes); `pedestrian_and_animal_crossing` pending a stalled download (student's network, not a bug) |
| Notebooks moved to `notebooks/`, labeled by pipeline stage; `fiftyone_review_processed.ipynb` built | 2026-08-13 | DEC-054 — `fiftyone_explore.ipynb` only supported raw COCO-style exports, not the intermediate schema every processed source now uses; new notebook builds the FiftyOne dataset manually from `images/`+`labels/` (canonical ids) rather than fighting FiftyOne's split-oriented YOLOv5 importer. Verified for real against `dataset_ninja_pothole_detection` (50 images, 142 detections, correct `Potholes` label). All 3 existing notebooks fixed for path-robustness after the move and given a clear top-of-notebook purpose label; `notebooks/README.md` added |
| `pedestrian_and_animal_crossing` fork retried successfully; Roboflow Stage 5.2 100% done | 2026-08-13 | DEC-055 — first attempt failed on a genuine dropped connection (`BrokenPipeError`), retried cleanly once the student's network stabilized. Real class name changed on export (Roboflow sanitizes special characters: `==============================` → `------------------------------`) — caught by checking the real `data.yaml`, not assumed. Converts to 2,158 images / 2,610 boxes, exact match to the API's instance count. Also fixed a real bug: `--projects` subset reruns were overwriting the full report instead of merging into it |
| CrowdHuman pulled for real — **Stage 5.1 and Stage 5.2 both 100% complete** | 2026-08-13 | DEC-056 — 0 errors, 21GB raw/11GB processed, 19,370 images / 439,046 boxes / 18,917 clipped. Exact match to the earlier annotation-only dry-run prediction. The one previously-flagged-unverified assumption (`Images/<ID>.jpg` zip layout) confirmed correct against real bytes. All 5 acquisition sources and both Stage 5.2 converters are now fully done — next up is Stage 5.3 (Box Audit) |
| Notebooks moved to `notebooks/`, labeled by pipeline stage; `fiftyone_review_processed.ipynb` built | 2026-08-13 | DEC-054 — `fiftyone_explore.ipynb` only supported raw COCO-style exports, not the intermediate schema every processed source now uses; new notebook builds the FiftyOne dataset manually from `images/`+`labels/` (canonical ids) rather than fighting FiftyOne's split-oriented YOLOv5 importer. Verified for real against `dataset_ninja_pothole_detection` (50 images, 142 detections, correct `Potholes` label). All 3 existing notebooks fixed for path-robustness after the move and given a clear top-of-notebook purpose label; `notebooks/README.md` added |

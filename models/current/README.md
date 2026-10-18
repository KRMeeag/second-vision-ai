# models/current/

Pointer to the currently deployed/production-candidate model — the artifact the `second-vision` production repository consumes.

This directory is the one exception carved out of `.gitignore`'s `models/` rules (`models/weights/`, `models/onnx/`, `models/hef/`, `models/archived/` are all ignored — trained artifacts are large and reproducible from `docs/experiments.md` + the pinned dataset/config, not meant to live in git history).

Keep this file updated with:
- Which experiment/run produced the current model (cross-reference `docs/experiments.md`)
- ONNX export date and Hailo HEF compilation date
- mAP@0.5 and per-class precision/recall summary
- Path or release tag where the actual `.hef` file can be retrieved (e.g. GitHub Release asset, shared drive) since the binary itself is not committed here

---

## Current model — `cap4500_yolov8s_v2-2` (v2, 14 classes)

| | |
|---|---|
| Run | `runs/detect/cap4500_yolov8s_v2-2`, epoch 78 of 100 (fitness-selected) |
| Condition | cap4500 on the v2 label rebuild — train 30,123 / val 6,644 / test 6,403 |
| Schema | **nc=14** (Shelf dropped, ids 5+ shifted down one — DEC-126) |
| Test mAP@0.5 | **0.7118** · mAP@0.5:0.95 **0.4812** |
| Per-class floors | Tricycle 0.839 PASS · Motorcycle 0.817 PASS · Vehicle 0.808 PASS · Bicycle 0.732 PASS · **Person 0.741 FAIL** (floor 0.75) |
| Weakest | Pole 0.385 (R 0.305) · Chairs 0.470 (R 0.437) · Tables 0.588 — still recall-limited |
| Strongest | Animals 0.914 · Trash Bins 0.897 · Tricycle 0.839 |
| False positives | 6,570 vs background at conf 0.25 |
| Verdict in `eval_fp32_test_summary.json` | **FAIL** — primary target met (0.7118 ≥ 0.70), per-class floors not |
| HEF | **compiled 2026-10-08** on an NVIDIA L4 (DEC-139). Two HEFs, `models/current/hef_v2/` |

### The compiled HEFs

**`models/current/hef_v2/`** — gitignored. Two HEFs, because conf/iou are baked into the
on-chip NMS at compile time and cannot be changed at runtime (DEC-120):

| | `nms_scores_th` | `nms_iou_th` | `classes` | sha256 | size |
|---|---|---|---|---|---|
| `hef_eval/best.hef` | **0.001** | 0.7 | 14 | `2ee394c8…` | 9,681,816 |
| `hef_deploy/best.hef` | **0.05** | 0.7 | 14 | `c6e4ec98…` | 9,681,398 |

`classes = 14` in both `nms_config.json` files confirms the nc=14 schema reached the chip.
Compiled from kit digest `6490abf1…` on an **NVIDIA L4** (23,034 MiB, driver 595.91.07) with
`torch 2.14.1+cu130` / `tensorflow 2.18.0` / `hailo_sdk 3.34.0` / `ultralytics 8.4.118`.
**3,044 s and 3,071 s** — ~10x DEC-114's 298 s, because that validation calibrated on 8 coco8
images and this calibrates on all 6,644.

`compile_record/` holds what dies with the pod: the compile venv's `pip_freeze` (202 packages —
the authoritative environment record per DEC-120, distinct from the kit's training freeze), the
venv build log, the five DFC compiler logs, the host record, and the compile report. `CHECKSUM`
carries a per-file sha256 plus an aggregate content digest (`9d89591f…` over 15 files).

A second copy lives on the EU-RO-1 network volume at `/workspace/hef_backup_v2/`, byte-verified
identical. That survives pod Terminate but **not** a balance lapse beyond ~48 h (DEC-137), so the
Mac remains the only durable store.

**Still pending:** no on-device measurement. `hailortcli parse-hef` on the Pi and the INT8
quantization gap against fp32 0.7118 / 0.4812 are unrun — the largest open gap in the thesis.

### Where the v2 artefacts live

**`models/current/deployment_kit_v2_cap4500/`** — self-contained, same layout as v1:
`best.pt` (sha256 `ab3678b6995b2b02…`), a `data.yaml` pointing at its own 6,644-image
INT8 calibration set, the frozen 14-class schema, `provenance.json` + `pip_freeze.txt`,
the measured results, and `RESTORE.md` with the exact two-HEF export commands.

Built by `scripts/train/build_deployment_kit_v2.py` on 2026-10-06 (DEC-135), ~24 s.
Rebuild it any time with `python3 scripts/train/build_deployment_kit_v2.py --tar`; the
builder refuses to write if the class schema disagrees across `data.yaml`,
`config/classes.yaml` and `provenance.json`, or if any label carries an id outside 0–13.

The kit is **gitignored** (703 MiB of files, 741 MB on disk — the same as v1 below, since
the calibration images are byte-identical), as is its `v2_kit.tar.gz` upload archive (700 MiB).

| location | status |
|---|---|
| laptop — `models/current/deployment_kit_v2_cap4500/` | present |
| laptop — `v2_kit.tar.gz` (repo root, rooted at `v2_kit/`) | present; uploaded and used for the 2026-10-08 compile |
| laptop — `v2_kit_skeleton.tar.gz` | present; the volume-sourced alternative (DEC-138), unusable until the DEC-123 JPEGs are repaired locally |
| RunPod pod | gone — terminated after the compile |
| GitHub Release asset | **TODO** — upload `best.pt` (22 MB, under the 100 MB limit) |

Integrity: `CHECKSUM` is a sha256 over path+size of all 13,297 files
(`6490abf10110e411bded8a1169932d1cc479298102ac28c77baec5cce9993a5f`).

> **The calibration images are 640-bounded re-encodes, not originals** (DEC-119 Decision 2 —
> ultralytics resizes the long side to `imgsz` at load time and Hailo calibration uses the
> same loader, so the pod never sees above 640 px). 1,678 MiB raw → 681 MiB here. They are
> byte-identical to the v1 kit's calibration set, so a v1-vs-v2 quantization comparison is
> not confounded by calibration data.

---

## Superseded — `cap4500_yolov8s` (v1, 15 classes)

| | |
|---|---|
| Run | `runs/detect/cap4500_yolov8s`, epoch 70 of 100 (fitness-selected) |
| Condition | cap4500 — train 30,123 / val 6,644 / test 6,403, box ratio 11.95 |
| Schema | **nc=15** (Shelf still present — the v2 rebuild drops it, see DEC-125) |
| Test mAP@0.5 | **0.6695** · mAP@0.5:0.95 **0.4530** |
| Per-class floors | Vehicle 0.803 PASS · Motorcycle 0.814 PASS · Tricycle 0.842 PASS · Bicycle 0.727 PASS · **Person 0.718 FAIL** (floor 0.75) |
| Weakest | Shelf 0.190 · Pole 0.379 · Chairs 0.436 — all recall-limited, see DEC-124 |
| False positives | 6,193 vs background at conf 0.25 (0.305 per instance) |
| ONNX / HEF | **not yet exported** |

### Where the v1 artefacts live

**`models/current/deployment_kit_v1_cap4500/`** — self-contained: `best.pt`, a
`data.yaml` pointing at its own 6,644-image INT8 calibration set, the frozen 15-class
schema, `provenance.json` + `pip_freeze.txt`, the measured results, and `RESTORE.md` with
the exact two-HEF export commands.

The kit is **gitignored** (741 MB). Copies:

| location | status |
|---|---|
| laptop — `models/current/deployment_kit_v1_cap4500/` | present |
| RunPod network volume (EU-RO-1) | **unreliable** — cut off by the 2026-10-07 balance lapse (DEC-137); treat the laptop as the only copy |
| GitHub Release asset | **TODO** — upload `best.pt` (22 MB, under the 100 MB limit) |

Integrity: `CHECKSUM` is a sha256 over path+size of all 13,297 files
(`eb0d1fdb46201ba639dcfe736f4b82c7da684f89f06bdf57e86f41eda51adfa3`).

> Do not mix this v1 kit with v2 outputs. v2 drops Shelf and renumbers ids 6-14 **down by
> one**, so a v1 HEF fed v2 class names mislabels nine classes silently — Potholes would
> read as Tricycle, which is the DEC-107 failure exactly.

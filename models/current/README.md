# models/current/

Pointer to the currently deployed/production-candidate model — the artifact the `second-vision` production repository consumes.

This directory is the one exception carved out of `.gitignore`'s `models/` rules (`models/weights/`, `models/onnx/`, `models/hef/`, `models/archived/` are all ignored — trained artifacts are large and reproducible from `docs/experiments.md` + the pinned dataset/config, not meant to live in git history).

Keep this file updated with:
- Which experiment/run produced the current model (cross-reference `docs/experiments.md`)
- ONNX export date and Hailo HEF compilation date
- mAP@0.5 and per-class precision/recall summary
- Path or release tag where the actual `.hef` file can be retrieved (e.g. GitHub Release asset, shared drive) since the binary itself is not committed here

---

## Current model — `cap4500_yolov8s` (v1, 15 classes)

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

### Where the artefacts live

**`models/current/deployment_kit_v1_cap4500/`** — self-contained: `best.pt`, a
`data.yaml` pointing at its own 6,644-image INT8 calibration set, the frozen 15-class
schema, `provenance.json` + `pip_freeze.txt`, the measured results, and `RESTORE.md` with
the exact two-HEF export commands.

The kit is **gitignored** (741 MB). Copies:

| location | status |
|---|---|
| laptop — `models/current/deployment_kit_v1_cap4500/` | present |
| RunPod network volume `bsj8rdxddm` (EU-RO-1) | via `/workspace/second-vision-ai/runs/` |
| GitHub Release asset | **TODO** — upload `best.pt` (22 MB, under the 100 MB limit) |

Integrity: `CHECKSUM` is a sha256 over path+size of all 13,297 files
(`eb0d1fdb46201ba639dcfe736f4b82c7da684f89f06bdf57e86f41eda51adfa3`).

> Do not mix this v1 kit with v2 outputs. v2 drops Shelf and renumbers ids 6-14 **down by
> one**, so a v1 HEF fed v2 class names mislabels nine classes silently — Potholes would
> read as Tricycle, which is the DEC-107 failure exactly.

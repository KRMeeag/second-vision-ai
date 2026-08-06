# models/current/

Pointer to the currently deployed/production-candidate model — the artifact the `second-vision` production repository consumes.

This directory is the one exception carved out of `.gitignore`'s `models/` rules (`models/weights/`, `models/onnx/`, `models/hef/`, `models/archived/` are all ignored — trained artifacts are large and reproducible from `docs/experiments.md` + the pinned dataset/config, not meant to live in git history).

Keep this file updated with:
- Which experiment/run produced the current model (cross-reference `docs/experiments.md`)
- ONNX export date and Hailo HEF compilation date
- mAP@0.5 and per-class precision/recall summary
- Path or release tag where the actual `.hef` file can be retrieved (e.g. GitHub Release asset, shared drive) since the binary itself is not committed here

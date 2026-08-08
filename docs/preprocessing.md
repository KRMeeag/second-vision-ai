# preprocessing.md — Second Vision AI

> Running log of manual pre-audit findings (ahead of `box_audit.py` / Stage 5.3 actually being built) and reusable preprocessing procedures. Strategic decisions derived from these findings are logged formally in `docs/DECISIONS.md`; this doc holds the supporting detail and anything procedural that would clutter the decision log.

---

## Procedure: Model-Assisted Pre-Labeling for Missing Classes

**Problem it solves:** a source dataset annotates only its own target class (e.g. Roboflow's `augmented_tricycle` labels tricycles) but the images also contain other canonical classes that went unlabeled (e.g. the tricycle's driver/passengers — Person). Re-annotating those by hand from scratch is wasted effort when a pretrained model can already find that class reliably.

**When it applies:** any source where a canonical class is visibly present in the images but absent from the source's own labels — most likely Person, since it appears incidentally in almost any street/vehicle scene and is the easiest class for any pretrained detector to find correctly.

**Steps (FiftyOne only — no paid tooling required):**

1. Load the source into a FiftyOne dataset from its downloaded export (e.g. `fo.Dataset.from_dir(dataset_type=fo.types.YOLOv5Dataset, ...)` for a Roboflow YOLOv8-format pull).
2. Load a pretrained detector from FiftyOne's Model Zoo — free, no API key beyond what's already set up: `model = foz.load_zoo_model("yolov8s-coco-torch")`.
3. Run inference into a *separate* label field so the source's original ground truth is untouched: `dataset.apply_model(model, label_field="predicted_person")`, then filter that field to just the `person` class.
4. Review in the FiftyOne App, sorted/filtered by confidence. Person detection from a COCO-pretrained checkpoint is reliable even on an unfamiliar domain (COCO's `person` class is enormous and visually generic), so most predictions should be usable as-is — this is a much lower-risk case than trying to auto-label a novel class the pretrained model has never seen.
5. Merge high-confidence predictions into the ground-truth label field (FiftyOne supports merging/renaming label fields directly).
6. Route low-confidence or visually ambiguous predictions to CVAT/Label Studio for manual correction — the same correction workflow already used for mistakenness (DEC-019), not a new process.

This is the same "model-assisted curation" philosophy as DEC-019, applied to *missing* labels for an entire class rather than *mistaken* labels for an existing one.

**Applies to (so far):** `augmented_tricycle` (missing Person). Likely also relevant to `traffico_y1` and other single-purpose Roboflow sources once reviewed — check for incidental people whenever a source's native class list doesn't include one.

---

## Per-Source Audit Findings

### Stairs — box shape doesn't match "object detection" labeling (`stairs_lusiz`, `stairs_hsatv`)

Both sources are labeled as object-detection format, but on inspection the ground-truth boxes **wrap tightly around the staircase's silhouette** rather than being a simple axis-aligned rectangle — consistent with annotations originally drawn as a polygon (or a very tightly-fit box) and only exported in a bounding-box-shaped format. Since staircases are frequently diagonal in-frame, a true axis-aligned box enclosing that silhouette should include a fair amount of "empty" surrounding space — a box that instead hugs the diagonal shape closely suggests the source coordinates may not be reliable axis-aligned boxes at all.

Both marked `audit_status: failed` in `datasets.yaml` — not discarded outright, since re-labeling is possible if the volume is worth recovering. Not being actively pursued right now; `stairs_i2yia` and `escalator_stairs` (both still `pending`) are prioritized first since they haven't shown this problem yet.

**Follow-up for `box_audit.py` (Stage 5.3, not yet built):** worth adding a heuristic check for this pattern generally — e.g. flag boxes whose fill ratio against a rotated/silhouette estimate looks abnormally tight for a nominally axis-aligned annotation. Diagonal-shaped classes (Stairs, Escalator, Pole) are the most likely to expose this.

### Elevator — `elevator_status_s4lrk` partially usable, same box-shape concern

Originally flagged (see `datasets.yaml` audit_note, pre-2026-08-08) as a possible scene/state classifier rather than true object detection. On inspection: it's primarily close-up shots of elevators, and does contain multiple state-related labels — these can be collapsed into the single canonical `elevator` class rather than preserved as distinct states, since this project doesn't need elevator-state discrimination. It also contains genuine object-detection-style samples: images with people inside the elevator that do have real bounding boxes, not just image-level classification tags.

Same box-shape concern as Stairs applies here too — the boxes cling to object shape rather than sitting as a clean axis-aligned rectangle. Still `pending` in `datasets.yaml`; not resolved to `approved`/`rejected` yet since isolating the genuinely-useful detection-style subset from the classifier-style images still needs the real Stage 5.3 audit pass, not just this initial look.

### Augmented Tricycle — needs dedup, missing Person labels

`augmented_tricycle` needs a duplicate-image pass (FiftyOne Brain, per DEC-025 — this is Stage 5.6's job, flagged here so it isn't forgotten when that source reaches merge) and is missing Person annotations for images where people are clearly visible (e.g. the tricycle driver). See the model-assisted pre-labeling procedure above for how to fill these in without full manual re-annotation.

### Traffico-y1 — class list resolved

Full native class list obtained: `Jeepney`, `Motorcycle`, `null`, `Tricycle`, `tricycle`. Resolution logged as DEC-032 in `docs/DECISIONS.md`. The duplicate `Tricycle`/`tricycle` entries (case variants) both map to the canonical `tricycle` class. `null` is left unresolved for now — likely an empty/background class artifact from the source project, not a real label to map.

"""
scripts/curate/run_mistakenness.py
─────────────────────────────────────
Stage 5.5 Model-Assisted Curation: FiftyOne Brain mistakenness computation.
Tool-agnostic about what happens next (docs/OPEN_QUESTIONS.md #5 — CVAT vs.
Label Studio for the *correction* step is a separate, still-open question
this script doesn't need answered) — it only scores and ranks samples for
human review.

Mistakenness needs two label fields per sample: a `predictions` field and
a `ground_truth` field to compare against. This project has no trained
model of its own yet (that's Phase 3, RunPod, not done here) — so
predictions come from a pretrained, COCO-trained YOLOv8n (`yolov8n.pt`,
zero-shot, already an existing `requirements.txt` dependency, just not yet
installed in this environment before this script needed it).

That only works for canonical classes with an unambiguous COCO analog.
The crosswalk below is not invented — every entry reuses `classes.yaml`'s
own already-decided `native_class` field for that class's Open Images
primary source, cross-referenced against COCO's fixed, standard 80-class
list (asserted present in yolov8n's own `model.names` before use, not
assumed):

    Person     -> person
    Vehicle    -> car, bus, truck      (classes.yaml's OI native_class list, verbatim)
    Motorcycle -> motorcycle
    Bicycle    -> bicycle
    Animals    -> dog, cat             (classes.yaml's OI native_class is Dog/Cat only —
                                         COCO's other animal classes, e.g. bird/horse/cow,
                                         are deliberately NOT pulled in)
    Chairs     -> chair
    Tables     -> dining table         (COCO's only table-like class; classes.yaml
                                         already groups Table + Coffee table as one
                                         canonical class, so this is a forced 1:1, not
                                         a choice between options)

The other 9 canonical classes (Pole, Stairs, Escalator, Doors, Tricycle,
Potholes, Trash Bins, Elevator, Pedestrian Lane) have no COCO equivalent —
mistakenness is genuinely not computable for them without this project's
own trained model. They're explicitly skipped and reported as such, not
silently dropped.

Operates on Stage 5.4's *capped* selection (dataset/reports/cap_report.json's
per-class "selected" lists), not the raw uncapped processed/ pool — that's
already the practically-relevant post-cap set, and keeps this bounded to a
size that actually finishes in one unattended run (~23k unique images
across the 7 eligible classes, measured ~60 img/s on this machine's MPS
backend -> ~6-7 minutes, not hours).

Read-only against dataset/processed/ and dataset/curated/ — writes only to
dataset/reports/. Does not flag samples for deletion; ranks by mistakenness
for the student's review, same posture as box_audit.py and cap_per_class.py.

Usage:
    python3 scripts/curate/run_mistakenness.py
    python3 scripts/curate/run_mistakenness.py --dry-run
    python3 scripts/curate/run_mistakenness.py --limit 500   # smoke-test on a subset
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

# Leaf script, not a shared library module — needs the repo root on
# sys.path to import sibling utils regardless of how it's invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.config_loader import get_class_id  # noqa: E402
from scripts.utils.file_utils import build_stem_index, ensure_dir, processed_dir, reports_dir  # noqa: E402

# canonical_class_key -> [COCO class names]. See module docstring for provenance.
COCO_CROSSWALK: dict[str, list[str]] = {
    "person": ["person"],
    "vehicle": ["car", "bus", "truck"],
    "motorcycle": ["motorcycle"],
    "bicycle": ["bicycle"],
    "animals": ["dog", "cat"],
    "chairs": ["chair"],
    "tables": ["dining table"],
}
CANONICAL_KEY_TO_NAME = {
    "person": "Person", "vehicle": "Vehicle", "motorcycle": "Motorcycle",
    "bicycle": "Bicycle", "animals": "Animals", "chairs": "Chairs", "tables": "Tables",
}
NO_COCO_ANALOG = [
    "Pole", "Stairs", "Escalator", "Doors", "Tricycle",
    "Potholes", "Trash Bins", "Elevator", "Pedestrian Lane",
]

ELIGIBLE_CANONICAL_IDS = {get_class_id(name) for name in CANONICAL_KEY_TO_NAME.values()}


def yolo_to_fo_bbox(cx: float, cy: float, w: float, h: float) -> list[float]:
    """YOLO center-based (cx, cy, w, h) -> FiftyOne top-left-based [x, y, w, h]."""
    return [cx - w / 2, cy - h / 2, w, h]


def load_union_images(cap_report_path: Path, limit: int | None) -> set[tuple[str, str]]:
    """Union of (source, filename) across the 7 COCO-eligible classes' capped selections."""
    cap_report = json.loads(cap_report_path.read_text(encoding="utf-8"))
    union: set[tuple[str, str]] = set()
    for key, class_name in CANONICAL_KEY_TO_NAME.items():
        selected = cap_report["classes"][class_name]["selected"]
        for entry in selected:
            source, filename = entry.split("/", 1)
            union.add((source, filename))
    images = sorted(union)
    if limit is not None:
        images = images[:limit]
    return set(images)


def load_eligible_ground_truth(source: str, filename: str, canonical_names: list[str]) -> list[dict[str, Any]]:
    """Ground-truth boxes for one image, filtered to the 7 COCO-eligible canonical classes only."""
    label_path = processed_dir(source) / "labels" / f"{filename}.txt"
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        if class_id not in ELIGIBLE_CANONICAL_IDS:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append({"label": canonical_names[class_id], "bounding_box": yolo_to_fo_bbox(cx, cy, w, h)})
    return boxes


def run(dry_run: bool = False, limit: int | None = None) -> dict[str, Any]:
    from scripts.utils.config_loader import get_canonical_names

    canonical_names = get_canonical_names()
    cap_report_path = reports_dir() / "cap_report.json"
    if not cap_report_path.is_file():
        raise FileNotFoundError(
            f"{cap_report_path} not found — run scripts/preprocess/cap_per_class.py (Stage 5.4) first."
        )

    images = load_union_images(cap_report_path, limit)
    print(f"COCO-eligible classes: {sorted(CANONICAL_KEY_TO_NAME.values())}")
    print(f"No COCO analog, skipped: {NO_COCO_ANALOG}")
    print(f"Unique images to score: {len(images)}")

    if dry_run:
        return {"unique_images": len(images), "eligible_classes": sorted(CANONICAL_KEY_TO_NAME.values())}

    from ultralytics import YOLO
    import torch
    import fiftyone as fo
    import fiftyone.brain as fob

    model = YOLO("yolov8n.pt")
    coco_names_lower = {v.lower(): v for v in model.names.values()}
    for key, coco_list in COCO_CROSSWALK.items():
        for c in coco_list:
            assert c in coco_names_lower, f"COCO class {c!r} not found in yolov8n.names"
    coco_to_canonical = {c: CANONICAL_KEY_TO_NAME[key] for key, cs in COCO_CROSSWALK.items() for c in cs}

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running yolov8n inference on {len(images)} images (device={device})...")

    image_index = {source: build_stem_index(source) for source in {source for source, _ in images}}
    image_records = []
    for source, filename in sorted(images):
        img_path = image_index.get(source, {}).get(filename)
        if img_path is None:
            continue
        image_records.append((source, filename, img_path))

    paths = [str(p) for _, _, p in image_records]
    BATCH = 16
    predictions_by_path: dict[str, list[dict[str, Any]]] = {}
    for i in range(0, len(paths), BATCH):
        batch = paths[i:i + BATCH]
        results = model.predict(batch, device=device, verbose=False)
        for path, result in zip(batch, results):
            dets = []
            names = result.names
            for box in result.boxes:
                coco_name = names[int(box.cls.item())]
                if coco_name not in coco_to_canonical:
                    continue
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                dets.append({
                    "label": coco_to_canonical[coco_name],
                    "bounding_box": [x1, y1, x2 - x1, y2 - y1],
                    "confidence": float(box.conf.item()),
                })
            predictions_by_path[path] = dets
        if (i // BATCH) % 20 == 0:
            print(f"  inferred {min(i + BATCH, len(paths))}/{len(paths)}...")

    print("Inference complete. Building FiftyOne dataset...")
    dataset_name = f"mistakenness_stage5_5_{uuid.uuid4().hex[:8]}"
    dataset = fo.Dataset(name=dataset_name)
    try:
        samples = []
        meta_by_filepath: dict[str, tuple[str, str]] = {}
        for source, filename, img_path in image_records:
            path_str = str(img_path)
            gt_boxes = load_eligible_ground_truth(source, filename, canonical_names)
            pred_boxes = predictions_by_path.get(path_str, [])
            sample = fo.Sample(filepath=path_str)
            sample["ground_truth"] = fo.Detections(detections=[fo.Detection(**b) for b in gt_boxes])
            sample["predictions"] = fo.Detections(detections=[fo.Detection(**b) for b in pred_boxes])
            samples.append(sample)
            meta_by_filepath[path_str] = (source, filename)
        dataset.add_samples(samples)

        print("Running fiftyone.brain.compute_mistakenness...")
        fob.compute_mistakenness(dataset, "predictions", label_field="ground_truth")

        ranked = []
        for sample in dataset.select_fields(["filepath", "mistakenness", "ground_truth", "predictions"]):
            source, filename = meta_by_filepath[sample.filepath]
            gt_classes = sorted({d.label for d in sample.ground_truth.detections})
            ranked.append({
                "source": source,
                "filename": filename,
                "mistakenness": sample.mistakenness,
                "canonical_classes": gt_classes,
                "gt_box_count": len(sample.ground_truth.detections),
                "pred_box_count": len(sample.predictions.detections),
            })
        ranked.sort(key=lambda r: r["mistakenness"], reverse=True)
    finally:
        # Scratch FiftyOne dataset, not persisted — try/finally so a mid-computation error
        # doesn't leave a ~23k-sample orphaned dataset registered in FiftyOne's backing store.
        dataset.delete()

    report = {
        "eligible_classes": sorted(CANONICAL_KEY_TO_NAME.values()),
        "no_coco_analog_classes_skipped": NO_COCO_ANALOG,
        "coco_crosswalk": COCO_CROSSWALK,
        "confidence_threshold": "ultralytics default (0.25) — tool default, not tuned",
        "unique_images_scored": len(ranked),
        "ranked": ranked,
    }

    ensure_dir(reports_dir())
    report_path = reports_dir() / "mistakenness_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n{len(ranked)} images scored. Report written to {report_path}")
    if ranked:
        print("Top 5 by mistakenness:")
        for r in ranked[:5]:
            print(f"  {r['mistakenness']:.4f}  {r['source']}/{r['filename']}  "
                  f"classes={r['canonical_classes']}  gt={r['gt_box_count']} pred={r['pred_box_count']}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print scope only — no inference, no report.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of images scored (smoke test).")
    args = parser.parse_args()

    print("=" * 60)
    print("run_mistakenness.py — Stage 5.5")
    print("=" * 60)

    run(dry_run=args.dry_run, limit=args.limit)

    if args.dry_run:
        print("\n--dry-run: no inference run, no report written.")


if __name__ == "__main__":
    main()

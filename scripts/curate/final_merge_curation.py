"""
scripts/curate/final_merge_curation.py
─────────────────────────────────────────
Stage 5.7 Final Pre-Split Curation Gate: same mistakenness analysis as
run_mistakenness.py (Stage 5.5), reused here against dataset/merged/ (the
Stage 5.6 pooled, source-prefixed output) as a final check before Stage
5.8 splits it. Same posture as every report-producing stage this
session: flags samples for the student to fix in place, doesn't touch
anything itself.

Reuses run_mistakenness.py's COCO crosswalk, eligible-class set, and box
conversion helper directly (same module, scripts/curate/) rather than
duplicating them — same 7 canonical classes with an unambiguous COCO
analog, same reasoning, see that module's docstring for the full
crosswalk provenance.

Does not block Stage 5.8/5.9: `split.py` and `generate_yaml.py` were
built and can run directly against dataset/merged/ without waiting for
this stage's flags to be resolved, the same way `merge.py` (Stage 5.6)
didn't wait on Stage 5.5's flags — none of Stages 5.5/5.7's findings
gate the physical pipeline, they're human-review inputs (see DEC-058's
reasoning, which applies here too).

Usage:
    python3 scripts/curate/final_merge_curation.py
    python3 scripts/curate/final_merge_curation.py --dry-run
    python3 scripts/curate/final_merge_curation.py --limit 500   # smoke test
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

from scripts.curate.run_mistakenness import (  # noqa: E402
    CANONICAL_KEY_TO_NAME, COCO_CROSSWALK, ELIGIBLE_CANONICAL_IDS, NO_COCO_ANALOG, yolo_to_fo_bbox,
)
from scripts.utils.config_loader import get_canonical_names  # noqa: E402
from scripts.utils.file_utils import ensure_dir, list_images, merged_dir, reports_dir  # noqa: E402


def load_eligible_ground_truth(label_path: Path, canonical_names: list[str]) -> list[dict[str, Any]]:
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
    canonical_names = get_canonical_names()
    img_dir = merged_dir() / "images"
    lbl_dir = merged_dir() / "labels"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"{img_dir} not found — run scripts/build/merge.py (Stage 5.6) first.")

    eligible_files = []
    for label_path in sorted(lbl_dir.glob("*.txt")):
        boxes = load_eligible_ground_truth(label_path, canonical_names)
        if boxes:
            eligible_files.append((label_path.stem, boxes))
    if limit is not None:
        eligible_files = eligible_files[:limit]

    print(f"COCO-eligible classes: {sorted(CANONICAL_KEY_TO_NAME.values())}")
    print(f"No COCO analog, skipped: {NO_COCO_ANALOG}")
    print(f"Merged-pool images with >=1 eligible-class box: {len(eligible_files)}")

    if dry_run:
        return {"eligible_images": len(eligible_files)}

    from ultralytics import YOLO
    import torch
    import fiftyone as fo
    import fiftyone.brain as fob

    model = YOLO("yolov8n.pt")
    coco_names_lower = {v.lower(): v for v in model.names.values()}
    for coco_list in COCO_CROSSWALK.values():
        for c in coco_list:
            assert c in coco_names_lower, f"COCO class {c!r} not found in yolov8n.names"
    coco_to_canonical = {c: CANONICAL_KEY_TO_NAME[key] for key, cs in COCO_CROSSWALK.items() for c in cs}

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running yolov8n inference on {len(eligible_files)} images (device={device})...")

    # list_images() (not a bare is_file() check) filters to IMAGE_EXTENSIONS — a stray
    # non-image file sharing a stem with a real image, or just present under
    # dataset/merged/images/, could otherwise get fed into model.predict() (DEC-061's
    # project-wide fix, applied here since this file's own inline lookup had drifted
    # from it). build_stem_index() itself operates per-source on dataset/processed/,
    # not the flat pooled dataset/merged/, so the same filtering is done inline here.
    image_index = {p.stem: p for p in list_images(img_dir, recursive=False)}
    records = [(stem, boxes, image_index[stem]) for stem, boxes in eligible_files if stem in image_index]
    missing_images = [stem for stem, _ in eligible_files if stem not in image_index]
    if missing_images:
        print(f"  WARNING: {len(missing_images)} eligible labels had no matching image file "
              f"(sample: {missing_images[:5]})")

    paths = [str(p) for _, _, p in records]
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
    dataset_name = f"final_curation_stage5_7_{uuid.uuid4().hex[:8]}"
    dataset = fo.Dataset(name=dataset_name)
    try:
        samples = []
        meta_by_filepath: dict[str, str] = {}
        for stem, boxes, img_path in records:
            path_str = str(img_path)
            pred_boxes = predictions_by_path.get(path_str, [])
            sample = fo.Sample(filepath=path_str)
            sample["ground_truth"] = fo.Detections(detections=[fo.Detection(**b) for b in boxes])
            sample["predictions"] = fo.Detections(detections=[fo.Detection(**b) for b in pred_boxes])
            samples.append(sample)
            meta_by_filepath[path_str] = stem
        dataset.add_samples(samples)

        print("Running fiftyone.brain.compute_mistakenness...")
        fob.compute_mistakenness(dataset, "predictions", label_field="ground_truth")

        ranked = []
        for sample in dataset.select_fields(["filepath", "mistakenness", "ground_truth", "predictions"]):
            stem = meta_by_filepath[sample.filepath]
            source = stem.split("__", 1)[0]
            gt_classes = sorted({d.label for d in sample.ground_truth.detections})
            ranked.append({
                "filename": stem,
                "source": source,
                "mistakenness": sample.mistakenness,
                "canonical_classes": gt_classes,
                "gt_box_count": len(sample.ground_truth.detections),
                "pred_box_count": len(sample.predictions.detections),
            })
        ranked.sort(key=lambda r: r["mistakenness"], reverse=True)
    finally:
        # Scratch FiftyOne dataset, not persisted — try/finally so a mid-computation error
        # doesn't leave an orphaned dataset registered in FiftyOne's backing store.
        dataset.delete()

    report = {
        "eligible_classes": sorted(CANONICAL_KEY_TO_NAME.values()),
        "no_coco_analog_classes_skipped": NO_COCO_ANALOG,
        "confidence_threshold": "ultralytics default (0.25) — tool default, not tuned",
        "eligible_labels_found": len(eligible_files),
        "missing_images": missing_images,
        "images_scored": len(ranked),
        "ranked": ranked,
    }

    ensure_dir(reports_dir())
    report_path = reports_dir() / "final_merge_curation_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n{len(ranked)} images scored. Report written to {report_path}")
    if ranked:
        print("Top 5 by mistakenness:")
        for r in ranked[:5]:
            print(f"  {r['mistakenness']:.4f}  {r['filename']}  classes={r['canonical_classes']}  "
                  f"gt={r['gt_box_count']} pred={r['pred_box_count']}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print scope only — no inference, no report.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of images scored (smoke test).")
    args = parser.parse_args()

    print("=" * 60)
    print("final_merge_curation.py — Stage 5.7")
    print("=" * 60)

    run(dry_run=args.dry_run, limit=args.limit)

    if args.dry_run:
        print("\n--dry-run: no inference run, no report written.")


if __name__ == "__main__":
    main()

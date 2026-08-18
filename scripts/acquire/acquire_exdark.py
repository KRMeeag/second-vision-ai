"""
scripts/acquire/acquire_exdark.py
──────────────────────────────────────
Stage 5.1 acquisition (parse-only, DEC-041): ExDark's archive is a
manual download — this script assumes the student has already
extracted it to dataset/raw/exdark/ExDark/<class>/ (images) and
dataset/raw/exdark/ExDark_Annno/<class>/ (annotations), one folder per
each of ExDark's 12 native classes. Its job starts at parsing.

Parsing logic (bbGt header skip, per-line class/x/y/w/h extraction) is
adapted from wraphex/ExDark2Yolo (github.com/wraphex/ExDark2Yolo, cited
as parser_reference in datasets.yaml) — used as a logic reference only,
not run as-is, because as published it does three things this project
can't use unmodified:
  1. Bakes in its own train/test/val split — conflicts with DEC-036
     (every source's split is discarded at acquisition; the only split
     that matters is the one Stage 5.8 computes on the merged pool).
  2. Emits all 12 ExDark classes with ITS OWN positional class ids —
     this project only maps 8 of them to canonical classes (per
     datasets.yaml's exdark_to_canonical_class_map); Boat/Bottle/Bus/Cup
     have no canonical slot and must be dropped, and kept classes need
     this project's canonical ids (config_loader.CANONICAL_NAMES), not
     ExDark's own list order.
  3. No box clamping — this project's bbox_utils.validate_bbox/
     clip_bbox already exists specifically to cover that gap.

Verified directly against the placed data (not assumed): all 7,363
images present, zero image/annotation filename mismatches when looked
up as `annotation_dir / (image_filename + ".txt")` — simpler and more
robust than the reference script's derive-image-path-from-annotation-
name approach, so that part isn't ported.

Output ("intermediate schema" for Stage 5.2, first defined here — see
DEC-046): canonical-class-remapped YOLO-format labels, ONE FLAT POOL
(no split), to dataset/processed/exdark/images/ + .../labels/. Images
whose only boxes were non-canonical classes are dropped entirely
(same "drop now-empty samples" pattern as acquire_openimages.py's
IsDepiction/IsGroupOf filter, DEC-043).

Usage:
    python3 scripts/acquire/acquire_exdark.py
    python3 scripts/acquire/acquire_exdark.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Leaf script, not a shared library module — needs the repo root on
# sys.path to import sibling utils regardless of how it's invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.bbox_utils import clip_bbox, validate_bbox, xywh_abs_to_yolo  # noqa: E402
from scripts.utils.config_loader import load_classes, get_source_config  # noqa: E402
from scripts.utils.file_utils import ensure_dir, processed_dir, raw_dir, reports_dir, safe_copy  # noqa: E402

IMAGES_SUBDIR = "ExDark"
ANNOTATIONS_SUBDIR = "ExDark_Annno"

# ExDark's 12 native class folder names (fixed, matches the archive's
# own directory names — not the same as our canonical class names).
EXDARK_NATIVE_CLASSES = (
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
)


def parse_annotation(
    ann_path: Path, class_map: dict[str, str]
) -> tuple[list[tuple[str, int, int, int, int]], int]:
    """
    Parse one ExDark bbGt annotation file.

    Skips the `% bbGt version=3` header line, then reads one object per
    line as `<native_class> <x> <y> <w> <h> <...ignored flags>` (absolute
    top-left pixel box). Lines whose native class has no entry in
    `class_map` (Boat/Bottle/Bus/Cup) are dropped and counted, not
    converted.

    Returns
    -------
    tuple[list[tuple[str, int, int, int, int]], int]
        ((canonical_class, x, y, w, h) for every box that maps to a
        canonical class), count of non-canonical boxes skipped.
    """
    boxes = []
    non_canonical_count = 0
    with ann_path.open("r", encoding="utf-8", errors="replace") as fh:
        next(fh, None)  # header line
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            native_class = parts[0]
            canonical_class = class_map.get(native_class)
            if canonical_class is None:
                non_canonical_count += 1
                continue
            x, y, w, h = (int(v) for v in parts[1:5])
            boxes.append((canonical_class, x, y, w, h))
    return boxes, non_canonical_count


def convert_image(
    img_path: Path,
    boxes: list[tuple[str, int, int, int, int]],
    class_id_map: dict[str, int],
) -> tuple[list[str], dict[str, int]]:
    """
    Convert one image's already-parsed canonical boxes to YOLO label lines.

    Parameters
    ----------
    class_id_map : dict[str, int]
        Maps a classes.yaml class key (e.g. "bicycle") to its YOLO class
        id. Note this is NOT the same as config_loader.CANONICAL_NAMES'
        title-case display strings — datasets.yaml's
        exdark_to_canonical_class_map values are classes.yaml dict keys.

    Returns
    -------
    tuple[list[str], dict[str, int]]
        YOLO label lines (empty if `boxes` is empty), and a tally of
        invalid-dropped / clipped boxes for the report.
    """
    from PIL import Image

    tally = {"invalid_dropped": 0, "clipped": 0}
    if not boxes:
        return [], tally

    with Image.open(img_path) as img:
        width, height = img.size

    lines = []
    for canonical_class, x, y, w, h in boxes:
        cx, cy, nw, nh = xywh_abs_to_yolo(x, y, w, h, width, height)
        reason = validate_bbox(cx, cy, nw, nh)
        if reason is not None:
            if nw <= 0 or nh <= 0:
                tally["invalid_dropped"] += 1
                continue
            cx, cy, nw, nh = clip_bbox(cx, cy, nw, nh)
            # A box entirely outside the frame clips down to degenerate
            # zero width/height, not a usable one — re-check (DEC-057).
            if nw <= 0 or nh <= 0:
                tally["invalid_dropped"] += 1
                continue
            tally["clipped"] += 1
        class_id = class_id_map[canonical_class]
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return lines, tally


def run(dry_run: bool = False) -> dict[str, Any]:
    exdark_config = get_source_config("exdark")
    class_map = exdark_config["exdark_to_canonical_class_map"]
    class_id_map = {key: entry["id"] for key, entry in load_classes()["classes"].items()}

    images_root = raw_dir("exdark") / IMAGES_SUBDIR
    annotations_root = raw_dir("exdark") / ANNOTATIONS_SUBDIR
    out_images_dir = processed_dir("exdark") / "images"
    out_labels_dir = processed_dir("exdark") / "labels"

    stats: dict[str, Any] = {
        "per_native_class": {},
        "images_converted": 0,
        "images_dropped_empty": 0,
        "boxes_kept": 0,
        "boxes_dropped_non_canonical": 0,
        "boxes_invalid_dropped": 0,
        "boxes_clipped": 0,
    }

    if not dry_run:
        ensure_dir(out_images_dir)
        ensure_dir(out_labels_dir)

    for native_class in EXDARK_NATIVE_CLASSES:
        img_dir = images_root / native_class
        ann_dir = annotations_root / native_class
        if not img_dir.is_dir():
            print(f"  WARNING: missing image folder for {native_class}, skipping")
            continue

        image_paths = sorted(p for p in img_dir.iterdir() if p.is_file() and not p.name.startswith("."))
        kept_here = 0

        for img_path in image_paths:
            ann_path = ann_dir / f"{img_path.name}.txt"
            if not ann_path.is_file():
                continue

            boxes, non_canonical_count = parse_annotation(ann_path, class_map)
            stats["boxes_dropped_non_canonical"] += non_canonical_count

            if not boxes:
                stats["images_dropped_empty"] += 1
                continue

            if dry_run:
                kept_here += 1
                stats["boxes_kept"] += len(boxes)
                continue

            lines, tally = convert_image(img_path, boxes, class_id_map)
            stats["boxes_invalid_dropped"] += tally["invalid_dropped"]
            stats["boxes_clipped"] += tally["clipped"]

            if not lines:
                stats["images_dropped_empty"] += 1
                continue

            safe_copy(img_path, out_images_dir / img_path.name, overwrite=True)
            label_path = out_labels_dir / f"{img_path.stem}.txt"
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            kept_here += 1
            stats["boxes_kept"] += len(lines)
            stats["images_converted"] += 1

        stats["per_native_class"][native_class] = {
            "total_images": len(image_paths),
            "kept": kept_here,
        }
        print(f"  {native_class:10s} {len(image_paths):4d} images -> {kept_here:4d} kept")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and tally without writing any converted output.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("acquire_exdark.py — Stage 5.1 (parse-only, DEC-041)")
    print("=" * 60)
    print()

    stats = run(dry_run=args.dry_run)

    print()
    print(f"Images converted : {stats['images_converted']}")
    print(f"Images dropped (no canonical boxes left): {stats['images_dropped_empty']}")
    print(f"Boxes kept       : {stats['boxes_kept']}")
    print(f"Boxes dropped (non-canonical class): {stats['boxes_dropped_non_canonical']}")
    print(f"Boxes clipped    : {stats['boxes_clipped']}")
    print(f"Boxes invalid, dropped: {stats['boxes_invalid_dropped']}")

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return

    report_path = ensure_dir(reports_dir()) / "acquire_exdark_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\nReport written to {report_path}")
    print(f"Output: {processed_dir('exdark')}")


if __name__ == "__main__":
    main()

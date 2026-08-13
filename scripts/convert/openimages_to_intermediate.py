"""
scripts/convert/openimages_to_intermediate.py
────────────────────────────────────────────────
Stage 5.2 conversion for Open Images: turns the 8 per-class COCO-style
exports under dataset/raw/open_images/<class_key>/ (built by
acquire_openimages.py) into DEC-046's intermediate schema — one flat
pool of dataset/processed/open_images/images/ + .../labels/, YOLO-format
labels using canonical class ids.

Two things this script has to handle that a naive per-folder conversion
would miss (both verified directly against the real exported data, not
assumed):

1. **Each class-folder's labels.json is NOT pre-filtered to that class.**
   acquire_openimages.py's `classes=native_classes` argument to the Zoo
   loader only selects which *images* to include (any image containing
   that class); every other object FiftyOne's ground_truth field has
   for those images comes along too — the "animals" folder's
   labels.json has 213 distinct category names in it, not just
   Dog/Cat. So this script filters annotations to each folder's own
   configured native_classes (same "drop what doesn't map" pattern
   `acquire_exdark.py` already uses), not just remaps ids.

2. **The same photo can appear in multiple class-folders.** Because
   each class is pulled independently, a photo containing both a chair
   and a table gets exported once under chairs/ and once under
   tables/ — verified directly: 1,578 of 3,236 chairs/ images are also
   in tables/, 706 also in person/, etc. (dataset/reports/
   openimages_to_intermediate_report.json has the full pairwise
   breakdown). Rather than letting the second folder's write silently
   clobber the first's label file, this script aggregates every
   canonical box for a given filename across ALL class-folders it
   appears in before writing a single merged label file — the photo
   genuinely contains both objects, so both belong in its ground truth.

Usage:
    python3 scripts/convert/openimages_to_intermediate.py
    python3 scripts/convert/openimages_to_intermediate.py --dry-run
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

from scripts.acquire.acquire_openimages import get_openimages_targets  # noqa: E402
from scripts.utils.bbox_utils import clip_bbox, validate_bbox, xywh_abs_to_yolo  # noqa: E402
from scripts.utils.file_utils import ensure_dir, processed_dir, raw_dir, reports_dir, safe_copy  # noqa: E402

IMAGES_SUBDIR = "data"


def collect_class_folder(
    class_key: str, class_id: int, native_classes: list[str], raw_root: Path
) -> tuple[list[tuple[str, int, int, int, list[float]]], dict[str, Any]]:
    """
    Load one class-folder's labels.json and keep only annotations whose
    category name is in this folder's own native_classes.

    Returns
    -------
    tuple
        (kept, stats) where kept is a list of
        (file_name, width, height, class_id, bbox) — bbox is COCO-style
        [x, y, w, h] absolute pixels — and stats tallies totals for the
        report.
    """
    label_path = raw_root / class_key / "labels.json"
    stats = {"annotations_total": 0, "annotations_kept": 0, "annotations_dropped_non_native": 0}

    if not label_path.is_file():
        print(f"  WARNING: missing {label_path}, skipping {class_key}")
        return [], stats

    with label_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    native_set = set(native_classes)
    img_by_id = {img["id"]: img for img in data["images"]}

    kept: list[tuple[str, int, int, int, list[float]]] = []
    for ann in data["annotations"]:
        stats["annotations_total"] += 1
        name = cat_id_to_name.get(ann["category_id"])
        if name not in native_set:
            stats["annotations_dropped_non_native"] += 1
            continue
        img = img_by_id[ann["image_id"]]
        kept.append((img["file_name"], img["width"], img["height"], class_id, ann["bbox"]))
        stats["annotations_kept"] += 1

    return kept, stats


def audit_class_counts(
    pool: dict[str, dict[str, Any]], targets: dict[str, dict[str, Any]]
) -> dict[str, dict[str, int]]:
    """
    For each canonical class, count both:
      - images: how many images in the merged pool carry at least one box
        of that class — non-exclusive, an image with both a Chair and a
        Table box counts toward both classes' image totals, not just one.
      - instances: total box count for that class across the merged pool
        (every box counts, even multiple of the same class in one image).

    These are the real numbers each class will actually draw on once
    merged (Stage 5.6), not the raw per-folder pull counts.
    """
    class_id_to_key = {info["class_id"]: class_key for class_key, info in targets.items()}
    counts = {class_key: {"images": 0, "instances": 0} for class_key in targets}
    for entry in pool.values():
        present_ids = [class_id for class_id, _ in entry["boxes"]]
        for class_id in set(present_ids):
            counts[class_id_to_key[class_id]]["images"] += 1
        for class_id in present_ids:
            counts[class_id_to_key[class_id]]["instances"] += 1
    return counts


def convert_pooled_image(
    width: int, height: int, boxes: list[tuple[int, list[float]]]
) -> tuple[list[str], dict[str, int]]:
    """
    Convert one image's aggregated (class_id, COCO bbox) pairs to YOLO
    label lines.

    Returns
    -------
    tuple[list[str], dict[str, int]]
        YOLO label lines, and a tally of invalid-dropped/clipped boxes.
    """
    tally = {"invalid_dropped": 0, "clipped": 0}
    lines = []
    for class_id, (x, y, w, h) in boxes:
        cx, cy, nw, nh = xywh_abs_to_yolo(x, y, w, h, width, height)
        reason = validate_bbox(cx, cy, nw, nh)
        if reason is not None:
            if nw <= 0 or nh <= 0:
                tally["invalid_dropped"] += 1
                continue
            cx, cy, nw, nh = clip_bbox(cx, cy, nw, nh)
            tally["clipped"] += 1
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines, tally


def run(dry_run: bool = False) -> dict[str, Any]:
    targets = get_openimages_targets()
    raw_root = raw_dir("open_images")
    out_images_dir = processed_dir("open_images") / "images"
    out_labels_dir = processed_dir("open_images") / "labels"

    stats: dict[str, Any] = {
        "per_class_folder": {},
        "images_converted": 0,
        "images_merged_across_classes": 0,
        "images_dropped_empty": 0,
        "boxes_kept": 0,
        "boxes_invalid_dropped": 0,
        "boxes_clipped": 0,
    }

    # Pass 1: collect every canonical box, per filename, across all 8
    # class-folders. dims are read once per filename (first-seen); a
    # mismatch would mean two different photos share a filename, which
    # shouldn't happen within one source's own id namespace — checked,
    # not assumed.
    pool: dict[str, dict[str, Any]] = {}
    for class_key, info in sorted(targets.items()):
        kept, folder_stats = collect_class_folder(
            class_key, info["class_id"], info["native_classes"], raw_root
        )
        stats["per_class_folder"][class_key] = folder_stats
        print(
            f"  {class_key:12s} {folder_stats['annotations_total']:5d} annotations -> "
            f"{folder_stats['annotations_kept']:5d} kept (native={info['native_classes']})"
        )
        for file_name, width, height, class_id, bbox in kept:
            entry = pool.setdefault(
                file_name, {"width": width, "height": height, "boxes": [], "source_folders": set()}
            )
            if entry["width"] != width or entry["height"] != height:
                print(
                    f"  WARNING: dimension mismatch for {file_name} "
                    f"({entry['width']}x{entry['height']} vs {width}x{height}) — "
                    f"keeping first-seen dims"
                )
            entry["boxes"].append((class_id, bbox))
            entry["source_folders"].add(class_key)

    print(f"\n{len(pool)} unique images across all classes "
          f"({sum(len(e['boxes']) for e in pool.values())} total boxes)")

    stats["final_per_class_counts"] = audit_class_counts(pool, targets)
    print("\nFinal per-class counts (images non-exclusive — an image can "
          "count toward more than one class; instances = total boxes):")
    for class_key, c in sorted(stats["final_per_class_counts"].items()):
        print(f"  {class_key:12s} images={c['images']:5d}  instances={c['instances']:5d}")

    if dry_run:
        merged = sum(1 for e in pool.values() if len(e["source_folders"]) > 1)
        stats["images_merged_across_classes"] = merged
        stats["images_converted"] = len(pool)
        stats["boxes_kept"] = sum(len(e["boxes"]) for e in pool.values())
        print("\n--dry-run: no files written.")
        return stats

    ensure_dir(out_images_dir)
    ensure_dir(out_labels_dir)

    # Pass 2: write one image + one merged label file per unique filename.
    for file_name, entry in pool.items():
        if len(entry["source_folders"]) > 1:
            stats["images_merged_across_classes"] += 1

        lines, tally = convert_pooled_image(entry["width"], entry["height"], entry["boxes"])
        stats["boxes_invalid_dropped"] += tally["invalid_dropped"]
        stats["boxes_clipped"] += tally["clipped"]

        if not lines:
            stats["images_dropped_empty"] += 1
            continue

        # Any source folder this image appeared in has an identical copy
        # of the same Open Images photo — first one found is as good as any.
        src_class_key = next(iter(entry["source_folders"]))
        src_path = raw_root / src_class_key / IMAGES_SUBDIR / file_name
        safe_copy(src_path, out_images_dir / file_name, overwrite=True)

        label_path = out_labels_dir / f"{Path(file_name).stem}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        stats["images_converted"] += 1
        stats["boxes_kept"] += len(lines)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tally without writing any converted output.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("openimages_to_intermediate.py — Stage 5.2")
    print("=" * 60)
    print()

    stats = run(dry_run=args.dry_run)

    print()
    print(f"Images converted : {stats['images_converted']}")
    print(f"  of which merged across >1 class-folder: {stats['images_merged_across_classes']}")
    print(f"Images dropped (no canonical boxes left): {stats['images_dropped_empty']}")
    print(f"Boxes kept       : {stats['boxes_kept']}")
    print(f"Boxes clipped    : {stats['boxes_clipped']}")
    print(f"Boxes invalid, dropped: {stats['boxes_invalid_dropped']}")

    if args.dry_run:
        return

    report_path = ensure_dir(reports_dir()) / "openimages_to_intermediate_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\nReport written to {report_path}")
    print(f"Output: {processed_dir('open_images')}")


if __name__ == "__main__":
    main()

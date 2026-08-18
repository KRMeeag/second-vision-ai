"""
scripts/acquire/acquire_datasetninja.py
──────────────────────────────────────
Stage 5.1 acquisition (parse-only, DEC-041): both Dataset Ninja sources
for Potholes — `pothole-detection` (co-primary, DEC-034) and
`road-damage-detector` (deprioritized fallback, DEC-034) — are manual
downloads via the website's own "Download" button (DEC-041; the
`dataset_tools` SDK route was rejected after a real Dropbox rate-limit
failure). This script assumes both are already extracted under
dataset/raw/dataset_ninja_pothole_detection/ and
dataset/raw/dataset_ninja_road_damage_detector/.

Format: verified directly against the real placed data (not assumed —
datasets.yaml previously assumed Pascal VOC, which turned out wrong).
Both are genuine Supervisely project exports:
  <source>/<ds_subdir>/img/<name>            — images
  <source>/<ds_subdir>/ann/<name>.json        — one annotation per image
  <source>/meta.json                          — class id/title definitions
Annotation JSON: `size.{width,height}` gives image dimensions directly
(no need to open the image), `objects[].points.exterior` is a
corner-pair rectangle `[[x1,y1],[x2,y2]]` — not the top-left+w/h format
ExDark/CrowdHuman use, so this needed bbox_utils.xyxy_to_yolo (promoted
from a private helper for this), not xywh_abs_to_yolo.

Considered and rejected using the `supervisely`/`dataset_tools` SDK to
convert this properly (as datasets.yaml originally suggested via
`sly.convert.to_pascal_voc()`) — that package already caused real
friction this project hit before (system-level libmagic dependency,
xmltodict downgrade below FiftyOne's floor) for a format simple enough
to parse directly in ~20 lines. Verified point ordering isn't always
top-left-first across the real data (1 degenerate zero-width box found
in 1,740 checked) — handled by letting bbox_utils.validate_bbox flag
non-positive width/height rather than trusting point order.

`road-damage-detector` has 4 native classes; only "pothole" maps to
the canonical class (DEC-034) — alligator/lateral/longitudinal crack
are dropped, same per-line filtering pattern as acquire_exdark.py.

Output: DEC-046 intermediate schema, one dir per source (merging
across sources happens later, Stage 5.6) —
dataset/processed/dataset_ninja_pothole_detection/{images,labels}/ and
dataset/processed/dataset_ninja_road_damage_detector/{images,labels}/.

Usage:
    python3 scripts/acquire/acquire_datasetninja.py
    python3 scripts/acquire/acquire_datasetninja.py --sources dataset_ninja_pothole_detection
    python3 scripts/acquire/acquire_datasetninja.py --dry-run
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

from scripts.utils.bbox_utils import clip_bbox, validate_bbox, xyxy_to_yolo  # noqa: E402
from scripts.utils.config_loader import get_class_id  # noqa: E402
from scripts.utils.file_utils import ensure_dir, processed_dir, raw_dir, reports_dir  # noqa: E402

# Per source: the Supervisely project's dataset subfolder name (varies
# per export — "ds" vs "ds0", verified against the real placed data),
# and which native classTitle(s) map to which canonical class key.
SOURCES: dict[str, dict[str, Any]] = {
    "dataset_ninja_pothole_detection": {
        "ds_subdir": "ds",
        "class_filter": {"pothole": "potholes"},
    },
    "dataset_ninja_road_damage_detector": {
        "ds_subdir": "ds0",
        "class_filter": {"pothole": "potholes"},
        # alligator crack / lateral crack / longitudinal crack: no
        # canonical class, dropped (DEC-034).
    },
}

POTHOLES_CLASS_ID = get_class_id("Potholes")


def convert_source(source_key: str, config: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    ds_root = raw_dir(source_key) / config["ds_subdir"]
    img_dir = ds_root / "img"
    ann_dir = ds_root / "ann"
    class_filter = config["class_filter"]

    out_images_dir = processed_dir(source_key) / "images"
    out_labels_dir = processed_dir(source_key) / "labels"
    if not dry_run:
        ensure_dir(out_images_dir)
        ensure_dir(out_labels_dir)

    stats = {
        "images_converted": 0,
        "images_dropped_empty": 0,
        "images_missing": 0,
        "boxes_kept": 0,
        "boxes_dropped_non_canonical": 0,
        "boxes_invalid_dropped": 0,
        "boxes_clipped": 0,
    }

    if not ann_dir.is_dir():
        print(f"  WARNING: {ann_dir} not found, skipping {source_key}")
        return stats

    for ann_path in sorted(ann_dir.iterdir()):
        if not ann_path.name.endswith(".json"):
            continue
        with ann_path.open("r", encoding="utf-8") as fh:
            entry = json.load(fh)

        width = entry["size"]["width"]
        height = entry["size"]["height"]

        lines = []
        for obj in entry["objects"]:
            canonical = class_filter.get(obj["classTitle"])
            if canonical is None:
                stats["boxes_dropped_non_canonical"] += 1
                continue

            (px1, py1), (px2, py2) = obj["points"]["exterior"]
            nx1, ny1, nx2, ny2 = px1 / width, py1 / height, px2 / width, py2 / height
            cx, cy, w, h = xyxy_to_yolo(nx1, ny1, nx2, ny2)

            reason = validate_bbox(cx, cy, w, h)
            if reason is not None:
                if w <= 0 or h <= 0:
                    stats["boxes_invalid_dropped"] += 1
                    continue
                cx, cy, w, h = clip_bbox(cx, cy, w, h)
                # A box entirely outside the frame clips down to degenerate
                # zero width/height, not a usable one — re-check (DEC-057).
                if w <= 0 or h <= 0:
                    stats["boxes_invalid_dropped"] += 1
                    continue
                stats["boxes_clipped"] += 1

            lines.append(f"{POTHOLES_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not lines:
            stats["images_dropped_empty"] += 1
            continue

        image_name = ann_path.name[: -len(".json")]
        img_path = img_dir / image_name
        if not img_path.is_file():
            stats["images_missing"] += 1
            continue

        if dry_run:
            stats["images_converted"] += 1
            stats["boxes_kept"] += len(lines)
            continue

        (out_images_dir / image_name).write_bytes(img_path.read_bytes())
        label_stem = Path(image_name).stem
        (out_labels_dir / f"{label_stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        stats["images_converted"] += 1
        stats["boxes_kept"] += len(lines)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help=f"Comma-separated source keys to convert. Defaults to all: {sorted(SOURCES)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and tally without writing any converted output.",
    )
    args = parser.parse_args()

    sources = SOURCES
    if args.sources:
        requested = [s.strip() for s in args.sources.split(",") if s.strip()]
        unknown = [s for s in requested if s not in SOURCES]
        if unknown:
            print(f"Unknown source key(s): {unknown}. Available: {sorted(SOURCES)}")
            sys.exit(1)
        sources = {k: v for k, v in SOURCES.items() if k in requested}

    print("=" * 60)
    print("acquire_datasetninja.py — Stage 5.1 (parse-only, DEC-041)")
    print("=" * 60)

    all_stats = {}
    for source_key, config in sources.items():
        print(f"\n[{source_key}]")
        stats = convert_source(source_key, config, args.dry_run)
        all_stats[source_key] = stats
        for k, v in stats.items():
            print(f"  {k}: {v}")

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return

    report_path = ensure_dir(reports_dir()) / "acquire_datasetninja_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(all_stats, fh, indent=2)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()

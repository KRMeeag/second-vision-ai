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

!! DIVERGENCE WARNING (2026-09-12) !!
    Re-running this script does NOT reproduce the dataset/processed/open_images
    tree that v2 was built from. Measured against it by --dry-run:

      shelf   9,518 -> 0      classes.yaml dropped Shelf (DEC-126), so the
                              converter no longer emits it at all. Regenerating
                              would GUT dataset/final_v2a, the 15-class
                              measurement tree that needs Shelf to exist.
      others  -2,939          converter-stage dedup sees the raw annotation
                              stream, so DEC-132/133 catch more there than the
                              equivalent post-pass caught on already-deduped
                              output.

    The shipped tree was produced by the older converter plus two validated
    post-passes:
        scripts/preprocess/dedupe_geometric.py --apply          (DEC-132)
        scripts/preprocess/find_hierarchy_duplicates.py --apply (DEC-133)
    Those post-passes are what was visually reviewed and approved. Do NOT
    re-run this converter before a v2 training run. If it is ever re-run, both
    v2 trees and the 15-class Shelf handling must be rebuilt deliberately.

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
from scripts.utils.config_loader import get_class_id  # noqa: E402
from scripts.utils.bbox_utils import clip_bbox, validate_bbox, xywh_abs_to_yolo  # noqa: E402
from scripts.utils.file_utils import ensure_dir, processed_dir, raw_dir, reports_dir, safe_copy  # noqa: E402

IMAGES_SUBDIR = "data"

# DEC-132. Same class + IoU >= this => the same physical object, keep one.
# At 0.90 two boxes are very nearly coincident; genuinely distinct objects
# do not overlap that much.
DEDUP_IOU = 0.90

# DEC-133. Below DEDUP_IOU the raw native label decides — see the dedup block.
HIER_IOU = 0.70
NEST_IOMIN = 0.90


def coco_iou(a: tuple, b: tuple) -> float:
    """IoU of two COCO [x, y, w, h] boxes (absolute pixels)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def coco_iomin(a: tuple, b: tuple) -> float:
    """Intersection over the SMALLER box. ~1.0 means one box sits inside the
    other — one object annotated at two extents. Genuine occlusion instead
    puts a SMALL box inside a large one, which scores low here."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    smaller = min(aw * ah, bw * bh)
    return inter / smaller if smaller > 0 else 0.0


def build_native_to_canonical(targets: dict[str, Any]) -> dict[str, int]:
    """Every Open Images native class name -> its canonical id, across the WHOLE schema.

    DEC-127. The original converter filtered each folder's labels.json to that
    folder's OWN native_classes and discarded the rest. But
    acquire_openimages.py pulled whole images, so a photo exported under
    `tables/` carries its Chair, Person and Bench boxes too — all of which
    belong to canonical classes this project trains. Discarding them did not
    make them absent from the image; it made them BACKGROUND, which is the
    sparse-annotation defect DEC-124 diagnosed and DEC-125 measured at up to
    -0.79 mAP@0.5 for Person.

    Ids come from the targets built out of config/classes.yaml, never
    hardcoded (AGENTS.md) — a stale literal here once rendered every Pothole a
    Tricycle (DEC-107).
    """
    mapping: dict[str, int] = {}
    for class_key, info in sorted(targets.items()):
        for native in info["native_classes"]:
            prior = mapping.get(native)
            if prior is not None and prior != info["class_id"]:
                raise SystemExit(
                    f"config/classes.yaml maps Open Images '{native}' to two different "
                    f"canonical ids ({prior} and {info['class_id']}, via {class_key}). "
                    "A native class must belong to exactly one canonical class."
                )
            mapping[native] = info["class_id"]
    return mapping


def collect_class_folder(
    class_key: str, native_to_canonical: dict[str, int], raw_root: Path
) -> tuple[list[tuple[str, int, int, int, list[float]]], dict[str, Any]]:
    """
    Load one class-folder's labels.json and keep every annotation that maps to
    ANY canonical class -- not just this folder's own (DEC-127).

    `annotations_dropped_non_native` still counts what is discarded, but that
    now means "belongs to no canonical class at all" (Wheel, Clothing, Tree,
    and 440 other Open Images categories), which is correct to drop.

    Returns
    -------
    tuple
        (kept, stats) where kept is a list of
        (file_name, width, height, class_id, bbox, native) — bbox is
        COCO-style [x, y, w, h] absolute pixels, `native` is the raw Open
        Images category name (DEC-133 needs it to tell a hierarchy duplicate
        from a genuine neighbour) — and stats tallies totals for the report.
    """
    label_path = raw_root / class_key / "labels.json"
    stats = {
        "annotations_total": 0,
        "annotations_kept": 0,
        "annotations_dropped_non_native": 0,
        "annotations_kept_foreign_class": 0,
        "kept_by_native_class": {},
    }

    if not label_path.is_file():
        print(f"  WARNING: missing {label_path}, skipping {class_key}")
        return [], stats

    with label_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    img_by_id = {img["id"]: img for img in data["images"]}
    kept: list[tuple[str, int, int, int, list[float], str]] = []
    for ann in data["annotations"]:
        stats["annotations_total"] += 1
        name = cat_id_to_name.get(ann["category_id"])
        class_id = native_to_canonical.get(name)
        if class_id is None:
            stats["annotations_dropped_non_native"] += 1
            continue
        img = img_by_id[ann["image_id"]]
        kept.append((img["file_name"], img["width"], img["height"], class_id, ann["bbox"], name))
        stats["annotations_kept"] += 1
        stats["kept_by_native_class"][name] = stats["kept_by_native_class"].get(name, 0) + 1

    return kept, stats


# DEC-115: Bench boxes stacked on a Tables/Chairs box are the same object.
#
# Open Images annotates a picnic table as BOTH "Bench" and "Table", and a
# bench-style seat in a dining set as both "Bench" and "Chair". DEC-052 merges
# boxes across class folders into one label file, so without this an image
# carries two contradictory boxes on one object -- exactly the wrong-class-on-
# top-of-right-class supervision DEC-109/110/111 were written to stop.
#
# Measured across all three Open Images splits before implementing: of 7,042
# post-DEC-043 Bench boxes, 758 (10.8%) collide at IoU >= 0.5. 307 images lose
# their last Bench box; 3,280 images still carry one, which is 2.2x DEC-042's
# 1,500 floor. Sensitivity is mild -- IoU 0.3 leaves 3,236 and IoU 0.7 leaves
# 3,373 -- so the threshold choice does not decide the class's viability. 0.5
# is used because it matches DEC-109/110/111's existing precedent in this repo.
#
# The Bench box is always the one dropped. Tables and Chairs predate this class
# and their counts must not move.
BENCH_FURNITURE_IOU = 0.5


def _coco_iou(a: list[float], b: list[float]) -> float:
    """IoU of two COCO [x, y, w, h] boxes in absolute pixels."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def drop_bench_on_furniture(pool: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Remove Bench boxes overlapping a same-image Tables/Chairs box.

    Mutates `pool` in place. Ids come from config_loader, never hand-copied
    (AGENTS.md) -- a stale literal here once rendered every Pothole as a
    Tricycle.
    """
    bench_id = get_class_id("Bench")
    furniture_ids = {get_class_id("Tables"), get_class_id("Chairs")}
    tally = {"bench_boxes_dropped": 0, "images_affected": 0, "images_emptied_of_bench": 0}

    for entry in pool.values():
        boxes = entry["boxes"]
        bench = [(i, bb) for i, (cid, bb, _nat) in enumerate(boxes) if cid == bench_id]
        if not bench:
            continue
        furniture = [bb for cid, bb, _nat in boxes if cid in furniture_ids]
        if not furniture:
            continue
        drop = {
            i for i, bb in bench
            if any(_coco_iou(bb, f) >= BENCH_FURNITURE_IOU for f in furniture)
        }
        if not drop:
            continue
        entry["boxes"] = [b for i, b in enumerate(boxes) if i not in drop]
        tally["bench_boxes_dropped"] += len(drop)
        tally["images_affected"] += 1
        if len(drop) == len(bench):
            tally["images_emptied_of_bench"] += 1

    return tally


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
        present_ids = [class_id for class_id, _bb, _nat in entry["boxes"]]
        for class_id in set(present_ids):
            counts[class_id_to_key[class_id]]["images"] += 1
        for class_id in present_ids:
            counts[class_id_to_key[class_id]]["instances"] += 1
    return counts


def convert_pooled_image(
    width: int, height: int, boxes: list[tuple[int, list[float], str]]
) -> tuple[list[str], dict[str, int]]:
    """
    Convert one image's aggregated (class_id, COCO bbox, native name) triples
    to YOLO label lines. The native name is carried only for DEC-133 dedup and
    is not written to the label file.

    Returns
    -------
    tuple[list[str], dict[str, int]]
        YOLO label lines, and a tally of invalid-dropped/clipped boxes.
    """
    tally = {"invalid_dropped": 0, "clipped": 0}
    lines = []
    for class_id, (x, y, w, h), _native in boxes:
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
        "boxes_deduplicated": 0,
    }

    # Pass 1: collect every canonical box, per filename, across all 8
    # class-folders. dims are read once per filename (first-seen); a
    # mismatch would mean two different photos share a filename, which
    # shouldn't happen within one source's own id namespace — checked,
    # not assumed.
    native_to_canonical = build_native_to_canonical(targets)
    print(f"  schema-wide filter: {len(native_to_canonical)} Open Images native classes "
          f"map into {len(set(native_to_canonical.values()))} canonical classes")

    pool: dict[str, dict[str, Any]] = {}
    for class_key, info in sorted(targets.items()):
        kept, folder_stats = collect_class_folder(class_key, native_to_canonical, raw_root)
        own = set(info["native_classes"])
        folder_stats["annotations_kept_foreign_class"] = sum(
            n for k, n in folder_stats["kept_by_native_class"].items() if k not in own
        )
        stats["per_class_folder"][class_key] = folder_stats
        print(
            f"  {class_key:12s} {folder_stats['annotations_total']:5d} annotations -> "
            f"{folder_stats['annotations_kept']:5d} kept "
            f"({folder_stats['annotations_kept_foreign_class']:5d} of them from OTHER "
            f"canonical classes, recovered by DEC-127)"
        )
        for file_name, width, height, class_id, bbox, native in kept:
            entry = pool.setdefault(
                file_name,
                {"width": width, "height": height, "boxes": [],
                 "source_folders": set()},
            )
            if entry["width"] != width or entry["height"] != height:
                print(
                    f"  WARNING: dimension mismatch for {file_name} "
                    f"({entry['width']}x{entry['height']} vs {width}x{height}) — "
                    f"keeping first-seen dims"
                )
            # DEC-127 dedup. Under the old per-folder filter each box could only
            # ever be collected once, so the pool appended blindly. With the
            # schema-wide filter the SAME annotation arrives from every folder
            # whose export contains that image -- measured: 7,071 of the 8,489
            # Chair boxes in tables/ are also in chairs/. Appending blindly would
            # write duplicate identical boxes into the label file.
            #
            # DEC-132: this comparison MUST be geometric. The original key
            # rounded the bbox to 1 decimal place, but `bbox` is COCO
            # [x, y, w, h] in ABSOLUTE PIXELS -- a 0.1 px tolerance, so it only
            # ever caught bit-identical copies. Open Images boxes the same
            # physical object slightly differently in each class-folder export
            # (measured: twins at IoU 0.98 offset by ~5 px), and 798 of those
            # survived into the v2 pool. IoU is scale-invariant, so it applies
            # directly to the pixel-space COCO box with no normalisation.
            # DEC-133: DEDUP_IOU alone (0.90) is too strict — duplicate
            # annotations of one object survive down to ~0.70. But a plain 0.70
            # cut destroys genuine neighbours, so the RAW NATIVE LABEL decides:
            #
            #   different native labels -> Open Images hierarchy parent/child/
            #     sibling collapsed onto one canonical class by DEC-127's
            #     mappings (Desk+Table 229x, Girl+Woman 161x, Man+Person 72x).
            #     One object under two names. Dedup at HIER_IOU.
            #   same native label -> only a duplicate if one box is NESTED in
            #     the other (IoMin >= NEST_IOMIN). Two people side by side at
            #     equal distance reach IoU 0.70 but only IoMin ~0.85, and are
            #     kept; a re-annotation of one object reaches IoMin ~1.0.
            #     Verified against 6 hand-checked cases: 5 duplicates caught,
            #     1 genuine pair spared.
            dup = False
            for c, b, nat in entry["boxes"]:
                if c != class_id:
                    continue
                v = coco_iou(b, bbox)
                if v >= DEDUP_IOU:
                    dup = True
                elif v >= HIER_IOU and nat != native:
                    dup = True
                elif v >= HIER_IOU and coco_iomin(b, bbox) >= NEST_IOMIN:
                    dup = True
                if dup:
                    break
            if dup:
                stats["boxes_deduplicated"] += 1
            else:
                entry["boxes"].append((class_id, bbox, native))
            entry["source_folders"].add(class_key)

    print(f"\n{len(pool)} unique images across all classes "
          f"({sum(len(e['boxes']) for e in pool.values())} total boxes)")

    bench_tally = drop_bench_on_furniture(pool)
    stats["bench_on_furniture"] = bench_tally
    if bench_tally["bench_boxes_dropped"]:
        print(
            f"\nDEC-115 Bench/furniture filter (IoU >= {BENCH_FURNITURE_IOU}): "
            f"{bench_tally['bench_boxes_dropped']} Bench box(es) dropped across "
            f"{bench_tally['images_affected']} image(s); "
            f"{bench_tally['images_emptied_of_bench']} image(s) lost their last Bench box."
        )

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

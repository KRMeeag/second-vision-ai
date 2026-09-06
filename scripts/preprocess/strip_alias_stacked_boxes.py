"""
Remove part-class boxes wrongly stacked on a composite-class box (DEC-109 cleanup).

WHY THIS EXISTS
---------------
DEC-098's cross-class alias rule switched itself off for any source that labels a
composite's parts separately. `dlsu_d_vehicle_type_detection` labels Vehicle,
Motorcycle AND Tricycle, so the guard was disabled there -- on the one source where
tricycles are dense. Vehicle/Motorcycle predictions boxing a *piece* of a tricycle
were therefore never tagged `dup_gt`, were auto-accepted, and write-back promoted
them into ground truth on top of the correct Tricycle box.

Measured damage: 2 such boxes before the review pass (DEC-098's own figure), 76 at
IoU >= 0.5 afterwards across 78 files.

DEC-109 fixes the rule going forward. This script repairs what the old rule already
wrote. It is NOT part of the pipeline and is safe to run once and forget.

WHAT IT REMOVES, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------
A box is removed only when ALL of:

  1. its class is a PART of a composite class (Vehicle/Motorcycle/Bicycle of Tricycle)
  2. at least ALIAS_CONTAINMENT of it lies inside a composite box in the same file
  3. it is ABSENT from a pre-review reference snapshot of the same source

Condition 3 is the safety catch, and it is the only one that works. An earlier
version tried "has no same-class twin" and a dry run exposed it as wrong: it flagged
454 boxes where only ~107 were ever added by review, because a genuine part box that
the source itself labelled is usually the only one of its class in that image and so
has no twin either. dlsu holds 576 genuine Vehicle/Motorcycle boxes overlapping a
Tricycle, 193 almost wholly inside one -- none of those may be touched, and only a
snapshot taken before the review pass can tell them apart.

Low-containment boxes are kept on purpose. A distant real vehicle merely clipping a
tricycle's box is exactly what DEC-098's author objected to suppressing, and that
objection still stands.

Usage
-----
    python3 scripts/preprocess/strip_alias_stacked_boxes.py --source roboflow_dlsu_d_vehicle_type_detection --dry-run
    python3 scripts/preprocess/strip_alias_stacked_boxes.py --source roboflow_dlsu_d_vehicle_type_detection
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.config_loader import get_canonical_names
from scripts.utils.file_utils import processed_dir

ALIAS_CONTAINMENT = 0.8

# --rule loose (DEC-110). A review-added box sitting inside a LOOSE author box of
# the same class: same object, two boxes. Bounded by area ratio because containment
# alone cannot separate that from a genuinely smaller object inside a larger one --
# measured on dlsu, 152 cases cluster at 2-5x with a valley at 5-10x and 33 sit
# above 10x, which are separate objects. The AUTHOR's box is always the one kept.
LOOSE_CONTAINMENT = 0.8
LOOSE_MAX_RATIO = 5.0
LOOSE_MAX_IOU = 0.5

# The reference snapshot predates DEC-100's promote+drop, so its ids are the old
# 16-class numbering. Compared by NAME, never by raw id.
LEGACY_NAMES = [
    "Person", "Vehicle", "Motorcycle", "Pole", "Animals", "Stairs", "Shelf", "Doors",
    "Chairs", "Tables", "Tricycle", "Potholes", "Trash Bins", "Elevator",
    "Pedestrian Lane", "Bicycle",
]
COMPOSITE_PARTS: dict[str, set[str]] = {"Tricycle": {"Vehicle", "Motorcycle", "Bicycle"}}


def to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def intersection(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def area_of(b) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def containment(inner, outer) -> float:
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return intersection(inner, outer) / area if area > 0 else 0.0


def iou(a, b) -> float:
    i = intersection(a, b)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def load_reference(path: Path) -> set[tuple]:
    """{(name, rounded box)} from a pre-review label file, using 16-class names."""
    out = set()
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) != 5:
            continue
        cid = int(f[0])
        name = LEGACY_NAMES[cid] if 0 <= cid < len(LEGACY_NAMES) else str(cid)
        out.add((name,) + tuple(round(float(v), 4) for v in f[1:]))
    return out


def clean_file(path: Path, names: list[str], reference: set[tuple],
               rule: str = "alias") -> tuple[list[str], list[tuple[str, float]]]:
    """Return (kept_lines, removed) for one label file, under `rule`."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) != 5:
            continue
        cid = int(f[0])
        cx, cy, w, h = (float(v) for v in f[1:])
        name = names[cid] if 0 <= cid < len(names) else str(cid)
        rows.append({"line": line, "cid": cid, "name": name, "box": to_xyxy(cx, cy, w, h),
                     "key": (name,) + tuple(round(float(v), 4) for v in f[1:])})

    kept, removed = [], []

    if rule == "loose":
        # Remove a REVIEW-ADDED box that duplicates an AUTHOR box of the same class
        # which is merely drawn loose around it. The author box is never touched.
        for r in rows:
            if r["key"] in reference:
                kept.append(r["line"])
                continue
            hosts = [
                o for o in rows
                if o["key"] in reference and o["name"] == r["name"]
                and containment(r["box"], o["box"]) >= LOOSE_CONTAINMENT
                and iou(r["box"], o["box"]) < LOOSE_MAX_IOU
                and area_of(o["box"]) / max(area_of(r["box"]), 1e-12) < LOOSE_MAX_RATIO
            ]
            if hosts:
                removed.append((r["name"], max(containment(r["box"], h["box"]) for h in hosts)))
            else:
                kept.append(r["line"])
        return kept, removed

    part_of = {p: c for c, parts in COMPOSITE_PARTS.items() for p in parts}
    for r in rows:
        composite = part_of.get(r["name"])
        if composite is None:
            kept.append(r["line"])
            continue
        hosts = [o["box"] for o in rows if o["name"] == composite]
        best = max((containment(r["box"], h) for h in hosts), default=0.0)
        if best < ALIAS_CONTAINMENT:
            kept.append(r["line"])
            continue
        # Condition 3: present in the pre-review snapshot => the source labelled it
        # itself => genuine => never remove. Only review-added boxes are eligible.
        if r["key"] in reference:
            kept.append(r["line"])
            continue
        removed.append((r["name"], best))
    return kept, removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="processed source key, e.g. roboflow_dlsu_d_vehicle_type_detection")
    ap.add_argument("--reference", required=True,
                    help="pre-review labels/ dir, e.g. dataset/backups/pre_class_drop_.../processed/<source>/labels")
    ap.add_argument("--rule", choices=("alias", "loose"), default="alias",
                    help="alias: part-class box stacked on a composite box (DEC-109). "
                         "loose: review-added box inside a loose same-class author box (DEC-110).")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args()

    names = get_canonical_names()
    base = processed_dir(args.source)
    targets = [d for d in (base / "labels", base / "labels_reviewed") if d.is_dir()]
    if not targets:
        raise SystemExit(f"No labels/ or labels_reviewed/ under {base}")

    ref_dir = Path(args.reference)
    if not ref_dir.is_dir():
        raise SystemExit(f"--reference is not a directory: {ref_dir}")
    print(f"rule: {args.rule}   reference (pre-review): {ref_dir}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for d in targets:
        files_changed = 0
        removed_total: list[tuple[str, float]] = []
        pending: list[tuple[Path, list[str]]] = []
        for p in sorted(d.glob("*.txt")):
            kept, removed = clean_file(p, names, load_reference(ref_dir / p.name), args.rule)
            if removed:
                files_changed += 1
                removed_total.extend(removed)
                pending.append((p, kept))

        by_class: dict[str, int] = {}
        for n, _ in removed_total:
            by_class[n] = by_class.get(n, 0) + 1
        print(f"\n{d.relative_to(base.parent.parent)}")
        print(f"  files affected : {files_changed}")
        print(f"  boxes removed  : {len(removed_total)}  {by_class or ''}")

        if args.dry_run or not pending:
            continue

        backup = d.parent / f"{d.name}_bak_{args.rule}_{stamp}"
        shutil.copytree(d, backup)
        print(f"  backed up to   : {backup}")
        for p, kept in pending:
            p.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        print(f"  rewrote {len(pending)} file(s)")

    if args.dry_run:
        print("\n--dry-run: nothing written.")


if __name__ == "__main__":
    main()

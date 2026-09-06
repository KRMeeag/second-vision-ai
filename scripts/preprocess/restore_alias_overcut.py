"""
Restore boxes the UNCAPPED alias rule removed but the capped one would have kept.

WHY THIS EXISTS
---------------
DEC-109's alias rule suppressed any part-class prediction sitting >= 0.8 inside a
composite box, with no bound on relative size. Measuring the 112 boxes it removed
from dlsu showed the cost: 70 sit below 2x the Tricycle's area (the model calling a
whole tricycle a "vehicle" -- correctly suppressed), but 10 sit ABOVE 10x, up to
472x. A box 472x smaller than the one containing it is not a part of that object; it
is a separate thing inside its bounding box -- typically a distant vehicle, which is
exactly what DEC-098 refused to discard and what the student named as the expensive
case to get wrong.

DEC-111 adds ALIAS_MAX_RATIO to the notebook so this cannot recur. This script
repairs the boxes the uncapped run already deleted.

WHAT IT RESTORES
----------------
A box is restored only when ALL of:

  1. it is present in the pre-alias-cleanup backup and absent from the live labels
  2. it is a part class (Vehicle/Motorcycle/Bicycle)
  3. it sits >= ALIAS_CONTAINMENT inside a composite (Tricycle) box in that file
  4. that composite box is >= ALIAS_MAX_RATIO times its area

Conditions 2-4 reconstruct exactly why the uncapped rule fired, so nothing removed
for a different reason -- notably DEC-110's loose-GT cleanup, which ran afterwards --
is touched. Restored boxes are appended; existing lines are never rewritten.

Usage
-----
    python3 scripts/preprocess/restore_alias_overcut.py --source roboflow_dlsu_d_vehicle_type_detection \\
        --backup-suffix bak_alias_20260905_171115 --dry-run
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
ALIAS_MAX_RATIO = 10.0
COMPOSITE_PARTS: dict[str, set[str]] = {"Tricycle": {"Vehicle", "Motorcycle", "Bicycle"}}


def to_xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def area_of(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def intersection(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def containment(inner, outer):
    a = area_of(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def parse(path: Path, names: list[str]):
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) != 5:
            continue
        cid = int(f[0])
        rows.append({
            "line": line,
            "name": names[cid] if 0 <= cid < len(names) else str(cid),
            "box": to_xyxy(*(float(v) for v in f[1:])),
            "key": (cid,) + tuple(round(float(v), 4) for v in f[1:]),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--backup-suffix", required=True,
                    help="e.g. bak_alias_20260905_171115 -> labels_<suffix>/ and labels_reviewed_<suffix>/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = get_canonical_names()
    base = processed_dir(args.source)
    part_of = {p: c for c, parts in COMPOSITE_PARTS.items() for p in parts}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dirname in ("labels", "labels_reviewed"):
        live = base / dirname
        bak = base / f"{dirname}_{args.backup_suffix}"
        if not live.is_dir() or not bak.is_dir():
            print(f"skip {dirname}: need both {live.name} and {bak.name}")
            continue

        pending: list[tuple[Path, list[str]]] = []
        restored: list[tuple[str, float]] = []
        for bp in sorted(bak.glob("*.txt")):
            lp = live / bp.name
            old = parse(bp, names)
            cur = parse(lp, names)
            cur_keys = {r["key"] for r in cur}
            tri = [r["box"] for r in old if r["name"] in COMPOSITE_PARTS]
            if not tri:
                continue
            add = []
            for r in old:
                if r["key"] in cur_keys or part_of.get(r["name"]) is None:
                    continue
                host = max(tri, key=lambda t: containment(r["box"], t))
                if containment(r["box"], host) < ALIAS_CONTAINMENT:
                    continue
                ratio = area_of(host) / max(area_of(r["box"]), 1e-12)
                if ratio < ALIAS_MAX_RATIO:
                    continue  # the capped rule would still suppress this one
                add.append(r["line"])
                restored.append((r["name"], ratio))
            if add:
                pending.append((lp, [c["line"] for c in cur] + add))

        by_class: dict[str, int] = {}
        for n, _ in restored:
            by_class[n] = by_class.get(n, 0) + 1
        print(f"\n{dirname}")
        print(f"  files affected  : {len(pending)}")
        print(f"  boxes restored  : {len(restored)}  {by_class or ''}")
        if restored:
            rs = sorted(r for _, r in restored)
            print(f"  area ratio      : min {rs[0]:.1f}x  median {rs[len(rs)//2]:.1f}x  max {rs[-1]:.1f}x")

        if args.dry_run or not pending:
            continue
        safety = live.parent / f"{dirname}_bak_prerestore_{stamp}"
        shutil.copytree(live, safety)
        print(f"  backed up to    : {safety}")
        for path, lines in pending:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  rewrote {len(pending)} file(s)")

    if args.dry_run:
        print("\n--dry-run: nothing written.")


if __name__ == "__main__":
    main()

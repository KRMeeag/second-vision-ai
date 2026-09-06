"""
Restore Stairs boxes that DEC-100 deleted from sources that are still ACTIVE.

WHY THIS EXISTS
---------------
DEC-100 dropped Stairs (16 -> 13 classes) and `drop_classes.py` deleted every
Stairs box from every label file. DEC-117 brought Stairs back at id 13, but only
from Open Images -- so the boxes this project had already collected AND reviewed
were left deleted.

Most of those sat in the three dedicated Roboflow stairs sources
(`escalator_stairs`, `stairs_i2yia`, `stair_gaptw`), which are benched on quality
grounds and must STAY benched -- the student's explicit instruction is that no
Roboflow stairs dataset feeds this class.

But two ACTIVE, general-purpose sources also carried Stairs incidentally:

    roboflow_revised_pedestrian_obstacle   338 images /  350 boxes
    roboflow_cv_project_hovyc               77 images /   87 boxes

Those are not "stairs datasets" -- they are sources already in the pool for other
classes that happen to contain staircases, and their boxes went through this
project's own review pass. That makes them higher-confidence than the unreviewed
Open Images pull, and they are pure gain: the images are already in the pool, so
this adds supervision without adding a single new image.

WHAT IT DOES
------------
For each label file in the pre-drop backup that carries a Stairs box:

  1. extract only the legacy-id-5 (Stairs) lines -- every other line is ignored,
     so the backup's stale 16-class ids for other classes can never leak back in
  2. remap to the current Stairs id, read from config (never hardcoded)
  3. append to BOTH dataset/processed/<source>/labels/<f> and labels_reviewed/<f>

Step 3's second half is not optional. All 415 files also exist in
labels_reviewed/, and `promote_reviews.py` copies labels_reviewed/ OVER labels/.
Patching only labels/ would leave a live landmine: the next promote run would
copy the unpatched reviewed file back and silently delete every box restored
here, with no error.

Boxes are deduplicated on (class_id, cx, cy, w, h) rounded to 6dp, so re-running
cannot double-add.

Usage
-----
    python3 scripts/preprocess/restore_stairs_from_predrop.py --dry-run
    python3 scripts/preprocess/restore_stairs_from_predrop.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.config_loader import get_class_id
from scripts.utils.file_utils import processed_dir

# The backup predates DEC-100, so its ids are the old 16-class numbering.
# Asserted against index 5 below rather than trusted -- a silent off-by-one here
# would restore the wrong class entirely.
LEGACY_NAMES = [
    "Person", "Vehicle", "Motorcycle", "Pole", "Animals", "Stairs", "Shelf", "Doors",
    "Chairs", "Tables", "Tricycle", "Potholes", "Trash Bins", "Elevator",
    "Pedestrian Lane", "Bicycle",
]
LEGACY_STAIRS_ID = 5

BACKUP = Path("dataset/backups/pre_class_drop_20260904_023304/processed")

# Deliberately NOT the three dedicated Roboflow stairs sources. Those are benched
# (DEC-082/100) and the student ruled them out explicitly; DEC-117's Stairs comes
# from Open Images. These two are active general sources, already in the pool.
SOURCES = [
    "roboflow_revised_pedestrian_obstacle",
    "roboflow_cv_project_hovyc",
]


def stairs_lines(path: Path, new_id: int) -> list[str]:
    """Legacy Stairs lines from a pre-drop label file, remapped to the current id."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) != 5 or int(f[0]) != LEGACY_STAIRS_ID:
            continue
        out.append(f"{new_id} {f[1]} {f[2]} {f[3]} {f[4]}")
    return out


def key_of(line: str) -> tuple:
    f = line.split()
    return (int(f[0]),) + tuple(round(float(v), 6) for v in f[1:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = ap.parse_args()

    assert LEGACY_NAMES[LEGACY_STAIRS_ID] == "Stairs", "legacy id map is wrong"
    new_id = get_class_id("Stairs")
    print(f"legacy Stairs id {LEGACY_STAIRS_ID} -> current id {new_id}")

    if not BACKUP.is_dir():
        raise SystemExit(f"pre-drop backup not found: {BACKUP}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grand = {"files": 0, "boxes": 0, "skipped_dup": 0}

    for source in SOURCES:
        bl = BACKUP / source / "labels"
        if not bl.is_dir():
            print(f"skip {source}: no backup labels/")
            continue
        base = processed_dir(source)

        pending: dict[Path, list[str]] = {}
        files = boxes = dup = 0
        unique_boxes = 0
        for bp in sorted(bl.glob("*.txt")):
            add = stairs_lines(bp, new_id)
            if not add:
                continue
            touched = False
            counted_unique = False
            for dirname in ("labels", "labels_reviewed"):
                target = base / dirname / bp.name
                if not target.is_file():
                    continue
                cur = [l for l in target.read_text(encoding="utf-8").splitlines() if l.strip()]
                have = {key_of(l) for l in cur if len(l.split()) == 5}
                new = [l for l in add if key_of(l) not in have]
                dup += len(add) - len(new)
                if not new:
                    continue
                pending[target] = cur + new
                boxes += len(new)
                if not counted_unique:
                    unique_boxes += len(new)   # count a box once, not once per directory
                    counted_unique = True
                touched = True
            if touched:
                files += 1

        print(f"\n{source}")
        print(f"  images patched : {files}")
        print(f"  Stairs boxes   : {unique_boxes} unique "
              f"({boxes} line-writes across labels/ + labels_reviewed/, "
              f"{dup} skipped as already present)")
        grand["files"] += files
        grand["boxes"] += unique_boxes
        grand["skipped_dup"] += dup

        if args.dry_run or not pending:
            continue

        for dirname in ("labels", "labels_reviewed"):
            live = base / dirname
            if live.is_dir():
                bak = live.parent / f"{dirname}_bak_stairsrestore_{stamp}"
                shutil.copytree(live, bak)
                print(f"  backed up      : {bak.name}")

        for path, lines in pending.items():
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {len(pending)} file(s) across labels/ + labels_reviewed/")

    print(f"\nTOTAL: {grand['boxes']} unique Stairs box(es) restored across "
          f"{grand['files']} image(s).")
    if args.dry_run:
        print("--dry-run: nothing written.")


if __name__ == "__main__":
    main()

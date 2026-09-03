"""
scripts/preprocess/apply_dedup_keeplist.py
──────────────────────────────────────────
Carries a completed dedup decision across a Roboflow RE-EXPORT, so forking a
project to escape its baked-in augmentation does not throw away the GPU dedup
run (DEC-089, ~8h on RunPod) or the manual false-positive review (DEC-090).

THE JOIN KEY IS THE BASE NAME. Roboflow exports as
`<original>_<ext>.rf.<hash>.<ext>`. Generating a new version changes the hash
on every file, so filenames cannot be matched across exports -- but `<original>`
is the name the image was UPLOADED under, which a re-generate does not touch.
roboflow_base_name() therefore yields a stable identity for the same photo in
both the old pool and the new one.

WHAT THIS IS AND IS NOT. It transfers a decision already made; it does not
re-verify it. The re-exported pixels differ from the ones dedup actually scored
(no augmentation, and possibly no resize), so:
  * Near-duplicate relationships between DISTINCT photos carry over soundly --
    "photo A duplicates photo B" is a statement about content, and content is
    what survives the re-export.
  * Exact (byte-identical) duplicates do NOT carry over -- different bytes.
    Run scripts/preprocess/dedup_extend_exact.py afterwards; it is CPU-only,
    filehash-based and cheap, and was built for exactly this situation.
A base is dropped only when EVERY file carrying it was flagged as a duplicate
AND none of them was a group's kept representative -- one surviving variant
keeps the whole base.

REVERSIBLE: files are moved to `images_dedup_dropped/` + `labels_dedup_dropped/`,
never deleted. --revert restores them.

Usage:
    python3 scripts/preprocess/apply_dedup_keeplist.py \\
        --source roboflow_revised_pedestrian_obstacle \\
        --keeplist dataset/reports/revised_pedestrian_obstacle_dedup_keeplist.json \\
        --dry-run
    python3 scripts/preprocess/apply_dedup_keeplist.py --source ... --keeplist ...
    python3 scripts/preprocess/apply_dedup_keeplist.py --source ... --revert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.file_utils import (  # noqa: E402
    ensure_dir, processed_dir, roboflow_base_name,
)

IMAGES_DROPPED = "images_dedup_dropped"
LABELS_DROPPED = "labels_dedup_dropped"


def load_keeplist(path: Path) -> tuple[set[str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload["keep"]), payload


def plan(source: str, keep_bases: set[str]) -> dict[str, Any]:
    """Which re-exported files match the keep-list. Pure -- touches nothing."""
    images_dir = processed_dir(source) / "images"
    keep: list[Path] = []
    move: list[Path] = []
    seen_bases: set[str] = set()
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.name.startswith("."):
            continue
        base = roboflow_base_name(image_path.stem)
        seen_bases.add(base)
        (keep if base in keep_bases else move).append(image_path)

    # A keep-list base that the re-export does not contain is the signal that the
    # join broke -- e.g. Roboflow renamed on fork, or the class filter differs
    # from the run the keep-list was built against. Surfaced, never silently
    # ignored: a near-total miss means the keep-list is meaningless here.
    return {
        "source": source,
        "files_seen": len(keep) + len(move),
        "bases_seen": len(seen_bases),
        "keep": keep,
        "move": move,
        "matched_bases": len(seen_bases & keep_bases),
        "keeplist_bases_missing_from_export": sorted(keep_bases - seen_bases),
    }


def apply_plan(result: dict[str, Any]) -> dict[str, int]:
    source = result["source"]
    images_dropped = processed_dir(source) / IMAGES_DROPPED
    labels_dropped = processed_dir(source) / LABELS_DROPPED
    labels_dir = processed_dir(source) / "labels"
    ensure_dir(images_dropped)
    ensure_dir(labels_dropped)

    moved_images = moved_labels = 0
    for image_path in result["move"]:
        image_path.rename(images_dropped / image_path.name)
        moved_images += 1
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            label_path.rename(labels_dropped / label_path.name)
            moved_labels += 1
    return {"moved_images": moved_images, "moved_labels": moved_labels}


def revert(source: str) -> dict[str, int]:
    restored_images = restored_labels = 0
    for sub, dest in ((IMAGES_DROPPED, "images"), (LABELS_DROPPED, "labels")):
        d = processed_dir(source) / sub
        if not d.is_dir():
            continue
        for path in list(d.iterdir()):
            if path.is_file():
                path.rename(processed_dir(source) / dest / path.name)
                if dest == "images":
                    restored_images += 1
                else:
                    restored_labels += 1
        d.rmdir()
    return {"restored_images": restored_images, "restored_labels": restored_labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Processed source key to filter.")
    parser.add_argument("--keeplist", help="Path to the keep-list JSON (required unless --revert).")
    parser.add_argument("--dry-run", action="store_true", help="Report only; change nothing.")
    parser.add_argument("--revert", action="store_true", help="Restore everything previously dropped.")
    args = parser.parse_args()

    print("=" * 72)
    print("apply_dedup_keeplist.py — carry a dedup decision across a re-export")
    print("=" * 72)

    if args.revert:
        result = revert(args.source)
        print(f"  restored {result['restored_images']} images / {result['restored_labels']} labels")
        return

    if not args.keeplist:
        parser.error("--keeplist is required unless --revert is given")

    keep_bases, payload = load_keeplist(Path(args.keeplist))
    print(f"  keep-list: {len(keep_bases)} bases (built from {payload.get('bases_in_old_pool')} in the old pool)")

    result = plan(args.source, keep_bases)
    matched = result["matched_bases"]
    coverage = 100.0 * matched / max(len(keep_bases), 1)
    print(f"  re-export: {result['files_seen']} files / {result['bases_seen']} bases")
    print(f"  keep-list bases found in re-export: {matched}/{len(keep_bases)} ({coverage:.1f}%)")
    print(f"  -> keep {len(result['keep'])} files, drop {len(result['move'])} files")

    if coverage < 50.0:
        print(
            "\n  ABORT: fewer than half the keep-list's bases appear in the re-export. "
            "The base-name join has broken (Roboflow renamed on fork, or a different "
            "class filter was applied) — applying this would delete real data on a bad "
            "match. Nothing was moved."
        )
        sys.exit(1)
    if result["keeplist_bases_missing_from_export"]:
        missing = result["keeplist_bases_missing_from_export"]
        print(f"  NOTE: {len(missing)} keep-list base(s) absent from the re-export, e.g. {missing[:3]}")

    if args.dry_run:
        print("\n--dry-run: nothing moved.")
        return

    stats = apply_plan(result)
    print(f"\n  moved aside: {stats['moved_images']} images / {stats['moved_labels']} labels")
    print(
        "\nNOTE: dataset/processed/ changed — re-run cap_per_class.py and merge.py. "
        "Then run scripts/preprocess/dedup_extend_exact.py for this source: the "
        "re-exported bytes were never checked for EXACT duplicates. Files were MOVED, "
        "not deleted; --revert undoes this."
    )


if __name__ == "__main__":
    main()

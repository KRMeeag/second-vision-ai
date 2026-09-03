"""
scripts/preprocess/build_dedup_keeplist.py
──────────────────────────────────────────
Freezes a source's dedup verdict into a base-name keep-list, so the decision
survives a Roboflow re-export (fork with augmentation off, or a re-pin to a
cleaner version) that changes every `.rf.<hash>` filename.

Run this BEFORE re-acquiring — it reads the CURRENT processed pool, which the
re-acquisition overwrites. Feed the output to apply_dedup_keeplist.py after
the new export has been converted.

THE RULE. A base is dropped only when every file carrying it was flagged as a
duplicate AND none of them was a dedup group's kept representative. One
surviving variant keeps the whole base. Near-duplicate entries listed in
near_duplicate_false_positives.json (the manual review, DEC-090) are excluded
from the flag set, exactly as hide_duplicates and the review notebook treat
them.

See apply_dedup_keeplist.py's docstring for why base names are the correct
join key across exports, and for what this does and does not transfer.

Usage:
    python3 scripts/preprocess/build_dedup_keeplist.py --source roboflow_door_detection_zqt59
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.file_utils import processed_dir, reports_dir, roboflow_base_name  # noqa: E402


def base_of(merged_filename: str) -> str:
    """`<source>__<roboflow name>` -> the stable base name."""
    _source, _, original = merged_filename.partition("__")
    return roboflow_base_name(Path(original).stem)


def build(source: str) -> dict:
    prefix = f"{source}__"
    report = json.loads((reports_dir() / "dedup_report.json").read_text(encoding="utf-8"))

    false_positive_path = reports_dir() / "near_duplicate_false_positives.json"
    false_positives: set[str] = set()
    if false_positive_path.is_file():
        false_positives = {
            entry["duplicate"]
            for entry in json.loads(false_positive_path.read_text(encoding="utf-8"))
        }

    kept_files: set[str] = set()
    duplicate_files: set[str] = set()

    for group in report.get("exact_duplicates", []):
        if group["kept"].startswith(prefix):
            kept_files.add(group["kept"])
        for name in group["duplicates"]:
            if name.startswith(prefix):
                duplicate_files.add(name)

    for group in report.get("near_duplicates", []):
        if group["kept"].startswith(prefix):
            kept_files.add(group["kept"])
        for entry in group["duplicates"]:
            name = entry["filename"]
            # Manually vindicated near-duplicates are not duplicates.
            if name.startswith(prefix) and name not in false_positives:
                duplicate_files.add(name)

    duplicate_bases = {base_of(name) for name in duplicate_files}
    kept_bases = {base_of(name) for name in kept_files}

    # Every base the source currently holds, including anything moved aside by
    # deaugment_sources.py -- an aside file still represents a real base.
    pool_bases: set[str] = set()
    for sub in ("images", "images_augmented_aside", "images_dedup_dropped"):
        directory = processed_dir(source) / sub
        if directory.is_dir():
            pool_bases |= {
                roboflow_base_name(path.stem)
                for path in directory.iterdir()
                if path.is_file() and not path.name.startswith(".")
            }

    drop = {b for b in pool_bases if b in duplicate_bases and b not in kept_bases}
    keep = pool_bases - drop

    return {
        "source": source,
        "generated": date.today().isoformat(),
        "basis": (
            "dedup_report.json minus near_duplicate_false_positives.json, collapsed to "
            "Roboflow base names so the verdict survives a re-export with new .rf hashes"
        ),
        "bases_in_old_pool": len(pool_bases),
        "bases_flagged_duplicate": len(duplicate_bases & pool_bases),
        "bases_kept_representative": len(kept_bases & pool_bases),
        "bases_dropped_as_duplicate": len(drop),
        "bases_to_keep": len(keep),
        "keep": sorted(keep),
        "drop": sorted(drop),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Processed source key.")
    parser.add_argument("--out", help="Output path (default: dataset/reports/<source>_dedup_keeplist.json).")
    args = parser.parse_args()

    payload = build(args.source)
    out_path = Path(args.out) if args.out else reports_dir() / f"{args.source}_dedup_keeplist.json"
    out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print("=" * 72)
    print(f"dedup keep-list — {args.source}")
    print("=" * 72)
    for field in (
        "bases_in_old_pool", "bases_flagged_duplicate",
        "bases_kept_representative", "bases_dropped_as_duplicate", "bases_to_keep",
    ):
        print(f"  {field:32s} {payload[field]:6d}")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()

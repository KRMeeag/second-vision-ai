"""
scripts/preprocess/deaugment_sources.py
─────────────────────────────────────────
Collapses Roboflow-generated augmentation in dataset/processed/<source>/:
keeps exactly ONE file per original source photo, moves every other
augmented variant aside (never deletes).

Why this exists (found 2026-08-26, docs/OPEN_QUESTIONS.md #17): most
Roboflow exports in this project ship pre-augmented. Roboflow's export
naming is `<original>_<ext>.rf.<hash>.<ext>`, so every augmented copy of
one photo shares a base name — e.g. base image `100` in
revised_pedestrian_obstacle exists as 7 files, two of which carry boxes
mirrored exactly around x=0.5 (a horizontal flip). Whole active pool:
97,933 files from only 75,679 distinct photos.

Why base-name matching rather than the existing GPU dedup: measured
recall of dedup.py's embedding check (threshold 0.2,
mobilenet-v2-imagenet-torch) on real augmented siblings is 46.6% — of
3,539 base-sibling pairs in revised_pedestrian_obstacle, it grouped
1,648. MobileNet features are not flip-invariant, so a flipped copy is
genuinely distant in that space; no threshold catches flips without
also flagging masses of unrelated images. The filename gives the same
answer at 100% precision and recall, instantly. This is a complement to
dedup.py, not a replacement — genuine (non-augmented) near-duplicates
still need the embedding check.

The three harms this addresses, in severity order:
  1. Train/val/test leakage — augmented siblings of one photo landing in
     different splits inflates apparent performance. split.py's
     duplicate-aware grouping only sees what the embedding threshold
     flagged, so it misses roughly half of them.
  2. Inflated counts corrupting DEC-042's floor/cap policy — a class
     "clearing" its 1,500 floor on augmented copies does not have 1,500
     distinct scenes.
  3. Multiplied review effort in fiftyone_review_processed.ipynb —
     door_detection_zqt59's 4,493 files show only 724 distinct photos.

WHICH VARIANT IS KEPT: the alphabetically-first filename per base group.
Deterministic and reproducible, but arbitrary among the variants — the
kept file may itself be a transformed copy (flipped/brightened) rather
than the pristine original. That is acceptable for training data (a
correctly-transformed image with correctly-transformed boxes is valid
ground truth), and there is no way to identify the true original from
Roboflow's naming. Choosing the *original* instead would require
re-exporting from Roboflow with augmentation disabled — option A in
docs/OPEN_QUESTIONS.md #17, deliberately not what this script does.

SCOPE, measured per-source 2026-08-26 (DEC-095). Two numbers place every
source: `rf%` (share of files carrying the `.rf.<hash>` suffix) and
`ratio` (files / distinct bases).
  * rf% 0, ratio 1.00   — not a Roboflow export (crowdhuman, exdark,
    open_images, dataset_ninja_*). This script is a no-op by construction.
  * rf% 100, ratio 1.00 — Roboflow export with augmentation OFF
    (crosswalk_detector_lz3hc, cv_project_hovyc, pothole_voxrl). No-op
    correctly; there is nothing to collapse.
  * rf% 100, ratio > 1  — augmentation ON. The 9 sources this acts on.
Roboflow augments ONLY the train split, verified against
dataset/raw/roboflow_projects/*/{train,valid,test}: trashcan_detection_pihfn
is train 172 bases all at 3x, valid 22 and test 21 all at 1x. So size-1
groups are val/test images, NOT files this script failed to match.

KNOWN LIMITATIONS — both bake the transform into the base name itself, so
no filename rule can see them (docs/OPEN_QUESTIONS.md #18):
  1. Pre-upload augmentation, e.g.
     `IMG_20210920_114223_output-jpg_flip_png.rf.<hash>.jpg` — `_flip` is
     part of the UPLOADED name, so it reads as its own base image.
     Measured: escalator_stairs 158/2663 bases, revised_pedestrian_obstacle
     56/2100, stairs_i2yia 29/1372.
  2. Export -> re-upload -> re-export cycles. door_detection_zqt59's valid
     and test splits carry group sizes 5 and 7, impossible under train-only
     augmentation, and 72.4% of its bases end in `_JPG` — pass 2 treated
     pass 1's augmented output as fresh originals.
Neither causes split leakage (split.py groups siblings AND cross-source
duplicates), but both mean those sources' distinct-photo counts are
overstated relative to DEC-042's floor.

NOT IN SCOPE: the same photo shipped by two DIFFERENT sources — this
script works within one source directory and cannot see across them.
That is handled by cross_source_duplicate_groups() in scripts/build/split.py
(173 base names / 346 files in the current merged pool).

REVERSIBLE by design: files are moved to a sibling directory, never
deleted. `--revert` moves them all back.

Usage:
    python3 scripts/preprocess/deaugment_sources.py --dry-run
    python3 scripts/preprocess/deaugment_sources.py
    python3 scripts/preprocess/deaugment_sources.py --source roboflow_wtf_dwvgm
    python3 scripts/preprocess/deaugment_sources.py --revert
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Leaf script, not a shared library module — needs the repo root on
# sys.path to import sibling utils regardless of how it's invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.config_loader import get_inactive_processed_source_keys  # noqa: E402
from scripts.utils.file_utils import (  # noqa: E402
    discover_processed_sources, ensure_dir, processed_dir, raw_dir, reports_dir, roboflow_base_name,
)

IMAGES_ASIDE = "images_augmented_aside"
LABELS_ASIDE = "labels_augmented_aside"


def raw_split_provenance(source: str) -> dict[str, str]:
    """
    Map each file stem to the split Roboflow exported it into.

    Roboflow's own export layout is the ground truth this script's `provenance`
    strategy rests on: dataset/raw/roboflow_projects/<project>/{train,valid,test}/.
    """
    project = source[len("roboflow_"):] if source.startswith("roboflow_") else source
    root = raw_dir() / "roboflow_projects" / project
    provenance: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        images = root / split / "images"
        if images.is_dir():
            for path in images.iterdir():
                if path.is_file() and not path.name.startswith("."):
                    provenance[path.stem] = split
    return provenance


def augmentation_multiplier(by_base: dict[str, list[Path]], provenance: dict[str, str]) -> int:
    """
    How many copies Roboflow made per train photo, measured rather than assumed.

    Taken as the most common size among groups lying ENTIRELY in train — val/test
    groups are excluded because Roboflow never augments them, so including them
    would bias the estimate toward 1. Returns 1 when export-time augmentation was
    off, which makes collapse_by_provenance() a no-op for that source (correct:
    there is no Roboflow augmentation to collapse).
    """
    train_sizes: Counter[int] = Counter(
        len(paths) for paths in by_base.values()
        if paths and all(provenance.get(p.stem) == "train" for p in paths)
    )
    return max(train_sizes, key=lambda size: train_sizes[size]) if train_sizes else 1


def plan_source(source: str, strategy: str = "filename") -> dict[str, Any]:
    """
    Which files this source would keep vs. move aside. Pure — touches nothing.

    strategy="filename"   — every base group collapses to one file. Maximises
        duplicate removal; accepts that some groups are filename COLLISIONS
        (two contributors uploading a different photo both named IMG_7497.jpg)
        rather than augmentation, so a genuine photo is sometimes moved aside.
    strategy="provenance" — collapse only groups PROVEN to be Roboflow
        augmentation: entirely inside train, and exactly the size this source's
        measured multiplier produces. Everything else is left in place. Costs
        some duplicate removal, never discards a distinct photo on a guess.

    Neither strategy deletes; both are reversible with --revert.
    """
    images_dir = processed_dir(source) / "images"
    by_base: dict[str, list[Path]] = defaultdict(list)
    for image_path in images_dir.iterdir():
        if image_path.is_file() and not image_path.name.startswith("."):
            by_base[roboflow_base_name(image_path.stem)].append(image_path)

    provenance = raw_split_provenance(source) if strategy == "provenance" else {}
    multiplier = augmentation_multiplier(by_base, provenance) if strategy == "provenance" else 0

    keep: list[Path] = []
    move: list[Path] = []
    collapsed_groups = 0
    for _base, paths in by_base.items():
        # Deterministic pick — see WHICH VARIANT IS KEPT in the module docstring.
        ordered = sorted(paths, key=lambda p: p.name)
        if strategy == "provenance":
            provable = (
                len(ordered) == multiplier
                and multiplier > 1
                and all(provenance.get(p.stem) == "train" for p in ordered)
            )
            if not provable:
                keep.extend(ordered)   # unproven -> leave the whole group alone
                continue
            collapsed_groups += 1
        keep.append(ordered[0])
        move.extend(ordered[1:])

    return {
        "source": source,
        "strategy": strategy,
        "multiplier": multiplier if strategy == "provenance" else None,
        "files_before": sum(len(v) for v in by_base.values()),
        "distinct_bases": len(by_base),
        "collapsed_groups": collapsed_groups if strategy == "provenance" else None,
        "keep": keep,
        "move": move,
    }


def apply_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Move this source's redundant variants (image + its label) aside."""
    source = plan["source"]
    images_aside = processed_dir(source) / IMAGES_ASIDE
    labels_aside = processed_dir(source) / LABELS_ASIDE
    labels_dir = processed_dir(source) / "labels"

    ensure_dir(images_aside)
    ensure_dir(labels_aside)

    moved_images = 0
    moved_labels = 0
    missing_labels: list[str] = []
    for image_path in plan["move"]:
        image_path.rename(images_aside / image_path.name)
        moved_images += 1
        # Label travels with its image — leaving it behind would make the label
        # an orphan that box_audit.py counts but no image backs.
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            label_path.rename(labels_aside / label_path.name)
            moved_labels += 1
        else:
            missing_labels.append(image_path.name)

    return {
        "moved_images": moved_images,
        "moved_labels": moved_labels,
        "images_without_label": missing_labels,
    }


def revert_source(source: str) -> dict[str, int]:
    """Move everything back out of the aside directories."""
    images_aside = processed_dir(source) / IMAGES_ASIDE
    labels_aside = processed_dir(source) / LABELS_ASIDE
    restored_images = restored_labels = 0

    if images_aside.is_dir():
        for path in list(images_aside.iterdir()):
            if path.is_file():
                path.rename(processed_dir(source) / "images" / path.name)
                restored_images += 1
        images_aside.rmdir()
    if labels_aside.is_dir():
        for path in list(labels_aside.iterdir()):
            if path.is_file():
                path.rename(processed_dir(source) / "labels" / path.name)
                restored_labels += 1
        labels_aside.rmdir()

    return {"restored_images": restored_images, "restored_labels": restored_labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report what would move; change nothing.")
    parser.add_argument(
        "--source", nargs="+",
        help="Only these source keys (default: every active source).",
    )
    parser.add_argument(
        "--strategy", choices=("filename", "provenance"), default="filename",
        help=(
            "filename (default): collapse every base group — maximum duplicate removal, "
            "occasionally discards a distinct photo that shares a filename. "
            "provenance: collapse only groups proven to be Roboflow augmentation "
            "(entirely in train, exactly the measured multiplier) — never discards a "
            "distinct photo, removes fewer duplicates."
        ),
    )
    parser.add_argument("--revert", action="store_true", help="Move every aside file back and remove the aside dirs.")
    parser.add_argument(
        "--include-inactive", action="store_true",
        help="Also process benched/failed sources (default: skip them, same as cap_per_class.py).",
    )
    args = parser.parse_args()

    sources = discover_processed_sources()
    if not args.include_inactive:
        inactive = get_inactive_processed_source_keys()
        sources = [s for s in sources if s not in inactive]
    if args.source:
        unknown = [s for s in args.source if s not in sources]
        if unknown:
            parser.error(
                f"not active processed source(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(sources))}"
            )
        sources = list(args.source)

    print("=" * 70)
    print("deaugment_sources.py — collapse Roboflow-generated augmentation")
    print("=" * 70)

    if args.revert:
        total = {"restored_images": 0, "restored_labels": 0}
        for source in sorted(sources):
            result = revert_source(source)
            if result["restored_images"] or result["restored_labels"]:
                print(f"  {source:45s} restored {result['restored_images']} images / {result['restored_labels']} labels")
            total["restored_images"] += result["restored_images"]
            total["restored_labels"] += result["restored_labels"]
        print(f"\nReverted: {total['restored_images']} images, {total['restored_labels']} labels.")
        return

    report: dict[str, Any] = {
        "sources": {},
        "strategy": args.strategy,
        "kept_variant_rule": "alphabetically-first filename per base",
    }
    total_before = total_after = total_moved = 0
    print(f"strategy: {args.strategy}\n")

    for source in sorted(sources):
        plan = plan_source(source, strategy=args.strategy)
        if not plan["move"]:
            print(f"  {source:45s} nothing collapsible under this strategy")
            continue

        files_after = plan["files_before"] - len(plan["move"])
        detail = f" [M={plan['multiplier']}, {plan['collapsed_groups']} groups]" if args.strategy == "provenance" else ""
        print(
            f"  {source:45s} {plan['files_before']:6d} -> {files_after:6d} "
            f"(aside {len(plan['move'])}){detail}"
        )
        entry: dict[str, Any] = {
            "files_before": plan["files_before"],
            "distinct_bases": plan["distinct_bases"],
            "files_after": files_after,
            "multiplier": plan["multiplier"],
            "collapsed_groups": plan["collapsed_groups"],
            "moved_aside": len(plan["move"]),
        }
        if not args.dry_run:
            entry.update(apply_plan(plan))
            if entry["images_without_label"]:
                print(f"      WARNING: {len(entry['images_without_label'])} moved image(s) had no label file")
        report["sources"][source] = entry

        total_before += plan["files_before"]
        total_after += files_after
        total_moved += len(plan["move"])

    report["totals"] = {
        "files_before": total_before,
        "files_after": total_after,
        "moved_aside": total_moved,
    }
    print(f"\n  {'TOTAL (augmented sources only)':45s} {total_before:6d} -> {total_after:6d} (aside {total_moved})")

    if args.dry_run:
        print("\n--dry-run: nothing moved, no report written.")
        return

    ensure_dir(reports_dir())
    report_path = reports_dir() / "deaugment_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    print(
        "\nNOTE: dataset/processed/ has changed — cap_per_class.py and merge.py must be re-run "
        "before split.py. Files were MOVED, not deleted; --revert undoes this completely."
    )


if __name__ == "__main__":
    main()

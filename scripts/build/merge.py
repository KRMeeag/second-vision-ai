"""
scripts/build/merge.py
─────────────────────────
Stage 5.6 Merge: pools Stage 5.4's capped selection into dataset/merged/,
source-prefixed filenames via file_utils.prefixed_filename() so no two
sources' images can collide once pooled.

Scope correction worth recording here rather than silently: this
session's handoff table describes merge.py tersely as "pool every
dataset/processed/<source>/ into dataset/merged/," which read as
"everything, uncapped" when cap_per_class.py (Stage 5.4, DEC-058) was
built — that script was deliberately left decision/report-only on the
assumption merge.py wouldn't consume its output. But README.md's own
directory table labels dataset/merged/ as **"Post-cap, post-merge,
pre-split"** — an explicit architectural statement that capping happens
before merging, missed when DEC-058 was written. This script follows the
README: it reads dataset/reports/cap_report.json (Stage 5.4's per-class
selected/excluded decision) and merges the UNION of every class's
"selected" images, not the raw uncapped dataset/processed/ pool.

Stage 5.5 (model-assisted curation + human correction) sits between 5.4
and 5.6 in the intended pipeline, and dataset/curated/ is specifically
earmarked for its re-imported corrections (README.md, PLAN.md) — but that
human-review step can't complete unattended (docs/OPEN_QUESTIONS.md #5,
CVAT/Label Studio still undecided). So this run merges the *capped,
pre-correction* pool — an accepted, documented interim state, not the
fully-realized pipeline. Re-running merge.py after Stage 5.5 actually
completes is expected, not a bug in this run.

An image can be selected by more than one class's cap decision (e.g. an
open_images photo with both a Chair and a Table box, each counted
separately in cap_report.json). Once merged, that image's FULL original
label file travels with it — every valid canonical box, not just the
one(s) that caused its selection. Dropping a correct, already-existing
box because its class's own cap decision happened not to pick this
particular image would throw away real ground truth for no benefit (same
tradeoff DEC-052 already made for Open Images' cross-folder merge). One
consequence: a class's real post-merge count can exceed its own
cap_report.json figure slightly — this script recomputes true post-merge
per-class counts from the merged labels themselves, which supersede
cap_report.json's pre-merge estimates.

Usage:
    python3 scripts/build/merge.py
    python3 scripts/build/merge.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

# Leaf script, not a shared library module — needs the repo root on
# sys.path to import sibling utils regardless of how it's invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.config_loader import get_canonical_names  # noqa: E402
from scripts.utils.file_utils import (  # noqa: E402
    build_stem_index, ensure_dir, merged_dir, prefixed_filename, processed_dir, reports_dir, safe_copy,
)


def load_selected_union(cap_report_path: Path) -> set[tuple[str, str]]:
    """Union of (source, filename) across every one of the 16 classes' capped selections."""
    cap_report = json.loads(cap_report_path.read_text(encoding="utf-8"))
    union: set[tuple[str, str]] = set()
    for entry in cap_report["classes"].values():
        for item in entry["selected"]:
            source, filename = item.split("/", 1)
            union.add((source, filename))
    return union


def run(dry_run: bool = False) -> dict[str, Any]:
    cap_report_path = reports_dir() / "cap_report.json"
    if not cap_report_path.is_file():
        raise FileNotFoundError(
            f"{cap_report_path} not found — run scripts/preprocess/cap_per_class.py (Stage 5.4) first."
        )

    union = load_selected_union(cap_report_path)
    print(f"{len(union)} unique (source, filename) pairs selected by Stage 5.4's cap decision.")

    per_source_counts: dict[str, int] = {}
    missing_images: list[str] = []
    missing_labels: list[str] = []

    out_images_dir = merged_dir() / "images"
    out_labels_dir = merged_dir() / "labels"
    if not dry_run:
        # Clear before writing — safe_copy(overwrite=True) below only overwrites
        # matching filenames, it doesn't remove files that are no longer part of
        # this run's selection. Without this, re-running merge.py after cap_per_class.py
        # produces a different selection (e.g. after a code fix, DEC-06x) leaves the
        # previous run's now-deselected files behind as silent orphans — the exact
        # "stale export" bug DEC-050 already found and fixed for acquire_openimages.py,
        # found again here the same way: by checking real file counts on disk, not
        # trusting the run's own tally.
        if out_images_dir.is_dir():
            shutil.rmtree(out_images_dir)
        if out_labels_dir.is_dir():
            shutil.rmtree(out_labels_dir)
        ensure_dir(out_images_dir)
        ensure_dir(out_labels_dir)

    image_index = {source: build_stem_index(source) for source in {source for source, _ in union}}

    merged_count = 0
    for source, filename in sorted(union):
        label_path = processed_dir(source) / "labels" / f"{filename}.txt"
        img_path = image_index.get(source, {}).get(filename)

        if img_path is None:
            missing_images.append(f"{source}/{filename}")
            continue
        if not label_path.is_file():
            missing_labels.append(f"{source}/{filename}")
            continue

        # Counted here, not earlier — this must mean "actually merged," not "attempted";
        # a missing image/label above would otherwise inflate this past merged_count.
        per_source_counts[source] = per_source_counts.get(source, 0) + 1

        if dry_run:
            merged_count += 1
            continue

        prefixed_image_name = prefixed_filename(source, img_path.name)
        prefixed_stem = Path(prefixed_image_name).stem

        safe_copy(img_path, out_images_dir / prefixed_image_name, overwrite=True)
        safe_copy(label_path, out_labels_dir / f"{prefixed_stem}.txt", overwrite=True)
        merged_count += 1

    print(f"Merged: {merged_count} / {len(union)}")
    if missing_images:
        print(f"  WARNING: {len(missing_images)} selected entries had no image file on disk (sample: {missing_images[:5]})")
    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} selected entries had no label file on disk (sample: {missing_labels[:5]})")

    stats: dict[str, Any] = {
        "selected_total": len(union),
        "merged_count": merged_count,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "per_source_counts": per_source_counts,
    }

    if dry_run:
        return stats

    # Recompute TRUE post-merge per-class counts from the merged labels themselves —
    # supersedes cap_report.json's pre-merge estimates (see module docstring).
    canonical_names = get_canonical_names()
    real_counts = {name: {"images": 0, "instances": 0} for name in canonical_names}
    unknown_class_ids: dict[str, int] = {}
    for label_path in out_labels_dir.glob("*.txt"):
        present_ids: dict[int, int] = {}
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            present_ids[class_id] = present_ids.get(class_id, 0) + 1
        for class_id, cnt in present_ids.items():
            # Files are already copied by this point (loop above) — a corrupt/out-of-range
            # class_id here must not crash mid-report; count it separately and keep going,
            # same defensive posture box_audit.py already uses for the same lookup.
            if not (0 <= class_id < len(canonical_names)):
                unknown_class_ids[str(class_id)] = unknown_class_ids.get(str(class_id), 0) + cnt
                continue
            name = canonical_names[class_id]
            real_counts[name]["images"] += 1
            real_counts[name]["instances"] += cnt
    stats["real_post_merge_class_counts"] = real_counts
    if unknown_class_ids:
        stats["unknown_class_ids_in_merged_labels"] = unknown_class_ids
        print(f"  WARNING: out-of-range class ids found in merged labels: {unknown_class_ids}")

    print("\nReal post-merge per-class counts (from merged labels, supersedes cap_report.json):")
    for name in canonical_names:
        c = real_counts[name]
        print(f"  {name:18s} images={c['images']:6d}  instances={c['instances']:7d}")

    ensure_dir(reports_dir())
    report_path = reports_dir() / "merge_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\nReport written to {report_path}")
    print(f"Output: {merged_dir()}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Tally without copying any files.")
    args = parser.parse_args()

    print("=" * 60)
    print("merge.py — Stage 5.6")
    print("=" * 60)

    run(dry_run=args.dry_run)

    if args.dry_run:
        print("\n--dry-run: no files copied.")


if __name__ == "__main__":
    main()

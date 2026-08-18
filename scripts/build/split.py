"""
scripts/build/split.py
──────────────────────────
Stage 5.8 Split: partitions dataset/merged/ into dataset/final/{train,val,test}/,
source-stratified, at a specific point within docs/PLAN.md's approved ratio
range (train 70-80%, val 10-15%, test 10-15%).

Ratio point (docs/OPEN_QUESTIONS.md #8 — a deliberate choice within an
already-approved range, not a new decision): TRAIN=0.70, VAL=0.15,
TEST=0.15, matching Ultralytics Academy's own stated baseline
("Split the Dataset Correctly," docs.ultralytics.com) — changed from an
earlier 75/12.5/12.5 pick, both points were always within docs/PLAN.md's
approved range. Recorded as a specific choice so it's not silently
arbitrary, per that entry's own framing.

Confirmed via Ultralytics' own docs before relying on this design at all
(2026-08-18): `ultralytics.data.utils.autosplit()` exists but is an
optional, manually-invoked utility — YOLOv8 training does NOT auto-split
an unsplit folder on its own. Standard training requires pre-organized
train/val(/test) directories referenced by data.yaml paths, which is
exactly this script's job. This script is necessary, not redundant with
anything Ultralytics does automatically.

Source-stratified: each of the 16 processed sources is split independently
at the same ratio, then pooled — this keeps every source proportionally
represented in train/val/test rather than, say, one source landing
entirely in train by chance. dataset/merged/'s filenames are already
source-prefixed (merge.py, DEC-059) via "<source>__<original>", so the
source is recovered by splitting on the first "__" — safe because source
keys themselves only ever contain single underscores.

Leakage prevention beyond "no file appears in two splits" (true by
construction, verified anyway): if dataset/reports/dedup_report.json
exists AND covers the full merged pool (not a partial/smoke-test run),
every near-duplicate and exact-duplicate GROUP it found is kept together
in one split — a near-duplicate pair straddling train/val would let the
model see a near-identical image at train time and again at eval time,
inflating apparent performance. If dedup_report.json is missing or only
covers a subset, this step is skipped and clearly logged as skipped, not
silently ignored.

Val doubles as the Hailo calibration set (AGENTS.md) — the same
source-stratification that keeps train representative keeps val
representative too; no separate logic needed for that requirement.

Usage:
    python3 scripts/build/split.py
    python3 scripts/build/split.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

# Leaf script, not a shared library module — needs the repo root on
# sys.path to import sibling utils regardless of how it's invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.config_loader import get_canonical_names  # noqa: E402
from scripts.utils.file_utils import ensure_dir, final_dir, merged_dir, reports_dir, safe_copy  # noqa: E402

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # remainder, not independently applied — see assign_splits()
SEED = 42


def load_duplicate_groups(merged_image_count: int) -> list[list[str]] | None:
    """
    Near/exact-duplicate groups (as merged-pool filenames) from dedup_report.json,
    if it exists and its exact-duplicate check (the only one required to be
    exhaustive) covers the full merged pool. Returns None if unavailable or only
    a partial (smoke-test) run.

    Coverage is NOT uniform between the two checks dedup.py (DEC-062) runs:
    exact-duplicates (filehash, no threshold) always covers the full pool, but
    near-duplicates (embedding-based) only covers a seeded stratified SAMPLE
    (near_duplicate_sample_size, e.g. 6,000 of 51,529 images) — a measured,
    documented scope bound, not an oversight (see dedup.py's module docstring
    and docs/OPEN_QUESTIONS.md #7). The groups returned here still include both:
    every near-duplicate group actually found is real and worth keeping intact,
    even though the sample doesn't claim to have found every near-duplicate pair
    that exists in the full pool.
    """
    report_path = reports_dir() / "dedup_report.json"
    if not report_path.is_file():
        print("  dedup_report.json not found — duplicate-aware split grouping skipped.")
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    images_checked = report.get("images_checked", 0)
    if images_checked < merged_image_count:
        print(
            f"  dedup_report.json only covers {images_checked} of "
            f"{merged_image_count} merged images for exact-duplicate detection (a partial/"
            f"smoke-test run) — duplicate-aware split grouping skipped, not applied to a subset."
        )
        return None

    groups = []
    for g in report.get("exact_duplicates", []):
        groups.append([g["kept"]] + g["duplicates"])
    for g in report.get("near_duplicates", []):
        groups.append([g["kept"]] + [d["filename"] for d in g["duplicates"]])
    near_dup_sample_size = report.get("near_duplicate_sample_size", images_checked)
    print(
        f"  dedup_report.json's exact-duplicate check covers the full pool "
        f"({images_checked}/{merged_image_count} images); its near-duplicate check covers a "
        f"{near_dup_sample_size}-image sample only (docs/OPEN_QUESTIONS.md #7) — "
        f"{len(groups)} duplicate groups found across both will be kept intact."
    )
    return groups


def assign_splits(filenames: list[str], duplicate_groups: list[list[str]] | None) -> dict[str, str]:
    """
    {filename: split} for every filename, source-stratified, seeded, with duplicate
    groups (if provided) assigned as a single unit.
    """
    source_of = {f: f.split("__", 1)[0] for f in filenames}

    # Real union-find, not last-write-wins: exact-duplicate and near-duplicate groups
    # can overlap (dedup.py/DEC-062's two checks have different coverage — a file can
    # legitimately be reported in both an exact-dup group and a separate near-dup
    # group). Naively doing `group_of[f] = rep` per group would let a later group's
    # processing silently overwrite an earlier group's link for a shared file,
    # breaking that file off from duplicates it's transitively chained to — verified
    # against the real dedup_report.json: 187 files sit in two groups with different
    # membership. Union-find merges these into one connected component instead, so
    # the whole chain moves as a single unit, matching this script's own stated goal.
    parent: dict[str, str] = {f: f for f in filenames}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    if duplicate_groups:
        for group in duplicate_groups:
            present = [f for f in group if f in source_of]
            for f in present[1:]:
                union(present[0], f)

    group_of: dict[str, str] = {f: find(f) for f in filenames}

    # Build per-source lists of *group representatives* (one entry per duplicate
    # group, or per standalone file), so a whole group moves as one unit.
    reps_by_source: dict[str, list[str]] = {}
    members_by_rep: dict[str, list[str]] = {}
    for f in filenames:
        rep = group_of[f]
        members_by_rep.setdefault(rep, []).append(f)
    seen_reps: set[str] = set()
    for f in filenames:
        rep = group_of[f]
        if rep in seen_reps:
            continue
        seen_reps.add(rep)
        reps_by_source.setdefault(source_of[rep], []).append(rep)

    rng = random.Random(SEED)
    assignment: dict[str, str] = {}
    for source, reps in sorted(reps_by_source.items()):
        reps = list(reps)
        rng.shuffle(reps)
        n = len(reps)
        n_train = round(n * TRAIN_RATIO)
        n_val = round(n * VAL_RATIO)
        splits_for_reps = ["train"] * n_train + ["val"] * n_val
        splits_for_reps += ["test"] * (n - len(splits_for_reps))
        for rep, split in zip(reps, splits_for_reps):
            for member in members_by_rep[rep]:
                assignment[member] = split

    return assignment


def run(dry_run: bool = False) -> dict[str, Any]:
    img_dir = merged_dir() / "images"
    lbl_dir = merged_dir() / "labels"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"{img_dir} not found — run scripts/build/merge.py (Stage 5.6) first.")

    filenames = sorted(p.name for p in img_dir.iterdir() if p.is_file())
    print(f"{len(filenames)} images in dataset/merged/images/ to split.")

    print("Checking for duplicate-group leakage prevention...")
    duplicate_groups = load_duplicate_groups(len(filenames))

    assignment = assign_splits(filenames, duplicate_groups)

    # Verify: every file assigned exactly once, to exactly one of the 3 splits.
    assert set(assignment) == set(filenames), "assignment mismatch — not every file was split"
    assert all(s in ("train", "val", "test") for s in assignment.values())

    # Verify: no duplicate group straddles more than one split.
    leakage: list[dict[str, Any]] = []
    if duplicate_groups:
        for group in duplicate_groups:
            present = [f for f in group if f in assignment]
            splits_seen = {assignment[f] for f in present}
            if len(splits_seen) > 1:
                leakage.append({"group": present, "splits": sorted(splits_seen)})

    split_counts = {"train": 0, "val": 0, "test": 0}
    for s in assignment.values():
        split_counts[s] += 1
    print(f"Split sizes: {split_counts}")
    if leakage:
        print(f"  WARNING: {len(leakage)} duplicate groups still straddle multiple splits (unexpected — investigate)")
        # A hard gate, not just a printed warning — the entire point of duplicate-aware
        # grouping (DEC-063) is preventing train/val/test leakage; a warning that's easy
        # to miss in an unattended overnight log while files get written anyway would
        # defeat that purpose. Real leakage here means assign_splits()'s union-find has
        # a bug or dedup_report.json changed underneath it — investigate, don't proceed.
        # --dry-run still reports it without raising, since no files are written in that mode.
        if not dry_run:
            raise RuntimeError(
                f"{len(leakage)} duplicate groups straddle multiple splits — refusing to write "
                f"dataset/final/ with a known leakage bug. See stats['cross_split_duplicate_leakage'] "
                f"for the affected groups."
            )

    stats: dict[str, Any] = {
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "seed": SEED,
        "duplicate_aware": duplicate_groups is not None,
        "duplicate_group_coverage_note": (
            "exact-duplicate groups cover the full merged pool; near-duplicate groups only "
            "cover dedup_report.json's stratified sample (see docs/OPEN_QUESTIONS.md #7) — "
            "groups outside the sample were never checked, not confirmed absent"
        ) if duplicate_groups is not None else None,
        "split_counts": split_counts,
        "cross_split_duplicate_leakage": leakage,
    }

    if dry_run:
        return stats

    # Clear before writing — same stale-output fix as merge.py (DEC-050/061): without
    # this, re-running split.py after a selection change (e.g. dedup_report.json or the
    # merged pool changes) leaves a prior run's now-differently-assigned files behind.
    # Worse here than in merge.py: an orphaned file could sit in a DIFFERENT split than
    # its current assignment (e.g. train from a stale run, val from the fresh one),
    # which is a real train/val/test leakage bug the in-memory leakage check can't see
    # because it never inspects disk state.
    for split in ("train", "val", "test"):
        split_images_dir = final_dir(split) / "images"
        split_labels_dir = final_dir(split) / "labels"
        if split_images_dir.is_dir():
            shutil.rmtree(split_images_dir)
        if split_labels_dir.is_dir():
            shutil.rmtree(split_labels_dir)
        ensure_dir(split_images_dir)
        ensure_dir(split_labels_dir)

    missing_labels: list[str] = []
    for filename, split in assignment.items():
        stem = Path(filename).stem
        safe_copy(img_dir / filename, final_dir(split) / "images" / filename, overwrite=True)
        label_src = lbl_dir / f"{stem}.txt"
        if label_src.is_file():
            safe_copy(label_src, final_dir(split) / "labels" / f"{stem}.txt", overwrite=True)
        else:
            # Tracked, not silently dropped — merge.py already treats this as report-worthy
            # for the identical situation (missing_labels); an image with no label file
            # would otherwise be an untraceable unlabeled training sample.
            missing_labels.append(filename)
    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} images had no label file in dataset/merged/labels/ "
              f"(sample: {missing_labels[:5]})")
    stats["missing_labels"] = missing_labels

    # Per-split per-class / per-source distribution, verified from the copied files.
    canonical_names = get_canonical_names()
    per_split_class: dict[str, dict[str, dict[str, int]]] = {}
    per_split_source: dict[str, dict[str, int]] = {}
    unknown_class_ids: dict[str, int] = {}
    for split in ("train", "val", "test"):
        class_counts = {name: {"images": 0, "instances": 0} for name in canonical_names}
        source_counts: dict[str, int] = {}
        for label_path in (final_dir(split) / "labels").glob("*.txt"):
            source = label_path.stem.split("__", 1)[0]
            source_counts[source] = source_counts.get(source, 0) + 1
            present: dict[int, int] = {}
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                present[class_id] = present.get(class_id, 0) + 1
            for class_id, cnt in present.items():
                if not (0 <= class_id < len(canonical_names)):
                    # Tracked, not silently dropped — matches merge.py's identical
                    # defensive pattern for the same corrupt/out-of-range id situation.
                    unknown_class_ids[str(class_id)] = unknown_class_ids.get(str(class_id), 0) + cnt
                    continue
                name = canonical_names[class_id]
                class_counts[name]["images"] += 1
                class_counts[name]["instances"] += cnt
        per_split_class[split] = class_counts
        per_split_source[split] = source_counts
    if unknown_class_ids:
        stats["unknown_class_ids_in_split_labels"] = unknown_class_ids
        print(f"  WARNING: out-of-range class ids found in split labels: {unknown_class_ids}")

    stats["per_split_class_counts"] = per_split_class
    stats["per_split_source_counts"] = per_split_source

    print("\nPer-split class distribution (images / instances):")
    for name in canonical_names:
        row = "  ".join(f"{s}={per_split_class[s][name]['images']}" for s in ("train", "val", "test"))
        print(f"  {name:18s} {row}")

    ensure_dir(reports_dir())
    report_path = reports_dir() / "split_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\nReport written to {report_path}")
    print(f"Output: {final_dir()}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Tally split assignment without copying files.")
    args = parser.parse_args()

    print("=" * 60)
    print("split.py — Stage 5.8")
    print("=" * 60)

    run(dry_run=args.dry_run)

    if args.dry_run:
        print("\n--dry-run: no files copied.")


if __name__ == "__main__":
    main()

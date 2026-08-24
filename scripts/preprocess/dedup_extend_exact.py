"""
scripts/preprocess/dedup_extend_exact.py
──────────────────────────────────────────
Extends dedup_report.json's EXACT-duplicate coverage (only) to a newly
added source, without re-running the GPU near-duplicate check.

Why this exists: DEC-089's real dedup.py run was a one-time GPU
investment (RunPod, "no intentions to rerun the dedup"). Adding a small
source afterward (e.g. crosswalk_detector_lz3hc, added to close
Pedestrian Lane's post-dedup floor shortfall) still deserves real
exact-duplicate coverage before merging — that's cheap, filehash-based,
CPU-only, no sampling risk, unlike near-duplicate detection which needs
the embedding model. This script does exactly that and nothing more.

What it deliberately does NOT do: touch near_duplicates,
near_duplicate_sample_size, or near_duplicate_covered_source_keys. A
newly-extended source's near-duplicate status against the rest of the
pool stays genuinely unknown — this script makes that gap survivable
(split.py/hide_duplicates only gate on images_checked, see their own
docstrings/guards), not smaller. See dedup_report.json's
near_duplicate_covered_source_keys_note for the full reasoning.

Idempotent: refuses to run twice against the same source (checks
whether that source's prefix already appears in exact_duplicates or
already counts toward images_checked beyond dataset/merged/'s own
current size), so re-running by mistake can't double-count.

Usage:
    python3 scripts/preprocess/dedup_extend_exact.py --source roboflow_crosswalk_detector_lz3hc
    python3 scripts/preprocess/dedup_extend_exact.py --source roboflow_crosswalk_detector_lz3hc --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.file_utils import list_images, merged_dir, processed_dir, prefixed_filename, reports_dir  # noqa: E402


def sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=str, required=True,
        help="Processed source key whose images (dataset/processed/<source>/images/) "
             "should be added to exact-duplicate coverage. Not yet merged into "
             "dataset/merged/ -- this script hashes them from their processed location "
             "directly, applying the same source__filename prefix merge.py would use.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute and print, write nothing.")
    args = parser.parse_args()

    print("=" * 60)
    print("dedup_extend_exact.py")
    print("=" * 60)

    report_path = reports_dir() / "dedup_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    merged_images_dir = merged_dir() / "images"
    merged_count = len(list_images(merged_images_dir, recursive=False))
    if report["images_checked"] < merged_count:
        raise RuntimeError(
            f"dedup_report.json's images_checked ({report['images_checked']}) is already "
            f"behind the live merged pool ({merged_count}) -- something else already changed "
            f"the pool since this report was last written. Investigate before extending further."
        )

    new_source_dir = processed_dir(args.source) / "images"
    new_images = list_images(new_source_dir, recursive=False)
    if not new_images:
        raise RuntimeError(f"No images found under {new_source_dir}.")
    new_prefixed_names = {prefixed_filename(args.source, p.name) for p in new_images}

    # Idempotency guard: has this source already been folded into exact_duplicates,
    # or already counted toward images_checked? (images_checked > merged_count is
    # exactly what a prior successful run of this script against this source would
    # leave behind, since the new source isn't in dataset/merged/ yet.)
    already_referenced = any(
        prefixed in new_prefixed_names
        for g in report["exact_duplicates"]
        for prefixed in [g["kept"], *g["duplicates"]]
    )
    if already_referenced or report["images_checked"] > merged_count:
        raise RuntimeError(
            f"{args.source} looks like it may already be reflected in dedup_report.json "
            f"(images_checked={report['images_checked']} vs live merged pool={merged_count}, "
            f"or a matching filename already appears in exact_duplicates). Refusing to "
            f"re-extend -- check dedup_report.json by hand before forcing this."
        )

    print(f"New source: {args.source} ({len(new_images)} images)")
    print(f"Existing merged pool: {merged_count} images")
    print("Hashing new source images...")
    new_hashes: dict[str, str] = {}  # prefixed filename -> sha256
    for p in new_images:
        new_hashes[prefixed_filename(args.source, p.name)] = sha256_of(p)

    print("Hashing existing merged pool (cheap, filehash only, no GPU)...")
    existing_hashes: dict[str, str] = {}  # prefixed filename -> sha256
    for p in list_images(merged_images_dir, recursive=False):
        existing_hashes[p.name] = sha256_of(p)

    # Group by hash across BOTH sets combined, so within-new-source and
    # new-vs-existing collisions are both caught by the same pass.
    by_hash: dict[str, list[str]] = {}
    for name, h in {**existing_hashes, **new_hashes}.items():
        by_hash.setdefault(h, []).append(name)

    new_groups = []
    for h, names in by_hash.items():
        if len(names) < 2:
            continue
        involves_new = any(n in new_hashes for n in names)
        if not involves_new:
            continue  # pre-existing group, already in the report -- don't touch
        names_sorted = sorted(names)
        new_groups.append({"kept": names_sorted[0], "duplicates": names_sorted[1:]})

    new_dup_files = sum(len(g["duplicates"]) for g in new_groups)
    print()
    print(f"Found {len(new_groups)} new exact-duplicate group(s) involving {args.source}, "
          f"{new_dup_files} duplicate file(s).")
    for g in new_groups:
        print(f"  kept={g['kept']}  duplicates={g['duplicates']}")

    if args.dry_run:
        print("\n--dry-run: report not written.")
        return

    report["exact_duplicates"].extend(new_groups)
    report["exact_duplicate_groups"] = len(report["exact_duplicates"])
    report["exact_duplicate_files"] = sum(len(g["duplicates"]) for g in report["exact_duplicates"])
    report["images_checked"] = merged_count + len(new_images)
    # near_duplicates / near_duplicate_sample_size / near_duplicate_covered_source_keys
    # deliberately untouched -- see module docstring.

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport updated: images_checked -> {report['images_checked']}, "
          f"exact_duplicate_groups -> {report['exact_duplicate_groups']}, "
          f"exact_duplicate_files -> {report['exact_duplicate_files']}")
    print("near_duplicate_sample_size UNCHANGED -- this source has no near-duplicate coverage.")


if __name__ == "__main__":
    main()

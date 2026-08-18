"""
scripts/preprocess/dedup.py
──────────────────────────────
Stage 5.6 Dedup: FiftyOne Brain near-duplicate + exact-duplicate detection
over the merged pool (dataset/merged/), per docs/OPEN_QUESTIONS.md #7.

Flag-and-report only, same posture as every other Stage 5.3-5.7 script —
does not delete or move anything. Nothing under dataset/merged/ is
modified; only dataset/reports/dedup_report.json is written.

Two checks, both real FiftyOne Brain methods (verified against the
installed package's actual signatures/docstrings before use, not assumed
from memory):

  1. compute_exact_duplicates() — filehash-based, no threshold, no
     judgment call. Byte-identical files only.
  2. compute_near_duplicates(threshold=0.2) — embedding-distance-based.
     0.2 is FiftyOne Brain's OWN default parameter value, and its
     docstring states outright: "Values in [0.1, 0.25] work well for the
     default setup." This is the "run it once with FiftyOne Brain's own
     suggested/default threshold" the handoff asked for — not a number
     invented for this project, the tool's own documented default.

     Embedding model: explicitly `mobilenet-v2-imagenet-torch` on MPS,
     not FiftyOne's silent default (measured directly, not assumed —
     the unspecified default model ran at ~3-7 img/s on this machine,
     which would take ~2-4.8 hours over the full 51k-image merged pool;
     mobilenet-v2 on MPS measured at ~28 img/s on a real 1,000-image
     slice, ~9x faster, making the full pool tractable in one run).
     Caveat worth flagging honestly: FiftyOne's "[0.1, 0.25] works well"
     guidance is presumably calibrated against its own default embedding
     space, not necessarily mobilenet-v2's — the threshold number is
     unchanged from the tool's default, but which embedding backbone it
     was tuned against is not 100% certain to transfer. Noted in the
     report, not hidden.

The resulting near-duplicate counts are empirical but UNREVIEWED — same
distinction this project drew for cap_per_class.py's trim method. A pair
being "near-duplicate" at distance 0.2 doesn't mean one of the pair should
be dropped; it means the pair is worth the student's eyes before anything
is.

Near-duplicate embedding computation is run on a seeded, per-source-
stratified sample (NEAR_DUP_SAMPLE_SIZE, default 6000) rather than the
full merged pool — measured directly, not assumed: an unbounded run
against all ~51.5k images degraded from ~89 img/s at the start to under
10 img/s within the first 90 seconds (projecting to an open-ended, likely
1+ hour runtime) once `num_workers` was left at FiftyOne's own default,
which spins up more DataLoader worker processes than this machine's MPS
device handles well concurrently. Fixed by pinning `num_workers=4`
(matches the isolated benchmark that measured ~28 img/s cleanly), but
kept the sample bound anyway as a safety margin against this exact class
of surprise recurring at full scale unattended. compute_exact_duplicates
(filehash-based, no embedding model, no degradation risk) still runs
against the FULL merged pool regardless — only the near-duplicate,
embedding-based check is sampled.

Usage:
    python3 scripts/preprocess/dedup.py
    python3 scripts/preprocess/dedup.py --dry-run
    python3 scripts/preprocess/dedup.py --limit 500   # smoke test (both checks)
    python3 scripts/preprocess/dedup.py --full-scale  # near-duplicate check over the WHOLE
                                                        # merged pool, not the 6000-sample —
                                                        # student's explicit call (2026-08-18):
                                                        # run it locally, accept 1hr+, don't
                                                        # defer to RunPod for this one.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path
from typing import Any

# Leaf script, not a shared library module — needs the repo root on
# sys.path to import sibling utils regardless of how it's invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.file_utils import ensure_dir, list_images, merged_dir, reports_dir  # noqa: E402

NEAR_DUPLICATE_THRESHOLD = 0.2  # FiftyOne Brain's own default — see module docstring
NEAR_DUP_SAMPLE_SIZE = 6000  # practical bound — see module docstring
NEAR_DUP_SEED = 42
EMBEDDING_NUM_WORKERS = 4  # pinned — FiftyOne's own default over-subscribes this machine's MPS device


def stratified_sample(images: list[Path], sample_size: int) -> list[Path]:
    """Seeded, per-source-proportional sample — not `images[:n]`, which would just take
    whichever source sorts first alphabetically."""
    if len(images) <= sample_size:
        return images
    by_source: dict[str, list[Path]] = {}
    for p in images:
        source = p.name.split("__", 1)[0]
        by_source.setdefault(source, []).append(p)
    rng = random.Random(NEAR_DUP_SEED)
    sampled: list[Path] = []
    for source, paths in sorted(by_source.items()):
        take = max(1, round(len(paths) / len(images) * sample_size))
        shuffled = list(paths)
        rng.shuffle(shuffled)
        sampled.extend(shuffled[:take])
    return sampled


def run(dry_run: bool = False, limit: int | None = None, full_scale: bool = False) -> dict[str, Any]:
    img_dir = merged_dir() / "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"{img_dir} not found — run scripts/build/merge.py (Stage 5.6) first.")

    # list_images() (not a bare iterdir()) filters to IMAGE_EXTENSIONS — a stray non-image
    # file (e.g. .DS_Store, which already exists elsewhere under dataset/ on this machine)
    # would otherwise be fed straight into fo.Sample()/compute_*_duplicates() and likely
    # error out partway through an unattended multi-hour run. Not recursive (False) —
    # dataset/merged/images/ is flat by construction (merge.py, DEC-059).
    images = list_images(img_dir, recursive=False)
    if limit is not None:
        images = images[:limit]
    if limit is not None:
        near_dup_images = images
    elif full_scale:
        # Student's explicit choice (2026-08-18, docs/OPEN_QUESTIONS.md #7): the sample
        # bound was a measured throughput precaution, not a hard requirement — running the
        # embedding check over the whole pool locally is an accepted 1hr+ cost this time.
        near_dup_images = images
    else:
        near_dup_images = stratified_sample(images, NEAR_DUP_SAMPLE_SIZE)
    print(f"{len(images)} images in dataset/merged/images/ (exact-duplicate check covers all of them).")
    print(f"{len(near_dup_images)} images {'(full scale, no sampling)' if full_scale and limit is None else 'sampled'} "
          f"for the near-duplicate embedding check.")

    if dry_run:
        return {"images_checked": len(images), "near_duplicate_sample_size": len(near_dup_images)}

    import torch
    import fiftyone as fo
    import fiftyone.brain as fob
    import fiftyone.zoo as foz

    dataset_name = f"dedup_stage5_6_{uuid.uuid4().hex[:8]}"
    # Both created inside the try — creating `dataset` successfully and then having
    # `near_dataset`'s construction raise (backing-store hiccup, disk pressure) would
    # otherwise leak the first dataset, since the try/finally cleanup below wouldn't
    # even be entered yet. Matches DEC-061's project-wide try/finally intent exactly.
    dataset = None
    near_dataset = None
    try:
        dataset = fo.Dataset(name=dataset_name)
        near_dataset = fo.Dataset(name=f"{dataset_name}_neardup_sample")
        sample_ids = dataset.add_samples([fo.Sample(filepath=str(p)) for p in images])
        id_to_name = dict(zip(sample_ids, (p.name for p in images)))

        print("Running compute_exact_duplicates (filehash, no threshold, full pool)...")
        exact_groups = fob.compute_exact_duplicates(dataset)
        # .get(id, id) fallback, not a bare [id] lookup — matches the near-duplicate
        # report below; both should degrade the same way if FiftyOne ever returns an
        # id outside id_to_name, rather than one path crashing and the other not.
        exact_report = [
            {"kept": id_to_name.get(keep_id, keep_id),
             "duplicates": [id_to_name.get(d, d) for d in dup_ids]}
            for keep_id, dup_ids in exact_groups.items()
        ]
        exact_duplicate_count = sum(len(g["duplicates"]) for g in exact_report)
        print(f"  {len(exact_report)} exact-duplicate groups, {exact_duplicate_count} duplicate files total.")

        near_ids = near_dataset.add_samples([fo.Sample(filepath=str(p)) for p in near_dup_images])
        near_id_to_name = dict(zip(near_ids, (p.name for p in near_dup_images)))

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        embedding_model = foz.load_zoo_model("mobilenet-v2-imagenet-torch", device=device)
        print(f"Running compute_near_duplicates on the {len(near_dup_images)}-image sample "
              f"(threshold={NEAR_DUPLICATE_THRESHOLD}, FiftyOne Brain default threshold; "
              f"mobilenet-v2-imagenet-torch embeddings on {device}, num_workers={EMBEDDING_NUM_WORKERS} — "
              f"see module docstring)...")
        near_results = fob.compute_near_duplicates(
            near_dataset, threshold=NEAR_DUPLICATE_THRESHOLD, model=embedding_model,
            num_workers=EMBEDDING_NUM_WORKERS,
        )
        neighbors_map = near_results.neighbors_map
        near_report = []
        for keep_id, neighbors in neighbors_map.items():
            if not neighbors:
                continue
            near_report.append({
                "kept": near_id_to_name.get(keep_id, keep_id),
                "duplicates": [
                    {"filename": near_id_to_name.get(dup_id, dup_id), "distance": round(dist, 4)}
                    for dup_id, dist in neighbors
                ],
            })
        near_duplicate_count = len(near_results.duplicate_ids)
        print(f"  {near_duplicate_count} images flagged as near-duplicates of another "
              f"(threshold={NEAR_DUPLICATE_THRESHOLD}), across {len(near_report)} groups.")
    finally:
        # Scratch FiftyOne datasets — report is the persisted artifact, not the fo DB entries.
        # try/finally so a mid-computation error doesn't leave orphaned datasets registered
        # in FiftyOne's backing store. None-guarded since either constructor call itself
        # (now inside this try) could be what failed.
        if dataset is not None:
            dataset.delete()
        if near_dataset is not None:
            near_dataset.delete()

    report = {
        "images_checked": len(images),
        "near_duplicate_sample_size": len(near_dup_images),
        "near_duplicate_sample_method": (
            "FULL POOL, no sampling — student's explicit choice (2026-08-18) to accept the "
            "longer runtime locally rather than defer to faster hardware, see docs/OPEN_QUESTIONS.md #7"
            if (full_scale and limit is None) else
            "seeded (42), per-source-proportional stratified sample — not exhaustive, see module "
            "docstring for why (measured MPS throughput degradation at full scale, num_workers "
            "over-subscription)"
        ),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "near_duplicate_threshold_source": "fiftyone.brain.compute_near_duplicates default; "
            "docstring states [0.1, 0.25] works well for the default embedding setup",
        "embedding_model": f"mobilenet-v2-imagenet-torch (device={device}) — chosen for measured speed "
            "(~9x faster than FiftyOne's unspecified default on this machine), not the tool's silent "
            "default. Caveat: the [0.1, 0.25] threshold guidance may be calibrated against the tool's "
            "own default embedding space, not necessarily this one — flagged, not hidden.",
        "exact_duplicate_groups": len(exact_report),
        "exact_duplicate_files": exact_duplicate_count,
        "exact_duplicates": exact_report,
        "near_duplicate_files": near_duplicate_count,
        "near_duplicate_groups": len(near_report),
        "near_duplicates": near_report,
    }

    ensure_dir(reports_dir())
    report_path = reports_dir() / "dedup_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport written to {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Count images only — no dedup computation.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of images checked (smoke test).")
    parser.add_argument(
        "--full-scale", action="store_true",
        help="Run the near-duplicate embedding check over the whole merged pool instead of the "
             f"{NEAR_DUP_SAMPLE_SIZE}-image sample. Slower (1hr+ plausible locally) — deliberate, not a default.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("dedup.py — Stage 5.6")
    print("=" * 60)

    run(dry_run=args.dry_run, limit=args.limit, full_scale=args.full_scale)

    if args.dry_run:
        print("\n--dry-run: no dedup computation run, no report written.")


if __name__ == "__main__":
    main()

"""
scripts/acquire/acquire_openimages.py
──────────────────────────────────────
Stage 5.1 acquisition: pulls Open Images V7 as the primary source for
every canonical class configured with `primary.source: open_images` in
config/classes.yaml (Person, Vehicle, Motorcycle, Bicycle, Animals,
Chairs, Tables, Trash Bins).

Uses the FiftyOne Zoo loader (already the project's chosen download
path — see datasets.yaml's open_images.download_strategy) rather than
manual annotation parsing. One zoo pull per canonical class, filtered
to that class's native Open Images label(s), bounded by max_samples
(DEC-027) rather than an unbounded download.

max_samples per class = ceil(classes.yaml cap * BUFFER_FACTOR), where
BUFFER_FACTOR implements DEC-030's ~1.3-1.4x raw-acquisition buffer
over the eventual post-merge cap. That total is split evenly across
Open Images' three splits (train/validation/test, per DEC-036 — source
splits are discarded and rebuilt once at Stage 5.8, so pulling only
`train` would just shrink the pool for no benefit).

Each class's images + COCO-style detection labels are exported to
dataset/raw/open_images/<class_key>/ (images/ + labels.json), matching
datasets.yaml's `annotation_format: coco_style` for this source. This
is Stage 5.1 output only — remapping native Open Images class names to
canonical class IDs happens in Stage 5.2 (openimages_to_intermediate.py,
not yet built), not here.

Usage:
    python3 scripts/acquire/acquire_openimages.py
    python3 scripts/acquire/acquire_openimages.py --classes person,chairs
    python3 scripts/acquire/acquire_openimages.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

# Leaf script, not a shared library module — needs the repo root on
# sys.path to import sibling utils regardless of how it's invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.config_loader import load_classes  # noqa: E402
from scripts.utils.file_utils import ensure_dir, raw_dir, reports_dir  # noqa: E402

# DEC-030: raw acquisition buffer of ~1.3-1.4x the post-merge cap
# (unvalidated estimate). 1.35 is the midpoint of that range.
BUFFER_FACTOR = 1.35

# Matches config/training.yaml's seed, for reproducible sampling.
SEED = 42

# DEC-036: source splits are discarded at acquisition and rebuilt once at
# Stage 5.8, so restricting to one Open Images split would just shrink the
# available pool for no benefit. Pull from all three.
ZOO_DATASET_NAME = "open-images-v7"
ZOO_SPLITS = ("train", "validation", "test")
LABEL_FIELD = "ground_truth"


def _normalize_native_classes(native_class: str | list[str]) -> list[str]:
    """classes.yaml stores native_class as either a single string or a list."""
    return [native_class] if isinstance(native_class, str) else list(native_class)


def get_openimages_targets() -> dict[str, dict[str, Any]]:
    """
    Find every canonical class in classes.yaml whose primary source is
    open_images, and return what's needed to pull it.

    Returns
    -------
    dict
        Keyed by class_key (e.g. "person"), each value has:
        `class_id`, `native_classes` (list[str]), `cap` (int | None).
    """
    classes_data = load_classes()["classes"]
    targets: dict[str, dict[str, Any]] = {}

    for class_key, entry in classes_data.items():
        primary = entry.get("primary")
        if not primary or primary.get("source") != "open_images":
            continue
        targets[class_key] = {
            "class_id": entry["id"],
            "native_classes": _normalize_native_classes(primary["native_class"]),
            "cap": entry.get("cap"),
        }

    return targets


def compute_max_samples(cap: int | None) -> int:
    """
    Buffered raw-acquisition target for one class, per DEC-030.

    Raises
    ------
    ValueError
        If cap is None — every open_images-primary class in classes.yaml
        currently has a cap set, so a missing one means the config
        changed in a way this script doesn't yet handle.
    """
    if cap is None:
        raise ValueError(
            "No cap set for this class in classes.yaml — can't compute a "
            "buffered max_samples without one."
        )
    return math.ceil(cap * BUFFER_FACTOR)


def pull_class(class_key: str, native_classes: list[str], max_samples: int) -> dict[str, Any]:
    """
    Download one class's Open Images subset via the FiftyOne Zoo and
    export it to dataset/raw/open_images/<class_key>/.

    Returns
    -------
    dict
        Summary record for the acquisition report.
    """
    import fiftyone as fo
    import fiftyone.zoo as foz
    from fiftyone import ViewField as F

    zoo_dataset_name = f"oi_v7_{class_key}"
    export_dir = raw_dir("open_images") / class_key

    # max_samples applies PER split when `splits` (plural) is passed, not as
    # a combined total (verified directly against the Zoo loader) — divide
    # the buffered total across the three splits accordingly.
    per_split_max_samples = math.ceil(max_samples / len(ZOO_SPLITS))

    dataset = foz.load_zoo_dataset(
        ZOO_DATASET_NAME,
        splits=list(ZOO_SPLITS),
        label_types=["detections"],
        classes=native_classes,
        max_samples=per_split_max_samples,
        shuffle=True,
        seed=SEED,
        dataset_name=zoo_dataset_name,
        drop_existing_dataset=True,
        label_field=LABEL_FIELD,
    )

    view = dataset.filter_labels(
        LABEL_FIELD,
        (F("IsDepiction") == False) & (F("IsGroupOf") == False)
    )
    view = view.match(F(f"{LABEL_FIELD}.detections").length() > 0)

    actual_count = len(view)
    if actual_count == 0:
        print(f"    WARNING: 0 images returned for {native_classes} — nothing to export.")
        dataset.delete()
        return {
            "class_key": class_key,
            "native_classes": native_classes,
            "requested_max_samples": max_samples,
            "actual_images": 0,
            "export_dir": str(export_dir),
        }

    # FiftyOne's export() MERGES into an existing directory rather than
    # replacing it — labels.json gets overwritten wholesale (correct), but
    # image files from a previous run that are no longer in this run's set
    # (e.g. after a filter change) are left behind as orphans (verified
    # directly, not assumed). Clear the directory first so every run is a
    # clean, reproducible snapshot of exactly what this run pulled.
    if export_dir.exists():
        shutil.rmtree(export_dir)
    ensure_dir(export_dir)
    view.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.COCODetectionDataset,
        label_field=LABEL_FIELD,
        export_media=True,
    )

    dataset.delete()

    return {
        "class_key": class_key,
        "native_classes": native_classes,
        "requested_max_samples": max_samples,
        "actual_images": actual_count,
        "export_dir": str(export_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help=(
            "Comma-separated canonical class keys to pull (e.g. "
            "'person,chairs'). Defaults to every open_images-primary "
            "class in classes.yaml."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned max_samples per class without downloading anything.",
    )
    args = parser.parse_args()

    targets = get_openimages_targets()

    if args.classes:
        requested = [c.strip() for c in args.classes.split(",") if c.strip()]
        unknown = [c for c in requested if c not in targets]
        if unknown:
            print(f"Unknown class key(s): {unknown}. Available: {sorted(targets)}")
            sys.exit(1)
        targets = {k: v for k, v in targets.items() if k in requested}

    print("=" * 60)
    print("acquire_openimages.py — Stage 5.1")
    print("=" * 60)
    print(f"\nClasses to pull: {sorted(targets)}\n")

    plan = []
    for class_key, info in targets.items():
        max_samples = compute_max_samples(info["cap"])
        plan.append((class_key, info, max_samples))
        print(
            f"  {class_key:12s} native={info['native_classes']!r:35} "
            f"cap={info['cap']:>5} -> max_samples={max_samples:>5}"
        )

    if args.dry_run:
        print("\n--dry-run: no downloads performed.")
        return

    print()
    results = []
    for class_key, info, max_samples in plan:
        print(f"[{class_key}] pulling up to {max_samples} images "
              f"for native class(es) {info['native_classes']}...")
        try:
            result = pull_class(class_key, info["native_classes"], max_samples)
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "class_key": class_key,
                "native_classes": info["native_classes"],
                "requested_max_samples": max_samples,
                "error": str(e),
            })
            continue
        print(f"    -> {result['actual_images']} images exported to {result['export_dir']}")
        results.append(result)

    report_path = ensure_dir(reports_dir()) / "acquire_openimages_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nReport written to {report_path}")
    print("\nDone. This is raw Stage 5.1 output — native Open Images class")
    print("names have NOT been remapped to canonical IDs yet (Stage 5.2).")


if __name__ == "__main__":
    main()

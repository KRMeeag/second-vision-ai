"""
Export a sample tag from a live FiftyOne review dataset to a durable JSON file.

WHY THIS EXISTS
---------------
Sample tags applied in the App live only in MongoDB. Rebuilding a review dataset
(or deleting one, which has happened twice in this project -- DEC-096) drops them
silently, with no error and nothing on disk to recover from. `labels_reviewed/`
protects box edits; nothing protected tags until this script.

The immediate use is DEC-108's "Option A" for Pedestrian Lane: the class was dropped
at 13 (DEC-100) but sits only ~68 distinct images below DEC-042's 1,500 floor, so it
may come back. Boxes for it cannot be drawn -- write-back raises on any label outside
CANONICAL_NAMES (a deliberate gate, not a bug). Tagging the SAMPLE instead records
"this image contains a pedestrian lane" without touching ground_truth at all, so the
images do not have to be hunted down again if the class is reinstated.

Write-back ignores every sample tag except `exclude` (verified against cell
`2343228f`), so any tag used here is inert with respect to the pipeline.

ACCUMULATES, NEVER TRUNCATES
----------------------------
Re-running merges into the existing file rather than replacing it, matching the
behaviour of `<source>_excluded.json` (DEC-078). A session that reviews half a source
therefore cannot erase the other half's marks recorded earlier. `--prune` opts into
removing entries whose tag has since been cleared in the App, for the case where a
mark was applied by mistake.

Usage
-----
    python3 scripts/preprocess/export_sample_tags.py --dataset review_dataset_ninja_road_damage_detector
    python3 scripts/preprocess/export_sample_tags.py --dataset <name> --tag has_pedestrian_lane
    python3 scripts/preprocess/export_sample_tags.py --list          # what's tagged, no writes
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fiftyone as fo

from scripts.utils.file_utils import ensure_dir, reports_dir

DEFAULT_TAG = "has_pedestrian_lane"


def source_key_from_dataset_name(name: str) -> str:
    """`review_<source_key>` -> `<source_key>`; anything else is returned unchanged."""
    return name[len("review_"):] if name.startswith("review_") else name


def report_path(source_key: str, tag: str) -> Path:
    return reports_dir() / f"{source_key}_tagged_{tag}.json"


def collect_tagged(dataset: fo.Dataset, tag: str) -> list[dict[str, str]]:
    """
    Every sample carrying `tag`, as {filename, filepath}.

    dataset.reload() first, ALWAYS (DEC-097): FiftyOne caches Sample objects by id
    inside this process, so iterating a dataset the App has touched returns the
    kernel's stale snapshot rather than what the App wrote to MongoDB. A tag applied
    in the App minutes ago is invisible without this call -- which is precisely the
    failure this script exists to guard against, so getting it wrong here would be
    self-defeating.
    """
    dataset.reload()
    out = []
    for sample in dataset.match_tags(tag):
        out.append({"filename": Path(sample.filepath).name, "filepath": sample.filepath})
    return sorted(out, key=lambda r: r["filename"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", help="FiftyOne dataset name, e.g. review_dataset_ninja_road_damage_detector")
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"Sample tag to export (default: {DEFAULT_TAG})")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="Show every dataset's tag counts and exit. No writes.")
    parser.add_argument("--prune", action="store_true",
                        help="Drop entries no longer tagged in the App (default: accumulate only).")
    args = parser.parse_args()

    if args.list_only:
        print("Sample tags across all FiftyOne datasets:\n")
        for name in sorted(fo.list_datasets()):
            try:
                d = fo.load_dataset(name)
                d.reload()
                tags = d.count_sample_tags()
            except Exception as exc:
                print(f"  {name}: <error: {exc}>")
                continue
            if tags:
                print(f"  {name}: {tags}")
        return

    if not args.dataset:
        parser.error("--dataset is required unless --list is given")

    if args.dataset not in fo.list_datasets():
        raise SystemExit(f"No such FiftyOne dataset: {args.dataset!r}. Run with --list to see what exists.")

    dataset = fo.load_dataset(args.dataset)
    source_key = source_key_from_dataset_name(args.dataset)
    tagged = collect_tagged(dataset, args.tag)
    path = report_path(source_key, args.tag)

    existing: dict[str, dict] = {}
    if path.is_file():
        prior = json.loads(path.read_text(encoding="utf-8"))
        existing = {e["filename"]: e for e in prior.get("samples", [])}
        print(f"Existing report has {len(existing)} entry/entries.")

    current = {e["filename"]: e for e in tagged}
    if args.prune:
        merged = current
        dropped = set(existing) - set(current)
        if dropped:
            print(f"--prune: dropping {len(dropped)} entry/entries no longer tagged in the App.")
    else:
        merged = {**existing, **current}

    added = set(merged) - set(existing)

    ensure_dir(reports_dir())
    payload = {
        "dataset": args.dataset,
        "source_key": source_key,
        "tag": args.tag,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(merged),
        "note": (
            "Sample tags exported from a live FiftyOne review dataset. These are NOT "
            "labels and no pipeline stage reads this file -- it records which images a "
            "human confirmed contain the tagged content, so they need not be found again."
        ),
        "samples": sorted(merged.values(), key=lambda r: r["filename"]),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nTag {args.tag!r} on {args.dataset}:")
    print(f"  currently tagged in the App : {len(current)}")
    print(f"  newly added to the report   : {len(added)}")
    print(f"  total recorded              : {len(merged)}")
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()

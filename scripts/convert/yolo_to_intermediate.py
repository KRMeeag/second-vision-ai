"""
scripts/convert/yolo_to_intermediate.py
─────────────────────────────────────────
Stage 5.2 conversion for Roboflow: turns each eligible project's native
YOLOv8 export under dataset/raw/roboflow_projects/<project_key>/
(built by acquire_roboflow.py) into DEC-046's intermediate schema — a
flat images/+labels/ pool per project, canonical class ids. Unlike
Open Images (DEC-052), Roboflow projects are kept as separate
processed outputs, one per project — they're independent sources with
independent licenses, and Stage 5.6 (not this script) is where
cross-source merging happens.

Roboflow's YOLOv8 export already gives normalized center-based boxes
(class_idx cx cy w h per line) — no coordinate math needed, just class
remapping — but every project's real data.yaml was checked against
config/datasets.yaml's assumed class list before writing this (DEC-053)
and 4 of 10 on-disk projects didn't match what was configured:

  - escalator_stairs / cv_project_hovyc: native_class_filter fixed to
    match real class names (previously either mismatched or missing,
    which would have matched nothing or over-included non-canonical
    classes).
  - me5_u6rvg: real data.yaml has placeholder names ("0".."7"), not
    real ones — a real Roboflow export bug. Real names recovered via
    the API and stored as `native_class_index_override` in
    datasets.yaml; this script substitutes them in before the normal
    name-based filter is applied.
  - pedestrian_and_animal_crossing: the only downloaded version had
    just one class ("people"), not the crosswalk-marking class the
    canonical Pedestrian Lane class actually needs. Forked + regenerated
    on Roboflow to pull in the real class (DEC-053); config now points
    at the forked project.

Class resolution per project, in priority order:
  1. `native_class_filter` as a dict: {canonical_key: native_name(s)} —
     explicit, only listed native names are kept.
  2. `native_class_filter` as a bare string: the one native name to
     keep, mapped to the project's singular `canonical_class`.
  3. No filter, singular `canonical_class`: every native class name in
     the project blanket-maps to it (real precedent: elevator_status_
     s4lrk's multiple state-labels all collapsing to "elevator").
  4. No filter, plural `canonical_classes`: case-insensitive match
     between each native name and the canonical class keys; anything
     that doesn't match is dropped (not guessed).

Usage:
    python3 scripts/convert/yolo_to_intermediate.py
    python3 scripts/convert/yolo_to_intermediate.py --projects pole_detection_z76mb
    python3 scripts/convert/yolo_to_intermediate.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Leaf script, not a shared library module — needs the repo root on
# sys.path to import sibling utils regardless of how it's invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml  # noqa: E402

from scripts.acquire.acquire_roboflow import get_eligible_projects  # noqa: E402
from scripts.utils.bbox_utils import clip_bbox, validate_bbox  # noqa: E402
from scripts.utils.config_loader import load_classes  # noqa: E402
from scripts.utils.file_utils import ensure_dir, processed_dir, raw_dir, reports_dir, safe_copy  # noqa: E402

SPLIT_DIRS = ("train", "valid", "test")


def get_native_names(project_dir: Path, entry: dict[str, Any]) -> list[str]:
    """
    Real class names for one project, index-ordered, from its data.yaml —
    substituted via `native_class_index_override` first if the project's
    own data.yaml has placeholder names (me5_u6rvg-style export bug).
    """
    data_yaml = yaml.safe_load((project_dir / "data.yaml").read_text(encoding="utf-8"))
    names: list[str] = data_yaml["names"]

    override = entry.get("native_class_index_override")
    if override:
        names = [override[str(i)] for i in range(len(names))]

    return names


def resolve_class_mapping(entry: dict[str, Any], native_names: list[str]) -> dict[str, str]:
    """
    Build native_name -> canonical_class_key for one project, per the
    4-case priority order in this module's docstring.
    """
    canonical_class = entry.get("canonical_class")
    native_filter = entry.get("native_class_filter")

    if isinstance(native_filter, dict):
        mapping: dict[str, str] = {}
        for canonical_key, native in native_filter.items():
            names = [native] if isinstance(native, str) else list(native)
            for name in names:
                mapping[name] = canonical_key
        return mapping

    if isinstance(native_filter, str):
        return {native_filter: canonical_class}

    if canonical_class is not None:
        return {name: canonical_class for name in native_names}

    canonical_classes = entry.get("canonical_classes", [])
    canonical_by_lower = {c.lower(): c for c in canonical_classes}
    mapping = {}
    for name in native_names:
        matched = canonical_by_lower.get(name.lower())
        if matched:
            mapping[name] = matched
    return mapping


def convert_project(project_key: str, entry: dict[str, Any], class_id_map: dict[str, int]) -> dict[str, Any]:
    """
    Convert one Roboflow project's train/valid/test YOLO export into a
    flat, canonical-id images/+labels/ pool.
    """
    project_dir = raw_dir(f"roboflow_projects/{project_key}")
    out_images_dir = processed_dir(f"roboflow_{project_key}") / "images"
    out_labels_dir = processed_dir(f"roboflow_{project_key}") / "labels"

    stats: dict[str, Any] = {
        "native_names": [],
        "class_mapping": {},
        "boxes_total": 0,
        "boxes_kept": 0,
        "boxes_dropped_non_canonical": 0,
        "boxes_invalid_dropped": 0,
        "boxes_clipped": 0,
        "images_converted": 0,
        "images_dropped_empty": 0,
    }

    if not (project_dir / "data.yaml").is_file():
        print(f"  WARNING: missing {project_dir}/data.yaml, skipping {project_key}")
        return stats

    native_names = get_native_names(project_dir, entry)
    mapping = resolve_class_mapping(entry, native_names)
    index_to_class_id = {
        i: class_id_map[mapping[name]]
        for i, name in enumerate(native_names)
        if name in mapping
    }
    stats["native_names"] = native_names
    stats["class_mapping"] = mapping

    ensure_dir(out_images_dir)
    ensure_dir(out_labels_dir)

    for split in SPLIT_DIRS:
        labels_dir = project_dir / split / "labels"
        images_dir = project_dir / split / "images"
        if not labels_dir.is_dir():
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            lines_out = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                stats["boxes_total"] += 1
                native_idx = int(parts[0])
                class_id = index_to_class_id.get(native_idx)
                if class_id is None:
                    stats["boxes_dropped_non_canonical"] += 1
                    continue

                cx, cy, w, h = (float(v) for v in parts[1:5])
                reason = validate_bbox(cx, cy, w, h)
                if reason is not None:
                    if w <= 0 or h <= 0:
                        stats["boxes_invalid_dropped"] += 1
                        continue
                    cx, cy, w, h = clip_bbox(cx, cy, w, h)
                    # A box entirely outside the frame clips down to
                    # degenerate zero width/height, not a usable one —
                    # re-check (DEC-057).
                    if w <= 0 or h <= 0:
                        stats["boxes_invalid_dropped"] += 1
                        continue
                    stats["boxes_clipped"] += 1

                lines_out.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats["boxes_kept"] += 1

            if not lines_out:
                stats["images_dropped_empty"] += 1
                continue

            image_path = images_dir / f"{label_path.stem}.jpg"
            if not image_path.is_file():
                candidates = list(images_dir.glob(f"{label_path.stem}.*"))
                if not candidates:
                    print(f"  WARNING: no image found for label {label_path}, skipping")
                    continue
                image_path = candidates[0]

            safe_copy(image_path, out_images_dir / image_path.name, overwrite=True)
            (out_labels_dir / f"{label_path.stem}.txt").write_text(
                "\n".join(lines_out) + "\n", encoding="utf-8"
            )
            stats["images_converted"] += 1

    return stats


def run(dry_run: bool = False, only: list[str] | None = None) -> dict[str, Any]:
    eligible = get_eligible_projects()
    if only:
        eligible = {k: v for k, v in eligible.items() if k in only}

    class_id_map = {key: entry["id"] for key, entry in load_classes()["classes"].items()}

    all_stats: dict[str, Any] = {}
    for project_key, entry in sorted(eligible.items()):
        project_dir = raw_dir(f"roboflow_projects/{project_key}")
        if not (project_dir / "data.yaml").is_file():
            print(f"  WARNING: missing {project_dir}/data.yaml, skipping {project_key}")
            continue

        native_names = get_native_names(project_dir, entry)
        mapping = resolve_class_mapping(entry, native_names)
        print(f"\n[{project_key}] native={native_names}")
        print(f"    mapping: {mapping}")

        if dry_run:
            all_stats[project_key] = {"native_names": native_names, "class_mapping": mapping}
            continue

        stats = convert_project(project_key, entry, class_id_map)
        all_stats[project_key] = stats
        print(
            f"    -> {stats['images_converted']} images, {stats['boxes_kept']} boxes kept "
            f"({stats['boxes_dropped_non_canonical']} dropped non-canonical, "
            f"{stats['boxes_clipped']} clipped, {stats['boxes_invalid_dropped']} invalid)"
        )

    return all_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projects",
        type=str,
        default=None,
        help="Comma-separated project keys to convert. Defaults to every eligible project.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each project's real native classes and resolved mapping without converting.",
    )
    args = parser.parse_args()

    only = [p.strip() for p in args.projects.split(",")] if args.projects else None

    print("=" * 60)
    print("yolo_to_intermediate.py — Stage 5.2")
    print("=" * 60)

    all_stats = run(dry_run=args.dry_run, only=only)

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return

    report_path = ensure_dir(reports_dir()) / "yolo_to_intermediate_report.json"
    if report_path.is_file():
        # Merge into the existing report rather than overwrite it wholesale —
        # a --projects subset run would otherwise silently discard every
        # other project's results from a prior full run (a real bug, found
        # after a --projects rerun clobbered a completed 11-project report).
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        existing.update(all_stats)
        all_stats = existing
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(all_stats, fh, indent=2)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()

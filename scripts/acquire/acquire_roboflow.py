"""
scripts/acquire/acquire_roboflow.py
──────────────────────────────────────
Stage 5.1 acquisition: pulls every eligible Roboflow project listed
under `roboflow_projects` in config/datasets.yaml, via pinned SDK
version (DEC-018r — Local Pull-and-Resolve). Native class remapping
happens later, at Stage 5.2 conversion — this script only downloads.

Eligibility: a project is pulled only if `audit_status: pending` AND
it has a non-null `pinned_version`. Everything else is skipped and
reported as skipped, not silently dropped:
  - `failed`  — relabel candidate (e.g. stairs_lusiz, stairs_hsatv,
                DEC-031), not pursued right now.
  - `benched` — nothing wrong with the source, just not currently
                chosen (e.g. jeep_hozhs, DEC-037).
  - `blocked` — no pinned_version to pull yet (e.g. trash_bins_secondary,
                and traffico_y1 which is `pending` but still has
                pinned_version: null — zero generated versions on
                Roboflow's side, DEC-032/DEC-037).

Each project downloads via `version.download(format, location=...)` in
its configured `download_format` (yolov8 for all current entries),
straight to dataset/raw/roboflow_projects/<project_key>/ — whatever
train/valid/test folders the SDK produces are kept as-is; DEC-036
already discards that split assignment at Stage 5.8, not here.

Usage:
    python3 scripts/acquire/acquire_roboflow.py
    python3 scripts/acquire/acquire_roboflow.py --projects pole_detection_z76mb,pothole_vhmow
    python3 scripts/acquire/acquire_roboflow.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Leaf script, not a shared library module — needs the repo root on
# sys.path to import sibling utils regardless of how it's invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv  # noqa: E402

from scripts.acquire.list_roboflow_versions import parse_workspace_project  # noqa: E402
from scripts.utils.config_loader import load_datasets  # noqa: E402
from scripts.utils.file_utils import ensure_dir, list_images, raw_dir, reports_dir  # noqa: E402


def get_eligible_projects() -> dict[str, dict[str, Any]]:
    """
    Find every Roboflow project entry eligible to pull right now:
    `audit_status: pending` and a non-null `pinned_version`.

    Returns
    -------
    dict
        Keyed by project key, each value is the raw datasets.yaml entry.
    """
    projects = load_datasets().get("roboflow_projects", {})
    return {
        key: entry
        for key, entry in projects.items()
        if entry.get("audit_status") == "pending" and entry.get("pinned_version") is not None
    }


def get_skipped_projects() -> dict[str, str]:
    """Every project NOT eligible right now, mapped to a human-readable reason."""
    projects = load_datasets().get("roboflow_projects", {})
    skipped = {}
    for key, entry in projects.items():
        status = entry.get("audit_status")
        version = entry.get("pinned_version")
        if status == "pending" and version is not None:
            continue
        if version is None:
            reason = f"{status} — no pinned_version"
        else:
            reason = status
        skipped[key] = reason
    return skipped


def pull_project(rf, project_key: str, entry: dict[str, Any]) -> dict[str, Any]:
    """
    Download one Roboflow project at its pinned version.

    Returns
    -------
    dict
        Summary record for the acquisition report.
    """
    dest_dir = raw_dir(f"roboflow_projects/{project_key}")
    canonical = entry.get("canonical_class") or entry.get("canonical_classes")
    workspace, project_slug = parse_workspace_project(entry["original_url"])
    version_num = entry["pinned_version"]
    download_format = entry["download_format"]

    ensure_dir(dest_dir)
    project = rf.workspace(workspace).project(project_slug)
    version = project.version(version_num)
    version.download(download_format, location=str(dest_dir), overwrite=True)

    actual_images = len(list_images(dest_dir))

    return {
        "project_key": project_key,
        "canonical": canonical,
        "workspace": workspace,
        "project_slug": project_slug,
        "pinned_version": version_num,
        "download_format": download_format,
        "dest_dir": str(dest_dir),
        "actual_images": actual_images,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projects",
        type=str,
        default=None,
        help=(
            "Comma-separated project keys to pull (e.g. "
            "'pole_detection_z76mb,pothole_vhmow'). Defaults to every "
            "eligible project."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pull/skip plan without downloading anything.",
    )
    args = parser.parse_args()

    eligible = get_eligible_projects()
    skipped = get_skipped_projects()

    if args.projects:
        requested = [p.strip() for p in args.projects.split(",") if p.strip()]
        unknown = [p for p in requested if p not in eligible and p not in skipped]
        if unknown:
            print(f"Unknown project key(s): {unknown}. Available: {sorted({**eligible, **skipped})}")
            sys.exit(1)
        still_skipped = [p for p in requested if p in skipped]
        if still_skipped:
            print(f"Requested project(s) not eligible, skipping: "
                  f"{[(p, skipped[p]) for p in still_skipped]}")
        eligible = {k: v for k, v in eligible.items() if k in requested}

    print("=" * 60)
    print("acquire_roboflow.py — Stage 5.1")
    print("=" * 60)

    print(f"\nEligible to pull ({len(eligible)}):")
    for key, entry in eligible.items():
        canonical = entry.get("canonical_class") or entry.get("canonical_classes")
        print(f"  {key:32s} v{entry['pinned_version']:<3} {entry['download_format']:8s} {canonical}")

    print(f"\nSkipped ({len(skipped)}):")
    for key, reason in skipped.items():
        print(f"  {key:32s} {reason}")

    if args.dry_run:
        print("\n--dry-run: no downloads performed.")
        return

    load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("\nROBOFLOW_API_KEY not set (checked environment and .env). Aborting.")
        sys.exit(1)

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)

    print()
    results = []
    for project_key, entry in eligible.items():
        print(f"[{project_key}] pulling v{entry['pinned_version']} "
              f"({entry['download_format']})...")
        try:
            result = pull_project(rf, project_key, entry)
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"project_key": project_key, "error": str(e)})
            continue
        print(f"    -> {result['actual_images']} images in {result['dest_dir']}")
        results.append(result)

    report_path = ensure_dir(reports_dir()) / "acquire_roboflow_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nReport written to {report_path}")
    print("\nDone. This is raw Stage 5.1 output — native class names have")
    print("NOT been remapped to canonical IDs yet (Stage 5.2), and source")
    print("train/valid/test splits are kept as downloaded (discarded at")
    print("Stage 5.8, per DEC-036), not merged here.")


if __name__ == "__main__":
    main()

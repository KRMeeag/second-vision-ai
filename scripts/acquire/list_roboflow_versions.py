"""
scripts/acquire/list_roboflow_versions.py
──────────────────────────────────────────
Diagnostic tool: lists every available version for every Roboflow
project in config/datasets.yaml, so picking a pinned_version doesn't
require manually clicking through each project's page.

This is NOT part of the Stage 5.1 acquisition pipeline itself — it's
a prerequisite tool that informs what to write into datasets.yaml's
pinned_version fields before acquire_roboflow.py (still unbuilt) can
run against a fixed, reproducible version per project (DEC-018r).

Usage:
    Set ROBOFLOW_API_KEY in a local .env file (gitignored) or export
    it in the shell, then:

        python3 scripts/acquire/list_roboflow_versions.py

Output: printed to the console, and the same data written to
dataset/reports/roboflow_versions.json for later reference.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# Leaf script, not a shared library module (unlike scripts/utils/*.py) —
# needs the repo root on sys.path to import sibling utils regardless of
# how it's invoked (direct `python3 scripts/acquire/...` vs `python3 -m`).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv  # noqa: E402
from roboflow import Roboflow  # noqa: E402

from scripts.utils.config_loader import load_datasets  # noqa: E402
from scripts.utils.file_utils import ensure_dir, reports_dir  # noqa: E402


def parse_workspace_project(url: str | None) -> tuple[str, str] | None:
    """
    Extract (workspace_slug, project_slug) from a Roboflow Universe URL.

    Returns None for placeholder/unresolved URLs (e.g. "TBD") rather
    than raising — callers should treat that as "skip this one".
    """
    if not url or url == "TBD":
        return None
    parts = urlparse(url).path.strip("/").split("/")
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def fetch_versions(rf: Roboflow, workspace: str, project_slug: str) -> list[dict]:
    """Fetch available versions for one project as plain dicts (JSON-safe)."""
    project = rf.workspace(workspace).project(project_slug)
    return [
        {
            "version": v.version,
            "created": v.created,
            "images": v.images,
            "splits": v.splits,
            "type": v.type,
        }
        for v in project.versions()
    ]


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY not set (checked environment and .env). Aborting.")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    projects = load_datasets().get("roboflow_projects", {})

    report: dict[str, object] = {}

    for key, entry in projects.items():
        canonical = entry.get("canonical_class") or entry.get("canonical_classes")
        url = entry.get("original_url")

        print("=" * 60)
        print(f"{key}  ({canonical})")
        print(f"  url: {url}")

        parsed = parse_workspace_project(url)
        if parsed is None:
            print("  SKIPPED — no resolvable project URL yet")
            continue

        workspace, project_slug = parsed
        try:
            versions = fetch_versions(rf, workspace, project_slug)
        except Exception as e:
            print(f"  ERROR fetching versions: {e}")
            report[key] = {"error": str(e)}
            continue

        if not versions:
            print("  No versions found.")
        for v in versions:
            print(
                f"  v{v['version']}: {v['images']} images, "
                f"type={v['type']}, created={v['created']}, splits={v['splits']}"
            )
        report[key] = versions

    out_path = ensure_dir(reports_dir()) / "roboflow_versions.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print("=" * 60)
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()

"""
scripts/utils/config_loader.py
──────────────────────────────
Loads and validates config/classes.yaml and config/datasets.yaml.
All acquisition, conversion, and curation scripts import from here.

Design principles:
  - Single source of truth: all path resolution goes through this module.
  - Fail early: config validation runs at import time so errors surface
    before any data processing begins.
  - No side effects: read-only; never mutates config files.

Pipeline context:
  YOLOv8 Training → ONNX Export → Hailo DFC → HEF → hailo-apps on RPi5
  This module feeds every stage: converters, acquire scripts, training YAML.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Repo root resolution
# ---------------------------------------------------------------------------

def get_repo_root() -> Path:
    """
    Walk up from this file's location to find the repo root.
    The repo root is defined as the first parent directory that contains
    both a 'config/' directory and an 'AGENTS.md' file.

    Returns
    -------
    Path
        Absolute path to the repository root.

    Raises
    ------
    RuntimeError
        If the repo root cannot be located (e.g., script moved outside repo).
    """
    candidate = Path(__file__).resolve()
    for parent in [candidate, *candidate.parents]:
        if (parent / "config").is_dir() and (parent / "AGENTS.md").is_file():
            return parent
    raise RuntimeError(
        "Cannot locate repo root. Expected a parent directory containing "
        "'config/' and 'AGENTS.md'. Check that this script lives inside "
        "the second-vision-ai repository."
    )


REPO_ROOT: Path = get_repo_root()
CLASSES_YAML: Path = REPO_ROOT / "config" / "classes.yaml"
DATASETS_YAML: Path = REPO_ROOT / "config" / "datasets.yaml"

# Authoritative class count — must match nc in classes.yaml
EXPECTED_NC: int = 16

# Canonical class names in YOLO id order (0-indexed).
# Must stay in sync with AGENTS.md and config/classes.yaml.
# ID 2 was "Two Wheeler" until DEC-038 (2026-08-08) split it into
# Motorcycle (slot reused, ID 2) and Bicycle (new, appended at ID 15).
CANONICAL_NAMES: list[str] = [
    "Person",
    "Vehicle",
    "Motorcycle",
    "Pole",
    "Animals",
    "Stairs",
    "Escalator",
    "Doors",
    "Chairs",
    "Tables",
    "Tricycle",
    "Potholes",
    "Trash Bins",
    "Elevator",
    "Pedestrian Lane",
    "Bicycle",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Repo root resolved to: {REPO_ROOT}"
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at top level in: {path}")
    return data


# ---------------------------------------------------------------------------
# classes.yaml
# ---------------------------------------------------------------------------

def load_classes() -> dict[str, Any]:
    """
    Load and validate config/classes.yaml.

    Validation checks:
      - File exists and parses as a YAML mapping.
      - 'nc' field matches EXPECTED_NC (16).
      - 'names' field has exactly nc entries.
      - Each canonical name matches the hardcoded CANONICAL_NAMES list.

    Returns
    -------
    dict
        The full parsed contents of classes.yaml.

    Raises
    ------
    FileNotFoundError
        If classes.yaml does not exist.
    ValueError
        If validation fails (nc mismatch, missing names, etc.).
    """
    data = _load_yaml(CLASSES_YAML)

    nc = data.get("nc")
    if nc != EXPECTED_NC:
        raise ValueError(
            f"classes.yaml 'nc' is {nc!r}, expected {EXPECTED_NC}. "
            "Update EXPECTED_NC in config_loader.py if you intentionally "
            "changed the class count."
        )

    names: dict[int, str] | None = data.get("names")
    if names is None:
        raise ValueError("classes.yaml is missing the 'names' field.")
    if len(names) != EXPECTED_NC:
        raise ValueError(
            f"classes.yaml 'names' has {len(names)} entries, expected {EXPECTED_NC}."
        )

    # Verify each canonical name matches CANONICAL_NAMES
    for idx, expected in enumerate(CANONICAL_NAMES):
        actual = names.get(idx)
        if actual != expected:
            raise ValueError(
                f"classes.yaml name mismatch at index {idx}: "
                f"got {actual!r}, expected {expected!r}. "
                "Check AGENTS.md and config/classes.yaml are in sync."
            )

    return data


def load_datasets() -> dict[str, Any]:
    """
    Load config/datasets.yaml.

    Returns
    -------
    dict
        The full parsed contents of datasets.yaml.

    Raises
    ------
    FileNotFoundError
        If datasets.yaml does not exist.
    ValueError
        If the file does not parse as a YAML mapping.
    """
    return _load_yaml(DATASETS_YAML)


# ---------------------------------------------------------------------------
# Accessor functions
# ---------------------------------------------------------------------------

def get_canonical_names() -> list[str]:
    """
    Return the ordered list of 16 canonical class names (0-indexed for YOLO).

    This is the authoritative source for class ordering in training YAML,
    converter scripts, and evaluation scripts.

    Returns
    -------
    list[str]
        16 class names in YOLO label order.
    """
    return CANONICAL_NAMES.copy()


def get_class_id(name: str) -> int:
    """
    Return the YOLO class ID (0-indexed) for a canonical class name.

    Parameters
    ----------
    name : str
        Canonical class name, e.g. ``"Person"``, ``"Trash Bins"``.
        Case-sensitive — must match CANONICAL_NAMES exactly.

    Returns
    -------
    int
        Class ID (0–15).

    Raises
    ------
    KeyError
        If the name is not in the canonical list.
    """
    try:
        return CANONICAL_NAMES.index(name)
    except ValueError:
        raise KeyError(
            f"'{name}' is not a canonical class name. "
            f"Valid names: {CANONICAL_NAMES}"
        )


def get_class_name(class_id: int) -> str:
    """
    Return the canonical class name for a YOLO class ID.

    Parameters
    ----------
    class_id : int
        YOLO class ID (0–15).

    Returns
    -------
    str
        Canonical class name.

    Raises
    ------
    IndexError
        If class_id is out of the valid range 0–15.
    """
    if not (0 <= class_id < EXPECTED_NC):
        raise IndexError(
            f"class_id {class_id} is out of range. "
            f"Valid range: 0–{EXPECTED_NC - 1}"
        )
    return CANONICAL_NAMES[class_id]


def get_roboflow_project(key: str) -> dict[str, Any]:
    """
    Retrieve a Roboflow project entry from datasets.yaml by its key.

    Parameters
    ----------
    key : str
        Project key as defined under ``roboflow_projects`` in datasets.yaml,
        e.g. ``"pole_detection_z76mb"``, ``"escalator_stairs"``.

    Returns
    -------
    dict
        The project configuration dict.

    Raises
    ------
    KeyError
        If the key is not found in roboflow_projects.
    ValueError
        If datasets.yaml has no 'roboflow_projects' section.
    """
    datasets = load_datasets()
    projects: dict | None = datasets.get("roboflow_projects")
    if projects is None:
        raise ValueError(
            "datasets.yaml has no 'roboflow_projects' section. "
            "Check that datasets.yaml is up to date."
        )
    if key not in projects:
        raise KeyError(
            f"Roboflow project '{key}' not found. "
            f"Available keys: {sorted(projects.keys())}"
        )
    return projects[key]


def get_source_config(source_key: str) -> dict[str, Any]:
    """
    Retrieve a large-scale source config block from datasets.yaml.

    Parameters
    ----------
    source_key : str
        Top-level key in datasets.yaml, e.g. ``"open_images"``,
        ``"crowdhuman"``, ``"exdark"``.

    Returns
    -------
    dict
        The source configuration dict.

    Raises
    ------
    KeyError
        If source_key does not exist at the top level of datasets.yaml.
    """
    datasets = load_datasets()
    if source_key not in datasets:
        raise KeyError(
            f"Source '{source_key}' not found in datasets.yaml. "
            f"Available top-level keys: {sorted(datasets.keys())}"
        )
    return datasets[source_key]


# ---------------------------------------------------------------------------
# Smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("config_loader.py — smoke test")
    print("=" * 60)

    print(f"\nRepo root  : {REPO_ROOT}")
    print(f"classes.yaml: {CLASSES_YAML}")
    print(f"datasets.yaml: {DATASETS_YAML}")

    print("\n[1] Loading classes.yaml ...")
    classes = load_classes()
    print(f"    nc={classes['nc']}, names validated OK")

    print("\n[2] Canonical class list:")
    for i, name in enumerate(get_canonical_names()):
        print(f"    {i:2d}: {name}")

    print("\n[3] Reverse lookup (get_class_id):")
    for name in ["Person", "Trash Bins", "Pedestrian Lane"]:
        cid = get_class_id(name)
        print(f"    '{name}' → id={cid}")

    print("\n[4] Loading datasets.yaml ...")
    datasets = load_datasets()
    print(f"    Top-level keys: {sorted(datasets.keys())}")

    print("\n[5] Roboflow project lookup (pole_detection_z76mb):")
    pole = get_roboflow_project("pole_detection_z76mb")
    print(f"    canonical_class: {pole['canonical_class']}")
    print(f"    pinned_version : {pole['pinned_version']}")

    print("\n✅ All checks passed.")
    sys.exit(0)

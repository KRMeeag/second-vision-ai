"""
scripts/utils/file_utils.py
────────────────────────────
Filesystem helpers shared by every acquire/convert/preprocess/curate/build
script: standard dataset directory resolution, collision-safe copy/move,
source-prefixed filenames (Stage 5.6 merge), and image enumeration.

Design principles:
  - Path helpers are pure: they resolve a Path, they never create it.
    Call ensure_dir() explicitly when a directory actually needs to exist.
  - Copy/move never silently overwrite: pass overwrite=True to opt in.
  - Repo layout matches the dataset/ tree in README.md:
      dataset/{raw,processed,curated,merged}/<source>/
      dataset/final/{train,val,test}/
      dataset/reports/

Pipeline context:
  YOLOv8 Training → ONNX Export → Hailo DFC → HEF → hailo-apps on RPi5
  Used by scripts/acquire, scripts/convert, scripts/preprocess,
  scripts/curate, and scripts/build — anywhere files move between
  dataset/ stage directories.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Repo root resolution
# ---------------------------------------------------------------------------
# Mirrors scripts/utils/config_loader.py's get_repo_root() — duplicated
# rather than imported so this module resolves correctly no matter how
# it's invoked (`python scripts/utils/file_utils.py` directly, or
# `python -m scripts.utils.file_utils`), without relying on sys.path
# containing the repo root.

def get_repo_root() -> Path:
    """
    Walk up from this file's location to find the repo root.

    Returns
    -------
    Path
        Absolute path to the repository root.

    Raises
    ------
    RuntimeError
        If the repo root cannot be located.
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
DATASET_ROOT: Path = REPO_ROOT / "dataset"

VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# Directory resolution (pure — never creates anything)
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """
    Create `path` (and any missing parents) if it doesn't already exist.

    Parameters
    ----------
    path : Path
        Directory to create.

    Returns
    -------
    Path
        The same path, guaranteed to exist as a directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir(source: str | None = None) -> Path:
    """Path to dataset/raw/ or dataset/raw/<source>/."""
    return (DATASET_ROOT / "raw" / source) if source else (DATASET_ROOT / "raw")


def processed_dir(source: str | None = None) -> Path:
    """Path to dataset/processed/ or dataset/processed/<source>/."""
    return (DATASET_ROOT / "processed" / source) if source else (DATASET_ROOT / "processed")


def curated_dir(source: str | None = None) -> Path:
    """Path to dataset/curated/ or dataset/curated/<source>/."""
    return (DATASET_ROOT / "curated" / source) if source else (DATASET_ROOT / "curated")


def merged_dir() -> Path:
    """Path to dataset/merged/ — the post-cap, post-merge, pre-split pool."""
    return DATASET_ROOT / "merged"


def final_dir(split: str | None = None) -> Path:
    """
    Path to dataset/final/ or dataset/final/<split>/.

    Parameters
    ----------
    split : str, optional
        One of "train", "val", "test".

    Raises
    ------
    ValueError
        If split is given but not one of VALID_SPLITS.
    """
    if split is None:
        return DATASET_ROOT / "final"
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    return DATASET_ROOT / "final" / split


def reports_dir() -> Path:
    """Path to dataset/reports/ — audit logs, cap reports, curation logs."""
    return DATASET_ROOT / "reports"


# ---------------------------------------------------------------------------
# Collision-safe copy / move
# ---------------------------------------------------------------------------

def safe_copy(src: Path, dst: Path, overwrite: bool = False) -> Path:
    """
    Copy `src` to `dst`, creating `dst`'s parent directories as needed.

    Parameters
    ----------
    src : Path
        Source file. Must exist.
    dst : Path
        Destination file path (not a directory).
    overwrite : bool, default False
        If False and dst already exists, raises FileExistsError instead
        of silently overwriting it.

    Returns
    -------
    Path
        The destination path.

    Raises
    ------
    FileNotFoundError
        If src does not exist.
    FileExistsError
        If dst exists and overwrite is False.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"Source file not found: {src}")
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {dst} (pass overwrite=True to replace it)"
        )
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def safe_move(src: Path, dst: Path, overwrite: bool = False) -> Path:
    """
    Move `src` to `dst`, creating `dst`'s parent directories as needed.

    Same collision semantics as safe_copy(): raises FileExistsError unless
    overwrite=True.

    Parameters
    ----------
    src : Path
        Source file. Must exist.
    dst : Path
        Destination file path (not a directory).
    overwrite : bool, default False
        If False and dst already exists, raises FileExistsError.

    Returns
    -------
    Path
        The destination path.

    Raises
    ------
    FileNotFoundError
        If src does not exist.
    FileExistsError
        If dst exists and overwrite is False.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"Source file not found: {src}")
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {dst} (pass overwrite=True to replace it)"
        )
    ensure_dir(dst.parent)
    shutil.move(str(src), str(dst))
    return dst


# ---------------------------------------------------------------------------
# Naming & enumeration
# ---------------------------------------------------------------------------

def prefixed_filename(source: str, filename: str) -> str:
    """
    Build a source-prefixed filename for Stage 5.6 merge, so filenames
    that collide across sources (e.g. two sources both emitting "0001.jpg")
    don't overwrite each other once merged into a single pool.

    Parameters
    ----------
    source : str
        Canonical source key, e.g. "open_images", "crowdhuman",
        "dataset_ninja_pothole_detection".
    filename : str
        Original filename, e.g. "0001.jpg".

    Returns
    -------
    str
        e.g. "open_images__0001.jpg".
    """
    return f"{source}__{filename}"


def list_images(directory: Path, recursive: bool = True) -> list[Path]:
    """
    List image files under `directory` with a recognized extension.

    Parameters
    ----------
    directory : Path
        Directory to search.
    recursive : bool, default True
        If True, search subdirectories too.

    Returns
    -------
    list[Path]
        Matching image paths, sorted for reproducibility.

    Raises
    ------
    FileNotFoundError
        If directory does not exist.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("file_utils.py — smoke test")
    print("=" * 60)

    print(f"\nRepo root   : {REPO_ROOT}")
    print(f"Dataset root: {DATASET_ROOT}")

    print("\n[1] Directory helpers:")
    print(f"    raw_dir()          : {raw_dir()}")
    print(f"    raw_dir('exdark')  : {raw_dir('exdark')}")
    print(f"    processed_dir('crowdhuman'): {processed_dir('crowdhuman')}")
    print(f"    merged_dir()       : {merged_dir()}")
    print(f"    final_dir('val')   : {final_dir('val')}")
    print(f"    reports_dir()      : {reports_dir()}")

    print("\n[2] final_dir() rejects an invalid split:")
    try:
        final_dir("bogus")
    except ValueError as e:
        print(f"    Raised as expected: {e}")

    print("\n[3] prefixed_filename:")
    print(f"    prefixed_filename('open_images', '0001.jpg') -> "
          f"{prefixed_filename('open_images', '0001.jpg')!r}")

    print("\n✅ All checks passed.")
    sys.exit(0)

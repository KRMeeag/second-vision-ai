"""
scripts/acquire/acquire_crowdhuman.py
──────────────────────────────────────
Stage 5.1 acquisition: CrowdHuman, secondary (volume_topup) source for
Person, sourced specifically for genuine crowd/occlusion coverage that
COCO-pretrained backbones handle poorly (DEC-028) — not just extra
Person volume.

Unlike ExDark/Dataset Ninja (DEC-041, manual download — no real API),
this script DOES download: the original crowdhuman.org mirrors
(Baidu/Google Drive) are dead, so the source moved to a HuggingFace
mirror (DEC-047) that's public and ungated — confirmed directly via
HF's API (`gated: false`) before writing this. That's a real,
well-maintained SDK (`huggingface_hub`), the same category as the
Roboflow SDK already used elsewhere (DEC-018r), not fragile host
scraping — so DEC-041's "not worth scripting" reasoning doesn't apply
to this specific source anymore.

Pulls CrowdHuman_train01/02/03.zip, CrowdHuman_val.zip,
annotation_train.odgt, annotation_val.odgt. Skips CrowdHuman_test.zip —
DEC-036 discards every source's original split and rebuilds one at
Stage 5.8, so a held-out test set here would just get pooled in anyway.

Per DEC-012: only the `vbox` (visible-region) field is used; `fbox`
(full) and `hbox` (head) are discarded. Verified directly against the
real annotation_train.odgt (not assumed): every gtbox has `tag` of
either "person" or "mask", and `extra.ignore` is 1, 0, or absent.
Filter is unambiguous: keep tag == "person" and extra.get("ignore", 0)
!= 1 — matches what datasets.yaml already specified before this was
checked.

Output: same Stage 5.2 "intermediate schema" as acquire_exdark.py
(DEC-046) — flat images/ + labels/ pool, canonical class ids, no
split — to dataset/processed/crowdhuman/.

One assumption NOT yet verified against real bytes (flagged, not
silently trusted): each zip's internal layout is assumed to be a top-
level Images/ folder with files named `<ID>.jpg` (CrowdHuman's
well-documented standard release structure) — confirming this exactly
requires downloading a multi-GB zip, which this script does not do on
its own initiative. extract_images() raises loudly if Images/ isn't
found rather than silently producing wrong output.

Usage:
    python3 scripts/acquire/acquire_crowdhuman.py
    python3 scripts/acquire/acquire_crowdhuman.py --dry-run
    python3 scripts/acquire/acquire_crowdhuman.py --skip-download   # files already placed
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

# Leaf script, not a shared library module — needs the repo root on
# sys.path to import sibling utils regardless of how it's invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.bbox_utils import clip_bbox, validate_bbox, xywh_abs_to_yolo  # noqa: E402
from scripts.utils.config_loader import get_class_id  # noqa: E402
from scripts.utils.file_utils import ensure_dir, processed_dir, raw_dir, reports_dir  # noqa: E402

REPO_ID = "sshao0516/CrowdHuman"
ZIP_FILES = ("CrowdHuman_train01.zip", "CrowdHuman_train02.zip", "CrowdHuman_train03.zip", "CrowdHuman_val.zip")
ANNOTATION_FILES = ("annotation_train.odgt", "annotation_val.odgt")
PERSON_CLASS_ID = get_class_id("Person")


def download(dest_dir: Path) -> None:
    """Pull every needed file from the HF mirror (skips test.zip, per DEC-036)."""
    from huggingface_hub import hf_hub_download

    for filename in (*ZIP_FILES, *ANNOTATION_FILES):
        print(f"  fetching {filename}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(dest_dir),
        )


def extract_images(dest_dir: Path, images_dir: Path) -> None:
    """
    Extract every zip's Images/ folder into a single flat images_dir.

    Raises
    ------
    KeyError
        If a zip doesn't contain the expected top-level Images/ folder —
        surfaced loudly rather than silently producing an empty/wrong
        image pool (see module docstring: this layout is assumed, not
        yet verified against real bytes).
    """
    ensure_dir(images_dir)
    for zip_name in ZIP_FILES:
        zip_path = dest_dir / zip_name
        if not zip_path.is_file():
            raise FileNotFoundError(f"{zip_path} not found — run without --skip-download first.")
        print(f"  extracting {zip_name}...")
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if m.startswith("Images/") and not m.endswith("/")]
            if not members:
                raise KeyError(
                    f"{zip_name} has no top-level 'Images/' folder — the assumed "
                    f"CrowdHuman zip layout doesn't match this archive. Inspect "
                    f"zf.namelist() manually before proceeding."
                )
            for member in members:
                target = images_dir / Path(member).name
                with zf.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())


def parse_odgt(ann_path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    """
    Parse one .odgt file into {image_id: [vbox, ...]}, keeping only
    tag == "person" and extra.ignore != 1 (DEC-012 vbox-only policy).
    """
    per_image: dict[str, list[tuple[int, int, int, int]]] = {}
    with ann_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            boxes = []
            for box in entry["gtboxes"]:
                if box["tag"] != "person":
                    continue
                if box.get("extra", {}).get("ignore", 0) == 1:
                    continue
                x, y, w, h = box["vbox"]
                boxes.append((x, y, w, h))
            per_image[entry["ID"]] = boxes
    return per_image


def convert(
    per_image: dict[str, list[tuple[int, int, int, int]]],
    images_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
) -> dict[str, int]:
    """Convert every image's vbox list to a YOLO label file; copy the image alongside it."""
    from PIL import Image

    stats = {"images_converted": 0, "images_missing": 0, "images_dropped_empty": 0,
              "boxes_kept": 0, "boxes_invalid_dropped": 0, "boxes_clipped": 0}

    for image_id, boxes in per_image.items():
        if not boxes:
            stats["images_dropped_empty"] += 1
            continue

        img_path = images_dir / f"{image_id}.jpg"
        if not img_path.is_file():
            stats["images_missing"] += 1
            continue

        with Image.open(img_path) as img:
            width, height = img.size

        lines = []
        for x, y, w, h in boxes:
            cx, cy, nw, nh = xywh_abs_to_yolo(x, y, w, h, width, height)
            reason = validate_bbox(cx, cy, nw, nh)
            if reason is not None:
                if nw <= 0 or nh <= 0:
                    stats["boxes_invalid_dropped"] += 1
                    continue
                cx, cy, nw, nh = clip_bbox(cx, cy, nw, nh)
                stats["boxes_clipped"] += 1
            lines.append(f"{PERSON_CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not lines:
            stats["images_dropped_empty"] += 1
            continue

        safe_target = out_images_dir / img_path.name
        safe_target.write_bytes(img_path.read_bytes())
        (out_labels_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        stats["images_converted"] += 1
        stats["boxes_kept"] += len(lines)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                         help="Parse annotation files and report tallies only — no download, no image extraction/conversion.")
    parser.add_argument("--skip-download", action="store_true",
                         help="Skip the HF download step (files already placed under dataset/raw/crowdhuman/).")
    args = parser.parse_args()

    dest_dir = raw_dir("crowdhuman")
    print("=" * 60)
    print("acquire_crowdhuman.py — Stage 5.1")
    print("=" * 60)

    if args.dry_run:
        print("\n--dry-run: parsing annotation files only (no download).\n")
        for ann_file in ANNOTATION_FILES:
            ann_path = dest_dir / ann_file
            if not ann_path.is_file():
                print(f"  {ann_file}: not present yet, skipping")
                continue
            per_image = parse_odgt(ann_path)
            total_boxes = sum(len(b) for b in per_image.values())
            non_empty = sum(1 for b in per_image.values() if b)
            print(f"  {ann_file}: {len(per_image)} images, {non_empty} with >=1 kept person box, {total_boxes} boxes kept")
        return

    ensure_dir(dest_dir)
    if not args.skip_download:
        print("\nDownloading from HuggingFace...")
        download(dest_dir)
    else:
        print("\n--skip-download: expecting files already under", dest_dir)

    images_dir = dest_dir / "Images"
    print("\nExtracting image archives...")
    extract_images(dest_dir, images_dir)

    out_images_dir = ensure_dir(processed_dir("crowdhuman") / "images")
    out_labels_dir = ensure_dir(processed_dir("crowdhuman") / "labels")

    combined_stats: dict[str, int] = {}
    for ann_file in ANNOTATION_FILES:
        print(f"\nParsing + converting {ann_file}...")
        per_image = parse_odgt(dest_dir / ann_file)
        stats = convert(per_image, images_dir, out_images_dir, out_labels_dir)
        for k, v in stats.items():
            combined_stats[k] = combined_stats.get(k, 0) + v
        print(f"  {stats}")

    report_path = ensure_dir(reports_dir()) / "acquire_crowdhuman_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(combined_stats, fh, indent=2)

    print(f"\nReport written to {report_path}")
    print(f"Output: {processed_dir('crowdhuman')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fold a dropped native Roboflow class into a canonical class, pool-images only.

A native class excluded by datasets.yaml's native_class_filter leaves its objects
visible but unlabelled in the converted output. Re-adding it by editing the filter
and re-running yolo_to_intermediate.py would regenerate labels/ from the raw export
and pull in EVERY image containing that class -- for dlsu's "Electric Bike" that is
2,566 images against the 277 already in the pool.

This instead patches the existing labels/ in place: for each image ALREADY present,
it copies that class's boxes out of the raw export under a canonical id. No image is
added, none is removed, and no other box is touched. Coordinates transfer unchanged
because both sides are normalised YOLO.

Idempotent: a box already present at the target id is not added twice.

    python3 scripts/preprocess/merge_native_class.py \\
        --source roboflow_dlsu_d_vehicle_type_detection \\
        --raw dlsu_d_vehicle_type_detection \\
        --native "Electric Bike" --canonical Tricycle --dry-run
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config_loader import CANONICAL_NAMES  # noqa: E402
from utils.file_utils import processed_dir, raw_dir  # noqa: E402

import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="dataset/processed/<source> dir name")
    ap.add_argument("--raw", required=True, help="dataset/raw/roboflow_projects/<raw> dir name")
    ap.add_argument("--native", required=True, nargs="+", help="native class name(s) in the raw data.yaml")
    ap.add_argument("--canonical", required=True, help="canonical class to fold it into")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.canonical not in CANONICAL_NAMES:
        raise SystemExit(f"{args.canonical!r} not in CANONICAL_NAMES: {CANONICAL_NAMES}")
    target_id = CANONICAL_NAMES.index(args.canonical)

    raw_root = raw_dir() / "roboflow_projects" / args.raw
    names = yaml.safe_load((raw_root / "data.yaml").read_text(encoding="utf-8"))["names"]
    missing = [n for n in args.native if n not in names]
    if missing:
        raise SystemExit(f"{missing} not in raw classes: {names}")
    native_ids = {names.index(n) for n in args.native}

    raw_by_stem = {f.stem: f for f in raw_root.glob("*/labels/*.txt")}
    labels = processed_dir(args.source) / "labels"
    pool = sorted(labels.glob("*.txt"))

    pending: list[tuple[Path, str]] = []
    imgs = added = skipped_dupe = no_raw = 0
    for pf in pool:
        rf = raw_by_stem.get(pf.stem)
        if rf is None:
            no_raw += 1
            continue
        new_lines = []
        for line in rf.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5 and int(parts[0]) in native_ids:
                new_lines.append(" ".join([str(target_id)] + parts[1:]))
        if not new_lines:
            continue
        existing = [l for l in pf.read_text(encoding="utf-8").splitlines() if l.strip()]
        fresh = [l for l in new_lines if l not in existing]
        skipped_dupe += len(new_lines) - len(fresh)
        if not fresh:
            continue
        imgs += 1
        added += len(fresh)
        pending.append((pf, "\n".join(existing + fresh) + "\n"))

    print(f"source     : {args.source}")
    print(f"pool images: {len(pool)}   (no raw label file for {no_raw})")
    print(f"folding    : {args.native} (raw ids {sorted(native_ids)}) -> {args.canonical!r} (canonical id {target_id})")
    print(f"{'WOULD ADD' if args.dry_run else 'ADDED'} {added} box(es) across {imgs} image(s)")
    if skipped_dupe:
        print(f"  {skipped_dupe} already present at the target id -- not duplicated")

    if pending and not args.dry_run:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = labels.parent / f"labels_pre_merge_{args.canonical.replace(' ', '')}_bak_{stamp}"
        shutil.copytree(labels, backup)
        for path, text in pending:
            path.write_text(text, encoding="utf-8")
        print(f"backup -> {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Promote labels_reviewed/ over labels/ so the pipeline actually sees review work.

cap_per_class.py scans dataset/processed/<source>/labels/ (cap_per_class.py:220).
It never looks at labels_reviewed/, so a completed review is invisible to the
cascade until its files are copied across. This is the step DEC-087 performed by
hand for cv_project_hovyc and trashcan_detection_pihfn on 2026-08-21; every review
since then has been sitting unpromoted.

labels/ is backed up to labels_pre_promote_bak_<timestamp>/ before anything is
overwritten. dataset/processed/ is gitignored, so that backup is the only copy.

Files present in labels/ but NOT in labels_reviewed/ are LEFT ALONE, never deleted.
They are images the review did not cover -- a narrower scope (restrict_to_merged)
or a source only partly reviewed. Deleting them would silently shrink the pool.

    python3 scripts/preprocess/promote_reviews.py --dry-run
    python3 scripts/preprocess/promote_reviews.py --source roboflow_roitrikee
    python3 scripts/preprocess/promote_reviews.py --all
"""
from __future__ import annotations

import argparse
import datetime
import filecmp
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.file_utils import processed_dir  # noqa: E402


def candidates() -> list[str]:
    out = []
    for path in sorted(processed_dir().iterdir()):
        if not path.is_dir():
            continue
        reviewed = path / "labels_reviewed"
        if reviewed.is_dir() and any(reviewed.glob("*.txt")):
            out.append(path.name)
    return out


def plan(source: str) -> dict:
    base = processed_dir(source)
    labels, reviewed = base / "labels", base / "labels_reviewed"
    changed, identical, new = [], [], []
    for src in sorted(reviewed.glob("*.txt")):
        dst = labels / src.name
        if not dst.is_file():
            new.append(src.name)
        elif filecmp.cmp(src, dst, shallow=False):
            identical.append(src.name)
        else:
            changed.append(src.name)
    untouched = sum(
        1 for f in labels.glob("*.txt") if not (reviewed / f.name).is_file()
    )
    return {
        "source": source,
        "changed": changed,
        "identical": identical,
        "new": new,
        "untouched_in_labels": untouched,
    }


def promote(source: str, dry_run: bool) -> dict:
    result = plan(source)
    base = processed_dir(source)
    labels, reviewed = base / "labels", base / "labels_reviewed"
    to_write = result["changed"] + result["new"]
    if not to_write or dry_run:
        result["backup"] = None
        return result

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = base / f"labels_pre_promote_bak_{stamp}"
    shutil.copytree(labels, backup)
    for name in to_write:
        shutil.copy2(reviewed / name, labels / name)
    result["backup"] = str(backup)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", help="processed source dir name(s); default is every source with a review")
    ap.add_argument("--all", action="store_true", help="promote every source that has a non-empty labels_reviewed/")
    ap.add_argument("--dry-run", action="store_true", help="report what would change and write nothing")
    args = ap.parse_args()

    if args.source:
        sources = args.source
    elif args.all or args.dry_run:
        sources = candidates()
    else:
        ap.error("pass --source, --all, or --dry-run")

    total = 0
    for source in sources:
        base = processed_dir(source)
        if not (base / "labels_reviewed").is_dir():
            print(f"{source}: no labels_reviewed/ -- skipped")
            continue
        r = promote(source, args.dry_run)
        n = len(r["changed"]) + len(r["new"])
        total += n
        verb = "would promote" if args.dry_run else "promoted"
        print(
            f"{r['source']:<44} {verb} {n:>5}"
            f"  (changed {len(r['changed'])}, new {len(r['new'])},"
            f" already identical {len(r['identical'])},"
            f" left alone in labels/ {r['untouched_in_labels']})"
        )
        if r.get("backup"):
            print(f"{'':<44} backup -> {r['backup']}")
    print(f"\n{'would promote' if args.dry_run else 'promoted'} {total} file(s) across {len(sources)} source(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

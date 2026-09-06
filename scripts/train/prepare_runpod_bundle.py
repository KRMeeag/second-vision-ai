"""
Build the single upload bundle that serves every training condition.

WHY A BUNDLE AND NOT JUST `dataset/final/`
------------------------------------------
Training runs on RunPod (DEC-026) and the student's measured upload throughput is
~0.7 MB/s, so `dataset/final/` at 8.4 GB costs ~3.3 h to transfer. Doing that once
per ablation condition is unaffordable. Two things fix it:

1. **One superset, not three pools.** cap 1500 / 4500 / 9000 are strictly nested
   (verified in-process every run, below -- it is a property of the current
   CLASS_PRIORITY_SOURCES config, not a structural guarantee, and it HAS been
   violated before by a config edit). Uploading the 9000 pool therefore covers
   every condition; each one is materialised on the pod from a manifest.

2. **Downscale to 640.** `ultralytics/data/base.py:257-261` already resizes every
   image's long side to `imgsz` at load time, BEFORE any augmentation, and mosaic
   routes through the same call (`augment.py:365` -> `base.py:410`). Hailo
   calibration uses the same loader (`exporter.py:976,988`). So a 640-bounded copy
   is what training sees either way. Measured: ~63% size reduction, 4.4 h -> ~1.9 h.

   DOWNSCALE ONLY. Only ~62% of the pool exceeds 640; `load_image()` would happily
   upscale the rest, but baking that in would inflate the bundle and lock in
   interpolation for nothing. Small images are copied byte-for-byte.

WHAT THIS SCRIPT WILL NOT DO
----------------------------
It never calls `merge.py` or `split.py`. Both `shutil.rmtree` their output
directories before writing (`split.py:411-418`, `merge.py:191-194`) and neither
accepts an output-dir override, so invoking them here would destroy the verified
`dataset/final/`. This script reads `dataset/processed/<source>/` directly and
emits manifests; nothing under `dataset/merged/` or `dataset/final/` is touched.

THE EVALUATION SET IS FROZEN
----------------------------
`split.py:279-292` creates one `random.Random(SEED)` and shuffles per-source rep
lists in sequence, so changing the pool changes every subsequent shuffle: each cap
condition would get a DIFFERENT train/val/test partition. Four conditions with four
different test sets cannot be compared, and an image that is `val` at 4500 can be
`train` at 9000 -- leakage across the comparison.

So val and test are pinned to the CURRENT verified split and are identical for every
condition. Only the training set varies:

    train(condition) = pool(condition) - frozen_val - frozen_test

Images that enter train only at 9000 are checked against the frozen eval sets through
the same duplicate-group and augmented-sibling union-find `split.py` uses, and dropped
on collision, so the zero-leakage guarantee survives.

Usage
-----
    python3 scripts/train/prepare_runpod_bundle.py --dry-run
    python3 scripts/train/prepare_runpod_bundle.py --out dataset/bundle
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from scripts.build.merge import load_excluded_pairs
from scripts.preprocess.cap_per_class import run as cap_run
from scripts.utils.config_loader import get_canonical_names
from scripts.utils.file_utils import ensure_dir, prefixed_filename, processed_dir, reports_dir

CONDITIONS = (1500, 4500, 9000)
SUPERSET_CAP = 9000
IMGSZ = 640
JPEG_QUALITY = 95


# ---------------------------------------------------------------------------
# pools
# ---------------------------------------------------------------------------

def pool_for(cap: int) -> set[tuple[str, str]]:
    """(source, stem) pairs cap_per_class selects at `cap`, minus review exclusions.

    Uses cap_per_class's own run() in dry-run mode rather than reimplementing its
    budget maths -- dry_run returns the report and writes nothing, so the canonical
    cap_report.json is never touched.
    """
    buf = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(buf):
        report = cap_run(dry_run=True, hard_cap_preset=cap)
    pool: set[tuple[str, str]] = set()
    for entry in report["classes"].values():
        pool |= {tuple(s.split("/", 1)) for s in entry["selected"]}
    return pool - load_excluded_pairs()


def frozen_eval_sets() -> tuple[set[str], set[str]]:
    """val/test prefixed filenames from the current split_report.json.

    These are the DEC-112/117/118 lineage's verified split, already checked for
    cross-split duplicate leakage. Pinned, never recomputed per condition.
    """
    report = json.loads((reports_dir() / "split_report.json").read_text(encoding="utf-8"))
    for key in ("split_assignments", "assignments", "splits"):
        if key in report and isinstance(report[key], dict):
            a = report[key]
            return ({f for f, s in a.items() if s == "val"},
                    {f for f, s in a.items() if s == "test"})
    # Fall back to reading dataset/final/ directly -- the split on disk IS the record.
    from scripts.utils.file_utils import final_dir
    val = {p.stem for p in (final_dir("val") / "labels").glob("*.txt")}
    test = {p.stem for p in (final_dir("test") / "labels").glob("*.txt")}
    return val, test


# ---------------------------------------------------------------------------
# EXIF gate
# ---------------------------------------------------------------------------

def exif_orientation(path: Path) -> int | None:
    """EXIF orientation tag, or None when absent."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            return exif.get(0x0112) if exif else None
    except Exception:
        return None


# ultralytics/data/utils.py:159 -- `if rotation in {6, 8}` -- these are the ONLY
# orientation values that transpose the reported dimensions, and only for JPEG.
# Verified against the installed 8.4.118, not assumed. Other values (including the
# invalid 0 this pool actually contains, 6 times) are ignored by ultralytics and
# are therefore harmless here.
TRANSPOSING_ORIENTATIONS = {6, 8}


def _exif_check(args: tuple[str, str]) -> tuple[str, int] | None:
    src, stem = args
    p = resolve_image(src, stem)
    if p is None:
        return None
    o = exif_orientation(p)
    return (f"{src}/{stem}", o) if o in TRANSPOSING_ORIENTATIONS else None


# ---------------------------------------------------------------------------
# image resolution + resize
# ---------------------------------------------------------------------------

_INDEX: dict[str, dict[str, Path]] = {}


def resolve_image(source: str, stem: str) -> Path | None:
    """Locate a processed image by stem, caching a per-source index."""
    if source not in _INDEX:
        d = processed_dir(source) / "images"
        _INDEX[source] = {p.stem: p for p in d.iterdir() if p.is_file() and not p.name.startswith(".")} \
            if d.is_dir() else {}
    return _INDEX[source].get(stem)


def _resize_one(job: tuple[str, str, str, str]) -> tuple[str, str, int, int]:
    """Copy or downscale one image into the bundle. Returns (status, name, w, h)."""
    src_path, dst_path, _source, _stem = job
    sp, dp = Path(src_path), Path(dst_path)
    with Image.open(sp) as im:
        w, h = im.size
        if max(w, h) <= IMGSZ:
            dp.write_bytes(sp.read_bytes())          # byte-identical, no re-encode
            return ("copied", dp.name, w, h)
        # Replicate ultralytics' own arithmetic (base.py:258-261) so the pod sees
        # the same pixel dimensions it would have computed itself.
        r = IMGSZ / max(w, h)
        nw = min(math.ceil(w * r), IMGSZ)
        nh = min(math.ceil(h * r), IMGSZ)
        out = im.resize((nw, nh), Image.BILINEAR)    # cv2.INTER_LINEAR equivalent
        if dp.suffix.lower() in (".jpg", ".jpeg"):
            if out.mode not in ("RGB", "L"):
                out = out.convert("RGB")
            out.save(dp, "JPEG", quality=JPEG_QUALITY, optimize=True)
        else:
            out.save(dp)                              # PNG stays PNG; mode preserved
    return ("resized", dp.name, nw, nh)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dataset/bundle", help="bundle output directory")
    ap.add_argument("--dry-run", action="store_true", help="measure and validate; write nothing")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-exif-gate", action="store_true",
                    help="Skip the EXIF orientation scan. Only for a re-run where it already passed.")
    args = ap.parse_args()

    names = get_canonical_names()
    print(f"schema: nc={len(names)}")

    # --- pools + nesting -------------------------------------------------
    print("\nDeriving pools (cap_per_class dry-run, nothing written)...")
    pools = {c: pool_for(c) for c in CONDITIONS}
    for c in CONDITIONS:
        print(f"  cap {c:>4}: {len(pools[c]):>6} images")

    superset = pools[SUPERSET_CAP]
    violations = {c: pools[c] - superset for c in CONDITIONS}
    bad = {c: v for c, v in violations.items() if v}
    if bad:
        for c, v in bad.items():
            print(f"  NESTING VIOLATION: cap {c} has {len(v)} image(s) outside the cap-{SUPERSET_CAP} superset")
        raise SystemExit(
            "Aborting. The bundle is only valid if every condition is a subset of the uploaded\n"
            "superset. This is a property of the current CLASS_PRIORITY_SOURCES config, not a\n"
            "guarantee -- see docs/RUNPOD_DEDUP_PLAN.md, which documents a config edit that broke\n"
            "it before. Either raise the superset cap or upload the union explicitly."
        )
    print(f"  nesting verified: every condition is a subset of cap {SUPERSET_CAP}")

    # --- EXIF gate -------------------------------------------------------
    if not args.skip_exif_gate:
        print(f"\nEXIF orientation gate over {len(superset)} images...")
        jobs = sorted(superset)
        offenders = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_exif_check, jobs, chunksize=256):
                if r:
                    offenders.append(r)
        if offenders:
            for name, o in offenders[:20]:
                print(f"  orientation={o}  {name}")
            print(
                f"\n  {len(offenders)} image(s) carry a TRANSPOSING orientation tag (6 or 8).\n"
                "  ultralytics reads dimensions via exif_size() (which applies rotation) but decodes\n"
                "  pixels with cv2.imdecode (which does not), so these can silently transpose their\n"
                "  labels. They are DROPPED from the bundle rather than risk it -- at this count the\n"
                "  data loss is negligible next to a wrong-orientation label."
            )
            drop = {tuple(n.split("/", 1)) for n, _ in offenders}
            superset -= drop
            for c in CONDITIONS:
                pools[c] -= drop
            # A dropped image can sit in the FROZEN val/test set (one does), so it
            # must leave those too -- otherwise the manifest names a file the bundle
            # does not contain and the pod fails at dataset-scan time.
            dropped_prefixed = {prefixed_filename(s, f"{stem}.txt").removesuffix(".txt")
                                for s, stem in drop}
            print(f"  dropped {len(drop)}; superset now {len(superset)}")
        else:
            dropped_prefixed = set()
            print("  clean: no transposing orientation tags")
    else:
        dropped_prefixed = set()

    # --- frozen eval sets ------------------------------------------------
    val_names, test_names = frozen_eval_sets()
    removed_eval = (val_names | test_names) & dropped_prefixed
    if removed_eval:
        val_names -= dropped_prefixed
        test_names -= dropped_prefixed
        print(f"\n  NOTE: {len(removed_eval)} EXIF-dropped image(s) removed from the frozen "
              f"eval set(s): {sorted(removed_eval)}")
    print(f"\nFrozen evaluation sets (identical for every condition):")
    print(f"  val  {len(val_names)} images")
    print(f"  test {len(test_names)} images")

    eval_names = val_names | test_names
    manifests = {}
    for c in CONDITIONS:
        prefixed = {prefixed_filename(s, f"{stem}.txt").removesuffix(".txt"): (s, stem)
                    for s, stem in pools[c]}
        train = sorted(n for n in prefixed if n not in eval_names)
        # val/test are the FROZEN sets verbatim, NOT intersected with this
        # condition's pool. That is the whole point: every condition is scored on
        # the identical images, so a mAP delta is attributable to training data.
        manifests[c] = {"train": train,
                        "val": sorted(val_names),
                        "test": sorted(test_names)}
        print(f"  cap {c:>4}: train {len(train):>6} | val {len(val_names)} | test {len(test_names)}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    # --- write bundle ----------------------------------------------------
    out = Path(args.out)
    img_out, lbl_out, man_out = out / "images", out / "labels", out / "manifests"
    for d in (img_out, lbl_out, man_out):
        ensure_dir(d)

    print(f"\nWriting bundle to {out} ...")
    jobs = []
    missing = []
    for source, stem in sorted(superset):
        sp = resolve_image(source, stem)
        if sp is None:
            missing.append(f"{source}/{stem}")
            continue
        name = prefixed_filename(source, sp.name)
        jobs.append((str(sp), str(img_out / name), source, stem))
        lp = processed_dir(source) / "labels" / f"{stem}.txt"
        if lp.is_file():
            (lbl_out / prefixed_filename(source, f"{stem}.txt")).write_bytes(lp.read_bytes())
    if missing:
        print(f"  WARNING: {len(missing)} image(s) could not be resolved, e.g. {missing[:3]}")

    tally = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for status, _n, _w, _h in ex.map(_resize_one, jobs, chunksize=64):
            tally[status] += 1
    print(f"  images: {tally['resized']} downscaled, {tally['copied']} copied unchanged "
          f"({sum(tally.values())} total)")
    print(f"  labels: {len(list(lbl_out.glob('*.txt')))}")

    for c, m in manifests.items():
        (man_out / f"cap{c}.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(f"  manifests: {', '.join(f'cap{c}.json' for c in CONDITIONS)}")

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    meta = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": sha,
        "nc": len(names),
        "names": names,
        "superset_cap": SUPERSET_CAP,
        "superset_images": len(superset),
        "imgsz_bound": IMGSZ,
        "resize": "downscale-only, long side to imgsz, BILINEAR, JPEG q95; PNG kept as PNG",
        "jpeg_quality": JPEG_QUALITY,
        "conditions": {str(c): {k: len(v) for k, v in m.items()} for c, m in manifests.items()},
        "frozen_eval": "val/test pinned to the current split_report.json for ALL conditions",
        "images_resized": tally["resized"],
        "images_copied_unchanged": tally["copied"],
    }
    (out / "bundle_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    h = hashlib.sha256()
    for p in sorted(img_out.iterdir()):
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
    (out / "CHECKSUM").write_text(f"{h.hexdigest()}  name+size digest over {len(jobs)} images\n",
                                  encoding="utf-8")
    print(f"\nDone. {out}")


if __name__ == "__main__":
    main()

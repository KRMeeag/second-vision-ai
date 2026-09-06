"""
Materialise one training condition on the pod from the shared bundle.

The bundle holds ONE copy of every image any condition needs (the cap-9000
superset) plus a manifest per condition. This script builds
`<root>/final_cap<N>/{train,val,test}/{images,labels}/` for one condition by
linking into that single copy, so three conditions cost one dataset on disk
rather than three.

LINKING, NOT COPYING
--------------------
Verified against the installed ultralytics 8.4.118 that links are safe here:

  * `data/base.py:167` lists images with `glob.glob(".../**/*.*", recursive=True)`,
    which matches symlinked files and follows symlinked directories.
  * `data/utils.py` `get_hash()` uses `os.stat`, which FOLLOWS links, so the label
    cache validates against real file sizes.
  * the cache path is `Path(label_files[0]).parent.with_suffix(".cache")`
    (`data/dataset.py:282`) -> `<split>/labels.cache`, a sibling of `labels/`.
    Per-condition roots differ, so caches never collide between conditions.

Three rules follow, and breaking any of them is silent rather than loud:

  1. **Link per FILE, never the `images/` directory.** A directory link shared
     across conditions makes every condition see the same file list, and the
     ablation quietly collapses into three runs of the same data.
  2. `<split>/` must be a REAL, writable directory — `labels.cache` is written
     there, not into `labels/`.
  3. Hardlinks are the default. They need the bundle and the output on one
     filesystem, which is the normal RunPod layout (both under the volume), and
     they survive tooling that resolves links oddly. `--link-mode symlink` works
     across filesystems; `copy` is the last resort and costs full disk.

A single link is created and read back before the other ~40,000 are attempted,
so a filesystem that cannot link fails in one second rather than halfway through.

Usage
-----
    python3 scripts/train/materialize_condition.py --bundle /workspace/bundle --cap 4500
    python3 scripts/train/materialize_condition.py --bundle /workspace/bundle --cap 4500 \\
        --out /workspace/data --link-mode symlink
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SPLITS = ("train", "val", "test")


def link_one(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def verify_link_mode(bundle_images: Path, out_root: Path, mode: str) -> None:
    """Create and read back ONE link before attempting tens of thousands."""
    probe_src = next(iter(sorted(bundle_images.iterdir())), None)
    if probe_src is None:
        raise SystemExit(f"bundle has no images: {bundle_images}")
    out_root.mkdir(parents=True, exist_ok=True)
    probe_dst = out_root / f".linkprobe{probe_src.suffix}"
    try:
        link_one(probe_src, probe_dst, mode)
        if probe_dst.stat().st_size != probe_src.stat().st_size:
            raise OSError("linked file reports a different size than its target")
    except OSError as exc:
        raise SystemExit(
            f"link-mode '{mode}' does not work here: {exc}\n"
            "Hardlinks need the bundle and output on ONE filesystem. On RunPod keep both\n"
            "under the network volume, or re-run with --link-mode symlink (or copy)."
        )
    finally:
        if probe_dst.exists() or probe_dst.is_symlink():
            probe_dst.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="bundle dir (images/, labels/, manifests/)")
    ap.add_argument("--cap", type=int, required=True, help="condition, e.g. 4500")
    ap.add_argument("--out", default=None, help="parent for final_cap<N>/ (default: alongside the bundle)")
    ap.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    ap.add_argument("--force", action="store_true", help="replace an existing output dir")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    b_img, b_lbl = bundle / "images", bundle / "labels"
    manifest_path = bundle / "manifests" / f"cap{args.cap}.json"
    for p in (b_img, b_lbl, manifest_path):
        if not p.exists():
            raise SystemExit(f"missing from bundle: {p}")

    meta = json.loads((bundle / "bundle_meta.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_root = Path(args.out or bundle.parent) / f"final_cap{args.cap}"

    if out_root.exists():
        if not args.force:
            raise SystemExit(f"{out_root} exists. Re-run with --force to replace it.")
        shutil.rmtree(out_root)

    print(f"bundle : {bundle}  (nc={meta['nc']}, {meta['superset_images']} superset images)")
    print(f"output : {out_root}")
    print(f"mode   : {args.link_mode}")
    verify_link_mode(b_img, out_root, args.link_mode)
    print("  link probe OK")

    # An index by stem: the manifest stores stems, the bundle keeps real extensions
    # (PNG stayed PNG -- never transcoded, because dedup/split records are filename-keyed).
    by_stem = {p.stem: p for p in b_img.iterdir() if p.is_file() and not p.name.startswith(".")}

    counts, missing = {}, []
    for split in SPLITS:
        names = manifest[split]
        img_dir, lbl_dir = out_root / split / "images", out_root / split / "labels"
        # Real directories, not links -- ultralytics writes <split>/labels.cache here.
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for name in names:
            src_img = by_stem.get(name)
            src_lbl = b_lbl / f"{name}.txt"
            if src_img is None or not src_lbl.is_file():
                missing.append(name)
                continue
            link_one(src_img, img_dir / src_img.name, args.link_mode)
            link_one(src_lbl, lbl_dir / f"{name}.txt", args.link_mode)
            n += 1
        counts[split] = n
        print(f"  {split:<6} {n:>6} images")

    if missing:
        print(f"\n  WARNING: {len(missing)} manifest entr(ies) missing from the bundle, "
              f"e.g. {missing[:3]}")

    # data.yaml written here rather than by generate_yaml.py, which is hardwired to
    # dataset/final/. No `path:` key -- DEC-066 verified ultralytics resolves the
    # splits relative to the yaml's own directory ONLY when `path` is absent, which
    # is what makes the bundle relocatable between machines.
    names = meta["names"]
    yaml_lines = [
        "# Generated by scripts/train/materialize_condition.py",
        f"# Condition: cap{args.cap}   bundle git sha: {meta.get('git_sha', 'unknown')}",
        "# val/test are the FROZEN evaluation sets, identical across every condition.",
        "# No 'path' key on purpose -- see DEC-066.",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {meta['nc']}",
        "names:",
        *[f"  {i}: {n}" for i, n in enumerate(names)],
    ]
    (out_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"\nwrote {out_root / 'data.yaml'} (nc={meta['nc']})")
    print("Next: validate it from an unrelated CWD before training --")
    print("  cd /tmp && python3 -c \"from ultralytics.data.utils import check_det_dataset; "
          f"print(check_det_dataset('{out_root / 'data.yaml'}')['nc'])\"")


if __name__ == "__main__":
    main()

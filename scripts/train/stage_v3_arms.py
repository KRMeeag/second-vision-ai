"""Package the three v3 arms for the pod as symlink+txt dataset roots.

RunPod's MooseFS quota counts every DIRECTORY ENTRY, not unique inodes (DEC-142) --
proven by test: 2,179 hardlinks totalling 201 MiB of entry-counted size took the
volume's writable headroom from 256 MiB to 0. So materialising 43,170 image entries
per arm would cost ~4 GiB per arm, not the ~0 that hardlinks cost on a normal
filesystem.

Instead each arm becomes:

    /workspace/<arm>/
        images -> /workspace/bundle/images      ONE symlink, one entry
        labels/                                  43,170 real .txt files, ~160 MB
        train.txt  val.txt  test.txt             absolute image paths
        data.yaml                                nc=13, no 'path' key (DEC-066)

Ultralytics derives a label path by replacing the last "/images/" in the image path
with "/labels/", so an entry of /workspace/<arm>/images/foo.jpg resolves its label at
/workspace/<arm>/labels/foo.txt -- which is why the symlink must be named `images`
and the labels must be flat. Split filenames were verified collision-free.

DEC-135 forbade symlinks because `tar czf` archived the LINKS and they resolved to
non-existent /Users/luna/... paths on the pod, silently calibrating INT8 on zero
images. That does not apply here: the symlink is created ON the pod, points at an
absolute path on the same volume, and is never archived. The tarballs this script
writes contain labels and text files only -- no symlink, no image.

Usage:
    python3 scripts/train/stage_v3_arms.py            # dry run
    python3 scripts/train/stage_v3_arms.py --apply
"""
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

APPLY = "--apply" in sys.argv
ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "dataset" / "bundle" / "images"
MANIFEST = ROOT / "dataset" / "bundle" / "manifests" / "cap4500.json"
OUT = ROOT / "dataset" / "staging"

REMOTE_ROOT = "/workspace"          # where the arms will live on the pod
REMOTE_IMAGES = "/workspace/bundle/images"

# Run order is PERMISSIVE -> LOOSE -> STRICT: strict is gated on what the first two
# show in Comet, so it must not be first.
ARMS = [
    ("v3p", "final_cap4500_nc13_v3p", "PERMISSIVE"),
    ("v3l", "final_cap4500_nc13_v3l", "LOOSE"),
    ("v3s", "final_cap4500_nc13_v3s", "STRICT"),
]


def main():
    man = json.loads(MANIFEST.read_text())
    splits = {s: man[s] for s in ("train", "val", "test")}

    # entry -> actual filename in bundle/ (extensions vary: jpg/jpeg/png)
    idx = {}
    for p in BUNDLE.iterdir():
        if p.is_file():
            idx[p.name[:p.name.rfind(p.suffix)] if p.suffix else p.name] = p.name
    missing = [e for ss in splits.values() for e in ss if e not in idx]
    if missing:
        sys.exit(f"ABORT: {len(missing)} manifest entries have no image in bundle/")

    if APPLY:
        OUT.mkdir(parents=True, exist_ok=True)

    print(f"{'DRY RUN - nothing written' if not APPLY else 'WRITING'}\n")
    for short, tree, label in ARMS:
        src = ROOT / "dataset" / tree
        if not src.is_dir():
            sys.exit(f"ABORT: {src} not found")
        names = [n for n in (src / "data.yaml").read_text().splitlines()]
        nc_line = next(l for l in names if l.startswith("nc:"))

        # data.yaml for the pod: text-file splits, still no 'path' key (DEC-066)
        name_lines = []
        seen = False
        for l in names:
            if l.startswith("names:"):
                seen = True
            if seen:
                name_lines.append(l)
        yaml_txt = (
            f"# v3 {label} arm, staged for the pod by scripts/train/stage_v3_arms.py\n"
            f"# Splits are TEXT FILES of absolute image paths, not directories: the\n"
            f"# MooseFS quota counts directory entries, so materialising images per-arm\n"
            f"# would cost ~4 GiB each (DEC-142).\n"
            f"# No 'path' key on purpose (DEC-066).\n"
            f"train: train.txt\nval: val.txt\ntest: test.txt\n{nc_line}\n"
            + "\n".join(name_lines[1:]) + "\n"
        )

        counts = Counter()
        label_files = []
        for split, entries in splits.items():
            for e in entries:
                lp = src / split / "labels" / f"{e}.txt"
                if not lp.is_file():
                    sys.exit(f"ABORT: missing label {lp}")
                label_files.append((e, lp))
                counts[split] += 1

        total_bytes = sum(lp.stat().st_size for _, lp in label_files)
        print(f"  {label:<11} {short}")
        print(f"    labels      {len(label_files):,} files, {total_bytes/1e6:.1f} MB")
        print(f"    splits      train={counts['train']:,} val={counts['val']:,} "
              f"test={counts['test']:,}")

        if not APPLY:
            continue

        tgz = OUT / f"{short}.tar.gz"
        with tarfile.open(tgz, "w:gz") as tf:
            for e, lp in label_files:
                tf.add(lp, arcname=f"{short}/labels/{e}.txt")
            for split, entries in splits.items():
                listing = "\n".join(f"{REMOTE_ROOT}/{short}/images/{idx[e]}"
                                    for e in entries) + "\n"
                info = tarfile.TarInfo(f"{short}/{split}.txt")
                data = listing.encode()
                info.size = len(data)
                import io
                tf.addfile(info, io.BytesIO(data))
            info = tarfile.TarInfo(f"{short}/data.yaml")
            data = yaml_txt.encode()
            info.size = len(data)
            import io
            tf.addfile(info, io.BytesIO(data))
        print(f"    -> {tgz.relative_to(ROOT)}  ({tgz.stat().st_size/1e6:.1f} MB)")

    if APPLY:
        print(f"\n  On the pod, for each arm:")
        print(f"    tar xzf <arm>.tar.gz -C {REMOTE_ROOT}")
        print(f"    ln -s {REMOTE_IMAGES} {REMOTE_ROOT}/<arm>/images")
        print(f"  The symlink is created THERE, never archived (DEC-135).")
    else:
        print("\n  Re-run with --apply to write the tarballs.")


if __name__ == "__main__":
    main()

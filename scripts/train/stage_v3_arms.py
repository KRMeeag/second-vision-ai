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
import io
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config_loader import CANONICAL_NAMES, EXPECTED_NC  # noqa: E402

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
        # data.yaml for the pod: text-file splits, still no 'path' key (DEC-066).
        #
        # PARSE, then RE-EMIT. An earlier version of this built the output by
        # slicing the source file's lines and dropped the `names:` key itself,
        # emitting `nc: 13` followed by a bare indented block -- not valid YAML at
        # all. Every arm staged that way was unloadable, and nothing downstream
        # could have trained on it. Round-tripping through the parser makes that
        # class of error impossible to ship: the emitted text is re-parsed and
        # checked against the repo schema below before it reaches a tarball.
        spec = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
        src_names = [spec["names"][i] for i in sorted(spec["names"])]
        if src_names != CANONICAL_NAMES or spec["nc"] != EXPECTED_NC:
            sys.exit(
                f"ABORT: {src}/data.yaml disagrees with config/classes.yaml\n"
                f"  tree : nc={spec.get('nc')} {src_names}\n"
                f"  repo : nc={EXPECTED_NC} {CANONICAL_NAMES}"
            )

        header = (
            f"# v3 {label} arm, staged for the pod by scripts/train/stage_v3_arms.py\n"
            f"# Splits are TEXT FILES of absolute image paths, not directories: the\n"
            f"# MooseFS quota counts directory entries, so materialising images per-arm\n"
            f"# would cost ~4 GiB each (DEC-142).\n"
            f"# No 'path' key on purpose (DEC-066).\n"
        )
        body = yaml.safe_dump(
            {"train": "train.txt", "val": "val.txt", "test": "test.txt",
             "nc": EXPECTED_NC,
             "names": {i: n for i, n in enumerate(CANONICAL_NAMES)}},
            sort_keys=False, default_flow_style=False, allow_unicode=True,
        )
        yaml_txt = header + body

        # Re-parse what we are about to ship, and require it to agree with the
        # repo -- the same check train.py runs on the pod, run here where fixing
        # it is free.
        back = yaml.safe_load(yaml_txt)
        back_names = [back["names"][i] for i in sorted(back["names"])]
        assert back.get("nc") == EXPECTED_NC and back_names == CANONICAL_NAMES, \
            f"emitted data.yaml failed its own round-trip check: {back}"
        assert "path" not in back, "emitted data.yaml must not carry a 'path' key (DEC-066)"

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
                tf.addfile(info, io.BytesIO(data))
            info = tarfile.TarInfo(f"{short}/data.yaml")
            data = yaml_txt.encode()
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        # ---- verify the ARCHIVE, not the variables that built it -------------
        # The failure this catches is a tarball whose data.yaml is unloadable or
        # whose label count silently fell short -- neither is visible from the
        # writing code, and both only surface on the pod.
        with tarfile.open(tgz, "r:gz") as tf:
            members = tf.getnames()
            got_yaml = yaml.safe_load(tf.extractfile(f"{short}/data.yaml").read())
            got_names = [got_yaml["names"][i] for i in sorted(got_yaml["names"])]
            assert got_names == CANONICAL_NAMES and got_yaml["nc"] == EXPECTED_NC, \
                f"{tgz}: shipped data.yaml does not match the repo schema"
            n_lab = sum(1 for m in members if m.startswith(f"{short}/labels/"))
            assert n_lab == len(label_files), \
                f"{tgz}: {n_lab} labels in archive, {len(label_files)} expected"
            for split, entries in splits.items():
                lines = tf.extractfile(f"{short}/{split}.txt").read().decode().splitlines()
                assert len(lines) == len(entries), \
                    f"{tgz}: {split}.txt has {len(lines)} lines, {len(entries)} expected"
                assert all(l.startswith(f"{REMOTE_ROOT}/{short}/images/") for l in lines), \
                    f"{tgz}: {split}.txt contains a path outside {REMOTE_ROOT}/{short}/images/"
        print(f"    -> {tgz.relative_to(ROOT)}  ({tgz.stat().st_size/1e6:.1f} MB)")
        print(f"    verified    data.yaml parses at nc={EXPECTED_NC}, "
              f"{n_lab:,} labels, all split lines absolute")

    if APPLY:
        print(f"\n  On the pod, for each arm:")
        print(f"    tar xzf <arm>.tar.gz -C {REMOTE_ROOT}")
        print(f"    ln -s {REMOTE_IMAGES} {REMOTE_ROOT}/<arm>/images")
        print(f"  The symlink is created THERE, never archived (DEC-135).")
    else:
        print("\n  Re-run with --apply to write the tarballs.")


if __name__ == "__main__":
    main()

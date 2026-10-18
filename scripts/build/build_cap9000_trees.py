"""
Build the cap9000 training trees from the frozen manifest + the authoritative
dataset/processed/<source>/labels/ (15-class) trees.

Produces two trees that differ ONLY in their label schema:

    dataset/final_cap9000_nc14/   14 classes - Shelf dropped (DEC-126 schema,
                                  identical to the trained final_v2b)
    dataset/final_cap9000_nc13/   13 classes - Shelf AND Pole dropped

Both use the SAME frozen val/test splits as cap4500, so every result stays
comparable to v1 and v2. Only the training set grows: 30,123 -> 41,467.

Design notes, each from a decision this project already paid for:

  * Images are HARDLINKS into dataset/bundle/images/, never symlinks.
    DEC-135: final_v2b/val/images holds absolute symlinks into dataset/final/,
    `tar czf` archived the links, and on the pod every one resolved to a
    /Users/luna/... path that does not exist - which would have calibrated INT8
    on ZERO images. Hardlinks survive tar, cost no disk, and cannot dangle.

  * data.yaml carries NO 'path' key. DEC-066: ultralytics resolves train/val/test
    relative to the yaml's own directory when 'path' is absent.

  * Empty label files ARE written for background images, matching final_v2b
    (888 train / 216 test backgrounds there).

  * The script ABORTS if a target tree exists. It never deletes or overwrites.

Usage:
    python3 scripts/build/build_cap9000_trees.py            # dry run
    python3 scripts/build/build_cap9000_trees.py --apply
"""
import json, os, sys
from pathlib import Path
from collections import Counter

APPLY = "--apply" in sys.argv
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "dataset" / "processed"
BUNDLE = ROOT / "dataset" / "bundle" / "images"
MANIFEST = ROOT / "dataset" / "bundle" / "manifests" / "cap9000.json"
REF_V2B = ROOT / "dataset" / "final_v2b"          # for the correctness check

NAMES15 = ["Person","Vehicle","Motorcycle","Pole","Animals","Shelf","Doors","Chairs",
           "Tables","Tricycle","Potholes","Trash Bins","Bicycle","Stairs","Bench"]
SHELF, POLE = 5, 3

TREES = {
    "final_cap9000_nc14": {"drop": {SHELF},       "nc": 14},
    "final_cap9000_nc13": {"drop": {SHELF, POLE}, "nc": 13},
}

SOURCES = sorted((p.name for p in PROC.iterdir() if p.is_dir()), key=len, reverse=True)

def split_source(entry):
    """Manifest entries carry no extension and may contain dots. Never use .stem."""
    for s in SOURCES:
        if entry.startswith(s + "__"):
            return s, entry[len(s) + 2:]
    return None, None

def remap(lines, drop):
    """15-class ids -> the target schema. Returns [] for a background image."""
    out = []
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        cid = int(parts[0])
        if cid in drop:
            continue
        parts[0] = str(cid - sum(1 for d in drop if d < cid))
        out.append(" ".join(parts))
    return out

def names_for(drop):
    return [n for i, n in enumerate(NAMES15) if i not in drop]

def main():
    man = json.loads(MANIFEST.read_text())
    index = {p.name.rsplit(".", 1)[0] if "." in p.name else p.name: p
             for p in BUNDLE.iterdir() if p.is_file()}
    # bundle filenames keep their extension; manifest entries do not
    index = {}
    for p in BUNDLE.iterdir():
        if p.is_file():
            index.setdefault(p.stem if p.suffix else p.name, p)
            index.setdefault(p.name[:p.name.rfind(p.suffix)] if p.suffix else p.name, p)

    for tname, cfg in TREES.items():
        dest = ROOT / "dataset" / tname
        if dest.exists():
            sys.exit(f"ABORT: {dest} already exists. This script never overwrites.")

    stats = {t: Counter() for t in TREES}
    missing_img, missing_lab = [], []

    for tname, cfg in TREES.items():
        drop, nc = cfg["drop"], cfg["nc"]
        dest = ROOT / "dataset" / tname
        for split in ("train", "val", "test"):
            entries = man[split]
            if APPLY:
                (dest / split / "images").mkdir(parents=True, exist_ok=True)
                (dest / split / "labels").mkdir(parents=True, exist_ok=True)
            for e in entries:
                src_img = index.get(e)
                if src_img is None:
                    missing_img.append(e); continue
                s, n = split_source(e)
                lab = PROC / s / "labels" / f"{n}.txt" if s else None
                if lab is None or not lab.exists():
                    missing_lab.append(e); continue
                lines = [l for l in lab.read_text().splitlines() if l.strip()]
                kept = remap(lines, drop)
                stats[tname][f"{split}_images"] += 1
                stats[tname]["boxes"] += len(kept)
                if not kept:
                    stats[tname][f"{split}_background"] += 1
                if APPLY:
                    di = dest / split / "images" / src_img.name
                    if not di.exists():
                        os.link(src_img, di)           # HARDLINK, not symlink
                    (dest / split / "labels" / f"{e}.txt").write_text(
                        ("\n".join(kept) + "\n") if kept else "")
        if APPLY:
            names = names_for(drop)
            (dest / "data.yaml").write_text(
                f"# cap9000 tree, nc={nc}. Built by scripts/build/build_cap9000_trees.py\n"
                "# No 'path' key on purpose - ultralytics resolves train/val/test relative to\n"
                "# THIS file's own directory when 'path' is absent (DEC-066).\n"
                "train: train/images\nval: val/images\ntest: test/images\n"
                f"nc: {nc}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))

    print(f"{'DRY RUN - nothing written' if not APPLY else 'APPLIED'}\n")
    for t, c in stats.items():
        nc = TREES[t]["nc"]
        print(f"  {t}  (nc={nc})")
        for sp in ("train", "val", "test"):
            print(f"    {sp:<6} {c[f'{sp}_images']:>7,} images   "
                  f"{c[f'{sp}_background']:>5,} background")
        print(f"    boxes  {c['boxes']:>7,}")
        print(f"    names  {', '.join(names_for(TREES[t]['drop']))}\n")
    if missing_img: print(f"  !! {len(missing_img):,} images not found in bundle/")
    if missing_lab: print(f"  !! {len(missing_lab):,} labels not found in processed/")
    if not missing_img and not missing_lab:
        print("  every manifest entry resolved to an image and a label.")

if __name__ == "__main__":
    main()

"""Generate notebooks/fiftyone_v3_schema_check.ipynb.

The notebook that has to pass before anyone trusts the nc=13 labels. Dropping
Pole (id 3) shifted TEN classes down by one, each onto its neighbour's former id
-- the DEC-107 signature, where a half-applied remap reads as ordinary
mislabelling rather than a schema fault.

READ-ONLY with respect to the dataset: writes one .ipynb and nothing else.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "fiftyone_v3_schema_check.ipynb"

CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": t.strip("\n").splitlines(True)})


def code(t):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": t.strip("\n").splitlines(True)})


md(r"""
**Notebook version: 1** — Claude will state the version number after editing any cell in this notebook.

# fiftyone_v3_schema_check.ipynb — prove the nc=14 → nc=13 remap is correct

Dropping **Pole (id 3)** shifted **ten** classes down by one:

```
Person      0 -> 0      Tables       7 -> 6
Vehicle     1 -> 1      Tricycle     8 -> 7
Motorcycle  2 -> 2      Potholes     9 -> 8
Pole        3 -> DROPPED  Trash Bins 10 -> 9
Animals     4 -> 3      Bicycle     11 -> 10
Doors       5 -> 4      Stairs      12 -> 11
Chairs      6 -> 5      Bench       13 -> 12
```

Every shifted class lands on its **neighbour's former id**. That is exactly the
DEC-107 failure mode: if the remap were applied to the ids but not the names (or
half the files), a Doors box reading as Chairs looks like an ordinary annotation
mistake, not a schema fault. Nothing crashes. You find out at write-up time.

So this notebook checks three independent things, weakest to strongest:

| # | check | catches |
|---|---|---|
| 1 | per-class box counts, nc=14 tree vs nc=13 arms | wholesale loss or duplication |
| 2 | **box-by-box name + geometry** on the same images | a shifted id that kept its count |
| 3 | **your eyes** on rendered samples per class | everything the first two cannot |

Check 2 is the one that matters. Counts can match while names are wrong — that is
precisely what makes this failure quiet. Check 3 exists because you have been
burned before and a number is not a picture.

### Reference

`dataset/reports/v3_renumber_map.json` holds the id map and per-class counts.
`config/classes.yaml` and `scripts/utils/config_loader.py` are the new authority
(nc=13); `dataset/final_v2b/` is the nc=14 tree the arms descend from.
""")

code(r'''
# Imports
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "config").is_dir() and (parent / "AGENTS.md").is_file():
            return parent
    raise RuntimeError("Could not locate repo root from notebook cwd.")


REPO_ROOT = _find_repo_root(Path.cwd())
sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.config_loader import EXPECTED_NC, CANONICAL_NAMES

# The OLD schema is deliberately hardcoded, not read from the repo: the repo now
# says nc=13, so there is nowhere left to read nc=14 from. This list is the thing
# being checked against.
OLD_NAMES = ["Person", "Vehicle", "Motorcycle", "Pole", "Animals", "Doors",
             "Chairs", "Tables", "Tricycle", "Potholes", "Trash Bins",
             "Bicycle", "Stairs", "Bench"]
NEW_NAMES = list(CANONICAL_NAMES)

OLD_TREE = REPO_ROOT / "dataset" / "final_v2b"               # nc=14
ARMS = {"v3s": "STRICT", "v3p": "PERMISSIVE", "v3l": "LOOSE"}
ARM_TREE = {a: REPO_ROOT / "dataset" / f"final_cap4500_nc13_{a}" for a in ARMS}

print(f"repo schema : nc={EXPECTED_NC}  {NEW_NAMES}")
print(f"old schema  : nc={len(OLD_NAMES)}  {OLD_NAMES}")
assert EXPECTED_NC == 13, "config_loader still says nc != 13"
assert "Pole" not in NEW_NAMES, "Pole is still in the canonical names"
''')

code(r'''
# ---- The id map, printed from the two name lists rather than retyped ----

print(f"{'class':<12}{'old id':>7}{'new id':>8}   shift")
for n in OLD_NAMES:
    o = OLD_NAMES.index(n)
    if n in NEW_NAMES:
        d = NEW_NAMES.index(n)
        print(f"{n:<12}{o:>7}{d:>8}   {'-1  SHIFTED' if d != o else ' 0'}")
    else:
        print(f"{n:<12}{o:>7}{'--':>8}   DROPPED")

shifted = [n for n in NEW_NAMES if NEW_NAMES.index(n) != OLD_NAMES.index(n)]
print(f"\n{len(shifted)} classes shifted: {shifted}")
''')

code(r'''
# ---- CHECK 1: per-class box counts, old tree vs each arm ----
#
# The five classes with NO recovery (no COCO class maps to them) must be
# IDENTICAL across v2b and all three arms. If a remap had gone wrong, those are
# the first numbers to move -- they have no other reason to change.

def counts(tree: Path, names: list[str]) -> Counter:
    c = Counter()
    for split in ("train", "val", "test"):
        d = tree / split / "labels"
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix != ".txt":
                continue
            for ln in f.read_text().splitlines():
                p = ln.split()
                if len(p) == 5:
                    c[names[int(p[0])]] += 1
    return c


old = counts(OLD_TREE, OLD_NAMES)
new = {a: counts(t, NEW_NAMES) for a, t in ARM_TREE.items()}

NO_RECOVERY = ["Doors", "Tricycle", "Potholes", "Trash Bins", "Stairs"]

print(f"{'class':<12}{'v2b(nc14)':>11}{'v3s':>9}{'v3p':>9}{'v3l':>9}   note")
ok = True
for n in NEW_NAMES:
    row = f"{n:<12}{old[n]:>11,}{new['v3s'][n]:>9,}{new['v3p'][n]:>9,}{new['v3l'][n]:>9,}"
    if n in NO_RECOVERY:
        same = old[n] == new['v3s'][n] == new['v3p'][n] == new['v3l'][n]
        ok &= same
        print(f"{row}   {'IDENTICAL as required' if same else '*** MOVED - INVESTIGATE ***'}")
    else:
        grew = new['v3s'][n] >= old[n]
        ok &= grew
        print(f"{row}   {'grew (recovery)' if grew else '*** SHRANK - INVESTIGATE ***'}")
print(f"\nPole in v2b: {old['Pole']:,} boxes, dropped entirely")
print("CHECK 1 PASS" if ok else "CHECK 1 FAIL")
''')

code(r'''
# ---- CHECK 2: box-by-box name and geometry, same images ----
#
# This is the check that actually catches a shifted id. For a sample of images,
# take the nc=14 labels, drop the Pole boxes, and require the result to equal the
# arm's ORIGINAL boxes exactly -- same class NAME, same coordinates, same order.
#
# `new_boxes.json` records how many boxes were original per image; everything
# after that index is a recovered box and is excluded here, because this check is
# about the remap, not about recovery.

ARM = "v3s"          # strict = originals + the fewest recovered boxes
N_IMAGES = 1500      # raise for more confidence; 1,500 takes ~15 s

def load(p: Path, names: list[str]):
    out = []
    for ln in p.read_text().splitlines():
        q = ln.split()
        if len(q) == 5:
            out.append((names[int(q[0])], tuple(round(float(v), 6) for v in q[1:])))
    return out


tree = ARM_TREE[ARM]
orig_count = json.loads((tree / "new_boxes.json").read_text())["original_box_count"]
random.seed(17)

checked = mismatches = pole_dropped = 0
examples = []
for split in ("train", "val", "test"):
    d = tree / split / "labels"
    files = sorted(p.name for p in d.iterdir() if p.suffix == ".txt")
    for fn in random.sample(files, min(N_IMAGES // 3, len(files))):
        o = OLD_TREE / split / "labels" / fn
        if not o.is_file():
            continue
        old_boxes = load(o, OLD_NAMES)
        pole_dropped += sum(1 for c, _ in old_boxes if c == "Pole")
        expect = [b for b in old_boxes if b[0] != "Pole"]

        entry = fn[:-4]
        got = load(d / fn, NEW_NAMES)
        n0 = orig_count.get(entry)
        if n0 is not None:
            got = got[:n0]                      # strip recovered boxes

        checked += len(expect)
        if expect != got:
            mismatches += 1
            if len(examples) < 5:
                examples.append((entry, expect[:4], got[:4]))

print(f"  arm                      : {ARM} ({ARMS[ARM]})")
print(f"  original boxes compared  : {checked:,}")
print(f"  Pole boxes dropped       : {pole_dropped:,}")
print(f"  images with a mismatch   : {mismatches}")
for e, a, b in examples:
    print(f"\n    {e}\n      v2b(nc14): {a}\n      v3 (nc13): {b}")
print("\nCHECK 2 PASS - every surviving box kept its class NAME and geometry"
      if mismatches == 0 else "\nCHECK 2 FAIL - do not train on this")
''')

md(r"""
## CHECK 3 — look at them

Counts and name-equality both pass without anyone having seen a box. The remaining
failure they cannot catch is one where the *source* labels were already wrong, or
where some upstream step mislabelled consistently.

The cell below renders a grid per class. **For each of the ten shifted classes,
confirm the pictures match the title.** `Animals` (now id 3, formerly Pole's
neighbour) is the one to look at hardest — if any remap were half-applied, Pole
boxes would surface here as `Animals`, and a street light in an Animals grid is
instantly obvious in a way a number never is.
""")

code(r'''
# ---- CHECK 3: render samples per class ----
#
# Crops are centred on the box with context, titled with the class the LABEL
# claims. Street lights appearing under "Animals" would mean Pole boxes survived
# the drop and inherited id 3.

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ARM = "v3s"
CLASSES = None      # None = every class; or e.g. ["Animals", "Doors", "Chairs"]
PER_CLASS = 8
SEED = 11

tree = ARM_TREE[ARM]
by_class = defaultdict(list)
for split in ("train", "val", "test"):
    d = tree / split / "labels"
    imgs = tree / split / "images"
    for f in d.iterdir():
        if f.suffix != ".txt":
            continue
        for ln in f.read_text().splitlines():
            p = ln.split()
            if len(p) != 5:
                continue
            by_class[NEW_NAMES[int(p[0])]].append(
                (imgs, f.stem, [float(v) for v in p[1:]]))

targets = CLASSES or NEW_NAMES
random.seed(SEED)
for cname in targets:
    pool = by_class.get(cname, [])
    if not pool:
        print(f"{cname}: no boxes"); continue
    sel = random.sample(pool, min(PER_CLASS, len(pool)))
    fig, axes = plt.subplots(1, len(sel), figsize=(2.2 * len(sel), 2.6), squeeze=False)
    fig.patch.set_facecolor("#121214")
    fig.suptitle(f"{cname}   (new id {NEW_NAMES.index(cname)}"
                 + (f", was {OLD_NAMES.index(cname)}" if cname in OLD_NAMES
                    and OLD_NAMES.index(cname) != NEW_NAMES.index(cname) else "")
                 + f")   n={len(pool):,}",
                 color="#e8e8e8", fontsize=11)
    for ax, (imgs, stem, box) in zip(axes.ravel(), sel):
        ax.axis("off")
        hit = next((p for p in imgs.iterdir() if p.stem == stem), None)
        if hit is None:
            continue
        im = Image.open(hit).convert("RGB")
        W, H = im.size
        cx, cy, w, h = box
        px, py, pw, ph = cx * W, cy * H, w * W, h * H
        m = max(pw, ph) * 1.8 + 24
        L, T = max(0, px - m / 2), max(0, py - m / 2)
        R, B = min(W, px + m / 2), min(H, py + m / 2)
        ax.imshow(im.crop((int(L), int(T), int(R), int(B))))
        ax.add_patch(Rectangle((px - pw / 2 - L, py - ph / 2 - T), pw, ph,
                               fill=False, lw=2.2, edgecolor="#ff3c3c"))
    plt.tight_layout()
    plt.show()
''')

code(r'''
# ---- Optional: the same thing in the FiftyOne App ----
#
# Useful for filtering to one class and paging through hundreds rather than eight.

import fiftyone as fo

ARM = "v3s"
MAX_PER_SPLIT = 4000          # keep the App snappy; None for everything

name = f"v3_schema_check_{ARM}"
if fo.dataset_exists(name):
    fo.delete_dataset(name)
ds = fo.Dataset(name)
ds.classes["ground_truth"] = list(NEW_NAMES)

tree = ARM_TREE[ARM]
samples = []
for split in ("train", "val", "test"):
    imgs, labs = tree / split / "images", tree / split / "labels"
    paths = sorted(p for p in imgs.iterdir() if p.is_file())
    if MAX_PER_SPLIT:
        paths = paths[:MAX_PER_SPLIT]
    for ip in paths:
        lp = labs / f"{ip.stem}.txt"
        if not lp.is_file():
            continue
        dets = []
        for ln in lp.read_text().splitlines():
            p = ln.split()
            if len(p) != 5:
                continue
            cid = int(p[0])
            cx, cy, w, h = (float(v) for v in p[1:])
            dets.append(fo.Detection(label=NEW_NAMES[cid],
                                     bounding_box=[cx - w / 2, cy - h / 2, w, h],
                                     class_id=cid))
        s = fo.Sample(filepath=str(ip))
        s["split"] = split
        s["ground_truth"] = fo.Detections(detections=dets)
        samples.append(s)
ds.add_samples(samples)
ds.persistent = False
for f in ("split", "ground_truth.detections.label", "ground_truth.detections.class_id"):
    try:
        ds.create_index(f)
    except Exception:
        pass
print(f"loaded {len(ds):,} samples")
print("In the App: filter ground_truth.label = Animals, and confirm no street lights.")
session = fo.launch_app(ds, auto=False)
session
''')

code(r'''
fo.close_app()
print("App closed.")
''')

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT.relative_to(ROOT)}  ({len(CELLS)} cells)")

"""Generate notebooks/fiftyone_v3_source_check.ipynb.

The PER-SOURCE companion to fiftyone_v3_schema_check.ipynb.

That notebook checks the nc=14 -> nc=13 remap dataset-wide. Dataset-wide totals are
the wrong resolution for the failure that actually happens: an index shift applied to
one source's conversion and not another's moves an ENTIRE source dataset onto the
wrong class, and the global total barely moves because the boxes are still there and
still counted -- just under a neighbour's name. "The whole pothole dataset reading as
Person" is invisible in a 156,506-box total and unmissable in a 664-image source.

The convention this leans on is merge.py's `prefixed_filename()`: every file in a
merged or final tree is named `<source>__<original>`, so the source survives in the
filename and can be recovered without a manifest. That is the same recovery
fiftyone_final_dataset.ipynb uses to make `source` a sidebar-filterable field, and
box_audit.py --pool merged relies on it too (DEC-075).

READ-ONLY with respect to the dataset: writes one .ipynb and nothing else.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "fiftyone_v3_source_check.ipynb"

CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": t.strip("\n").splitlines(True)})


def code(t):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": t.strip("\n").splitlines(True)})


md(r"""
**Notebook version: 1** — Claude will state the version number after editing any cell in this notebook.

# fiftyone_v3_source_check.ipynb — prove the nc=13 remap is correct **per source dataset**

`fiftyone_v3_schema_check.ipynb` checks the remap dataset-wide. This one checks it
**one source dataset at a time**, because that is the resolution the real failure
lives at.

### Why dataset-wide totals are the wrong resolution

An index shift does not usually hit everything. It hits **one conversion path** — one
source's labels written under a stale schema while the rest were written under the new
one. When that happens:

* the boxes are all still there, so the **total is unchanged**
* they are all still counted, so the **per-class totals move only slightly** — a few
  thousand boxes sliding from one class into its neighbour, inside a 156,506-box pool
* nothing crashes, nothing warns

But an entire source dataset is now wrong. The 664-image pothole set reads as Person.
At write-up time you discover the model never learned Potholes from it.

That is what this notebook is for. Twelve sources, each checked on its own.

### The five checks, weakest to strongest

| # | check | catches |
|---|---|---|
| 1 | per-source × per-class matrix, v2b vs each arm, compared **by name** | a source's boxes moving between classes |
| 2 | **monotonicity**: recovery can only ADD, so no `(source, class)` count may shrink | the shift directly — a shift always empties one class to fill another |
| 3 | **anchor counts**: the five classes no COCO class maps to must be *byte-identical* per source | a shift that happened to balance out |
| 4 | **composition similarity** per source, v2b vs arm | a redistribution too small for check 2 to flag |
| 5 | **your eyes**, per source, with the v2b and v3 labels on the same image | everything the first four cannot |

Check 2 is the decisive one, and it is worth saying why it is so strong here.
Recovery is purely **additive** — it proposes new boxes and never deletes an existing
one. So for every `(source, class)` pair, the v3 count must be **greater than or equal
to** the v2b count. An index shift violates this immediately and unavoidably: moving
Potholes onto Trash Bins makes the source's Potholes count **fall to zero**. There is
no way for a shift to hide from a monotonicity test.

Check 3 is the sharpest instrument. **Doors, Tricycle, Potholes, Trash Bins and
Stairs** have no COCO counterpart, so the yolov8x recovery pass could never propose
them — their counts are frozen. For a pothole source that means `Potholes` must still
read **exactly** its v2b count, not merely at least it. If that exact number turns up
under a different class name, you have found the shift and you know which source.

---

### This notebook is READ-ONLY, deliberately

It builds throwaway, non-persistent FiftyOne datasets and writes nothing to disk.
`dataset/final_cap4500_nc13_v3?` are **derived** trees regenerated wholesale by
`scripts/build/build_cap4500_v3_variants.py`; anything fixed here would be destroyed
on the next build. Corrections belong upstream, in
`notebooks/fiftyone_review_processed.ipynb`, which writes to
`dataset/processed/<source>/labels_reviewed/`.
""")

code(r'''
# ---- configuration ----------------------------------------------------------
import sys, yaml, math
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from scripts.utils.config_loader import CANONICAL_NAMES, EXPECTED_NC

# The nc=14 tree v2 trained on, and the three nc=13 arms built from it.
TREES = {
    "v2b":    ROOT / "dataset" / "final_v2b",
    "strict": ROOT / "dataset" / "final_cap4500_nc13_v3s",
    "perm":   ROOT / "dataset" / "final_cap4500_nc13_v3p",
    "loose":  ROOT / "dataset" / "final_cap4500_nc13_v3l",
}
SPLITS = ("train", "val", "test")

# Classes no COCO class maps to, so the yolov8x recovery pass could never propose
# them. Their per-source counts are FROZEN -- the sharpest check in this notebook.
ANCHOR_CLASSES = ["Doors", "Tricycle", "Potholes", "Trash Bins", "Stairs"]

for k, p in TREES.items():
    print(f"{k:<7} {'OK  ' if p.is_dir() else 'MISSING'} {p}")
print(f"\nrepo schema: nc={EXPECTED_NC}  {CANONICAL_NAMES}")
''')

code(r'''
# ---- load every tree, labelled with ITS OWN names ---------------------------
#
# Each tree is read with the names from its OWN data.yaml, never the live config.
# That is the whole point: labelling a 14-class tree with the 13-class schema is
# exactly the DEC-107 bug ("every Pothole shows as Tricycle"), and doing it inside
# the notebook meant to detect that bug would make the bug invisible.
#
# fiftyone_final_dataset.ipynb makes the same choice for the same reason.

def tree_names(tree: Path) -> list[str]:
    d = yaml.safe_load((tree / "data.yaml").read_text(encoding="utf-8"))
    return [d["names"][i] for i in sorted(d["names"])]


def source_of(filename: str) -> str:
    """Recover the source key from merge.py's `<source>__<original>` prefix.

    A name with no "__" means the file was not written by merge.py. Surfaced as
    its own bucket rather than silently folded into another source.
    """
    return filename.split("__", 1)[0] if "__" in filename else "<unprefixed>"


def scan(tree: Path):
    """-> (boxes[(source, class_name)], images[source][split], names, bad_ids)"""
    names = tree_names(tree)
    boxes = Counter()
    images = defaultdict(Counter)
    bad_ids = Counter()
    for split in SPLITS:
        for f in sorted((tree / split / "labels").iterdir()):
            if f.suffix != ".txt":
                continue
            src = source_of(f.name)
            images[src][split] += 1
            images[src]["total"] += 1
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                cid = int(line.split()[0])
                if not 0 <= cid < len(names):
                    bad_ids[(src, cid)] += 1     # an id the tree's own schema cannot name
                    continue
                boxes[(src, names[cid])] += 1
    return boxes, images, names, bad_ids


DATA = {}
for k, p in TREES.items():
    DATA[k] = scan(p)
    b, im, nm, bad = DATA[k]
    print(f"{k:<7} nc={len(nm):<3} {sum(b.values()):>8,} boxes  "
          f"{sum(c['total'] for c in im.values()):>7,} images  "
          f"{len(im):>3} sources" + (f"  !! {sum(bad.values())} UNNAMEABLE IDS" if bad else ""))

# Unpacked here, not in the first check cell: every cell below needs these, and
# making cell N depend on cell N-2 having been run is how a notebook ends up
# with a NameError the first time someone runs only the part they care about.
v2b_boxes, v2b_images, v2b_names, v2b_bad = DATA["v2b"]

SOURCES = sorted(v2b_images, key=lambda s: -v2b_images[s]["total"])
ARMS = ["strict", "perm", "loose"]
print(f"\n{len(SOURCES)} sources: {', '.join(SOURCES)}")
''')

md(r"""
## Check 1 & 2 — the per-source × per-class matrix, and monotonicity

One block per source. Every class that appears in *any* tree for that source gets a
row, so nothing is hidden by being zero somewhere.

Read the **note** column:

* `DROPPED ok` — Pole went to exactly 0, as DEC-143 requires
* `frozen ok` — an anchor class whose count is identical everywhere, as it must be
* `NEW` — a class recovery added to a source that had none. Legitimate: yolov8x can
  propose a Person in a pothole dataset.
* `!! SHRANK` — **a violation.** Recovery only adds, so a count cannot fall. This is
  what an index shift looks like.
* `!! ANCHOR MOVED` — **a violation.** A frozen class changed, which is impossible
  without a schema fault.
""")

code(r'''
# ---- CHECK 1 + 2: per-source x per-class, compared BY NAME -------------------
violations = []

for src in SOURCES:
    # every class name seen for this source in ANY tree
    seen = {n for (s, n) in v2b_boxes if s == src}
    for arm in ARMS:
        seen |= {n for (s, n) in DATA[arm][0] if s == src}
    im = v2b_images[src]
    print(f"\n{src}   {im['total']:,} images "
          f"(train {im['train']:,} / val {im['val']:,} / test {im['test']:,})")
    print(f"  {'class':<13}{'v2b':>9}{'strict':>9}{'perm':>9}{'loose':>9}   note")
    print("  " + "-" * 62)

    # canonical order first, then anything only the old schema had (i.e. Pole)
    order = [n for n in CANONICAL_NAMES if n in seen] + \
            [n for n in v2b_names if n in seen and n not in CANONICAL_NAMES]
    for n in order:
        a = v2b_boxes[(src, n)]
        row = [DATA[k][0][(src, n)] for k in ARMS]
        note = ""
        if n not in CANONICAL_NAMES:                       # retired in the new schema
            if all(r == 0 for r in row):
                note = "DROPPED ok"
            else:
                note = "!! SURVIVED A DROPPED CLASS"
        elif any(r < a for r in row):
            note = "!! SHRANK — recovery can only ADD"
        elif n in ANCHOR_CLASSES:
            note = "frozen ok" if all(r == a for r in row) else "!! ANCHOR MOVED"
        elif a == 0 and any(row):
            note = "NEW (recovery added)"
        if note.startswith("!!"):
            violations.append((src, n, a, row, note))
        print(f"  {n:<13}{a:>9,}{row[0]:>9,}{row[1]:>9,}{row[2]:>9,}   {note}")

print("\n" + "=" * 70)
print(f"CHECK 1+2: {'PASS — no source moved a box between classes' if not violations else f'FAIL — {len(violations)} violation(s)'}")
for src, n, a, row, note in violations:
    print(f"  {src} / {n}: v2b={a:,} -> {row}   {note}")
''')

md(r"""
## Check 3 — the anchor classes, per source, must be *byte-identical*

Doors, Tricycle, Potholes, Trash Bins and Stairs have no COCO counterpart, so the
recovery pass could never propose one. Every count below must be **exactly** equal
across v2b and all three arms — not merely non-decreasing.

This is the check that would have caught "the pothole dataset reads as Person". If a
shift moved `Potholes` onto a neighbour, the pothole source's `Potholes` row drops to
zero while some other row gains that exact number. The second table makes that
searchable: it lists, per source, any class whose v3 count equals another class's
missing v2b count.
""")

code(r'''
# ---- CHECK 3: anchor classes are frozen, per source -------------------------
print(f"{'source':<40}" + "".join(f"{c:>13}" for c in ANCHOR_CLASSES))
print("-" * (40 + 13 * len(ANCHOR_CLASSES)))
anchor_fail = []
for src in SOURCES:
    cells = []
    for c in ANCHOR_CLASSES:
        a = v2b_boxes[(src, c)]
        row = [DATA[k][0][(src, c)] for k in ARMS]
        if a == 0 and all(r == 0 for r in row):
            cells.append("        -    ")
        elif all(r == a for r in row):
            cells.append(f"{a:>12,} ")
        else:
            cells.append(f"{a:,}->{row}".rjust(13))
            anchor_fail.append((src, c, a, row))
    print(f"{src:<40}" + "".join(cells))

print()
if anchor_fail:
    print(f"CHECK 3: FAIL — {len(anchor_fail)} anchor count(s) moved")
    for src, c, a, row in anchor_fail:
        print(f"  {src} / {c}: v2b={a:,} -> {row}")
else:
    print("CHECK 3: PASS — every anchor class is byte-identical in every source")

# ---- the smoking gun, if there is one ---------------------------------------
# A shift moves a class's boxes WHOLESALE onto a neighbour, so the number that
# vanished from one class reappears, intact, under another. Look for that.
print("\nsearching for a vanished count reappearing under another class name...")
hits = 0
for src in SOURCES:
    for arm in ARMS:
        lost = {n: v2b_boxes[(src, n)] - DATA[arm][0][(src, n)]
                for n in CANONICAL_NAMES
                if v2b_boxes[(src, n)] - DATA[arm][0][(src, n)] > 0}
        gained = {n: DATA[arm][0][(src, n)] - v2b_boxes[(src, n)]
                  for n in CANONICAL_NAMES
                  if DATA[arm][0][(src, n)] - v2b_boxes[(src, n)] > 0}
        for ln, lv in lost.items():
            for gn, gv in gained.items():
                if lv == gv and lv > 0:
                    print(f"  !! {src} [{arm}]: {ln} lost {lv:,} and {gn} gained exactly {gv:,}")
                    hits += 1
print(f"  {'none — no class handed its boxes to another' if not hits else f'{hits} suspicious pair(s) above'}")
''')

md(r"""
## Check 4 — composition similarity per source

A redistribution too small for check 2 to flag would still change the *shape* of a
source's class distribution. This compares each source's class-proportion vector in
v2b (with the dropped class removed, since it is gone by design) against the same
vector in each arm.

Cosine similarity near **1.000** means the source still looks like itself. A shift
produces a near-orthogonal vector and a visibly low number. Recovery alone moves this
only slightly, and always by adding to the classes COCO can propose.
""")

code(r'''
# ---- CHECK 4: per-source composition similarity -----------------------------
def cosine(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    num = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else float("nan")

print(f"{'source':<40}{'strict':>10}{'perm':>10}{'loose':>10}   verdict")
print("-" * 80)
LOW = 0.95
low_hits = []
for src in SOURCES:
    # the dropped class is excluded: it is gone on purpose, not redistributed
    base = {n: v2b_boxes[(src, n)] for n in CANONICAL_NAMES if v2b_boxes[(src, n)]}
    sims = []
    for arm in ARMS:
        cur = {n: DATA[arm][0][(src, n)] for n in CANONICAL_NAMES if DATA[arm][0][(src, n)]}
        sims.append(cosine(base, cur))
    verdict = "ok" if all(s >= LOW for s in sims) else "!! LOW — inspect this source"
    if verdict.startswith("!!"):
        low_hits.append(src)
    print(f"{src:<40}" + "".join(f"{s:>10.4f}" for s in sims) + f"   {verdict}")

print()
print(f"CHECK 4: {'PASS — every source still looks like itself' if not low_hits else f'INSPECT — {low_hits}'}")
print(f"(threshold {LOW}; loose is expected lowest, it adds the most boxes)")
''')

md(r"""
## Check 5 — your eyes, one source at a time

The three checks above are arithmetic on class *names*. They cannot tell you whether
the box labelled `Potholes` is actually drawn around a pothole. Only you can.

The cell below builds a FiftyOne dataset with **`source` and `split` as real,
sidebar-filterable fields** — the gap `fiftyone_review_processed.ipynb` leaves, and
the reason `fiftyone_final_dataset.ipynb` exists — and puts **two label layers on the
same image**:

* `v2b` — the nc=14 labels, named with v2b's own schema
* `v3` — the selected arm's nc=13 labels, named with its own schema

A shift is then immediate and visual: the same box carries two different class names,
one above the other, on the same image.

**Sampling is per source, not global.** A global random sample would draw ~62% of its
images from `open_images` and might show you none at all from the 375-image
`crowdhuman` set. `PER_SOURCE` images are taken from each source so every one of the
twelve is represented.

### What to look at, by source

Work down the sidebar's `source` filter. For each, the question is whether the class
name matches what the source *is*:

| source | what every box should be | the shift would show as |
|---|---|---|
| `dataset_ninja_pothole_detection`, `roboflow_pothole_voxrl` | road surface damage labelled **Potholes** | potholes labelled Trash Bins or Tricycle |
| `roboflow_door_detection_zqt59`, `roboflow_cv_project_hovyc` | doorways labelled **Doors** | doors labelled Chairs |
| `roboflow_trashcan_detection_pihfn` | bins labelled **Trash Bins** | bins labelled Bicycle |
| `roboflow_roitrikee` | tricycles labelled **Tricycle** | tricycles labelled Potholes |
| `crowdhuman` | people labelled **Person** | — id 0 never moves, so this is the control |
| `open_images` | the mixed pool; **Animals** is the one to scrutinise | Pole's old neighbour, so a surviving street light appears here as "Animals" |
""")

code(r'''
# ---- CHECK 5: per-source visual review --------------------------------------
import fiftyone as fo

ARM = "perm"          # "strict" | "perm" | "loose"
PER_SOURCE = 25       # images sampled FROM EACH SOURCE
SPLIT_FILTER = None   # None = all splits, or "train" / "val" / "test"

arm_tree, arm_names = TREES[ARM], DATA[ARM][2]
v2b_tree = TREES["v2b"]
print(f"arm {ARM}: {arm_tree.name}, nc={len(arm_names)}")


def read_boxes(label_file: Path, names: list[str]) -> list[tuple[str, int, list[float]]]:
    out = []
    if not label_file.is_file():
        return out
    for line in label_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = line.split()
        cid = int(p[0])
        xc, yc, w, h = (float(v) for v in p[1:5])
        # YOLO is centre-based and normalised; FiftyOne wants top-left [x, y, w, h],
        # also normalised. Getting this inverse wrong draws plausible boxes in the
        # wrong place, which reads as bad annotation rather than a bug.
        out.append((names[cid] if 0 <= cid < len(names) else f"<id {cid}>",
                    cid, [xc - w / 2, yc - h / 2, w, h]))
    return out


# stratified sample: PER_SOURCE images from each source, deterministic
import random
rng = random.Random(42)
picked = []
by_src = defaultdict(list)
for split in SPLITS:
    if SPLIT_FILTER and split != SPLIT_FILTER:
        continue
    for f in sorted((arm_tree / split / "labels").iterdir()):
        if f.suffix == ".txt":
            by_src[source_of(f.name)].append((split, f.stem))
for src, items in by_src.items():
    picked += [(src, s, stem) for s, stem in rng.sample(items, min(PER_SOURCE, len(items)))]
print(f"sampled {len(picked):,} images across {len(by_src)} sources "
      f"({PER_SOURCE} per source, seed 42)")

name = f"v3_source_check_{ARM}"
if name in fo.list_datasets():
    fo.delete_dataset(name)
ds = fo.Dataset(name, persistent=False)

samples, missing_img = [], 0
for src, split, stem in picked:
    img = None
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        c = arm_tree / split / "images" / f"{stem}{ext}"
        if c.is_file():
            img = c
            break
    if img is None:
        missing_img += 1
        continue
    s = fo.Sample(filepath=str(img))
    s["source"] = src
    s["split"] = split
    for layer, tree, names in (("v2b", v2b_tree, v2b_names), ("v3", arm_tree, arm_names)):
        dets = [fo.Detection(label=lbl, bounding_box=bb, class_id=cid)
                for lbl, cid, bb in read_boxes(tree / split / "labels" / f"{stem}.txt", names)]
        s[layer] = fo.Detections(detections=dets)
    # The disagreement set, computed here so the App can filter straight to it:
    # same geometry, different class name between the two layers.
    v2b_by_box = {tuple(round(v, 5) for v in d.bounding_box): d.label
                  for d in s["v2b"].detections}
    conflicts = [d.label for d in s["v3"].detections
                 if v2b_by_box.get(tuple(round(v, 5) for v in d.bounding_box), d.label) != d.label]
    s["name_conflicts"] = len(conflicts)
    samples.append(s)

ds.add_samples(samples)
if missing_img:
    print(f"  ({missing_img} sampled images had no file on disk)")
for f in ("source", "split", "name_conflicts",
          "v3.detections.label", "v2b.detections.label"):
    try:
        ds.create_index(f)
    except Exception:
        pass

conf = sum(s["name_conflicts"] for s in ds)
print(f"loaded {len(ds):,} samples")
print(f"same-geometry name disagreements between v2b and {ARM}: {conf}")
print("  0 is the expected answer. Anything above 0 is a shifted id on a box that did")
print("  not move — filter name_conflicts > 0 in the App to go straight to them.")
''')

code(r'''
# ---- launch ------------------------------------------------------------------
# In the App:
#   1. sidebar -> source -> pick one dataset at a time, and work down the list
#   2. toggle the v2b and v3 layers on the same image; the names must agree
#      everywhere except Pole, which exists only in v2b
#   3. filter name_conflicts > 0 to jump straight to any disagreement
session = fo.launch_app(ds, auto=False)
session
''')

code(r'''
fo.close_app()
print("App closed.")
''')

md(r"""
## Verdict

Checks 1–4 are arithmetic and have a definite answer; re-run their cells and read the
PASS/FAIL lines. Check 5 is yours.

If all four pass and the rendered boxes carry the right names in every source, the
nc=13 remap is correct per source, and the only thing that changed between v2b and the
arms is what the recovery pass added. That is the claim the three-arm comparison rests
on.

If **check 2 or 3 fails**, the failing row names both the source and the class, which
is enough to find the conversion that went wrong. Do not fix it in this notebook —
these trees are regenerated wholesale by
`scripts/build/build_cap4500_v3_variants.py`. Fix the conversion, rebuild, re-run
this notebook, and re-stage the arms with `scripts/train/stage_v3_arms.py --apply`.
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT.relative_to(ROOT)}  ({len(CELLS)} cells)")

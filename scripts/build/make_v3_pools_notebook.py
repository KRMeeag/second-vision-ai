"""Generate notebooks/fiftyone_v3_pool_comparison.ipynb.

Kept as a script rather than hand-edited JSON so the notebook can be regenerated
after a rebuild without escaping ipynb JSON by hand.

READ-ONLY with respect to the dataset: this writes one .ipynb and nothing else.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "fiftyone_v3_pool_comparison.ipynb"

CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": t.strip("\n").splitlines(True)})


def code(t):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": t.strip("\n").splitlines(True)})


md(r"""
**Notebook version: 1** — Claude will state the version number after editing any cell in this notebook.

# fiftyone_v3_pool_comparison.ipynb — what is actually in each training pool

The three v3 variants **nest**: `STRICT ⊂ PERMISSIVE ⊂ LOOSE` (verified, not assumed —
the conflict chain is byte-identical across all three, so only the tier gate differs).
That means they are best viewed as **incremental layers on one image**, not as three
separate datasets.

| layer | boxes | what it is |
|---|---:|---|
| `ground_truth` | 147,686 | the labels that already existed |
| `strict` | 15,311 | tiers A and C — yolov8x proposed **and** v2 confirmed at IoU ≥ 0.5 |
| `permissive_add` | **+6,393** | tier D partial agreement, tier E strict agreement (both v2_conf ≥ 0.25) |
| `loose_add` | **+30,924** | everything else that survived the conflict filters — **no v2 corroboration at all** |
| `rejected` | 12,544 | blocked by the conflict chain, in **no** pool |

So each training arm is a running total:

```
STRICT      = ground_truth + strict                              = 162,997
PERMISSIVE  = ground_truth + strict + permissive_add             = 167,311
LOOSE       = ground_truth + strict + permissive_add + loose_add = 189,833
```

Tick the layers on in order in the App's Labels panel and you watch each pool grow.

### The conflict chain, identical in all three

```
same-class nested IoMin >= 0.90            8,893
inside a Tricycle (alias)                  3,273
duplicate of another proposal IoU >= 0.70    284
relabels a rare-class box IoU >= 0.70         94
```

### What no pool can ever contain

Four constraints upstream in the proposal generator bound every pool, and appear in no
rejection tally because they ran before anything became a proposal:

* `yolov8x conf >= 0.25` — lower detections were never recorded
* the `C2O` class map — only **8 of 13** classes can be proposed at all; **Doors,
  Tricycle, Potholes, Trash Bins and Stairs get zero in every variant**
* a same-class GT IoU ≥ 0.5 check at emit time
* `v2 predict conf=0.10` — below that v2 could never corroborate

### Nothing here modifies anything

`dataset/processed/` is authoritative and was never opened for writing. Tree images are
hardlinks into `dataset/bundle/images/`. This notebook is read-only.
""")

code(r'''
# Imports
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` to find the repo root (has config/ + AGENTS.md)."""
    for parent in [start, *start.parents]:
        if (parent / "config").is_dir() and (parent / "AGENTS.md").is_file():
            return parent
    raise RuntimeError("Could not locate repo root from notebook cwd.")


REPO_ROOT = _find_repo_root(Path.cwd())
sys.path.insert(0, str(REPO_ROOT))

import fiftyone as fo
import yaml

print(f"repo root : {REPO_ROOT}")
print(f"fiftyone  : {fo.__version__}")
''')

code(r'''
# ---- Load all three accepted sets and split them into incremental layers ----

PROPOSALS = REPO_ROOT / "dataset" / "proposals"
VARIANTS = {"strict": "final_cap4500_nc13_v3s",
            "permissive": "final_cap4500_nc13_v3p",
            "loose": "final_cap4500_nc13_v3l"}

# The LOOSE tree supplies the images and the original labels; it is the superset, so
# every box from every pool appears somewhere on these same images.
TREE_ROOT = REPO_ROOT / "dataset" / VARIANTS["loose"]
assert TREE_ROOT.is_dir(), f"no such tree: {TREE_ROOT}"
_dy = yaml.safe_load((TREE_ROOT / "data.yaml").read_text(encoding="utf-8"))
TREE_NAMES = [_dy["names"][i] for i in sorted(_dy["names"])]


def _key(e, cls, box):
    """Geometry key, rounded the way the build wrote the label line (%.6f)."""
    return (e, cls, tuple(round(float(v), 6) for v in box))


accepted = {}
for short, name in VARIANTS.items():
    accepted[short] = json.loads(
        (PROPOSALS / f"accepted_{name}.json").read_text(encoding="utf-8"))

keys = {s: {_key(e, d["cls"], d["box"]) for e, ds in accepted[s].items() for d in ds}
        for s in VARIANTS}

# Verify the nesting rather than trusting it: if it ever breaks, the incremental
# layers below would silently mislabel boxes.
assert keys["strict"] <= keys["permissive"] <= keys["loose"], (
    "pools are NOT nested -- the incremental layer view is invalid")

LAYER_OF = {}
for k in keys["strict"]:
    LAYER_OF[k] = "strict"
for k in keys["permissive"] - keys["strict"]:
    LAYER_OF[k] = "permissive_add"
for k in keys["loose"] - keys["permissive"]:
    LAYER_OF[k] = "loose_add"

# Proposal metadata (tier, confidences) keyed the same way, plus the conflict rejects.
META = {}
REJECTED = defaultdict(list)
with (PROPOSALS / f"dispositions_{VARIANTS['loose']}.jsonl").open() as fh:
    for line in fh:
        d = json.loads(line)
        k = _key(d["image"], d["cls"], d["box"])
        META[k] = d
        if d["disposition"] != "accepted":
            REJECTED[d["image"]].append(d)

print(f"ground truth          147,686")
print(f"strict                {len(keys['strict']):>7,}")
print(f"permissive adds       {len(keys['permissive'] - keys['strict']):>7,}")
print(f"loose adds            {len(keys['loose'] - keys['permissive']):>7,}")
print(f"rejected (no pool)    {sum(len(v) for v in REJECTED.values()):>7,}")
print(f"\nnesting verified: strict subset of permissive subset of loose")
''')

code(r'''
# ---- Build one dataset with five independently toggleable layers ----
#
# FiftyOne colours by FIELD, so each layer renders in its own colour and gets its own
# tick box. Ticking strict -> permissive_add -> loose_add in order shows each training
# pool growing on the same image.

LOAD_SCOPE = "any_proposal"   # "any_proposal" (images with any box in any pool),
                              # "loose_only"   (only images loose adds something to),
                              # "all"          (the whole 43,170-image pool)
SOURCE_FILTER = None
DATASET_NAME = "v3_pool_comparison"

FIELDS = ["ground_truth", "strict", "permissive_add", "loose_add", "rejected"]


def _source_of(fn):
    return fn.split("__", 1)[0] if "__" in fn else "<unprefixed>"


def _det(label, box, tags=None, **attrs):
    cx, cy, w, h = box
    d = fo.Detection(label=label, bounding_box=[cx - w / 2, cy - h / 2, w, h])
    if tags:
        d.tags = list(tags)
    for k, v in attrs.items():
        d[k] = v
    return d


if fo.dataset_exists(DATASET_NAME):
    fo.delete_dataset(DATASET_NAME)
    print(f"Dropped existing '{DATASET_NAME}' (re-derived from disk).")

dataset = fo.Dataset(DATASET_NAME)
for f in FIELDS:
    dataset.classes[f] = list(TREE_NAMES)

# entry -> number of ORIGINAL boxes in the loose tree's label file
SIDECAR = json.loads((TREE_ROOT / "new_boxes.json").read_text(encoding="utf-8"))
ORIG_COUNT = SIDECAR["original_box_count"]

samples = []
counts = Counter()

for split in ("train", "val", "test"):
    images_dir = TREE_ROOT / split / "images"
    labels_dir = TREE_ROOT / split / "labels"
    if not images_dir.is_dir():
        continue
    kept = 0
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue
        entry = image_path.name[:image_path.name.rfind(image_path.suffix)]
        source = _source_of(image_path.name)
        if SOURCE_FILTER is not None and source not in SOURCE_FILTER:
            continue

        in_any = entry in ORIG_COUNT or entry in REJECTED
        adds_loose = any(LAYER_OF.get(_key(entry, d["cls"], d["box"])) == "loose_add"
                         for d in accepted["loose"].get(entry, []))
        if LOAD_SCOPE == "any_proposal" and not in_any:
            continue
        if LOAD_SCOPE == "loose_only" and not adds_loose:
            continue

        label_path = labels_dir / f"{entry}.txt"
        if not label_path.is_file():
            continue

        # Original labels = the prefix of the loose tree's label file.
        lines = [l for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        n_orig = ORIG_COUNT.get(entry, len(lines))
        buckets = {f: [] for f in FIELDS}
        for ln in lines[:n_orig]:
            p = ln.split()
            if len(p) != 5:
                continue
            cid = int(p[0])
            if 0 <= cid < len(TREE_NAMES):
                buckets["ground_truth"].append(
                    _det(TREE_NAMES[cid], [float(v) for v in p[1:]]))

        for d in accepted["loose"].get(entry, []):
            k = _key(entry, d["cls"], d["box"])
            layer = LAYER_OF.get(k)
            if layer is None:
                continue
            m = META.get(k, {})
            buckets[layer].append(_det(
                d["cls"], d["box"],
                tags=[layer, f"tier_{d['tier']}"],
                tier=d["tier"], x_conf=float(d["x_conf"]),
                v2_conf=float(d["v2_conf"]), agree=bool(m.get("agree", False))))
            counts[layer] += 1

        for d in REJECTED.get(entry, []):
            short = (d["disposition"].split("(")[0].strip()
                     .replace("same-class nested >=0.90", "nested")
                     .replace("inside a Tricycle", "tricycle")
                     .replace("duplicate of another proposal", "dup_proposal")
                     .replace("relabels a rare-class box", "rare_relabel"))
            buckets["rejected"].append(_det(
                d["cls"], d["box"], tags=["rejected", f"rej_{short}"],
                tier=d["tier"], x_conf=float(d["x_conf"]),
                v2_conf=float(d["v2_conf"]), reason=d["disposition"]))
            counts["rejected"] += 1

        s = fo.Sample(filepath=str(image_path))
        s["source"] = source
        s["split"] = split
        for f in FIELDS:
            s[f] = fo.Detections(detections=buckets[f])
            s[f"n_{f}"] = len(buckets[f])
        s["n_total_loose"] = sum(len(buckets[f]) for f in
                                 ("ground_truth", "strict", "permissive_add", "loose_add"))
        tags = [f for f in ("strict", "permissive_add", "loose_add") if buckets[f]]
        s.tags = tags
        samples.append(s)
        counts["ground_truth"] += len(buckets["ground_truth"])
        kept += 1
    print(f"  {split}: {kept:,} samples")

dataset.add_samples(samples)
dataset.persistent = False

# Index what the sidebar filters on. NOT an optimisation: FiftyOne scans unindexed
# fields with a bounded budget and silently returns a PARTIAL value list when it runs
# out, so real classes and tags can vanish from the dropdown.
for f in ("source", "split", "tags", "n_loose_add", "n_total_loose"):
    try:
        dataset.create_index(f)
    except Exception:
        pass
for f in FIELDS:
    for sub in ("label", "tags", "tier"):
        try:
            dataset.create_index(f"{f}.detections.{sub}")
        except Exception:
            pass

print(f"\nLoaded {len(dataset):,} samples")
for f in FIELDS:
    print(f"  {f:<16}{counts[f]:>9,}")
''')

code(r'''
# ---- Pool composition: what each layer is made of ----

per_class = defaultdict(Counter)
per_tier = defaultdict(Counter)
per_source = defaultdict(Counter)
areas = defaultdict(list)
gt_area = defaultdict(list)

LAYERS = ["strict", "permissive_add", "loose_add", "rejected"]

for smp in dataset.select_fields(["source"] + FIELDS):
    for d in (smp.ground_truth.detections if smp.ground_truth else []):
        per_class["ground_truth"][d.label] += 1
        gt_area[d.label].append(d.bounding_box[2] * d.bounding_box[3])
    for f in LAYERS:
        for d in (smp[f].detections if smp[f] else []):
            per_class[f][d.label] += 1
            per_tier[f][d["tier"]] += 1
            per_source[f][smp["source"]] += 1
            areas[f].append((d.label, d.bounding_box[2] * d.bounding_box[3]))

import statistics as st

print("BOXES PER CLASS, cumulative across the three training pools\n")
print(f"{'class':<12}{'original':>10}{'STRICT':>10}{'PERMISS':>10}{'LOOSE':>10}"
      f"{'rejected':>10}")
tot = Counter()
for c in TREE_NAMES:
    o = per_class["ground_truth"][c]
    s = o + per_class["strict"][c]
    p = s + per_class["permissive_add"][c]
    l = p + per_class["loose_add"][c]
    print(f"{c:<12}{o:>10,}{s:>10,}{p:>10,}{l:>10,}{per_class['rejected'][c]:>10,}")
    tot["o"] += o; tot["s"] += s; tot["p"] += p; tot["l"] += l
print(f"{'TOTAL':<12}{tot['o']:>10,}{tot['s']:>10,}{tot['p']:>10,}{tot['l']:>10,}")

print("\n\nTIER COMPOSITION OF EACH LAYER\n")
print(f"{'layer':<16}" + "".join(f"{t:>9}" for t in "ABCDE"))
for f in LAYERS:
    print(f"{f:<16}" + "".join(f"{per_tier[f][t]:>9,}" for t in "ABCDE"))

print("\n\nBOX SIZE vs the class's own GT median (1.00 = typical)\n")
print(f"{'layer':<16}{'n':>9}{'median ratio':>15}{'below GT p5':>13}")
for f in LAYERS:
    rs, tiny = [], 0
    for lab, a in areas[f]:
        g = gt_area.get(lab)
        if not g or len(g) < 50:
            continue
        med = st.median(g)
        p5 = sorted(g)[int(0.05 * (len(g) - 1))]
        rs.append(a / med)
        if a < p5:
            tiny += 1
    if rs:
        print(f"{f:<16}{len(rs):>9,}{st.median(rs):>15.2f}{100*tiny/len(rs):>12.1f}%")

print("\n\nTOP SOURCES PER LAYER\n")
for f in LAYERS:
    top = ", ".join(f"{k}={v:,}" for k, v in per_source[f].most_common(4))
    print(f"  {f:<16}{top}")

print("""

Doors, Tricycle, Potholes, Trash Bins and Stairs show ZERO in every pool: no COCO
class maps to them, so yolov8x cannot propose them at any threshold. Whatever the
three arms show, those five classes are identical in all of them.""")
''')

md(r"""
## Reading the layers in the App

Each layer is its own field with its own colour and tick box in the **Labels** panel.
Tick them on in order:

| ticked | you are looking at |
|---|---|
| `ground_truth` | the dataset as it exists today |
| `+ strict` | what the **STRICT** arm trains on |
| `+ permissive_add` | what the **PERMISSIVE** arm trains on |
| `+ loose_add` | what the **LOOSE** arm trains on |
| `+ rejected` | boxes blocked by the conflict chain, in no arm |

### Questions worth asking of each layer

- **`loose_add`** is the big one — 30,924 boxes with **no v2 corroboration at all**.
  Filter `loose_add` + `tier_E` and ask whether those belong in a training set.
- **`permissive_add`** is 6,393 boxes where v2 did back the object, either partially
  (tier D) or strictly but weakly (tier E). Compare its look against `loose_add`.
- **`rejected`** — for `rej_nested`, a `ground_truth` box should be visible underneath.
  If it is not, the filter misfired and is worth reporting.

### Useful filters

| goal | filter |
|---|---|
| what loose adds, worst case | `loose_add` + `tier_E` + sort by `n_loose_add` desc |
| images loose changes most | SAMPLE TAGS `loose_add`, sort `n_loose_add` descending |
| one class across all layers | `*.detections.label` = e.g. `Chairs` |
| the eval splits (identical in all arms) | `split` = `test` |
""")

code(r'''
# Launch the App, busiest images first.
from fiftyone import ViewField as F

view = dataset.sort_by("n_loose_add", reverse=True)
session = fo.launch_app(view, auto=False)
session
''')

code(r'''
# ---- Contact sheet: one layer at a time, every other layer drawn for context ----
#
#   GREEN   ground_truth
#   RED     strict
#   ORANGE  permissive_add
#   YELLOW  loose_add
#   GREY    rejected (dashed)

import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

FOCUS = "loose_add"     # "strict" | "permissive_add" | "loose_add" | "rejected"
TIER = None             # "A".."E", or None
CLASS = None            # e.g. "Chairs", or None
N, COLS, SEED = 24, 6, 11

COLOURS = {"ground_truth": ("#3ce65a", False), "strict": ("#ff3c3c", False),
           "permissive_add": ("#ff9a2e", False), "loose_add": ("#ffe24a", False),
           "rejected": ("#9aa0a6", True)}

pool = []
for smp in dataset.select_fields(["filepath"] + FIELDS):
    for d in (smp[FOCUS].detections if smp[FOCUS] else []):
        if TIER and d["tier"] != TIER:
            continue
        if CLASS and d.label != CLASS:
            continue
        pool.append((smp.filepath, d, smp))

print(f"pool: {len(pool):,} boxes in `{FOCUS}`"
      + (f", tier {TIER}" if TIER else "") + (f", class {CLASS}" if CLASS else ""))
random.seed(SEED)
sel = random.sample(pool, min(N, len(pool)))

rows = (len(sel) + COLS - 1) // COLS
fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 2.9, rows * 3.1), squeeze=False)
fig.patch.set_facecolor("#121214")
for ax in axes.ravel():
    ax.axis("off")

for k, (fp, det, smp) in enumerate(sel):
    ax = axes.ravel()[k]
    im = Image.open(fp).convert("RGB")
    W, H = im.size
    bx, by, bw, bh = det.bounding_box
    cx, cy = (bx + bw / 2) * W, (by + bh / 2) * H
    m = max(bw * W, bh * H) * 1.8 + 26
    L, T = max(0, cx - m / 2), max(0, cy - m / 2)
    R, B = min(W, cx + m / 2), min(H, cy + m / 2)
    ax.imshow(im.crop((int(L), int(T), int(R), int(B))))
    for f in FIELDS:
        colour, dashed = COLOURS[f]
        for o in (smp[f].detections if smp[f] else []):
            ox, oy, ow, oh = o.bounding_box
            ax.add_patch(Rectangle((ox * W - L, oy * H - T), ow * W, oh * H, fill=False,
                                   lw=2.4 if o is det else 1.2, edgecolor=colour,
                                   linestyle="--" if dashed else "-"))
    ax.set_title(f"{det.label} {det['tier']} x{det['x_conf']:.2f} v2:{det['v2_conf']:.2f}",
                 fontsize=7, color="#e8e8e8", pad=3)

plt.tight_layout()
plt.show()
print("GREEN ground_truth  RED strict  ORANGE permissive_add  "
      "YELLOW loose_add  GREY-dashed rejected")
''')

code(r'''
# Run when finished. The dataset is non-persistent, so it also disappears on kernel
# shutdown -- rebuild from the cells above whenever you want it back.
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

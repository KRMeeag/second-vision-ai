"""Generate notebooks/fiftyone_v3_proposed_dataset.ipynb.

Kept as a script (rather than hand-edited JSON) so the notebook can be
regenerated after a tree rebuild without hand-escaping ipynb JSON.

READ-ONLY with respect to the dataset: this writes one .ipynb and nothing else.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "fiftyone_v3_proposed_dataset.ipynb"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(True)})


md(r"""
**Notebook version: 4** — Claude will state the version number after editing any cell in this notebook.

# fiftyone_v3_proposed_dataset.ipynb — strict vs permissive, side by side

Browses a v3 variant tree with existing labels and proposed boxes as **three separate,
independently toggleable layers**, and tags every accepted box with which variant it
belongs to.

## The two variants

Both share **identical val and test labels** (the strict set). Only TRAINING labels
differ, so a strict-vs-permissive training comparison varies one thing and the yardstick
never moves. A false box in an eval split is an unmatchable false negative — it caps
measured recall permanently and cannot be absorbed the way training noise can.

| tree | train boxes | total | rule |
|---|---:|---:|---|
| `final_cap4500_nc13_v3s` **STRICT** | +10,102 | 162,740 | tiers A and C only |
| `final_cap4500_nc13_v3p` **PERMISSIVE** | +13,992 | 166,630 | A, C, **+ tier D partial agreement, + tier E strict agreement** (both v2_conf ≥ 0.25) |

The permissive tree is a **superset**, so browsing it shows both: boxes tagged
`in_both` are in strict too, and **`permissive_only`** is exactly what permissiveness
buys — **+3,890 training boxes**.

## Corrections applied to BOTH variants

These were rule errors, not permissiveness:

- **`rej_6` now fires only on a same-extent relabel (IoU ≥ 0.70)**, not any IoMin ≥ 0.30
  touch. 80.9% of what the old rule dropped was legitimate — a Person on Stairs (401),
  a Bench beside Stairs (130), a Person in a doorway (38). Rejections there: 1,339 → 94.
- **`rej_3` (re-box) and `rej_4` (partial nest) removed.** Adjacent objects — chairs in a
  row, people in a crowd — routinely half-overlap. `rej_3` was worth only 7 boxes alone.

Still rejected in both: same-class nesting ≥ 0.90, the Tricycle alias (DEC-109/111),
proposal-vs-proposal self-overlap, tier B, and tier E without agreement.

## Label tags

| tag | meaning |
|---|---|
| `in_both` / `permissive_only` | which variant an accepted box belongs to |
| `rej_conflict` / `rej_unconfirmed` | the rejection family — **read this first** |
| `v2_agreed` / `v2_partial` | on a *rejected* box: v2 did back it, but a tier gate still dropped it |
| `tier_A` `tier_C` `tier_D` `tier_E` | which tier an accepted box came from |
| `xconf_*` | yolov8x confidence band — categorisation only, never a drop reason |

`rej_unconfirmed` boxes have **no geometric conflict by construction**. They were dropped
for lack of a second opinion, so finding them clean is expected and is not evidence a
filter misfired. The filters that can misfire are `rej_conflict` — for those a green
`ground_truth` box should be visible underneath.

### Nothing here modifies anything

`dataset/processed/` is authoritative and was never opened for writing. Tree images are
**hardlinks** into `dataset/bundle/images/`. This notebook is read-only.
""")

code(r'''
# Imports
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` to find the repo root (has config/ + AGENTS.md).

    Same helper the other notebooks use -- notebooks live in notebooks/, and
    Jupyter's working directory depends on how it was launched.
    """
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
# ---- What to load ----

# Which tree to browse:
#   "final_cap4500_nc13_v3p"  PERMISSIVE -- A+C, plus tier D partial agreement and
#                                           tier E strict agreement (both v2_conf >= 0.25)
#   "final_cap4500_nc13_v3s"  STRICT     -- A+C only
#   "final_cap4500_nc13_v3"   as first built (DEC-140), before the rej_6 correction
#
# Default to PERMISSIVE because it is a SUPERSET: every strict box is in it, and the
# extra ones are tagged `permissive_only`. That makes strict-vs-permissive one sidebar
# click instead of two App windows.
TREE = "final_cap4500_nc13_v3p"
COMPARE_TREE = "final_cap4500_nc13_v3s"   # None to skip the strict/permissive split

SPLITS = ["train", "val", "test"]

# Which images to pull in:
#   "enriched"     -> only images that GAINED a box (6,369). Fastest; rejected boxes on
#                     those same images still show, so most review happens here.
#   "any_proposal" -> every image either detector proposed anything on (18,554). Use this
#                     to judge rejections on images that gained nothing.
#   "all"          -> the whole 43,170-image pool.
LOAD_SCOPE = "enriched"

SHOW_REJECTED = True      # draw the proposed_rejected layer
SOURCE_FILTER = None      # e.g. {"open_images"}
CLASS_FILTER = None       # e.g. {"Chairs"} -- keeps images with >=1 box of these

DATASET_NAME = "v3_proposed_review"

TREE_ROOT = REPO_ROOT / "dataset" / TREE
assert TREE_ROOT.is_dir(), f"no such tree: {TREE_ROOT}"

_dy = yaml.safe_load((TREE_ROOT / "data.yaml").read_text(encoding="utf-8"))
TREE_NAMES = [_dy["names"][i] for i in sorted(_dy["names"])]

# Sidecar from the build: entry -> how many boxes were ORIGINAL. Everything in the label
# file at an index >= that value is a proposed box the build appended.
SIDECAR = json.loads((TREE_ROOT / "new_boxes.json").read_text(encoding="utf-8"))
ORIG_COUNT = SIDECAR["original_box_count"]

# Every proposal with its disposition, written by
# scripts/curate/write_proposal_dispositions.py using the build script's own filter chain.
# Per-variant, so the reasons always match the tree being browsed. The original
# dispositions.jsonl predates the rej_6 correction and would show stale reasons.
DISPOSITIONS = REPO_ROOT / "dataset" / "proposals" / f"dispositions_{TREE}.jsonl"
if not DISPOSITIONS.is_file():
    DISPOSITIONS = REPO_ROOT / "dataset" / "proposals" / "dispositions.jsonl"
assert DISPOSITIONS.is_file(), (
    "No dispositions file -- run scripts/curate/write_variant_dispositions.py first."
)

# Boxes the STRICT variant also accepted, keyed by rounded geometry. Anything accepted
# in the browsed tree but absent here is exactly what permissiveness bought.
STRICT_KEYS = set()
if COMPARE_TREE:
    _cmp = REPO_ROOT / "dataset" / "proposals" / f"accepted_{COMPARE_TREE}.json"
    if _cmp.is_file():
        for _e, _ds in json.loads(_cmp.read_text(encoding="utf-8")).items():
            for _d in _ds:
                STRICT_KEYS.add((_e, _d["cls"], tuple(round(float(v), 6) for v in _d["box"])))

PROPOSALS_BY_IMAGE = defaultdict(list)
with DISPOSITIONS.open() as fh:
    for line in fh:
        d = json.loads(line)
        PROPOSALS_BY_IMAGE[d["image"]].append(d)

_disp = Counter(d["disposition"] for ds in PROPOSALS_BY_IMAGE.values() for d in ds)
print(f"tree      : {TREE_ROOT}")
print(f"schema    : nc={len(TREE_NAMES)}  {TREE_NAMES}")
print(f"proposals : {sum(_disp.values()):,} over {len(PROPOSALS_BY_IMAGE):,} images")
print(f"  kept    : {_disp['accepted']:,}  (these ARE in the tree)")
print(f"  rejected: {sum(v for k, v in _disp.items() if k != 'accepted'):,}  (these are NOT)")
print(f"scope     : {LOAD_SCOPE}")
if STRICT_KEYS:
    print(f"compare   : {COMPARE_TREE} -- {len(STRICT_KEYS):,} of these boxes are in STRICT too;")
    print( "            the rest are tagged `permissive_only`")
''')

code(r'''
# ---- Build the dataset: three layers, with the reason as a LABEL TAG ----
#
# Detection ATTRIBUTES (what `reason` used to be) are buried inside the field expansion
# in the App and are awkward to filter on. LABEL TAGS get their own sidebar section and
# are one click, so every proposed box is tagged instead:
#
#   kept     ->  kept  +  tier_A | tier_C                 +  xconf_*
#   rejected ->  rej_conflict | rej_unconfirmed           <- the family; read this first
#                + rej_1_same_class_nested | rej_2_tricycle_alias | rej_3_rebox
#                | rej_4_partial_nest | rej_5_self_overlap | rej_6_rare_class
#                | rej_tier_B | rej_tier_D | rej_tier_E
#                + xconf_*
#
# `rej_unconfirmed` boxes have NO geometric conflict by construction -- they were dropped
# for lack of a second opinion, not for clashing with anything. Judge those on whether one
# model is enough evidence, not on whether they overlap something.

# THREE families, not two. The earlier scheme lumped every tier rejection into
# `rej_unconfirmed` ("dropped for lack of a second opinion"), which was wrong for
# 6,365 boxes that v2 DID agree with and that were dropped only because v2's own
# confidence sat under the 0.25 gate. Those now get their own family, so a box that
# was confirmed-but-weakly is never confused with one v2 never saw.
REASON_TAGS = {
    # --- geometric clash with something concrete ---
    "same-class nested >=0.90":            ("rej_conflict",    "rej_same_class_nested"),
    "inside a Tricycle (alias)":           ("rej_conflict",    "rej_tricycle_alias"),
    "relabels a rare-class box (IoU>=0.70)":("rej_conflict",   "rej_rare_class_relabel"),
    "proposal vs proposal >=0.30":         ("rej_conflict",    "rej_self_overlap"),
    # --- v2 DID back it; the confidence gate is what dropped it ---
    "tier E: v2 AGREED but below the v2_conf gate":
                                           ("rej_below_gate",  "rej_tier_E_agreed_weak"),
    "tier D: v2 signal below the gate":    ("rej_below_gate",  "rej_tier_D_weak"),
    # --- v2 genuinely did not back it ---
    "tier B: v2 silent (no agreement at any confidence)":
                                           ("rej_unconfirmed", "rej_tier_B"),
    "tier E: v2 did not agree":            ("rej_unconfirmed", "rej_tier_E_no_agree"),
    "tier E: v2 saw nothing":              ("rej_unconfirmed", "rej_tier_E_v2_blind"),
    "tier D: v2 saw nothing":              ("rej_unconfirmed", "rej_tier_D_v2_blind"),
    # --- excluded by the strict variant's rule, not by evidence ---
    "tier D excluded (strict variant)":    ("rej_strict_rule", "rej_tier_D_strict"),
    "tier E excluded (strict variant)":    ("rej_strict_rule", "rej_tier_E_strict"),
    # --- original chain, so the as-built tree stays browsable ---
    "tier B not accepted":                 ("rej_unconfirmed", "rej_tier_B"),
    "tier D not accepted":                 ("rej_unconfirmed", "rej_tier_D"),
    "tier E not accepted":                 ("rej_unconfirmed", "rej_tier_E"),
    # original chain, so the as-built tree stays browsable
    "1 same-class nested >=0.90":          ("rej_conflict",    "rej_same_class_nested"),
    "2 inside a Tricycle (alias)":         ("rej_conflict",    "rej_tricycle_alias"),
    "3 same-class re-box >=0.30 IoU":      ("rej_conflict",    "rej_rebox"),
    "4 same-class partial nest >=0.50":    ("rej_conflict",    "rej_partial_nest"),
    "5 proposal vs proposal >=0.30":       ("rej_conflict",    "rej_self_overlap"),
    "6 lands on a rare-class box":         ("rej_conflict",    "rej_rare_class_relabel"),
}


def _xconf_tag(x):
    if x >= 0.85:
        return "xconf_85plus"
    if x >= 0.70:
        return "xconf_70_85"
    if x >= 0.50:
        return "xconf_50_70"
    return "xconf_25_50"


def _source_of(filename: str) -> str:
    """Recover the source key from merge.py's `<source>__<name>` filename prefix."""
    return filename.split("__", 1)[0] if "__" in filename else "<unprefixed>"


def _det(label, box, tags=None, **attrs):
    """YOLO centre-relative xywh -> FiftyOne top-left-relative, plus tags/attributes."""
    cx, cy, w, h = box
    d = fo.Detection(label=label, bounding_box=[cx - w / 2, cy - h / 2, w, h])
    if tags:
        d.tags = list(tags)
    for k, v in attrs.items():
        d[k] = v
    return d


if fo.dataset_exists(DATASET_NAME):
    fo.delete_dataset(DATASET_NAME)
    print(f"Dropped existing '{DATASET_NAME}' (re-derived from disk, no review work lost).")

dataset = fo.Dataset(DATASET_NAME)
# DEC-081: without this the App blocks on an "Import your dataset schema" prompt.
for _f in ("ground_truth", "proposed_kept", "proposed_rejected"):
    dataset.classes[_f] = list(TREE_NAMES)

samples = []
n_gt = n_kept = n_rej = 0
skipped_missing_label = 0
unknown_disposition = Counter()

for split in SPLITS:
    images_dir = TREE_ROOT / split / "images"
    labels_dir = TREE_ROOT / split / "labels"
    if not images_dir.is_dir():
        print(f"  {split}: not found -- skipped.")
        continue

    kept_n = 0
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue
        entry = image_path.name[:image_path.name.rfind(image_path.suffix)]

        props = PROPOSALS_BY_IMAGE.get(entry, [])
        if LOAD_SCOPE == "enriched" and entry not in ORIG_COUNT:
            continue
        if LOAD_SCOPE == "any_proposal" and not props:
            continue

        source = _source_of(image_path.name)
        if SOURCE_FILTER is not None and source not in SOURCE_FILTER:
            continue

        label_path = labels_dir / f"{entry}.txt"
        if not label_path.is_file():
            skipped_missing_label += 1
            continue

        # ---- layer 1: the labels that were already there ----
        # The tree's label file is [originals..., accepted proposals...]; the sidecar says
        # where the originals stop. For an image that gained nothing, all lines are original.
        lines = [l for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        n_orig = ORIG_COUNT.get(entry, len(lines))
        gt = []
        for line in lines[:n_orig]:
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            if 0 <= cid < len(TREE_NAMES):
                gt.append(_det(TREE_NAMES[cid], [float(v) for v in parts[1:]]))

        # ---- layers 2 and 3: proposals, split by disposition ----
        kept, rejected = [], []
        for p in props:
            xc, vc = float(p["x_conf"]), float(p["v2_conf"])
            if p["disposition"] == "accepted":
                if STRICT_KEYS:
                    in_strict = (entry, p["cls"],
                                 tuple(round(float(v), 6) for v in p["box"])) in STRICT_KEYS
                    vtag = "in_both" if in_strict else "permissive_only"
                else:
                    vtag = "kept"
                kept.append(_det(p["cls"], p["box"],
                                 tags=["kept", vtag, f"tier_{p['tier']}", _xconf_tag(xc)],
                                 tier=p["tier"], x_conf=xc, v2_conf=vc, variant=vtag))
            else:
                if not SHOW_REJECTED:
                    continue
                fam_specific = REASON_TAGS.get(p["disposition"])
                if fam_specific is None:
                    # Surfaced, never silently bucketed -- an unmapped disposition means
                    # the filter chain grew a reason this notebook does not know about.
                    unknown_disposition[p["disposition"]] += 1
                    fam_specific = ("rej_unknown", "rej_unknown")
                rtags = [*fam_specific, _xconf_tag(xc)]
                if p.get("agree"):
                    rtags.append("v2_agreed")      # dropped DESPITE strict agreement
                elif vc >= 0.25:
                    rtags.append("v2_partial")     # v2 fired on the class, box misaligned
                rejected.append(_det(p["cls"], p["box"], tags=rtags,
                                     tier=p["tier"], x_conf=xc, v2_conf=vc,
                                     reason=p["disposition"]))

        if CLASS_FILTER is not None and not any(
            d.label in CLASS_FILTER for d in (*gt, *kept, *rejected)
        ):
            continue

        sample = fo.Sample(filepath=str(image_path))
        sample["source"] = source
        sample["split"] = split
        sample["n_ground_truth"] = len(gt)
        sample["n_proposed_kept"] = len(kept)
        sample["n_proposed_rejected"] = len(rejected)
        sample["ground_truth"] = fo.Detections(detections=gt)
        sample["proposed_kept"] = fo.Detections(detections=kept)
        sample["proposed_rejected"] = fo.Detections(detections=rejected)
        tags = []
        if kept:
            tags.append("gained_boxes")
        if any("rej_conflict" in (d.tags or []) for d in rejected):
            tags.append("has_conflict_reject")
        if any("rej_unconfirmed" in (d.tags or []) for d in rejected):
            tags.append("has_unconfirmed_reject")
        sample.tags = tags
        samples.append(sample)

        n_gt += len(gt); n_kept += len(kept); n_rej += len(rejected)
        kept_n += 1

    print(f"  {split}: {kept_n:,} samples")

dataset.add_samples(samples)
dataset.persistent = False   # a viewer, not review state -- rebuilt from disk in ~1 min

# Index the fields the sidebar filters on. NOT an optimisation but a correctness fix:
# FiftyOne searches UNINDEXED fields with a bounded scan and, when the budget runs out,
# returns a PARTIAL value list under a small "Incomplete search" note -- so tags and
# classes that DO exist can silently fail to appear in the dropdown.
for _f in ("source", "split", "tags",
           "n_proposed_kept", "n_proposed_rejected",
           "ground_truth.detections.label",
           "proposed_kept.detections.label",
           "proposed_kept.detections.tags",
           "proposed_kept.detections.variant",
           "proposed_rejected.detections.label",
           "proposed_rejected.detections.tags"):
    try:
        dataset.create_index(_f)
    except Exception:
        pass

print(f"\nLoaded {len(dataset):,} samples")
print(f"  ground_truth      {n_gt:,}")
print(f"  proposed_kept     {n_kept:,}   <- in the v3 tree")
print(f"  proposed_rejected {n_rej:,}   <- NOT in the tree")
if skipped_missing_label:
    print(f"WARNING: {skipped_missing_label:,} image(s) had no matching label file.")
if unknown_disposition:
    print(f"WARNING: unmapped disposition(s): {dict(unknown_disposition)}")
''')

code(r'''
# ---- The two rejection families, and what each one actually means ----

fam = Counter(); specific = defaultdict(Counter); fam_conf = defaultdict(Counter)
kept_tier = Counter()

for smp in dataset.select_fields(["proposed_kept", "proposed_rejected"]):
    for d in (smp.proposed_kept.detections if smp.proposed_kept else []):
        kept_tier[d["tier"]] += 1
    for d in (smp.proposed_rejected.detections if smp.proposed_rejected else []):
        tags = d.tags or []
        f = next((t for t in tags if t in ("rej_conflict", "rej_below_gate",
                                           "rej_unconfirmed", "rej_strict_rule")),
                 "rej_unknown")
        fam[f] += 1
        for t in tags:
            if t.startswith("rej_") and t != f:
                specific[f][t] += 1
        fam_conf[f][next((t for t in tags if t.startswith("xconf_")), "?")] += 1

total = sum(fam.values()) or 1
print("REJECTION FAMILIES (label tags -- click these in the App's LABEL TAGS section)\n")
for f, v in fam.most_common():
    print(f"  {f:<20}{v:>8,}  {100*v/total:>5.1f}%")
    for t, c in specific[f].most_common():
        print(f"      {t:<30}{c:>8,}")
    print(f"      x_conf: {dict(sorted(fam_conf[f].items()))}")
    print()

print("HOW TO READ THESE\n")
print("  rej_conflict     -> clashes with something concrete: duplicates an existing box,")
print("                      duplicates another proposal, or relabels a rare-class object.")
print("                      A green ground_truth box should be visible underneath. If it")
print("                      is NOT, that is a misfire and worth reporting.")
print()
print("  rej_below_gate   -> v2 DID back this box. It was dropped only because v2's own")
print("                      confidence fell under the 0.25 gate. These are the closest")
print("                      calls in the whole set: a kept tier-E box at v2_conf 0.283")
print("                      and a rejected one at 0.101 differ ONLY by that number.")
print("                      Loosening the gate promotes exactly these.")
print()
print("  rej_unconfirmed  -> v2 genuinely did not back it: either no same-class detection")
print("                      overlapped at IoU>=0.5, or v2 saw nothing at all. No geometric")
print("                      conflict exists, so one of these looking 'fine, nothing")
print("                      clashing' is EXPECTED and is not evidence of a misfire.")
print()
print("  rej_strict_rule  -> excluded by the STRICT variant's rule, not by evidence.")
print("                      Only appears when browsing the strict tree.")
print()
print(f"  kept by tier: {dict(sorted(kept_tier.items()))}")
print()
print("  The slice most worth your judgement: rej_tier_B -- yolov8x >= 0.70 but v2 silent.")
print("  Click rej_tier_B, then xconf_85plus, and decide whether those belong in the tree.")
''')

md(r"""
## Using the three layers

In the App's **Labels** panel each field has its own tick box and colour:

- untick **`proposed_kept`** → the dataset as it is **today**
- tick it → what v3 would **train on**
- tick **`proposed_rejected`** → everything the detectors suggested but a filter removed

### LABEL TAGS — where the rejection reasons live

The sidebar's **LABEL TAGS** section lists every tag. Start with the family:

| tag | what it tells you |
|---|---|
| `rej_unconfirmed` | no conflict; dropped for lack of a second opinion |
| `rej_conflict` | clashes with an existing box or another proposal |
| `rej_1_same_class_nested` / `rej_3_rebox` / `rej_4_partial_nest` | the object is **already labelled**, just drawn at another extent |
| `rej_2_tricycle_alias` | DEC-109/111: a tricycle *is* a motorcycle plus a cab |
| `rej_5_self_overlap` | yolov8x duplicating itself (its NMS runs at IoU 0.7) |
| `rej_6_rare_class` | landed on a Stairs/Potholes/Doors/Trash Bins/Tricycle box |
| `rej_tier_B` / `rej_tier_D` / `rej_tier_E` | v2 did not confirm (B, D), or yolov8x < 0.50 (E) |
| `xconf_85plus` / `xconf_70_85` / `xconf_50_70` / `xconf_25_50` | yolov8x confidence band |
| `kept`, `tier_A`, `tier_C` | on the boxes that made it in |

### When you see an object that looks like it should be labelled

1. **Green box on it?** → already labelled; nothing wrong.
2. **Amber box with `rej_conflict`?** → check for a green box underneath. Present means the
   filter was right. **Absent means the filter misfired** — tell me the tag and the image.
3. **Amber box with `rej_unconfirmed`?** → no conflict by design. The judgement call is
   whether yolov8x alone is enough. `rej_tier_B` + `xconf_85plus` is the strongest such slice.
4. **Nothing at all?** → neither detector proposed it. No threshold change recovers it;
   that object needs manual labelling.

### Checking for duplicates

Tick `ground_truth` and `proposed_kept` together and look for a red box on an object that
already has a green one. ~5% of kept boxes still overlap a **different-class** ground-truth
box at IoU ≥ 0.50 (Bench↔Tables, Motorcycle↔Bicycle), which is not geometrically separable
from legitimate cases like a Person on a Motorcycle.
""")

code(r'''
# Launch the App. If a server is already running (it survives kernel restarts),
# this attaches to it rather than starting a second one.
from fiftyone import ViewField as F

view = dataset.sort_by("n_proposed_kept", reverse=True)
session = fo.launch_app(view, auto=False)
session
''')

code(r'''
# ---- Jump straight to one rejection reason ----
#
# Set TAG to any label tag and re-run to retarget the App at exactly that slice, with
# every other box still drawn for context. Useful pairs:
#
#   "rej_tier_B"              yolov8x confident, v2 silent  <- the judgement call
#   "rej_1_same_class_nested" should ALWAYS sit on a green box; if not, a misfire
#   "rej_6_rare_class"        near Stairs/Potholes/Doors/Trash Bins/Tricycle
#   "rej_5_self_overlap"      yolov8x duplicating itself

from fiftyone import ViewField as F

TAG = "rej_tier_B"

view = (dataset
        .match(F("proposed_rejected.detections").filter(F("tags").contains(TAG)).length() > 0)
        .sort_by("n_proposed_rejected", reverse=True))
print(f"{len(view):,} images contain at least one `{TAG}` box")
session.view = view
session
''')

code(r'''
# ---- Offline contact sheet: the same three layers, rendered inline ----
#
#   GREEN  ground truth (already labelled)
#   RED    proposed and KEPT      -> in the v3 tree
#   AMBER  proposed and REJECTED  -> not in the tree, dashed
#
# Centres each tile on a box carrying FOCUS_TAG, so you can eyeball one rejection reason
# at a time without the App.

import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

FOCUS_TAG = "rej_tier_B"     # any label tag, e.g. "kept", "rej_1_same_class_nested"
CLASS = None                 # e.g. "Chairs", or None
N = 24
COLS = 6
SEED = 11

pool = []
for smp in dataset.select_fields(
    ["filepath", "ground_truth", "proposed_kept", "proposed_rejected"]
):
    for fname in ("proposed_kept", "proposed_rejected"):
        for d in (smp[fname].detections if smp[fname] else []):
            if FOCUS_TAG not in (d.tags or []):
                continue
            if CLASS and d.label != CLASS:
                continue
            pool.append((smp.filepath, d, smp))

print(f"pool: {len(pool):,} boxes tagged `{FOCUS_TAG}`" + (f" of class {CLASS}" if CLASS else ""))
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
    m = max(bw * W, bh * H) * 1.7 + 24
    L, T = max(0, cx - m / 2), max(0, cy - m / 2)
    R, B = min(W, cx + m / 2), min(H, cy + m / 2)
    ax.imshow(im.crop((int(L), int(T), int(R), int(B))))

    for fname, colour, dashed in (("ground_truth", "#3ce65a", False),
                                  ("proposed_kept", "#ff3c3c", False),
                                  ("proposed_rejected", "#ffb020", True)):
        for o in (smp[fname].detections if smp[fname] else []):
            ox, oy, ow, oh = o.bounding_box
            ax.add_patch(Rectangle(
                (ox * W - L, oy * H - T), ow * W, oh * H, fill=False,
                lw=2.4 if o is det else 1.3, edgecolor=colour,
                linestyle="--" if dashed else "-"))

    ax.set_title(f"{det.label} {det['tier']}  x{det['x_conf']:.2f} v2:{det['v2_conf']:.2f}",
                 fontsize=7, color="#e8e8e8", pad=3)

plt.tight_layout()
plt.show()
print("GREEN = ground truth   RED = proposed & kept   AMBER dashed = proposed & rejected")
''')

code(r'''
# Run when finished. The dataset is non-persistent, so it also disappears on
# kernel shutdown -- rebuild from the cells above whenever you want it back.
fo.close_app()
print("App closed.")
''')

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT.relative_to(ROOT)}  ({len(CELLS)} cells)")

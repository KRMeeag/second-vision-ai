"""
Build the v3 training tree: cap4500 + recovered boxes, 13 classes.

This is cap4500 (NOT cap9000 - the 11,344 extra images carry zero boxes for
Doors/Tricycle/Potholes/Trash Bins/Stairs/Bench and are 76% Person+Vehicle, so
they dilute exactly the classes that already need help) with the recovered
boxes from the dual-detector proposal run merged in.

ACCEPTED = tiers A and C only: yolov8x proposed it AND our own v2 model
independently drew a same-class box over it at IoU >= 0.5. Tiers B and D
(v2 silent) and E (low confidence) are NOT promoted - the visual audit found
confident errors throughout them (a bollard as Chairs at 0.78, a stone wall as
Bench at 0.71, a blurred smear as Vehicle at 0.90).

NOTHING EXISTING IS TOUCHED. Specifically:

  * dataset/processed/<source>/labels/ is opened READ-ONLY. It remains the
    authoritative 15-class store. The recovered boxes live only in the new tree
    and in the accepted-proposals JSON, so the original annotation state is
    always recoverable by rebuilding without this script.

  * Images are HARDLINKS into dataset/bundle/images/, never symlinks, never
    copies. DEC-135: final_v2b/val/images held absolute symlinks, `tar czf`
    archived the links, and on the pod every one resolved to a /Users/luna/...
    path that does not exist - which would have calibrated INT8 on ZERO images.
    Hardlinks survive tar, cost no disk, and cannot dangle. A hardlink also
    means this tree adds ~0 bytes of image data: it shares inodes with bundle/.

  * The script ABORTS if the target tree exists. It never deletes or overwrites.

  * data.yaml carries NO 'path' key (DEC-066: ultralytics resolves train/val/test
    relative to the yaml's own directory when 'path' is absent).

  * Empty label files ARE written for background images, matching final_v2b.

Duplicate suppression applied to every proposal before acceptance (each of these
was measured; the counts are printed at the end):

  1. same-class fully nested      IoMin >= 0.90   - re-annotation of one object
  2. nested inside a Tricycle box                 - DEC-109/111 alias: a tricycle
                                                    IS a motorcycle plus a cab,
                                                    so yolov8x re-proposes its
                                                    motorcycle half
  3. same-class re-box            IoU   >= 0.30   - same object, different extent
  4. same-class partial nest      IoMin >= 0.50   - under the 0.90 bar, still a dup
  5. proposal vs proposal         IoU   >= 0.30   - yolov8x's own NMS runs at 0.7,
                                                    so it emits its own duplicates
  6. overlaps a RARE-class box    IoMin >= 0.30   - Stairs/Potholes/Doors/Trash
                                                    Bins/Tricycle get ZERO recovery
                                                    (no COCO class maps to them),
                                                    so a proposal landing on one
                                                    can only damage it. 655 landed
                                                    on Stairs alone.

Usage:
    python3 scripts/build/build_cap4500_v3_tree.py                 # dry run
    python3 scripts/build/build_cap4500_v3_tree.py --apply
    python3 scripts/build/build_cap4500_v3_tree.py --apply --splits train
"""
import json, os, sys
from pathlib import Path
from collections import Counter, defaultdict

APPLY = "--apply" in sys.argv
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "dataset" / "processed"
BUNDLE = ROOT / "dataset" / "bundle" / "images"
MANIFEST = ROOT / "dataset" / "bundle" / "manifests" / "cap4500.json"
PROPOSALS = [ROOT / "dataset" / "proposals" / "recovery_cap4500.jsonl",
             ROOT / "dataset" / "proposals" / "recovery_cap9000_extra.jsonl"]
DEST = ROOT / "dataset" / "final_cap4500_nc13_v3"
ACCEPTED_OUT = ROOT / "dataset" / "proposals" / "accepted_v3_AC.json"

NAMES15 = ["Person","Vehicle","Motorcycle","Pole","Animals","Shelf","Doors","Chairs",
           "Tables","Tricycle","Potholes","Trash Bins","Bicycle","Stairs","Bench"]
SHELF, POLE = 5, 3
DROP = {SHELF, POLE}
NC = 13

ACCEPT_TIERS = {"A", "C"}
RARE = {"Stairs", "Potholes", "Doors", "Trash Bins", "Tricycle"}
ALIAS = {"Tricycle"}

SPLITS = ["train", "val", "test"]
for a in sys.argv:
    if a.startswith("--splits"):
        SPLITS = a.split("=", 1)[1].split(",") if "=" in a else SPLITS
if "--splits" in sys.argv:
    i = sys.argv.index("--splits")
    if i + 1 < len(sys.argv):
        SPLITS = sys.argv[i + 1].split(",")

SOURCES = sorted((p.name for p in PROC.iterdir() if p.is_dir()), key=len, reverse=True)


def split_source(entry):
    """Manifest entries carry no extension and may contain dots. Never use .stem.

    Roboflow filenames look like `..._jpg.rf.<32-hex>`; Path.stem would strip the
    hash as if it were an extension and silently mis-key every one of them.
    """
    for s in SOURCES:
        if entry.startswith(s + "__"):
            return s, entry[len(s) + 2:]
    return None, None


def _inter(a, b):
    return (max(0, min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2))
            * max(0, min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2)))


def iou(a, b):
    i = _inter(a, b)
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


def iomin(a, b):
    i = _inter(a, b)
    m = min(a[2] * a[3], b[2] * b[3])
    return i / m if m > 0 else 0.0


_gt_cache = {}


def gt15(entry):
    """The authoritative 15-class labels for one entry. READ ONLY."""
    if entry in _gt_cache:
        return _gt_cache[entry]
    s, n = split_source(entry)
    out = []
    if s:
        p = PROC / s / "labels" / f"{n}.txt"
        if p.exists():
            for ln in p.read_text().splitlines():
                q = ln.split()
                if len(q) == 5:
                    out.append((NAMES15[int(q[0])], tuple(map(float, q[1:]))))
    _gt_cache[entry] = out
    return out


def remap15_to_nc13(lines):
    """15-class id lines -> nc13 ids. Returns [] for a background image."""
    out = []
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        cid = int(parts[0])
        if cid in DROP:
            continue
        parts[0] = str(cid - sum(1 for d in DROP if d < cid))
        out.append(" ".join(parts))
    return out


NAMES13 = [n for i, n in enumerate(NAMES15) if i not in DROP]
NAME_TO_ID13 = {n: i for i, n in enumerate(NAMES13)}


def classify_proposals(pool):
    """Apply every duplicate/safety filter and record a DISPOSITION for every proposal.

    Returns {entry: [proposal + "disposition"]}, where disposition is either
    "accepted" or the reason the proposal was dropped.

    Keeping the rejects (rather than counting and discarding them) is what lets the
    review notebook answer the question that matters when you spot an obviously
    unlabelled object: was it proposed and filtered out, or did BOTH detectors miss
    it? Those look identical on the image and have opposite fixes.
    """
    by_img = defaultdict(list)
    for f in PROPOSALS:
        if not f.exists():
            sys.exit(f"ABORT: missing proposal file {f}")
        with f.open() as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d["image"] in pool:
                    by_img[d["image"]].append(d)

    out = {}
    for entry, ds in by_img.items():
        g = gt15(entry)
        decided, survivors = [], []
        for d in ds:
            box, cls = tuple(d["box"]), d["cls"]
            nested_any = {gc for gc, gb in g if iomin(box, gb) >= 0.90}
            same = [gb for gc, gb in g if gc == cls]
            if cls in nested_any:
                d["disposition"] = "1 same-class nested >=0.90"
            elif nested_any & ALIAS:
                d["disposition"] = "2 inside a Tricycle (alias)"
            elif max([iou(box, x) for x in same] + [0.0]) >= 0.30:
                d["disposition"] = "3 same-class re-box >=0.30 IoU"
            elif max([iomin(box, x) for x in same] + [0.0]) >= 0.50:
                d["disposition"] = "4 same-class partial nest >=0.50"
            elif any(gc in RARE and iomin(box, gb) >= 0.30 for gc, gb in g):
                d["disposition"] = "6 lands on a rare-class box"
            else:
                survivors.append(d)
                continue
            decided.append(d)

        # greedy NMS across surviving proposals of the same class, highest conf wins
        survivors.sort(key=lambda d: -d["x_conf"])
        deduped = []
        for d in survivors:
            if any(d["cls"] == o["cls"] and iou(tuple(d["box"]), tuple(o["box"])) >= 0.30
                   for o in deduped):
                d["disposition"] = "5 proposal vs proposal >=0.30"
                decided.append(d)
                continue
            deduped.append(d)

        for d in deduped:
            d["disposition"] = ("accepted" if d["tier"] in ACCEPT_TIERS
                                else f"tier {d['tier']} not accepted")
            decided.append(d)
        out[entry] = decided
    return out


def accept_proposals(pool):
    """Thin wrapper over classify_proposals(): the accepted subset + a reject tally.

    Returns (accepted_by_image, rejection_counter). The dicts handed back carry no
    "disposition" key, so accepted_v3_AC.json stays byte-comparable with the one
    written when the tree was built.
    """
    rej = Counter()
    accepted = {}
    for entry, ds in classify_proposals(pool).items():
        keep = []
        for d in ds:
            if d["disposition"] == "accepted":
                keep.append({k: v for k, v in d.items() if k != "disposition"})
            else:
                rej[d["disposition"]] += 1
        if keep:
            accepted[entry] = keep
    return accepted, rej


def main():
    if DEST.exists():
        sys.exit(f"ABORT: {DEST} already exists. This script never overwrites.\n"
                 f"       Move it aside first if you really mean to rebuild.")

    man = json.loads(MANIFEST.read_text())
    pool = set(man["train"]) | set(man["val"]) | set(man["test"])

    index = {}
    for p in BUNDLE.iterdir():
        if p.is_file():
            index.setdefault(p.name[:p.name.rfind(p.suffix)] if p.suffix else p.name, p)

    print("Filtering proposals ...")
    accepted, rej = accept_proposals(pool)
    n_new = sum(len(v) for v in accepted.values())
    print(f"  accepted {n_new:,} boxes over {len(accepted):,} images\n")

    stats = Counter()
    per_class_new = Counter()
    per_class_gt = Counter()
    new_box_index = {}          # entry -> number of ORIGINAL boxes (new ones follow)
    missing_img, missing_lab = [], []

    for split in SPLITS:
        if APPLY:
            (DEST / split / "images").mkdir(parents=True, exist_ok=True)
            (DEST / split / "labels").mkdir(parents=True, exist_ok=True)
        for e in man[split]:
            src_img = index.get(e)
            if src_img is None:
                missing_img.append(e)
                continue
            s, n = split_source(e)
            lab = PROC / s / "labels" / f"{n}.txt" if s else None
            if lab is None or not lab.exists():
                missing_lab.append(e)
                continue

            orig = remap15_to_nc13([l for l in lab.read_text().splitlines() if l.strip()])
            for ln in orig:
                per_class_gt[NAMES13[int(ln.split()[0])]] += 1

            added = []
            for d in accepted.get(e, []):
                cid = NAME_TO_ID13.get(d["cls"])
                if cid is None:          # class not in the nc13 schema - cannot happen
                    continue
                x, y, w, h = d["box"]
                added.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                per_class_new[d["cls"]] += 1

            lines = orig + added
            stats[f"{split}_images"] += 1
            stats["boxes_original"] += len(orig)
            stats["boxes_added"] += len(added)
            if not lines:
                stats[f"{split}_background"] += 1
            if added:
                stats[f"{split}_enriched"] += 1
                new_box_index[e] = len(orig)

            if APPLY:
                di = DEST / split / "images" / src_img.name
                if not di.exists():
                    os.link(src_img, di)                      # HARDLINK, not symlink
                (DEST / split / "labels" / f"{e}.txt").write_text(
                    ("\n".join(lines) + "\n") if lines else "")

    if APPLY:
        (DEST / "data.yaml").write_text(
            f"# cap4500 + recovered boxes (tiers A and C), nc={NC}.\n"
            "# Built by scripts/build/build_cap4500_v3_tree.py\n"
            "# No 'path' key on purpose - ultralytics resolves train/val/test relative to\n"
            "# THIS file's own directory when 'path' is absent (DEC-066).\n"
            "train: train/images\nval: val/images\ntest: test/images\n"
            f"nc: {NC}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES13)))
        # Sidecar: for every enriched image, how many boxes were ORIGINAL. Everything
        # at an index >= that value is a recovered box. The review notebook uses this
        # to tag new boxes individually without re-deriving any geometry.
        (DEST / "new_boxes.json").write_text(json.dumps({
            "built_from": "dataset/final_cap4500_nc13_v3",
            "accept_tiers": sorted(ACCEPT_TIERS),
            "splits_enriched": SPLITS,
            "n_new_boxes": stats["boxes_added"],
            "original_box_count": new_box_index,
        }, indent=2))
        ACCEPTED_OUT.write_text(json.dumps(accepted, indent=2))

    head = "DRY RUN - nothing written" if not APPLY else "APPLIED"
    print(f"{head}\n")
    print(f"  tree   {DEST.relative_to(ROOT)}  (nc={NC})")
    print(f"  splits enriched: {', '.join(SPLITS)}")
    for sp in ("train", "val", "test"):
        if stats[f"{sp}_images"]:
            print(f"    {sp:<6} {stats[f'{sp}_images']:>7,} images   "
                  f"{stats[f'{sp}_background']:>5,} background   "
                  f"{stats[f'{sp}_enriched']:>6,} enriched")
    tot = stats["boxes_original"] + stats["boxes_added"]
    print(f"\n    boxes original {stats['boxes_original']:>8,}")
    print(f"    boxes added    {stats['boxes_added']:>8,}   "
          f"(+{100*stats['boxes_added']/max(stats['boxes_original'],1):.1f}%)")
    print(f"    boxes total    {tot:>8,}")

    print("\n  per class:")
    print(f"    {'class':<12}{'original':>10}{'added':>8}{'+%':>8}{'total':>10}")
    for c in NAMES13:
        o, a = per_class_gt[c], per_class_new[c]
        print(f"    {c:<12}{o:>10,}{a:>8,}{(100*a/o if o else 0):>7.1f}%{o+a:>10,}")

    print("\n  proposals rejected:")
    for k in sorted(rej):
        print(f"    {rej[k]:>8,}  {k}")

    if missing_img:
        print(f"\n  !! {len(missing_img):,} images not found in bundle/")
    if missing_lab:
        print(f"  !! {len(missing_lab):,} labels not found in processed/")
    if not missing_img and not missing_lab:
        print("\n  every manifest entry resolved to an image and a label.")
    if not APPLY:
        print("\n  Re-run with --apply to write the tree.")


if __name__ == "__main__":
    main()

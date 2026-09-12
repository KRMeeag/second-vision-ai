"""
Find Open Images boxes that are the SAME physical object annotated under two
different native labels which both map to one canonical class (DEC-133).

DEC-127 extended the mappings so that Open Images hierarchy parents, children
and siblings collapse onto one canonical class:

    Person <- Person, Man, Woman, Boy, Girl
    Tables <- Table, Desk, Kitchen & dining room table, Coffee table
    Chairs <- Chair, Couch, Stool, Sofa bed
    Vehicle <- Car, Van, Taxi, Bus, Truck

Open Images frequently annotates one object under several of these at once
(measured: Desk+Table 230x, Girl+Woman 164x, Man+Person 72x). After mapping,
that is two boxes on one object.

The discriminator is the RAW native label, not IoU:
  * different native labels + high IoU -> one object under two names  -> DROP one
  * same native label      + high IoU  -> two real adjacent objects   -> KEEP both

Measured across IoU bands, different-native pairs dominate above ~0.70 and
same-native pairs dominate below it, so a plain IoU threshold at 0.70 would
destroy 132 genuinely distinct objects. This rule does not.

Which box survives: the one that also exists in the v1 tree (dataset/final),
so the v1-vs-v2 comparison is not perturbed by which twin was kept. If neither
or both are in v1, the first is kept.

Writes dataset/reports/dup_hierarchy_candidates.json. Changes no labels.

Usage:
    python3 scripts/preprocess/find_hierarchy_duplicates.py
    python3 scripts/preprocess/find_hierarchy_duplicates.py --iou 0.60
"""
import argparse
import collections
import json
from pathlib import Path

import yaml

RAW = Path("dataset/raw/open_images")
V2 = Path("dataset/final_v2b")
V1 = Path("dataset/final")


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def iomin(a, b):
    """Intersection over the SMALLER box. ~1.0 means one box is nested inside
    the other -- one object annotated at two extents, never two adjacent
    objects. This is what separates a same-label duplicate from a same-label
    neighbour, which plain IoU cannot do."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    smaller = min(aw * ah, bw * bh)
    return inter / smaller if smaller > 0 else 0.0


def load(p, names):
    out = []
    if not p.is_file():
        return out
    for line in p.read_text().split("\n"):
        line = line.strip()
        if not line:
            continue
        f = line.split()
        c = int(f[0])
        cx, cy, w, h = map(float, f[1:5])
        out.append((names[c], (cx - w / 2, cy - h / 2, w, h), line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.70)
    ap.add_argument("--iomin", type=float, default=0.90,
                    help="same-native pairs nested at or above this are ALSO duplicates")
    ap.add_argument("--apply", action="store_true",
                    help="remove the proposed boxes from dataset/processed/open_images/labels")
    args = ap.parse_args()

    n2 = {int(k): v for k, v in yaml.safe_load((V2 / "data.yaml").read_text())["names"].items()}
    n1 = {int(k): v for k, v in yaml.safe_load((V1 / "data.yaml").read_text())["names"].items()}

    cand = {}
    for f in V2.rglob("*/labels/open_images__*.txt"):
        boxes = load(f, n2)
        if len(boxes) < 2:
            continue
        pairs = [(i, j, iou(boxes[i][1], boxes[j][1]))
                 for i in range(len(boxes)) for j in range(i + 1, len(boxes))
                 if boxes[i][0] == boxes[j][0] and iou(boxes[i][1], boxes[j][1]) >= args.iou]
        if pairs:
            cand[f.stem.split("__", 1)[1]] = (f, boxes, pairs)

    # resolve every candidate box to its raw native Open Images label
    raw = collections.defaultdict(list)
    for folder in sorted(p.name for p in RAW.iterdir() if p.is_dir()):
        lp = RAW / folder / "labels.json"
        if not lp.is_file():
            continue
        j = json.loads(lp.read_text())
        cats = {c["id"]: c["name"] for c in j["categories"]}
        info = {i["id"]: (Path(i["file_name"]).stem, i["width"], i["height"]) for i in j["images"]}
        for a in j["annotations"]:
            st, W, H = info[a["image_id"]]
            if st not in cand:
                continue
            x, y, w, h = a["bbox"]
            raw[st].append((cats.get(a["category_id"], "?"), (x / W, y / H, w / W, h / H)))

    out = []
    protected = []
    kept_same_native = 0
    reasons = collections.Counter()
    for stem, (f, boxes, pairs) in cand.items():
        split = f.parts[2]
        v1boxes = load(V1 / split / "labels" / f.name, n1)
        dropped = set()
        for i, j, v in pairs:
            if i in dropped or j in dropped:
                continue
            ni = max(raw[stem], key=lambda r: iou(boxes[i][1], r[1]), default=("?", None))[0]
            nj = max(raw[stem], key=lambda r: iou(boxes[j][1], r[1]), default=("?", None))[0]
            if ni == nj and iomin(boxes[i][1], boxes[j][1]) >= args.iomin:
                # same label, but one box sits inside the other: one object
                # annotated at two extents (measured: Girl+Girl IoMin 1.00,
                # Table+Table 0.997, Bicycle+Bicycle 1.00). A duplicate.
                ni = ni + " (outer)"
                nj = nj + " (nested)"
            elif ni == nj:
                kept_same_native += 1
                protected.append({
                    "stem": stem, "split": split, "canonical": boxes[i][0],
                    "native": ni, "iou": round(v, 4),
                    "line_a": boxes[i][2], "line_b": boxes[j][2],
                })
                continue
            in_v1 = lambda b: any(l == b[0] and iou(b[1], vb) >= 0.99 for l, vb, _ in v1boxes)
            drop = j if (in_v1(boxes[i]) or not in_v1(boxes[j])) else i
            keep = i if drop == j else j
            dropped.add(drop)
            reasons[tuple(sorted((ni, nj)))] += 1
            out.append({
                "stem": stem, "split": split, "canonical": boxes[drop][0],
                "native_dropped": nj if drop == j else ni,
                "native_kept": ni if drop == j else nj,
                "iou": round(v, 4),
                "line_dropped": boxes[drop][2], "line_kept": boxes[keep][2],
            })

    rep = Path("dataset/reports/dup_hierarchy_candidates.json")
    rep.write_text(json.dumps(
        {"iou_threshold": args.iou, "n_boxes_to_drop": len(out),
         "same_native_pairs_protected": kept_same_native,
         "protected": protected,
         "by_native_pair": {f"{a} + {b}": n for (a, b), n in reasons.most_common()},
         "candidates": out}, indent=2))

    print(f"IoU threshold: {args.iou}")
    n_nested = sum(1 for o in out if "(nested)" in o["native_dropped"] or "(outer)" in o["native_dropped"])
    print(f"boxes proposed for removal        : {len(out):,}")
    print(f"   of which hierarchy (two names) : {len(out) - n_nested:,}")
    print(f"   of which nested  (same name)   : {n_nested:,}")
    print(f"same-native pairs PROTECTED       : {kept_same_native:,}  <- side-by-side, not nested")
    print(f"images affected                   : {len({o['stem'] for o in out}):,}")
    print("\nby native-label pair:")
    for (a, b), n in reasons.most_common(12):
        print(f"   {a + ' + ' + b:52s} {n:5,}")
    print("\nby canonical class:")
    for k, n in collections.Counter(o["canonical"] for o in out).most_common():
        print(f"   {k:12s} {n:5,}")
    print(f"\nreport -> {rep}")

    if not args.apply:
        print("\n(dry run -- pass --apply to remove them from processed/open_images/labels)")
        return

    # processed/ is the 15-class tree; the report's lines are v2b 14-class ids.
    # Shelf (id 5) was dropped, so every v2b id >= 5 maps back to id + 1 (DEC-126).
    proc = Path("dataset/processed/open_images/labels")
    removed = 0
    missing = []
    by_stem = collections.defaultdict(list)
    for o in out:
        by_stem[o["stem"]].append(o["line_dropped"])

    for stem, lines in by_stem.items():
        f = proc / f"{stem}.txt"
        if not f.is_file():
            missing.append((stem, "no processed label file"))
            continue
        cur = [l.strip() for l in f.read_text().split("\n") if l.strip()]
        for ln in lines:
            parts = ln.split()
            cid = int(parts[0])
            parts[0] = str(cid + 1 if cid >= 5 else cid)
            target = " ".join(parts)
            if target in cur:
                cur.remove(target)          # removes ONE occurrence
                removed += 1
            else:
                missing.append((stem, target))
        f.write_text(("\n".join(cur) + "\n") if cur else "")

    print(f"\nAPPLIED: removed {removed:,} boxes from {len(by_stem):,} files in {proc}")
    if missing:
        print(f"  !! {len(missing)} not found -- NOTHING was skipped silently, inspect:")
        for m in missing[:10]:
            print("     ", m)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

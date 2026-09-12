"""
Remove geometric near-duplicate boxes from dataset/processed/<source>/ (DEC-132).

The Open Images converter deduplicates on a key built from ABSOLUTE COCO
pixel coords rounded to 1 decimal -- a 0.1 px tolerance, which only ever
catches bit-identical copies. Open Images boxes the same physical object
slightly differently in each class-folder export, so the cross-folder
merge (DEC-127) left near-twins that differ by a few pixels.

This pass is geometric: within one label file, same class, IoU >= --iou,
keep the FIRST occurrence and drop the later one. Keeping the first
preserves the v1 box wherever a v1 box exists, so the v1-vs-v2 comparison
is not perturbed by which twin survived.

Writes a report to dataset/reports/dedupe_geometric_report.json.

Usage:
    python3 scripts/preprocess/dedupe_geometric.py                 # dry run
    python3 scripts/preprocess/dedupe_geometric.py --apply
    python3 scripts/preprocess/dedupe_geometric.py --iou 0.95 --apply
"""
import argparse
import json
import collections
from pathlib import Path

import yaml

PROC = Path("dataset/processed")
TREES = ("labels", "labels_reviewed")


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.90)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    names = yaml.safe_load(Path("dataset/final/data.yaml").read_text())["names"]
    id2name = {int(k): v for k, v in names.items()}

    by_source = collections.Counter()
    by_class = collections.Counter()
    by_tree = collections.Counter()
    files_touched = 0
    boxes_before = 0

    for src_dir in sorted(p for p in PROC.iterdir() if p.is_dir()):
        for tree in TREES:
            d = src_dir / tree
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.txt")):
                lines = [l.strip() for l in f.read_text().split("\n") if l.strip()]
                boxes_before += len(lines)
                if len(lines) < 2:
                    continue
                parsed = []
                for l in lines:
                    p = l.split()
                    c = int(p[0])
                    cx, cy, w, h = map(float, p[1:5])
                    parsed.append((c, (cx - w / 2, cy - h / 2, w, h), l))
                drop = [False] * len(parsed)
                for i in range(len(parsed)):
                    if drop[i]:
                        continue
                    for j in range(i + 1, len(parsed)):
                        if drop[j] or parsed[i][0] != parsed[j][0]:
                            continue
                        if iou(parsed[i][1], parsed[j][1]) >= args.iou:
                            drop[j] = True
                            by_source[src_dir.name] += 1
                            by_class[id2name.get(parsed[j][0], f"?{parsed[j][0]}")] += 1
                            by_tree[tree] += 1
                if any(drop):
                    files_touched += 1
                    if args.apply:
                        kept = [parsed[k][2] for k in range(len(parsed)) if not drop[k]]
                        f.write_text("\n".join(kept) + "\n")

    total = sum(by_source.values())
    print(f"IoU threshold: {args.iou}")
    print(f"boxes scanned: {boxes_before:,}")
    print(f"boxes dropped: {total:,}  across {files_touched:,} files")
    print(f"\nby tree:   {dict(by_tree)}")
    print("\nby source:")
    for k, v in by_source.most_common():
        print(f"   {k:44s} {v:5,}")
    print("\nby class:")
    for k, v in by_class.most_common():
        print(f"   {k:16s} {v:5,}")

    if args.apply:
        rep = {
            "iou_threshold": args.iou,
            "boxes_scanned": boxes_before,
            "boxes_dropped": total,
            "files_touched": files_touched,
            "by_tree": dict(by_tree),
            "by_source": dict(by_source),
            "by_class": dict(by_class),
        }
        out = Path("dataset/reports/dedupe_geometric_report.json")
        out.write_text(json.dumps(rep, indent=2))
        print(f"\nAPPLIED -- report written to {out}")
    else:
        print("\n(dry run -- pass --apply to write)")


if __name__ == "__main__":
    main()

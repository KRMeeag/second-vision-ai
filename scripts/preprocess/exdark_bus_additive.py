"""
Additively add ExDark `Bus` boxes as canonical `vehicle` (DEC-129).

Only images ALREADY present in dataset/final_v2a are touched. ExDark
images that carry Bus boxes but are not in the frozen pool are skipped
deliberately: admitting them would grow the image set, break the
v1-vs-v2 controlled comparison, and force a ~3 h image re-upload.

Idempotent — a Bus line already present in a label file is counted as
`already_present` and not appended twice.

Usage:
    python3 scripts/preprocess/exdark_bus_additive.py            # dry run
    python3 scripts/preprocess/exdark_bus_additive.py --apply
"""
import sys, collections
from pathlib import Path
sys.path.insert(0, "scripts")
from PIL import Image

APPLY = "--apply" in sys.argv
root = Path("dataset/raw/exdark")
IMG, ANN = root/"ExDark", root/"ExDark_Annno"
proc_lab = Path("dataset/processed/exdark/labels")
proc_img = Path("dataset/processed/exdark/images")
pool = {p.stem.split("__",1)[1] for p in Path("dataset/final_v2a").rglob("exdark__*.txt")}

def yolo(x,y,w,h,W,H):
    return ((x+w/2)/W, (y+h/2)/H, w/W, h/H)

stats = collections.Counter()
to_write = collections.defaultdict(list)
bus_only_imgs = []

for folder in sorted(p.name for p in IMG.iterdir() if p.is_dir()):
    adir = ANN/folder
    if not adir.is_dir(): continue
    for ipath in sorted(p for p in (IMG/folder).iterdir() if p.is_file() and not p.name.startswith(".")):
        apath = adir/f"{ipath.name}.txt"
        if not apath.is_file(): continue
        stem = ipath.stem
        buses = []
        with apath.open(errors="replace") as fh:
            next(fh, None)
            for line in fh:
                p_ = line.split()
                if p_ and p_[0] == "Bus":
                    buses.append(tuple(int(v) for v in p_[1:5]))
        if not buses: continue
        stats["images_with_bus"] += 1
        stats["bus_boxes_total"] += len(buses)
        if stem not in pool:
            bus_only_imgs.append(stem)
            stats["images_not_in_pool"] += 1
            stats["bus_boxes_skipped_not_pooled"] += len(buses)
            continue
        lf = proc_lab/f"{stem}.txt"
        if not lf.exists():
            stats["pooled_but_no_label_file"] += 1; continue
        cands = sorted(proc_img.glob(stem + ".*"))
        if not cands:
            stats["pooled_but_no_image"] += 1; continue
        ip = cands[0]
        with Image.open(ip) as im: W,H = im.size
        existing = [l.strip() for l in lf.read_text().split("\n") if l.strip()]
        for (x,y,w,h) in buses:
            cx,cy,nw,nh = yolo(x,y,w,h,W,H)
            cx,cy = min(max(cx,0),1), min(max(cy,0),1)
            nw,nh = min(nw,1), min(nh,1)
            if nw<=0 or nh<=0:
                stats["invalid_dropped"] += 1; continue
            line = f"1 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
            if line in existing:
                stats["already_present"] += 1; continue
            to_write[stem].append(line)
            stats["bus_boxes_to_add"] += 1

print("=== ExDark Bus measurement ===")
for k,v in sorted(stats.items()): print(f"  {k:32s} {v}")
print(f"  images receiving new boxes        {len(to_write)}")
print(f"\n  Bus-only images that would be EXCLUDED: {len(bus_only_imgs)}")

if APPLY:
    for stem, lines in to_write.items():
        for tree in ["labels"]:
            f = Path("dataset/processed/exdark")/tree/f"{stem}.txt"
            cur = f.read_text()
            if cur and not cur.endswith("\n"): cur += "\n"
            f.write_text(cur + "\n".join(lines) + "\n")
    print(f"\nAPPLIED: appended {stats['bus_boxes_to_add']} Bus->Vehicle boxes to {len(to_write)} files in processed/exdark/labels")
else:
    print("\n(dry run — pass --apply to write)")

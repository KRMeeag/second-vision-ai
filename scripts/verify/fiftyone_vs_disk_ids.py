"""
Compare FiftyOne review-dataset label NAMES against on-disk label IDS
for every reviewed source (DEC-130).

This is the cheapest available detector for a schema-shift defect: it
compares two representations that a shifted class id cannot corrupt in
the same direction. `processed/` is the 15-class tree, so disk ids are
resolved through dataset/final/data.yaml.

Run this after ANY write to dataset/processed/.

Raw count mismatches are mostly benign and are NOT the thing to read:
  * dropped classes FiftyOne still holds (Pedestrian Lane, Elevator,
    Escalator) -> negative delta
  * the review datasets predate the Aug-26 dedup and DEC-132's geometric
    dedup -> negative delta
  * a source promoted after its review dataset was last rebuilt -> disk
    legitimately has MORE -> positive delta

**The signature to read is the SHIFT SUSPECTS section at the end**: a
source where disk holds a class FiftyOne never had at all. That is what
DEC-130 looked like (disk showed Doors/Trash Bins; FiftyOne had Chairs/
Tables/Bicycle). Sources not present in any final tree are excluded --
they carry stale pre-class-drop ids by design and never reach training.

Usage:
    python3 scripts/verify/fiftyone_vs_disk_ids.py
"""
import sys, collections
from pathlib import Path

import fiftyone as fo
import yaml

SKIP_SUFFIXES = ("_bak_20260911", "_bak_prededupe_20260912")


def main() -> int:
    names15 = yaml.safe_load(Path("dataset/final/data.yaml").read_text())["names"]
    id2name = {int(k): v for k, v in names15.items()}

    rows = []
    for ds_name in sorted(fo.list_datasets()):
        if not ds_name.startswith("review_"):
            continue
        if ds_name.endswith(SKIP_SUFFIXES):
            continue
        src = ds_name[len("review_"):]
        proc = Path("dataset/processed") / src
        if not proc.exists():
            rows.append((src, "NO PROCESSED DIR", None, None))
            continue

        lab_dir = proc / "labels_reviewed"
        if not lab_dir.exists():
            lab_dir = proc / "labels"

        d = fo.load_dataset(ds_name)
        fo_counts, disk_counts = collections.Counter(), collections.Counter()
        for s in d.select_fields(["filepath", "ground_truth"]):
            f = lab_dir / f"{Path(s.filepath).stem}.txt"
            if not f.exists():
                continue
            gt = s["ground_truth"]
            if gt and gt.detections:
                for det in gt.detections:
                    fo_counts[det.label] += 1
            for line in f.read_text().split("\n"):
                line = line.strip()
                if line:
                    cid = int(line.split()[0])
                    disk_counts[id2name.get(cid, f"?{cid}")] += 1
        rows.append((src, None, fo_counts, disk_counts))

    # sources whose labels reach no final tree carry stale pre-class-drop
    # ids by design (e.g. roboflow_escalator_stairs) -- never trained on.
    pooled = set()
    for tree in ("final", "final_v2a", "final_v2b"):
        root = Path("dataset") / tree
        if root.exists():
            for f in root.rglob("*/labels/*.txt"):
                for cand in sorted((p.name for p in Path("dataset/processed").iterdir()
                                    if p.is_dir()), key=len, reverse=True):
                    if f.stem.startswith(cand + "__"):
                        pooled.add(cand)
                        break

    suspects = []
    n_mismatch = 0
    for src, err, fc, dc in rows:
        if err:
            print(f"{src:46s} {err}")
            continue
        diffs = [(k, fc.get(k, 0), dc.get(k, 0))
                 for k in sorted(set(fc) | set(dc))
                 if fc.get(k, 0) != dc.get(k, 0)]
        if not diffs:
            print(f"{src:46s} OK  ({sum(dc.values())} boxes)")
        else:
            n_mismatch += 1
            print(f"{src:46s} MISMATCH")
            for k, f_, d_ in diffs:
                print(f"{'':48s}{k:16s} fiftyone={f_:6d}  disk={d_:6d}  delta={d_ - f_:+d}")
                if f_ == 0 and d_ > 0 and src in pooled:
                    suspects.append((src, k, d_))

    print(f"\n{len(rows) - n_mismatch}/{len(rows)} sources with no count difference")
    print("\n=== SHIFT SUSPECTS (the DEC-130 signature) ===")
    if not suspects:
        print("  NONE -- no pooled source holds a class its review dataset never had.")
        return 0
    for src, k, d_ in suspects:
        print(f"  !! {src}: disk has {d_} x {k!r}, FiftyOne has none")
    return 1


if __name__ == "__main__":
    sys.exit(main())

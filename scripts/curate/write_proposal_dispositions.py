"""Record what happened to EVERY proposal, accepted or not.

The v3 tree only contains the 13,383 boxes that survived. When you then look at a
v3 image and see an obviously unlabelled object, you cannot tell which of three
things happened:

  1. it already has a ground-truth box      -> visible as an original box
  2. it was proposed and then FILTERED OUT  -> invisible, and the reason is lost
  3. neither detector ever proposed it      -> invisible, and nothing to show

(2) and (3) look identical on screen and have opposite fixes: (2) means a filter is
too aggressive, (3) means the detectors are blind there and only manual labelling or
a different model will help.

This writes every proposal with its disposition so the review notebook can draw the
rejects in their own colour and name the filter that dropped each one.

Reuses the build script's own filter chain by import -- the dispositions cannot
drift from what the tree was actually built with.

Output: dataset/proposals/dispositions.jsonl
        one record per proposal: image, cls, box, tier, x_conf, v2_conf, disposition

Usage:
    python3 scripts/curate/write_proposal_dispositions.py
"""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "scripts" / "build" / "build_cap4500_v3_tree.py"
OUT = ROOT / "dataset" / "proposals" / "dispositions.jsonl"

# Import the build script as a module WITHOUT running main(); it guards on
# __name__, so the filter chain and its constants come across untouched.
spec = importlib.util.spec_from_file_location("build_v3", BUILD)
build = importlib.util.module_from_spec(spec)
sys.modules["build_v3"] = build
spec.loader.exec_module(build)


def main():
    man = json.loads(build.MANIFEST.read_text())
    pool = set(man["train"]) | set(man["val"]) | set(man["test"])

    # Which split each entry belongs to, so the notebook need not re-derive it.
    split_of = {}
    for sp in ("train", "val", "test"):
        for e in man[sp]:
            split_of[e] = sp

    print("Classifying every proposal through the build script's own filter chain ...")
    by_img = build.classify_proposals(pool)

    tally = Counter()
    per_class_rejected = Counter()
    n = 0
    with OUT.open("w") as fh:
        for entry, ds in by_img.items():
            for d in ds:
                tally[d["disposition"]] += 1
                if d["disposition"] != "accepted":
                    per_class_rejected[d["cls"]] += 1
                fh.write(json.dumps({
                    "image": entry,
                    "split": split_of.get(entry),
                    "cls": d["cls"],
                    "box": d["box"],
                    "tier": d["tier"],
                    "x_conf": d["x_conf"],
                    "v2_conf": d["v2_conf"],
                    "disposition": d["disposition"],
                }) + "\n")
                n += 1

    acc = tally["accepted"]
    print(f"\n  {n:,} proposals over {len(by_img):,} images -> {OUT.relative_to(ROOT)}\n")
    print(f"    {'disposition':<36}{'count':>9}{'share':>8}")
    for k in sorted(tally, key=lambda k: (k != "accepted", k)):
        print(f"    {k:<36}{tally[k]:>9,}{100*tally[k]/n:>7.1f}%")
    print(f"\n  accepted {acc:,}  (must equal the 13,383 already written into the tree)")
    if acc != 13383:
        print("  !! MISMATCH -- the filter chain has drifted from the built tree.")
    else:
        print("  OK -- matches the tree exactly.")

    print("\n  rejected proposals per class (what review will now be able to SEE):")
    for c, v in per_class_rejected.most_common():
        print(f"    {c:<12}{v:>8,}")


if __name__ == "__main__":
    main()

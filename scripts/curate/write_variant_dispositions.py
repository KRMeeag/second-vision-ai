"""Write per-variant proposal dispositions for the review notebook.

`dataset/proposals/dispositions.jsonl` was produced under the ORIGINAL filter chain
(DEC-140, before the rej_6 correction and the rej_3/rej_4 removals), so its reasons
are stale for the corrected variants. This regenerates them from the variant build's
own filter chain, so what the notebook draws always matches what the tree contains.

Output, one file per variant:
    dataset/proposals/dispositions_<variant>.jsonl

Each record: image, split, cls, box, tier, x_conf, v2_conf, agree, disposition.

Usage:
    python3 scripts/curate/write_variant_dispositions.py
"""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VAR = ROOT / "scripts" / "build" / "build_cap4500_v3_variants.py"

spec = importlib.util.spec_from_file_location("build_variants", VAR)
V = importlib.util.module_from_spec(spec)
sys.modules["build_variants"] = V
spec.loader.exec_module(V)
B = V.B


def main():
    man = json.loads(B.MANIFEST.read_text())
    pool = set(man["train"]) | set(man["val"]) | set(man["test"])
    split_of = {e: sp for sp in ("train", "val", "test") for e in man[sp]}

    for name, cfg in V.VARIANTS.items():
        out = ROOT / "dataset" / "proposals" / f"dispositions_{name}.jsonl"
        print(f"\n{cfg['label']}  ->  {out.relative_to(ROOT)}")
        by_img = V.classify_full(pool, cfg)
        tally = Counter()
        n = 0
        with out.open("w") as fh:
            for entry, ds in by_img.items():
                for d in ds:
                    tally[d["disposition"]] += 1
                    fh.write(json.dumps({
                        "image": entry,
                        "split": split_of.get(entry),
                        "cls": d["cls"],
                        "box": d["box"],
                        "tier": d["tier"],
                        "x_conf": d["x_conf"],
                        "v2_conf": d["v2_conf"],
                        "agree": bool(d["agree"]),
                        "disposition": d["disposition"],
                    }) + "\n")
                    n += 1
        acc = tally["accepted"]
        print(f"  {n:,} proposals, {acc:,} accepted")
        for k, v in tally.most_common():
            if k != "accepted":
                print(f"      {v:>8,}  {k}")

        # The accepted count must match what the tree was actually built with.
        tree = ROOT / "dataset" / name / "new_boxes.json"
        if tree.is_file():
            built = json.loads(tree.read_text())["n_new_boxes"]
            # val/test come from STRICT in both variants, so only the strict tree's
            # total is directly comparable to its own accepted set.
            note = "matches tree" if (name == V.STRICT_KEY and acc == built) else \
                   f"tree holds {built:,} (val/test taken from STRICT)"
            print(f"  {note}")


if __name__ == "__main__":
    main()

"""Build the STRICT and PERMISSIVE v3 label variants, side by side.

Separate from build_cap4500_v3_tree.py on purpose: that script is the provenance
record for `dataset/final_cap4500_nc13_v3` as originally built (DEC-140), and is
left alone. Its filter helpers are imported here so the geometry cannot drift.

Two kinds of change are in play and they are NOT the same thing:

  CORRECTIONS -- applied to BOTH variants, because the original rule was wrong:

    * rej_6 now fires only on a SAME-EXTENT overlap (IoU >= 0.70) with a rare-class
      box, not on any IoMin >= 0.30 touch. Measured: 80.9% of the boxes the old rule
      dropped were legitimate -- a Person standing on Stairs (401), a Bench beside
      Stairs (130), a Person in a doorway (38). Only 6.9% were genuine relabels of
      the rare object. The old justification ("a proposal there can only contradict
      it") was simply false for a person standing on a staircase.

    * rej_3 (same-class re-box, IoU >= 0.30) removed. Worth only 7 boxes on its own --
      almost everything it caught is caught by partial-nest or self-overlap anyway.

    * rej_4 (same-class partial nest, IoMin >= 0.50) removed. Adjacent objects --
      chairs in a row, people in a crowd -- routinely half-overlap each other.

  PERMISSIVENESS -- the actual experimental variable, PERMISSIVE only:

    * tier D with PARTIAL agreement (v2_conf >= 0.25). Note tier D with STRICT
      agreement does not exist: tier() is `'C' if agree else 'D'`, so an agreeing box
      at that confidence is already tier C. Partial agreement means v2 fired on the
      same class nearby but under the IoU 0.5 matching bar -- corroboration of
      existence, disagreement on extent.

    * tier E with STRICT agreement and v2_conf >= 0.25. tier() discarded agreement
      below 0.50 confidence (`return 'E'` regardless), collapsing 12,260 agreeing and
      27,624 silent proposals into one bucket. This recovers the agreeing half.

  STILL REJECTED IN BOTH: same-class nesting >= 0.90, the Tricycle alias (DEC-109/111),
  proposal-vs-proposal self-overlap, tier B, and any tier E without agreement.

VAL AND TEST ARE IDENTICAL IN BOTH VARIANTS, and always use the STRICT rule.
A false box in an evaluation split is an unmatchable false negative: it caps measured
recall permanently and cannot be absorbed the way training noise can. Holding the
eval splits fixed is also what makes a strict-vs-permissive training comparison mean
anything -- only the training labels differ, so the yardstick never moves.

Usage:
    python3 scripts/build/build_cap4500_v3_variants.py            # dry run
    python3 scripts/build/build_cap4500_v3_variants.py --apply
"""
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts" / "build" / "build_cap4500_v3_tree.py"

spec = importlib.util.spec_from_file_location("build_v3_base", BASE)
B = importlib.util.module_from_spec(spec)
sys.modules["build_v3_base"] = B
spec.loader.exec_module(B)

APPLY = "--apply" in sys.argv

VARIANTS = {
    "final_cap4500_nc13_v3s": dict(
        label="STRICT",
        d_partial=None,      # tier D excluded
        e_agree=None,        # tier E excluded
    ),
    "final_cap4500_nc13_v3p": dict(
        label="PERMISSIVE",
        d_partial=0.25,      # tier D, v2 fired on the same class at >= this conf
        e_agree=0.25,        # tier E, v2 strictly agreed at >= this conf
    ),
    "final_cap4500_nc13_v3l": dict(
        label="LOOSE",
        d_partial=None,
        e_agree=None,
        all_tiers=True,      # accept ANY proposal that survived the conflict filters,
                             # with no v2 corroboration required at all
    ),
}
STRICT_KEY = "final_cap4500_nc13_v3s"


def tier_verdict(d, cfg):
    """Tier gate for one proposal. Returns (accepted, reason).

    The reason has to name what v2 actually did, not just "tier E not accepted".
    A tier-E box that v2 STRICTLY AGREED with but whose v2_conf was 0.101 is not
    "unconfirmed" -- it was confirmed, weakly, and the gate is what dropped it.
    6,365 rejected boxes are in exactly that state, and calling them unconfirmed
    makes them indistinguishable from the 7,111 boxes v2 never saw at all.
    """
    t = d["tier"]
    agree, vc = bool(d["agree"]), d["v2_conf"]

    # LOOSE: the conflict filters already ran; nothing else is required.
    if cfg.get("all_tiers"):
        return True, "accepted"

    if t in ("A", "C"):
        return True, "accepted"

    if t == "B":
        return False, "tier B: v2 silent (no agreement at any confidence)"

    if t == "D":
        # tier D cannot have STRICT agreement: tier() is "C if agree else D".
        # Its gate is PARTIAL agreement -- v2 fired on the class, box misaligned.
        if cfg["d_partial"] is None:
            return False, "tier D excluded (strict variant)"
        if vc >= cfg["d_partial"]:
            return True, "accepted"
        if vc > 0:
            return False, "tier D: v2 signal below the gate"
        return False, "tier D: v2 saw nothing"

    if t == "E":
        if cfg["e_agree"] is None:
            return False, "tier E excluded (strict variant)"
        if not agree:
            return (False, "tier E: v2 did not agree"
                    if vc > 0 else "tier E: v2 saw nothing")
        if vc >= cfg["e_agree"]:
            return True, "accepted"
        return False, "tier E: v2 AGREED but below the v2_conf gate"

    return False, f"tier {t} not accepted"


def accept(d, cfg):
    """Boolean form of tier_verdict(), for callers that only need the decision."""
    return tier_verdict(d, cfg)[0]


def classify_full(pool, cfg):
    """Corrected filter chain + this variant's tier gate, recording EVERY proposal.

    Returns {entry: [proposal + "disposition"]} where disposition is "accepted" or
    the reason it was dropped. The review notebook needs the rejects, not just a
    tally: an unlabelled-looking object is diagnosable only if you can see whether a
    proposal was dropped there and why.
    """
    by_img = defaultdict(list)
    for f in B.PROPOSALS:
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
        g = B.gt15(entry)
        decided, surv = [], []
        for d in ds:
            box, cls = tuple(d["box"]), d["cls"]
            nested = {gc for gc, gb in g if B.iomin(box, gb) >= 0.90}
            if cls in nested:
                d["disposition"] = "same-class nested >=0.90"
            elif nested & B.ALIAS:
                d["disposition"] = "inside a Tricycle (alias)"
            # CORRECTED: same-extent relabel only, not any touch
            elif any(gc in B.RARE and B.iou(box, gb) >= 0.70 for gc, gb in g):
                d["disposition"] = "relabels a rare-class box (IoU>=0.70)"
            else:
                surv.append(d)
                continue
            decided.append(d)

        # Proposal-vs-proposal dedup at 0.70, NOT 0.30.
        #
        # Every proposal for one image comes from a single yolov8x predict call, so
        # ultralytics NMS has already deduplicated them -- at IoU 0.7, and CLASS-AWARE.
        # A threshold of 0.30 therefore second-guesses NMS and deletes boxes it
        # deliberately kept as distinct objects: 3,859 pairs sit in 0.30-0.70, of which
        # 1,374 are Person (people standing close in a crowd).
        #
        # The >=0.70 tail is real and must still go: 402 pairs, 304 of them Vehicle,
        # 73 Chairs, 24 Animals, 1 Person. Those are the classes where C2O maps SEVERAL
        # COCO ids onto one of ours (car/bus/truck -> Vehicle, chair/couch -> Chairs).
        # Class-aware NMS never compares a `car` box with a `truck` box on the same
        # vehicle, so both survive and both collapse onto Vehicle as a near-identical
        # pair. Same failure as DEC-133's hierarchy collapse, one layer earlier.
        surv.sort(key=lambda d: -d["x_conf"])
        ded = []
        for d in surv:
            if any(d["cls"] == o["cls"] and B.iou(tuple(d["box"]), tuple(o["box"])) >= 0.70
                   for o in ded):
                d["disposition"] = "duplicate of another proposal (IoU>=0.70, class collapse)"
                decided.append(d)
                continue
            ded.append(d)

        for d in ded:
            _, d["disposition"] = tier_verdict(d, cfg)
            decided.append(d)
        out[entry] = decided
    return out


def classify(pool, cfg):
    """The accepted subset + a reject tally, on top of classify_full()."""
    out, rej = {}, Counter()
    for entry, ds in classify_full(pool, cfg).items():
        keep = []
        for d in ds:
            if d["disposition"] == "accepted":
                keep.append({k: v for k, v in d.items() if k != "disposition"})
            else:
                rej[d["disposition"]] += 1
        if keep:
            out[entry] = keep
    return out, rej


def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    targets = [n for n in VARIANTS if only is None or n in only]
    existing = [n for n in targets if (ROOT / "dataset" / n).exists()]
    if existing:
        sys.exit("ABORT: these already exist and this script never overwrites:\n  "
                 + "\n  ".join(f"dataset/{n}" for n in existing)
                 + "\n       Use --only to build just the new ones.")
    if not targets:
        sys.exit("Nothing to build.")

    man = json.loads(B.MANIFEST.read_text())
    pool = set(man["train"]) | set(man["val"]) | set(man["test"])

    index = {}
    for p in B.BUNDLE.iterdir():
        if p.is_file():
            index.setdefault(p.name[:p.name.rfind(p.suffix)] if p.suffix else p.name, p)

    print("Classifying under each variant ...")
    accepted = {}
    rejects = {}
    # The strict set is always needed: it supplies val/test for EVERY variant.
    for name, cfg in VARIANTS.items():
        if name not in targets and name != STRICT_KEY:
            continue
        accepted[name], rejects[name] = classify(pool, cfg)
        n = sum(len(v) for v in accepted[name].values())
        print(f"  {cfg['label']:<11} {n:>7,} boxes over {len(accepted[name]):,} images")

    summary = {}
    for name in targets:
        cfg = VARIANTS[name]
        dest = ROOT / "dataset" / name
        # val/test ALWAYS come from the strict variant -- identical across both builds
        src_for = {"train": accepted[name],
                   "val": accepted[STRICT_KEY],
                   "test": accepted[STRICT_KEY]}

        stats = Counter()
        per_new = Counter()
        per_orig = Counter()
        new_index = {}

        for split in ("train", "val", "test"):
            acc = src_for[split]
            if APPLY:
                (dest / split / "images").mkdir(parents=True, exist_ok=True)
                (dest / split / "labels").mkdir(parents=True, exist_ok=True)
            for e in man[split]:
                src_img = index.get(e)
                if src_img is None:
                    continue
                s, nm = B.split_source(e)
                lab = B.PROC / s / "labels" / f"{nm}.txt" if s else None
                if lab is None or not lab.exists():
                    continue
                orig = B.remap15_to_nc13(
                    [l for l in lab.read_text().splitlines() if l.strip()])
                for ln in orig:
                    per_orig[B.NAMES13[int(ln.split()[0])]] += 1

                added = []
                for d in acc.get(e, []):
                    cid = B.NAME_TO_ID13.get(d["cls"])
                    if cid is None:
                        continue
                    x, y, w, h = d["box"]
                    added.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    per_new[d["cls"]] += 1

                lines = orig + added
                stats[f"{split}_images"] += 1
                stats["orig"] += len(orig)
                stats["added"] += len(added)
                stats[f"{split}_added"] += len(added)
                if added:
                    new_index[e] = len(orig)
                if APPLY:
                    di = dest / split / "images" / src_img.name
                    if not di.exists():
                        os.link(src_img, di)          # HARDLINK, not symlink (DEC-135)
                    (dest / split / "labels" / f"{e}.txt").write_text(
                        ("\n".join(lines) + "\n") if lines else "")

        if APPLY:
            (dest / "data.yaml").write_text(
                f"# cap4500 v3 {cfg['label']} variant, nc={B.NC}.\n"
                "# Built by scripts/build/build_cap4500_v3_variants.py\n"
                "# val/test are the STRICT labels in BOTH variants, so a strict-vs-permissive\n"
                "# training comparison varies only the TRAINING labels.\n"
                "# No 'path' key on purpose (DEC-066).\n"
                "train: train/images\nval: val/images\ntest: test/images\n"
                f"nc: {B.NC}\nnames:\n"
                + "".join(f"  {i}: {n}\n" for i, n in enumerate(B.NAMES13)))
            (dest / "new_boxes.json").write_text(json.dumps({
                "variant": cfg["label"],
                "d_partial_min_v2conf": cfg["d_partial"],
                "e_agree_min_v2conf": cfg["e_agree"],
                "eval_splits_from": "STRICT",
                "n_new_boxes": stats["added"],
                "original_box_count": new_index,
            }, indent=2))
            (ROOT / "dataset" / "proposals" / f"accepted_{name}.json").write_text(
                json.dumps(accepted[name], indent=2))

        summary[name] = (cfg, stats, per_orig, per_new)

    print(f"\n{'DRY RUN - nothing written' if not APPLY else 'APPLIED'}\n")
    for name, (cfg, stats, per_orig, per_new) in summary.items():
        tot = stats["orig"] + stats["added"]
        print(f"  dataset/{name}   [{cfg['label']}]")
        print(f"    train +{stats['train_added']:,}   val +{stats['val_added']:,}   "
              f"test +{stats['test_added']:,}   (val/test from STRICT in both)")
        print(f"    boxes {stats['orig']:,} + {stats['added']:,} = {tot:,}"
              f"   (+{100*stats['added']/stats['orig']:.1f}%)\n")

    # Per-class comparison across whichever variants were built this run. Counted from
    # the ACCEPTED sets rather than the per-tree stats, so it still works when only one
    # tree is being built (the others already exist on disk).
    cols = [n for n in VARIANTS if n in accepted]
    if cols:
        per = {}
        for n in cols:
            cnt = Counter()
            for ds in accepted[n].values():
                for d in ds:
                    cnt[d["cls"]] += 1
            per[n] = cnt
        head = "".join(f"{VARIANTS[n]['label']:>12}" for n in cols)
        print(f"  {'class':<12}{head}")
        for c in B.NAMES13:
            print(f"  {c:<12}" + "".join(f"{per[n][c]:>12,}" for n in cols))
        print(f"  {'TOTAL':<12}" + "".join(f"{sum(per[n].values()):>12,}" for n in cols))
        print("  (whole-pool accepted sets; each tree takes val/test from STRICT)")

    for n in cols:
        if n in rejects:
            print(f"\n  rejections under {VARIANTS[n]['label']}:")
            for k, v in rejects[n].most_common():
                print(f"    {v:>8,}  {k}")
    if not APPLY:
        print("\n  Re-run with --apply to write the tree(s).")


if __name__ == "__main__":
    main()

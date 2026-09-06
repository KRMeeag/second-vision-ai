"""
Evaluate a trained model on the FROZEN test split and emit paper-ready metrics.

Run after training. Produces a per-class CSV whose row set is identical for every
ablation condition, a JSON summary, and a PASS/FAIL report line against the
pre-registered criterion (DEC-122).

THREE TRAPS THIS SCRIPT EXISTS TO AVOID
---------------------------------------
All three were read out of the installed ultralytics 8.4.118, not inferred.

1. `metrics.box.maps` IS NOT PER-CLASS mAP FOR ABSENT CLASSES.
   `utils/metrics.py:1000-1005`:

       maps = np.zeros(self.nc) + self.map      # every class seeded with OVERALL mAP
       for i, c in enumerate(self.ap_class_index):
           maps[c] = self.ap[i]                 # only PRESENT classes overwritten

   A class with zero test instances therefore reports the overall mAP as its own
   score. On this dataset that would silently invent a plausible number for a
   class the model was never tested on. `maps` is never used here.

2. `metrics.summary()` SILENTLY OMITS CLASSES WITH NO TEST INSTANCES.
   `utils/metrics.py:1232-1265` builds one dict per entry of `ap_class_index`, so
   an absent class produces no row at all. Writing that straight to CSV gives
   different row sets per condition, and the ablation's CSVs stop lining up --
   column-aligned comparison across cap1500/4500/9000 would silently compare
   different classes. Every row here is reindexed against `get_canonical_names()`
   and absent classes are written with an explicit `present=False` and EMPTY
   metric cells -- never 0.0, which would read as "detected nothing" rather than
   "was never tested".

3. THE CONFUSION MATRIX USES A DIFFERENT CONFIDENCE THRESHOLD THAN mAP.
   `models/yolo/detect/val.py:54`:

       self.confusion_matrix_conf = 0.25 if conf is None else conf

   `args.conf` defaults to None for val, so mAP is computed over the full PR curve
   (conf 0.001) while the confusion matrix is built at 0.25. Passing `--conf 0.001`
   to "be precise" would rebuild the matrix at 0.001 and inflate the false-positive
   count with every near-zero-confidence detection -- corrupting the exact number
   DEC-120 says to report. So conf is left UNSET by default. 0.25 is also the
   threshold baked into `hef_deploy`, so the reported FP rate matches what ships.

WHY FALSE POSITIVES COME FROM THE CONFUSION MATRIX AT ALL
---------------------------------------------------------
The dataset contains zero empty-label (background) images, and DEC-120 declined to
add any rather than disturb the pool every nesting guarantee is built on. The agreed
substitute is the background column. `utils/metrics.py` indexes `matrix[detected, truth]`:

    matrix[dc, nc] += 1   # FP -- predicted class dc, ground truth background
    matrix[nc, gc] += 1   # FN -- predicted background, ground truth class gc

so column `nc` is per-class false positives and row `nc` is per-class misses.

NOTHING HERE GATES ANYTHING
---------------------------
DEC-122 pre-registers the criterion as documentation only. This script prints
PASS/FAIL and exits 0 either way. A run under target is a result to report and
discuss, not an error condition. `--strict` is available for scripted use but is
never the default.

Usage
-----
    python3 scripts/train/evaluate.py \
        --weights runs/detect/cap4500_yolov8s/weights/best.pt \
        --data /workspace/final_cap4500/data.yaml
    python3 scripts/train/evaluate.py ... --int8 --tag hef_eval   # quantized run
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.config_loader import REPO_ROOT, get_canonical_names

# ── Pre-registered criterion, DEC-122. Declared before the first run. ────────
PRIMARY_MAP50_FP32 = 0.70
PRIMARY_MAP50_INT8 = 0.65      # 5-point quantization allowance
PER_CLASS_FLOORS = {
    "Person": 0.75, "Vehicle": 0.75,
    "Motorcycle": 0.65, "Tricycle": 0.65, "Bicycle": 0.65,
}
EXPECTED_WEAK = {"Pole", "Potholes"}   # pre-registered, deliberately no floor


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=REPO_ROOT).strip()
    except Exception:
        return "unknown"


def reindex_per_class(summary_rows, canonical, nt_per_class, nt_per_image,
                      fp_col, fn_row) -> list[dict]:
    """One row per CANONICAL class, in canonical order, always.

    `summary_rows` is `metrics.summary()`, which contains a row only for classes
    present in the split. Classes absent from it get `present_in_split=False` and
    EMPTY metric cells -- deliberately not 0.0, which a reader would take as
    "the model detected nothing", and deliberately not `box.maps[cid]`, which
    ultralytics fills with the overall mAP for exactly these classes.
    """
    by_name = {r["Class"]: r for r in summary_rows}
    rows = []
    for cid, name in enumerate(canonical):
        r = by_name.get(name)
        rows.append({
            "class_id": cid,
            "class": name,
            "present_in_split": bool(r),
            "instances": int(nt_per_class[cid]) if nt_per_class is not None else 0,
            "images": int(nt_per_image[cid]) if nt_per_image is not None else 0,
            "precision": r["Box-P"] if r else "",
            "recall": r["Box-R"] if r else "",
            "f1": r["Box-F1"] if r else "",
            "mAP50": r["mAP50"] if r else "",
            "mAP50_95": r["mAP50-95"] if r else "",
            "false_positives_vs_background": int(fp_col[cid]) if fp_col is not None else "",
            "missed_detections": int(fn_row[cid]) if fn_row is not None else "",
            "expected_weak": name in EXPECTED_WEAK,
            "floor": PER_CLASS_FLOORS.get(name, ""),
        })
    return rows


def self_test() -> int:
    """Synthetic checks for the reindexing, including a demonstration of the
    `box.maps` trap this script exists to avoid."""
    import numpy as np

    canonical = get_canonical_names()
    nc = len(canonical)
    absent = {"Stairs", "Bench"}
    overall_map = 0.6123

    summary_rows = [
        {"Class": n, "Images": 10, "Instances": 20, "Box-P": 0.8, "Box-R": 0.7,
         "Box-F1": 0.75, "mAP50": 0.71, "mAP50-95": 0.5}
        for n in canonical if n not in absent
    ]
    nt_per_class = np.array([0 if n in absent else 20 for n in canonical])
    nt_per_image = np.array([0 if n in absent else 10 for n in canonical])
    matrix = np.zeros((nc + 1, nc + 1))
    matrix[3, nc] = 42        # 42 false positives for class 3 (Pole)
    matrix[nc, 3] = 17        # 17 missed Poles
    fp_col, fn_row = matrix[:nc, nc], matrix[nc, :nc]

    rows = reindex_per_class(summary_rows, canonical, nt_per_class, nt_per_image,
                             fp_col, fn_row)
    checks = []

    checks.append(("row set is the full canonical schema", len(rows) == nc))
    checks.append(("rows are in canonical order",
                   [r["class"] for r in rows] == canonical))
    checks.append(("summary() really did omit the absent classes",
                   len(summary_rows) == nc - len(absent)))

    stairs = next(r for r in rows if r["class"] == "Stairs")
    checks.append(("absent class still gets a row", stairs is not None))
    checks.append(("absent class mAP50 is EMPTY, not 0.0", stairs["mAP50"] == ""))
    checks.append(("absent class is not silently 0.0", stairs["mAP50"] != 0.0))
    checks.append(("absent class flagged present_in_split=False",
                   stairs["present_in_split"] is False))
    checks.append(("absent class instances are a genuine 0", stairs["instances"] == 0))

    # The trap, demonstrated rather than asserted in a comment.
    maps = np.zeros(nc) + overall_map
    for cid, n in enumerate(canonical):
        if n not in absent:
            maps[cid] = 0.71
    checks.append(("box.maps WOULD have reported the overall mAP for Stairs",
                   abs(maps[canonical.index("Stairs")] - overall_map) < 1e-9))
    checks.append(("...and this script reports nothing instead",
                   stairs["mAP50"] == ""))

    pole = next(r for r in rows if r["class"] == "Pole")
    checks.append(("FP read from the background COLUMN",
                   pole["false_positives_vs_background"] == 42))
    checks.append(("misses read from the background ROW",
                   pole["missed_detections"] == 17))
    checks.append(("Pole carries the expected-weak flag", pole["expected_weak"] is True))
    person = next(r for r in rows if r["class"] == "Person")
    checks.append(("Person carries its 0.75 floor", person["floor"] == 0.75))
    checks.append(("present class metrics pass through", person["mAP50"] == 0.71))

    ok = 0
    for desc, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {desc}")
        ok += bool(passed)
    print(f"\n{ok}/{len(checks)} checks passed")
    return 0 if ok == len(checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="trained weights (required unless --self-test)")
    ap.add_argument("--data", help="the condition's data.yaml (required unless --self-test)")
    ap.add_argument("--split", default="test", choices=("test", "val"),
                    help="test is the frozen evaluation set (DEC-119). val only for debugging.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None, help="output dir (default: the run dir above weights/)")
    ap.add_argument("--tag", default=None, help="label for these results, e.g. fp32 / hef_eval")
    ap.add_argument("--int8", action="store_true",
                    help="judge against the INT8 secondary target (0.65) instead of 0.70")
    ap.add_argument("--conf", type=float, default=None,
                    help="LEAVE UNSET. Setting it also moves the confusion matrix off 0.25 "
                         "and corrupts the false-positive count -- see the module docstring.")
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic checks on the reindexing logic and exit")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if the criterion is not met. Off by default (DEC-122).")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.weights or not args.data:
        ap.error("--weights and --data are required unless --self-test is given")

    weights = Path(args.weights).resolve()
    data = Path(args.data).resolve()
    for p in (weights, data):
        if not p.exists():
            raise SystemExit(f"not found: {p}")

    out_dir = Path(args.out) if args.out else weights.parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("int8" if args.int8 else "fp32")

    if args.conf is not None:
        print("WARNING: --conf was set explicitly. The confusion matrix is now built at "
              f"{args.conf} instead of 0.25, so the false-positive counts below are NOT "
              "comparable with other runs and should not be reported.")

    from ultralytics import YOLO

    canonical = get_canonical_names()
    model = YOLO(str(weights))

    # Schema check before spending time on inference. A model head that disagrees
    # with classes.yaml produces per-class rows labelled with the wrong classes.
    model_names = [model.names[i] for i in sorted(model.names)]
    if model_names != canonical:
        raise SystemExit(
            "model classes disagree with config/classes.yaml:\n"
            f"  model      : nc={len(model_names)} {model_names}\n"
            f"  classes.yaml: nc={len(canonical)} {canonical}"
        )
    print(f"schema check: model agrees with classes.yaml at nc={len(canonical)}")

    val_kwargs = dict(data=str(data), split=args.split, batch=args.batch,
                      plots=True, project=str(out_dir), name=f"eval_{tag}", exist_ok=True)
    if args.device is not None:
        val_kwargs["device"] = args.device
    if args.conf is not None:
        val_kwargs["conf"] = args.conf

    print(f"\nevaluating {weights.name} on split='{args.split}' ...")
    m = model.val(**val_kwargs)

    # ── reindex per-class results against the canonical schema ──────────────
    by_name = {row["Class"]: row for row in m.summary()}
    present_names = set(by_name)
    missing = [n for n in canonical if n not in present_names]

    cm = getattr(m, "confusion_matrix", None)
    matrix = cm.matrix if cm is not None else None
    nc = len(canonical)
    fp_col = matrix[:nc, nc] if matrix is not None else None   # predicted X, truth background
    fn_row = matrix[nc, :nc] if matrix is not None else None   # predicted background, truth X

    rows = reindex_per_class(m.summary(), canonical, m.nt_per_class, m.nt_per_image,
                             fp_col, fn_row)

    csv_path = out_dir / f"eval_{tag}_{args.split}_per_class.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ── criterion, DEC-122 -- reported, never enforced ───────────────────────
    target = PRIMARY_MAP50_INT8 if args.int8 else PRIMARY_MAP50_FP32
    map50, map5095 = float(m.box.map50), float(m.box.map)
    primary_ok = map50 >= target

    floor_results = []
    for name, floor in PER_CLASS_FLOORS.items():
        r = by_name.get(name)
        got = float(r["mAP50"]) if r else None
        floor_results.append((name, floor, got, (got is not None and got >= floor)))
    floors_ok = all(ok for _, _, _, ok in floor_results)

    total_fp = int(fp_col.sum()) if fp_col is not None else None
    total_inst = int(m.nt_per_class.sum()) if m.nt_per_class is not None else 0

    # ── report ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}\n  {weights.parents[1].name}  ·  split={args.split}  ·  {tag}\n{'='*72}")
    print(f"  mAP@0.5      {map50:.4f}     target {target:.2f}   "
          f"{'PASS' if primary_ok else 'FAIL'}")
    print(f"  mAP@0.5:0.95 {map5095:.4f}")
    if total_fp is not None:
        print(f"\n  false positives vs background (conf {args.conf or 0.25}): {total_fp}")
        print(f"  over {total_inst} ground-truth instances = {total_fp/max(total_inst,1):.3f} per instance")
        print("  (the dataset has no background images by design -- DEC-120)")

    print("\n  per-class floors:")
    for name, floor, got, ok in floor_results:
        shown = f"{got:.4f}" if got is not None else "  ABSENT"
        print(f"    {name:<12} {shown}  >= {floor:.2f}   {'PASS' if ok else 'FAIL'}")
    print(f"\n  expected-weak (pre-registered, no floor): {', '.join(sorted(EXPECTED_WEAK))}")
    for name in sorted(EXPECTED_WEAK):
        r = by_name.get(name)
        print(f"    {name:<12} {float(r['mAP50']):.4f}" if r else f"    {name:<12}   ABSENT")

    if missing:
        print(f"\n  WARNING: {len(missing)} class(es) have NO instances in this split: {missing}")
        print("  Their CSV rows are written with empty metrics, not zeros. Cross-condition")
        print("  comparison for these classes is not meaningful.")

    verdict = "PASS" if (primary_ok and floors_ok) else "FAIL"
    print(f"\n  {'-'*68}\n  CRITERION (DEC-122): {verdict}")
    print("  Reported only -- nothing is gated on this. A run under target is a result.")
    print(f"{'='*72}")

    summary = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": git_sha(),
        "weights": str(weights),
        "run": weights.parents[1].name,
        "data_yaml": str(data),
        "split": args.split,
        "tag": tag,
        "precision_mode": "int8" if args.int8 else "fp32",
        "nc": len(canonical),
        "map50": map50,
        "map50_95": map5095,
        "target_map50": target,
        "primary_pass": primary_ok,
        "per_class_floors": [
            {"class": n, "floor": f, "map50": g, "pass": ok} for n, f, g, ok in floor_results
        ],
        "floors_pass": floors_ok,
        "expected_weak": sorted(EXPECTED_WEAK),
        "classes_absent_from_split": missing,
        "false_positives_vs_background": total_fp,
        "confusion_matrix_conf": args.conf if args.conf is not None else 0.25,
        "ground_truth_instances": total_inst,
        "verdict": verdict,
        "speed_ms": dict(m.speed) if hasattr(m, "speed") else None,
        "per_class_csv": csv_path.name,
    }
    json_path = out_dir / f"eval_{tag}_{args.split}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nwrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"\nexperiments.md row:\n"
          f"| {weights.parents[1].name} | {tag} | {args.split} | {map50:.4f} | "
          f"{map5095:.4f} | {total_fp if total_fp is not None else '-'} | {verdict} |")

    return 1 if (args.strict and verdict == "FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())

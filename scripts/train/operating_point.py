"""
Recall-oriented operating point for a VI safety device (Phase 1, item 3).

WHY THIS EXISTS
---------------
A missed obstacle and a spurious one do not cost the same thing to a blind user.
Walking into an unannounced pole is an injury; an extra beep is an annoyance. But
every number ultralytics prints is chosen to balance them EQUALLY.

`utils/metrics.py:892` picks the single confidence at which P/R/F1 are reported:

    i = smooth(f1_curve.mean(0), 0.1).argmax()   # max mean-F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]

F1 is the harmonic mean with beta=1 -- precision and recall weighted the same. So
the headline recall in every training log is the recall AT THE PRECISION-BALANCED
THRESHOLD, not the recall this device should actually ship at. This script reports
the same curves under F2 (beta=2, recall weighted 4x in the harmonic mean) and as
recall subject to an explicit precision floor, so the deployment threshold is a
stated engineering choice rather than an inherited default.

NO EXTRA INFERENCE
------------------
`ap_per_class` already computes the full curves over a 1000-point confidence grid
(`metrics.py:842-884`) and `DetMetrics.curves_results` exposes them. Everything here
is read off ONE val pass -- F2 and recall-at-precision cost nothing beyond it.

THE INDEXING TRAP (same family as the ones in evaluate.py)
----------------------------------------------------------
`p_curve` and `r_curve` are `np.zeros((nc, 1000))` where nc is the number of classes
PRESENT in the split, and row `ci` follows `unique_classes` -- i.e. rows are indexed
by position in `ap_class_index`, NOT by canonical class id. Indexing them with a
canonical id silently reads another class's curve whenever any class is absent from
the split. Every lookup here goes through `ap_class_index`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.config_loader import get_canonical_names

PRECISION_FLOORS = (0.5, 0.7, 0.9)


def fbeta(p: np.ndarray, r: np.ndarray, beta: float, eps: float = 1e-16) -> np.ndarray:
    """F-beta. beta>1 weights RECALL more; beta=1 is ultralytics' F1."""
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r + eps)


def recall_at_precision(p_row: np.ndarray, r_row: np.ndarray, px: np.ndarray,
                        floor: float) -> tuple[float, float]:
    """Best recall achievable while precision stays >= floor.

    Returns (recall, confidence). (nan, nan) if the floor is never met -- which is
    itself a finding and must not be reported as 0.0.
    """
    ok = p_row >= floor
    if not ok.any():
        return float("nan"), float("nan")
    idx = int(np.argmax(np.where(ok, r_row, -1.0)))
    return float(r_row[idx]), float(px[idx])


def self_test() -> int:
    checks = []
    p = np.array([1.0, 0.5]); r = np.array([0.5, 1.0])
    f1 = fbeta(p, r, 1.0); f2 = fbeta(p, r, 2.0)
    checks.append(("F1 is symmetric in P and R", abs(f1[0] - f1[1]) < 1e-9))
    checks.append(("F2 prefers the HIGH-RECALL point", f2[1] > f2[0]))
    checks.append(("F2 matches the closed form 5PR/(4P+R)",
                   abs(f2[0] - (5 * 1.0 * 0.5) / (4 * 1.0 + 0.5)) < 1e-9))
    checks.append(("F1 equals the harmonic mean", abs(f1[0] - 2 / 3) < 1e-9))

    px = np.linspace(0, 1, 5)
    p_row = np.array([0.20, 0.55, 0.75, 0.95, 1.00])
    r_row = np.array([0.90, 0.80, 0.60, 0.30, 0.05])
    rec, conf = recall_at_precision(p_row, r_row, px, 0.5)
    checks.append(("recall@P>=0.5 takes the best FEASIBLE recall", abs(rec - 0.80) < 1e-9))
    checks.append(("...and reports the conf that achieves it", abs(conf - 0.25) < 1e-9))
    rec9, _ = recall_at_precision(p_row, r_row, px, 0.9)
    checks.append(("a stricter floor cannot increase recall", rec9 <= rec))
    rec_imp, conf_imp = recall_at_precision(p_row, r_row, px, 1.5)
    checks.append(("an unreachable floor yields NaN, never 0.0",
                   np.isnan(rec_imp) and np.isnan(conf_imp)))

    # the indexing trap, demonstrated
    canonical = get_canonical_names()
    ap_class_index = np.array([i for i, n in enumerate(canonical) if n != canonical[0]])
    curve = np.arange(len(ap_class_index), dtype=float)[:, None] * np.ones((1, 3))
    target_cid = canonical.index(canonical[-1])
    row_pos = int(np.where(ap_class_index == target_cid)[0][0])
    checks.append(("curve rows follow ap_class_index, not class id",
                   row_pos != target_cid))
    checks.append(("...so the mapped row is the right one",
                   curve[row_pos, 0] == float(row_pos)))

    ok = 0
    for desc, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {desc}")
        ok += bool(passed)
    print(f"\n{ok}/{len(checks)} checks passed")
    return 0 if ok == len(checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights")
    ap.add_argument("--data")
    ap.add_argument("--split", default="test", choices=("test", "val"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iou", type=float, default=None, help="NMS IoU (val default 0.7)")
    ap.add_argument("--max-det", type=int, default=None, dest="max_det")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="operating_point")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.weights or not args.data:
        ap.error("--weights and --data are required unless --self-test is given")

    from ultralytics import YOLO
    from ultralytics.utils.metrics import smooth

    weights, data = Path(args.weights).resolve(), Path(args.data).resolve()
    for p in (weights, data):
        if not p.exists():
            raise SystemExit(f"not found: {p}")
    out_dir = Path(args.out) if args.out else weights.parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical = get_canonical_names()
    model = YOLO(str(weights))
    model_names = [model.names[i] for i in sorted(model.names)]
    if model_names != canonical:
        raise SystemExit(f"model classes disagree with classes.yaml:\n  {model_names}\n  {canonical}")

    kw = dict(data=str(data), split=args.split, batch=args.batch, plots=False,
              project=str(out_dir), name=f"val_{args.tag}", exist_ok=True)
    if args.device is not None:
        kw["device"] = args.device
    if args.iou is not None:
        kw["iou"] = args.iou
    if args.max_det is not None:
        kw["max_det"] = args.max_det
    m = model.val(**kw)

    box = m.box
    px = np.asarray(box.px, dtype=float)                 # confidence grid, 1000 pts
    p_curve = np.asarray(box.p_curve, dtype=float)       # (n_present, 1000)
    r_curve = np.asarray(box.r_curve, dtype=float)
    f1_curve = np.asarray(box.f1_curve, dtype=float)
    ap_class_index = np.asarray(box.ap_class_index, dtype=int)
    f2_curve = fbeta(p_curve, r_curve, 2.0)

    # ── global operating points, using ultralytics' own selection convention ──
    i_f1 = int(smooth(f1_curve.mean(0), 0.1).argmax())
    i_f2 = int(smooth(f2_curve.mean(0), 0.1).argmax())

    def at(i):
        return dict(conf=float(px[i]),
                    precision=float(p_curve[:, i].mean()),
                    recall=float(r_curve[:, i].mean()),
                    f1=float(f1_curve[:, i].mean()),
                    f2=float(f2_curve[:, i].mean()))

    op_f1, op_f2 = at(i_f1), at(i_f2)

    rows = []
    for cid, name in enumerate(canonical):
        hit = np.where(ap_class_index == cid)[0]
        if len(hit) == 0:
            rows.append({"class_id": cid, "class": name, "present_in_split": False,
                         **{k: "" for k in ("recall_at_f1_opt", "recall_at_f2_opt",
                                            "precision_at_f2_opt", "recall_gain_f1_to_f2")},
                         **{f"recall_at_p{int(f*100)}": "" for f in PRECISION_FLOORS},
                         **{f"conf_at_p{int(f*100)}": "" for f in PRECISION_FLOORS}})
            continue
        j = int(hit[0])
        row = {"class_id": cid, "class": name, "present_in_split": True,
               "recall_at_f1_opt": round(float(r_curve[j, i_f1]), 4),
               "recall_at_f2_opt": round(float(r_curve[j, i_f2]), 4),
               "precision_at_f2_opt": round(float(p_curve[j, i_f2]), 4),
               "recall_gain_f1_to_f2": round(float(r_curve[j, i_f2] - r_curve[j, i_f1]), 4)}
        for f in PRECISION_FLOORS:
            rec, conf = recall_at_precision(p_curve[j], r_curve[j], px, f)
            row[f"recall_at_p{int(f*100)}"] = "" if np.isnan(rec) else round(rec, 4)
            row[f"conf_at_p{int(f*100)}"] = "" if np.isnan(conf) else round(conf, 4)
        rows.append(row)

    csv_path = out_dir / f"{args.tag}_{args.split}_per_class.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"\n{'='*78}\n  RECALL-ORIENTED OPERATING POINT  ·  {weights.parents[1].name}"
          f"  ·  split={args.split}\n{'='*78}")
    print(f"  {'objective':<12} {'conf':>6} {'precision':>10} {'recall':>8} {'F1':>7} {'F2':>7}")
    print(f"  {'-'*54}")
    print(f"  {'F1 (default)':<12} {op_f1['conf']:>6.3f} {op_f1['precision']:>10.4f} "
          f"{op_f1['recall']:>8.4f} {op_f1['f1']:>7.4f} {op_f1['f2']:>7.4f}")
    print(f"  {'F2 (VI)':<12} {op_f2['conf']:>6.3f} {op_f2['precision']:>10.4f} "
          f"{op_f2['recall']:>8.4f} {op_f2['f1']:>7.4f} {op_f2['f2']:>7.4f}")
    print(f"\n  moving to the F2 operating point changes mean recall by "
          f"{op_f2['recall']-op_f1['recall']:+.4f} "
          f"and mean precision by {op_f2['precision']-op_f1['precision']:+.4f}")

    print("\n  per class:")
    hdr = f"  {'class':<12} {'R@F1':>7} {'R@F2':>7} {'delta':>7}" + \
          "".join(f" {'R@P'+str(int(f*100)):>7}" for f in PRECISION_FLOORS)
    print(hdr); print(f"  {'-'*(len(hdr)-2)}")
    for r in rows:
        if not r["present_in_split"]:
            print(f"  {r['class']:<12} {'ABSENT':>7}"); continue
        cells = ""
        for f in PRECISION_FLOORS:
            v = r[f"recall_at_p{int(f*100)}"]
            cells += f" {v:>7.4f}" if v != "" else f" {'--':>7}"
        print(f"  {r['class']:<12} {r['recall_at_f1_opt']:>7.4f} {r['recall_at_f2_opt']:>7.4f} "
              f"{r['recall_gain_f1_to_f2']:>+7.4f}{cells}")
    print("  ('--' = that precision floor is UNREACHABLE for the class at any confidence)")
    print("="*78)

    summary = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weights": str(weights), "data_yaml": str(data), "split": args.split,
        "nms_iou": args.iou if args.iou is not None else 0.7,
        "max_det": args.max_det if args.max_det is not None else 300,
        "map50": float(box.map50), "map50_95": float(box.map),
        "operating_point_f1": op_f1, "operating_point_f2": op_f2,
        "precision_floors": list(PRECISION_FLOORS),
        "per_class_csv": csv_path.name,
    }
    json_path = out_dir / f"{args.tag}_{args.split}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {csv_path}\nwrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

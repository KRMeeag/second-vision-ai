"""Summarise whichever v3 arms have finished. Safe to run at any point, reads only.

    python3 scripts/train/compare_v3.py /workspace/runs_v3

The numbers in TRAIN_BOXES are not decoration: preflight_v3.py check #8 counts the
boxes actually on disk and FAILS if they disagree with these, so the figure printed
here is the figure that was trained on, not a number copied out of a document once.
"""
import json
import sys
from pathlib import Path

P = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/runs_v3")
ORDER = ("v3p", "v3l", "v3s")
LABEL = {"v3p": "PERMISSIVE", "v3l": "LOOSE", "v3s": "STRICT"}
TRAIN_BOXES = {"v3p": 118279, "v3l": 140801, "v3s": 113965}

rows = []
notes = []

b = P / "baseline_result.json"
if b.is_file():
    d = json.loads(b.read_text())
    rows.append(dict(name="v2 baseline", boxes="-", t=d["test"], hours=None,
                     fp=None, parity=None, verdict=None))
    if d.get("caveat"):
        notes.append(("v2 baseline", d["caveat"]))

for arm in ORDER:
    f = P / f"{arm}_result.json"
    if not f.is_file():
        continue
    d = json.loads(f.read_text())
    rows.append(dict(name=LABEL[arm], boxes=f"{TRAIN_BOXES[arm]:,}", t=d["test"],
                     hours=d.get("train_hours"),
                     fp=d.get("false_positives_vs_background"),
                     parity=d.get("args_parity_with_v2"),
                     verdict=d.get("criterion_verdict")))
    if d.get("args_parity_with_v2") is False:
        notes.append((LABEL[arm],
                      "HYPERPARAMETERS DIFFER FROM v2 -- this arm's gap may be a "
                      "schedule effect, not a label effect. See check_args_parity.py "
                      "output in its log before reporting it."))

# A smoke result is a 2-epoch model on 2% of the data. Shown, clearly marked, so a
# finished smoke phase is visible -- and never mistaken for an arm.
for arm in ORDER:
    f = P / f"{arm}_smoke_result.json"
    if f.is_file():
        d = json.loads(f.read_text())
        notes.append((f"{LABEL[arm]} smoke",
                      f"2 epochs on 2% -- mAP50 {d['test']['mAP50']:.4f}. Proves the "
                      f"path works end to end. NOT a result."))

if not rows:
    print("no results yet")
    sys.exit(0)

print(f"\n{'arm':<13}{'train boxes':>13}{'mAP50':>9}{'mAP50-95':>10}"
      f"{'P':>8}{'R':>8}{'FP/bg':>8}{'hours':>7}")
for r in rows:
    t = r["t"]
    p = f"{t['precision']:.4f}" if t.get("precision") is not None else "-"
    rc = f"{t['recall']:.4f}" if t.get("recall") is not None else "-"
    fp = f"{r['fp']:,}" if r["fp"] is not None else "-"
    hrs = f"{r['hours']:.2f}" if r["hours"] is not None else "-"
    flag = "" if r["parity"] in (None, True) else "  !!"
    print(f"{r['name']:<13}{r['boxes']:>13}{t['mAP50']:>9.4f}{t['mAP50_95']:>10.4f}"
          f"{p:>8}{rc:>8}{fp:>8}{hrs:>7}{flag}")

for who, msg in notes:
    print(f"\nNOTE on {who}: {msg}")

arms = {r["name"]: r["t"] for r in rows if r["name"] in LABEL.values()}
if "PERMISSIVE" in arms and "LOOSE" in arms:
    d50 = arms["LOOSE"]["mAP50"] - arms["PERMISSIVE"]["mAP50"]
    d95 = arms["LOOSE"]["mAP50_95"] - arms["PERMISSIVE"]["mAP50_95"]
    print(f"\nLOOSE - PERMISSIVE   mAP50 {d50:+.4f}   mAP50-95 {d95:+.4f}")
    print(f"  Loose adds {TRAIN_BOXES['v3l'] - TRAIN_BOXES['v3p']:,} training boxes with")
    print("  NO v2 corroboration. If this is negative, label noise is costing more")
    print("  than the extra volume buys.")
if "STRICT" in arms and "PERMISSIVE" in arms:
    d50 = arms["PERMISSIVE"]["mAP50"] - arms["STRICT"]["mAP50"]
    d95 = arms["PERMISSIVE"]["mAP50_95"] - arms["STRICT"]["mAP50_95"]
    print(f"\nPERMISSIVE - STRICT  mAP50 {d50:+.4f}   mAP50-95 {d95:+.4f}")
    print(f"  Permissive adds {TRAIN_BOXES['v3p'] - TRAIN_BOXES['v3s']:,} boxes that tier D/E")
    print("  rejected but v2 corroborated. This is the agreement-boundary question.")

missing = [a for a in ORDER if not (P / f"{a}_result.json").is_file()]
if missing:
    # Not "incomplete": permissive alone is the planned first run. The arms share
    # byte-identical val and test labels, so one trained later is comparable to one
    # trained today -- nothing here has to be re-run to add it.
    print(f"\nnot trained yet: {', '.join(LABEL[a] for a in missing)}")
    print("  These are comparable to the arm(s) above whenever you run them -- the val")
    print("  and test labels are byte-identical across all three and pre-flight checks")
    print("  it. Add one with:  run_v3.sh " + " ".join(missing))

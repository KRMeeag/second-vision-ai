"""Summarise whichever arms have finished. Safe to run at any point."""
import json, sys
from pathlib import Path

P = Path(sys.argv[1])
LABEL = {"v3p": "PERMISSIVE", "v3l": "LOOSE", "v3s": "STRICT"}
TRAIN_BOXES = {"v3p": 118279, "v3l": 140801, "v3s": 113965}

rows = []
b = P / "baseline_result.json"
if b.is_file():
    d = json.loads(b.read_text())
    rows.append(("v2 baseline", "-", d["test"], d.get("caveat")))
for arm in ("v3p", "v3l", "v3s"):
    f = P / f"{arm}_result.json"
    if f.is_file():
        d = json.loads(f.read_text())
        rows.append((LABEL[arm], f"{TRAIN_BOXES[arm]:,}", d["test"], None))

if not rows:
    print("no results yet"); sys.exit(0)

print(f"\n{'arm':<13}{'train boxes':>13}{'mAP50':>9}{'mAP50-95':>10}{'P':>8}{'R':>8}")
for name, nb, t, _ in rows:
    print(f"{name:<13}{nb:>13}{t['mAP50']:>9.4f}{t['mAP50_95']:>10.4f}"
          f"{t['precision']:>8.4f}{t['recall']:>8.4f}")
for name, _, _, caveat in rows:
    if caveat:
        print(f"\nNOTE on {name}: {caveat}")

arms = {n: t for n, nb, t, _ in rows if n in LABEL.values()}
if "PERMISSIVE" in arms and "LOOSE" in arms:
    d = arms["LOOSE"]["mAP50_95"] - arms["PERMISSIVE"]["mAP50_95"]
    print(f"\nLOOSE - PERMISSIVE on mAP50-95: {d:+.4f}")
    print("  Loose adds 22,522 training boxes with NO v2 corroboration. If this is")
    print("  negative, label noise is costing more than the extra volume buys.")

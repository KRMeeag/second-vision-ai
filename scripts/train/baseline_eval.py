"""Score the EXISTING v2 weights against the new shared test labels.

Without this the three arms are comparable only to each other. v2's published mAP50
of 0.7118 was measured on the OLD test labels; those labels changed, so that number
cannot be compared with anything produced here. Two minutes of GPU time fixes it.

THE SCHEMA PROBLEM, AND WHY THIS SCRIPT BUILDS A SHADOW DATASET
---------------------------------------------------------------
v2 is a 14-CLASS model. The arms are 13-class: Pole was dropped and the ten classes
above it each shifted down one id (DEC-143). Handing v2's weights a 13-class
data.yaml does not fail -- it misaligns every class above the dropped one, so every
Pothole in the test set is scored against v2's Tricycle output. That is DEC-107
happening again, one layer further out, and it would report a plausible, wrong,
quietly catastrophic baseline.

An earlier version of this file did exactly that: it noted the mismatch in a caveat
string and then passed the arms' data.yaml anyway.

So the test labels are REMAPPED UP into v2's own schema instead. The remap is safe
and lossless in this one direction, for one reason: the new labels contain no Pole
boxes at all, so nothing has to be invented or discarded.

THE REMAP IS BY NAME, NEVER BY ARITHMETIC
-----------------------------------------
`new_id -> new_id if new_id < 3 else new_id + 1` is the correct arithmetic today and
is exactly the kind of expression that silently stops being correct after the next
schema change. The mapping here is built by looking each canonical class name up in
the model's OWN `names` dict, read out of the checkpoint. A name that is not there
aborts the script rather than guessing.

What remains, and is reported in the output and in baseline_result.json: v2's head
still carries a Pole output with no Pole ground truth, so its Pole predictions land
in the background column as false positives. The overall mAP is therefore a slight
UNDER-statement of v2, averaged over 14 classes where one cannot score. Treat it as
a reference point with a known direction of bias, not a number to subtract.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ROOT = Path("/workspace")
PROJECT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs_v3")
W = ROOT / "model_backups" / "cap4500_yolov8s_v2-2_20260913T164555Z" / "weights" / "best.pt"
SRC_ARM = ROOT / "v3p"            # any arm: the test labels are byte-identical across all
SHADOW = ROOT / "v2_baseline"     # the 14-class shadow dataset this script builds


def main() -> int:
    import yaml
    from ultralytics import YOLO

    from scripts.utils.config_loader import get_canonical_names

    if not W.is_file():
        print(f"v2 weights not found at {W} -- skipping the baseline. The arms stay "
              f"comparable to each other, just not to where the project stands.")
        return 1
    if not (SRC_ARM / "test.txt").is_file():
        print(f"no test listing at {SRC_ARM/'test.txt'} -- run setup_pod_v3.sh first")
        return 1

    model = YOLO(str(W))
    v2_names = {int(i): n for i, n in model.names.items()}
    v2_order = [v2_names[i] for i in sorted(v2_names)]
    canonical = get_canonical_names()
    print(f"v2 weights  : {W}")
    print(f"  model nc  : {len(v2_order)} {v2_order}")
    print(f"  repo  nc  : {len(canonical)} {canonical}")

    # Build the by-name remap. Abort rather than guess.
    inv = {n: i for i, n in v2_names.items()}
    missing = [n for n in canonical if n not in inv]
    if missing:
        print(f"\nABORT: these canonical classes have no counterpart in the v2 head: "
              f"{missing}\nThe baseline cannot be made comparable; skipping it is the "
              f"honest outcome.")
        return 1
    remap = {new_id: inv[name] for new_id, name in enumerate(canonical)}
    v2_only = [n for n in v2_order if n not in canonical]
    print(f"  remap     : {remap}")
    print(f"  in v2 only: {v2_only} (no ground truth -- its predictions become "
          f"background false positives)")

    # ---- build the shadow dataset ------------------------------------------------
    # Small: 6,403 label files of a few hundred bytes. The images are reached by the
    # same single symlink the arms use, so no image entries are created (DEC-142).
    (SHADOW / "labels").mkdir(parents=True, exist_ok=True)
    if not (SHADOW / "images").exists():
        (SHADOW / "images").symlink_to(ROOT / "bundle" / "images")

    lines = [l for l in (SRC_ARM / "test.txt").read_text().split() if l.strip()]
    written = boxes = 0
    for img in lines:
        stem = Path(img).name.rsplit(".", 1)[0]
        src = SRC_ARM / "labels" / f"{stem}.txt"
        out = []
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            out.append(" ".join([str(remap[int(parts[0])])] + parts[1:]))
        (SHADOW / "labels" / f"{stem}.txt").write_text("\n".join(out) + ("\n" if out else ""))
        written += 1
        boxes += len(out)
    # Point the listing at the shadow root so ultralytics derives shadow label paths.
    (SHADOW / "test.txt").write_text(
        "\n".join(f"{SHADOW}/images/{Path(l).name}" for l in lines) + "\n")
    print(f"\nshadow dataset: {written:,} label files, {boxes:,} boxes remapped into "
          f"v2's schema at {SHADOW}")

    data = SHADOW / "data.yaml"
    data.write_text(
        "# GENERATED by scripts/train/baseline_eval.py -- not hand-maintained.\n"
        "# The v3 test labels remapped UP into v2's 14-class schema, so the v2\n"
        "# checkpoint can be scored on the same images and boxes as the arms.\n"
        "# No 'path' key on purpose (DEC-066).\n"
        + yaml.safe_dump({"train": "test.txt", "val": "test.txt", "test": "test.txt",
                          "nc": len(v2_order),
                          "names": {i: n for i, n in enumerate(v2_order)}},
                         sort_keys=False, allow_unicode=True),
        encoding="utf-8")

    # Prove the generated file round-trips before spending GPU time on it.
    back = yaml.safe_load(data.read_text())
    assert [back["names"][i] for i in sorted(back["names"])] == v2_order, \
        "generated data.yaml does not match the model's own class order"

    # A stale cache here would serve the PREVIOUS remap's boxes. Renamed, not deleted.
    for c in SHADOW.glob("*.cache"):
        c.rename(c.with_suffix(c.suffix + ".stale"))
        print(f"  moved stale {c.name} aside")

    # ---- evaluate ----------------------------------------------------------------
    r = model.val(data=str(data), split="test", imgsz=640, batch=32, device="0",
                  project=str(PROJECT), name="baseline_v2_test", exist_ok=True, plots=False)

    per_class = {}
    try:
        for i, c in enumerate(r.box.ap_class_index):
            per_class[v2_names.get(int(c), str(c))] = {
                "AP50": round(float(r.box.ap50[i]), 4),
                "AP50_95": round(float(r.box.ap[i]), 4),
            }
    except Exception as e:
        print(f"per-class extraction failed: {e}")

    caveat = (
        f"v2 is a {len(v2_order)}-class model scored on the arms' test IMAGES and BOXES, "
        f"remapped by NAME into its own schema. Classes present only in v2 ({v2_only}) "
        f"have no ground truth, so their predictions count as background false positives "
        f"and the overall mAP is averaged over {len(v2_order)} classes of which "
        f"{len(v2_only)} cannot score. It is a reference point biased DOWNWARD, not a "
        f"number to subtract from an arm's mAP. Compare per-class AP50 for the 13 shared "
        f"classes instead."
    )
    out = {
        "weights": str(W), "data": str(data),
        "model_nc": len(v2_order), "data_nc": len(v2_order),
        "remap_new_to_v2": remap,
        "classes_in_v2_only": v2_only,
        "test_images": len(lines), "test_boxes": boxes,
        "caveat": caveat,
        "test": {"mAP50": round(float(r.box.map50), 4),
                 "mAP50_95": round(float(r.box.map), 4),
                 "precision": round(float(r.box.mp), 4),
                 "recall": round(float(r.box.mr), 4)},
        "per_class": per_class,
    }
    (PROJECT / "baseline_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    t = out["test"]
    print(f"\nBASELINE v2 on the new test labels: mAP50={t['mAP50']} "
          f"mAP50-95={t['mAP50_95']} P={t['precision']} R={t['recall']}")
    print(f"\n{caveat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

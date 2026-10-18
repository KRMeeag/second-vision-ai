"""Score the EXISTING v2 weights against the new shared test labels.

Without this the three arms are comparable only to each other. v2's published 0.7118
was measured on the OLD test labels; those labels changed, so that number cannot be
compared to anything produced here. Two minutes of GPU time fixes that.
"""
import json, os, sys
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
from pathlib import Path

PROJECT = Path(sys.argv[1])
W = "/workspace/model_backups/cap4500_yolov8s_v2-2_20260913T164555Z/weights/best.pt"

# v2 is a 14-class model; the arms are 13-class. Scoring it on a 13-class data.yaml
# would misalign every id above the dropped class. It must be evaluated on ITS OWN
# schema, over the same images -- so this reports a reference point, NOT a number
# that can be subtracted from the arms' mAP.
DATA = "/workspace/v3p/data.yaml"

def main():
    from ultralytics import YOLO
    assert Path(W).is_file(), f"missing {W}"
    m = YOLO(W)
    nc_model = len(m.names)
    print(f"v2 weights: {W}\n  classes in model: {nc_model} -> {list(m.names.values())}",
          flush=True)
    import yaml
    nc_data = yaml.safe_load(Path(DATA).read_text())["nc"]
    note = None
    if nc_model != nc_data:
        note = (f"SCHEMA MISMATCH: v2 has {nc_model} classes, the arms' data.yaml has "
                f"{nc_data}. Reported as a reference point only -- it is NOT directly "
                f"comparable to the arms' mAP.")
        print(f"  !! {note}", flush=True)
    r = m.val(data=DATA, split="test", imgsz=640, batch=32, device="0",
              project=str(PROJECT), name="baseline_v2_test", exist_ok=True, plots=False)
    out = {"weights": W, "data": DATA, "model_nc": nc_model, "data_nc": nc_data,
           "caveat": note,
           "test": {"mAP50": round(float(r.box.map50), 4),
                    "mAP50_95": round(float(r.box.map), 4),
                    "precision": round(float(r.box.mp), 4),
                    "recall": round(float(r.box.mr), 4)}}
    (PROJECT / "baseline_result.json").write_text(json.dumps(out, indent=2))
    t = out["test"]
    print(f"\nBASELINE v2 on new test labels: mAP50={t['mAP50']} "
          f"mAP50-95={t['mAP50_95']} P={t['precision']} R={t['recall']}", flush=True)

if __name__ == "__main__":
    main()

"""Train and evaluate ONE v3 arm. Called by run_v3.sh; not meant to be run directly.

Hyperparameters are pinned to v2's exactly (read back from its args.yaml) so the only
difference between arms is the TRAINING LABELS. val and test labels are byte-identical
across all three arms, so the evaluation yardstick cannot move.

Usage: python3 train_one_arm.py <arm> <project_dir>
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

ARM = sys.argv[1]
PROJECT = Path(sys.argv[2])
DATA = Path(f"/workspace/{ARM}/data.yaml")

# Exactly v2's settings (runs/detect/cap4500_yolov8s_v2-2/args.yaml).
# Any change here breaks comparability with the 0.7118 v2 result.
CFG = dict(
    model="yolov8s.pt",
    epochs=100,
    batch=32,
    imgsz=640,
    device="0",
    workers=8,
    optimizer="auto",
    lr0=0.01,
    lrf=0.01,
    seed=42,
    deterministic=True,
    close_mosaic=10,
    patience=100000,
    save_period=10,
    cache=False,
    amp=True,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    pretrained=True,
    exist_ok=True,
)


def main():
    from ultralytics import YOLO
    import ultralytics
    import torch

    print(f"=== ARM {ARM} ===", flush=True)
    print(f"ultralytics {ultralytics.__version__}  torch {torch.__version__}", flush=True)
    print(f"cuda={torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}", flush=True)
    assert DATA.is_file(), f"missing {DATA}"

    name = f"v3_{ARM}"
    t0 = time.time()

    model = YOLO(CFG["model"])
    model.train(data=str(DATA), project=str(PROJECT), name=name,
                **{k: v for k, v in CFG.items() if k != "model"})
    train_s = time.time() - t0
    print(f"\n[{ARM}] training finished in {train_s/3600:.2f} h", flush=True)

    # Evaluate the BEST checkpoint on the shared test split. split="test" makes
    # ultralytics read the `test:` key, not `val:` -- using val here would silently
    # report the wrong number.
    best = PROJECT / name / "weights" / "best.pt"
    assert best.is_file(), f"no best.pt at {best}"
    m = YOLO(str(best))
    r = m.val(data=str(DATA), split="test", imgsz=CFG["imgsz"], batch=CFG["batch"],
              device=CFG["device"], project=str(PROJECT), name=f"{name}_test",
              exist_ok=True, plots=False)

    names = r.names if hasattr(r, "names") else {}
    per_class = {}
    try:
        for i, c in enumerate(r.box.ap_class_index):
            per_class[names.get(int(c), str(c))] = {
                "AP50": round(float(r.box.ap50[i]), 4),
                "AP50_95": round(float(r.box.ap[i]), 4),
            }
    except Exception as e:  # surfaced, never silently skipped
        print(f"[{ARM}] per-class extraction failed: {e}", flush=True)

    out = {
        "arm": ARM,
        "data": str(DATA),
        "weights": str(best),
        "train_hours": round(train_s / 3600, 3),
        "test": {
            "mAP50": round(float(r.box.map50), 4),
            "mAP50_95": round(float(r.box.map), 4),
            "precision": round(float(r.box.mp), 4),
            "recall": round(float(r.box.mr), 4),
        },
        "per_class": per_class,
    }
    (PROJECT / f"{ARM}_result.json").write_text(json.dumps(out, indent=2))
    print(f"\n[{ARM}] TEST mAP50={out['test']['mAP50']}  "
          f"mAP50-95={out['test']['mAP50_95']}  "
          f"P={out['test']['precision']}  R={out['test']['recall']}", flush=True)
    print(f"[{ARM}] -> {PROJECT / f'{ARM}_result.json'}", flush=True)


if __name__ == "__main__":
    main()

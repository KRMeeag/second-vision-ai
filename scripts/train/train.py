"""
Train one condition. Thin wrapper over ultralytics that adds provenance.

The training call itself is a few lines. Everything else here exists so the run
is reproducible six months later when the pod is long gone:

  * hyperparameters come from `config/training.yaml`, never from flags typed once
  * the resolved data.yaml path, git SHA and full `pip freeze` land IN the run dir
  * Comet and TensorBoard are switched on explicitly and reported, so a run that
    silently logged nowhere is visible immediately rather than at write-up time

WHY `patience` IS OVERRIDDEN FOR ABLATION ARMS
----------------------------------------------
`close_mosaic` (default 10) disables mosaic for the final N epochs, which reliably
buys a mAP bump because the model finally calibrates to un-composited images. With
`epochs=100, close_mosaic=10` that phase starts at epoch 90.

Early stopping breaks this. If one arm stops at epoch 62 it never gets the
mosaic-off phase while an arm that runs to 100 does, so the arms differ in
SCHEDULE as well as in data -- and the ablation no longer measures what it claims
to. `--patience 0` (mapped to a very large value) keeps every arm on an identical
100-epoch schedule. `best.pt` is still selected by fitness, so over-training costs
GPU time, not model quality.

Set `--patience` explicitly for exploratory one-off runs where cost matters more
than comparability.

Usage
-----
    python3 scripts/train/train.py --data /workspace/data/final_cap4500/data.yaml --device 0
    python3 scripts/train/train.py --data ... --smoke          # 2 epochs, 2% of data
    python3 scripts/train/train.py --resume runs/detect/<name>/weights/last.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from scripts.utils.config_loader import REPO_ROOT, get_canonical_names

TRAINING_YAML = REPO_ROOT / "config" / "training.yaml"


def load_training_config() -> dict:
    cfg = yaml.safe_load(TRAINING_YAML.read_text(encoding="utf-8")) or {}
    cfg.pop("data", None)   # always supplied explicitly; see --data
    return cfg


def enable_loggers(comet: bool, tensorboard: bool) -> dict[str, bool]:
    """Turn on the trackers and report what actually armed.

    Both are opt-in in ultralytics SETTINGS and both fail SOFT -- a missing package
    or an unset API key means the run trains fine and logs nowhere, which you would
    otherwise discover only when writing up. Hence the explicit report.
    """
    from ultralytics import settings

    state = {}
    if tensorboard:
        settings.update({"tensorboard": True})
        try:
            import tensorboard  # noqa: F401
            state["tensorboard"] = True
        except ImportError:
            state["tensorboard"] = False
    if comet:
        settings.update({"comet": True})
        try:
            import comet_ml  # noqa: F401
            import os
            state["comet"] = bool(os.environ.get("COMET_API_KEY"))
        except ImportError:
            state["comet"] = False
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="path to the condition's data.yaml (required unless --resume)")
    ap.add_argument("--device", default=None, help="e.g. 0, 0,1, cpu, mps. Left to ultralytics if unset.")
    ap.add_argument("--name", default=None, help="run name; defaults to config/training.yaml's")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32,
                    help="Must be IDENTICAL across ablation arms. 32 fits yolov8s@640 on 24GB.")
    ap.add_argument("--patience", type=int, default=0,
                    help="0 = effectively disabled (ablation default, see module docstring).")
    ap.add_argument("--save-period", type=int, default=10,
                    help="Checkpoint every N epochs. Insurance against a pod dying mid-run.")
    ap.add_argument("--resume", default=None, help="path to last.pt to resume from")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on 2%% of the data. Exercises dataloader, AMP, loggers, "
                         "checkpoint writing and plots in ~2 minutes before a real run.")
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--no-tensorboard", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        print(f"Resuming from {args.resume}")
        YOLO(args.resume).train(resume=True)
        return

    if not args.data:
        ap.error("--data is required unless --resume is given")

    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        raise SystemExit(f"data.yaml not found: {data_path}")

    # Fail loudly if the data.yaml disagrees with the repo schema. A silent nc
    # mismatch trains a model whose head does not match the labels -- and this
    # project has already had a stale-schema incident render every Pothole as a
    # Tricycle (DEC-107).
    dy = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = get_canonical_names()
    dy_names = [dy["names"][i] for i in sorted(dy["names"])]
    if dy_names != names:
        raise SystemExit(
            f"data.yaml disagrees with config/classes.yaml.\n"
            f"  data.yaml    : nc={dy.get('nc')} {dy_names}\n"
            f"  classes.yaml : nc={len(names)} {names}"
        )
    print(f"schema check: data.yaml agrees with classes.yaml at nc={len(names)}")

    cfg = load_training_config()
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    cfg["batch"] = args.batch
    # ultralytics has no "off" for patience; a value above `epochs` never fires.
    cfg["patience"] = 100000 if args.patience == 0 else args.patience
    cfg["save_period"] = args.save_period
    if args.device is not None:
        cfg["device"] = args.device
    if args.name:
        cfg["name"] = args.name
    if args.smoke:
        cfg.update({"epochs": 2, "fraction": 0.02, "name": (cfg.get("name", "run") + "_smoke")})
        print("SMOKE MODE: 2 epochs on 2% of the data — not a real run.")

    loggers = enable_loggers(comet=not args.no_comet, tensorboard=not args.no_tensorboard)
    for k, v in loggers.items():
        print(f"logger {k}: {'ARMED' if v else 'NOT ARMED (package or API key missing)'}")

    model = YOLO(cfg.pop("model", "yolov8s.pt"))
    print(f"\nstarting: {json.dumps({k: v for k, v in cfg.items() if not isinstance(v, dict)}, default=str)}")
    results = model.train(data=str(data_path), **cfg)

    # Provenance into the run dir, beside ultralytics' own args.yaml.
    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(model.trainer.save_dir)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT).strip()
    except Exception:
        sha = "unknown"
    (run_dir / "provenance.json").write_text(json.dumps({
        "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_yaml": str(data_path),
        "data_yaml_contents": dy,
        "git_sha": sha,
        "loggers": loggers,
        "resolved_args": {k: str(v) for k, v in cfg.items()},
        "smoke": args.smoke,
    }, indent=2), encoding="utf-8")
    try:
        (run_dir / "pip_freeze.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    except Exception as exc:
        print(f"  (pip freeze failed: {exc})")

    print(f"\nrun dir: {run_dir}")
    print("Next: python3 scripts/train/evaluate.py --weights "
          f"{run_dir / 'weights' / 'best.pt'} --data {data_path}")


if __name__ == "__main__":
    main()

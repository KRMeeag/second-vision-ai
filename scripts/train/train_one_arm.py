"""Train and evaluate ONE v3 arm. Called by run_v3.sh; runnable alone for a retry.

This is an ORCHESTRATOR, not a trainer. It shells out to the two scripts that
already exist and are already the tried-and-tested path for this project:

    scripts/train/train.py        hyperparameters from config/training.yaml, the
                                  nc-vs-classes.yaml schema gate, provenance.json,
                                  pip_freeze.txt, and ARMED/NOT ARMED logger reporting
    scripts/train/evaluate.py     the frozen test split, per-class rows reindexed
                                  against the canonical schema, background-column
                                  false positives, DEC-122 criterion

An earlier version of this file carried its OWN copy of the hyperparameters, hand-
transcribed from v2's args.yaml. That is the thing to avoid: a second source of truth
for how this project trains, which drifts the moment config/training.yaml changes and
reports nothing when it does. Parity with v2 is now CHECKED (check_args_parity.py)
rather than re-typed.

WHAT MAKES THE ARMS COMPARABLE
------------------------------
Nothing in this file varies per arm except the data.yaml path. The val and test
labels are byte-identical across all three arms, verified in pre-flight, so the
yardstick cannot move; the only independent variable is the training labels.

Usage
-----
    python3 scripts/train/train_one_arm.py v3p /workspace/runs_v3
    python3 scripts/train/train_one_arm.py v3p /workspace/runs_v3 --smoke
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ARM_LABEL = {"v3p": "PERMISSIVE", "v3l": "LOOSE", "v3s": "STRICT"}


def run(cmd: list[str], env: dict | None = None) -> int:
    """Run a child with its output INHERITED, so run_v3.sh's tee sees it live.

    Capturing and re-printing would hold a three-hour training run's output until
    it finished, which is the opposite of what a detached run needs.
    """
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd, cwd=str(REPO), env=env)


def mean_present(csv_path: Path, field: str) -> float | None:
    """Mean of `field` over classes PRESENT in the split.

    This reproduces ultralytics' own box.mp / box.mr, which average over
    `ap_class_index` -- i.e. present classes only. Averaging over all 13 rows
    instead would divide by classes that were never tested and silently drag
    every arm's precision down by the same wrong amount.
    """
    if not csv_path.is_file():
        return None
    vals = []
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("present_in_split") == "True" and row.get(field):
                vals.append(float(row[field]))
    return round(sum(vals) / len(vals), 4) if vals else None


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    smoke = "--smoke" in sys.argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    arm, project = argv[0], Path(argv[1])
    data = Path(f"/workspace/{arm}/data.yaml")
    project.mkdir(parents=True, exist_ok=True)

    if not data.is_file():
        raise SystemExit(f"missing {data} -- run setup_pod_v3.sh first")

    label = ARM_LABEL.get(arm, arm)
    # train.py appends "_smoke" itself, so the run dirs never collide.
    name = f"v3_{arm}"
    # ...and neither do the result files: a smoke result landing on <arm>_result.json
    # would make run_v3.sh --status report a 2-epoch model as the arm's result.
    result_name = f"{arm}_smoke_result.json" if smoke else f"{arm}_result.json"
    print(f"=== ARM {arm} ({label}){' SMOKE' if smoke else ''} ===", flush=True)

    env = dict(os.environ)
    env.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
    # train.py writes the REAL run dir here. It cannot be predicted from --name:
    # ultralytics appends a numeric suffix when the name is already taken, which
    # is exactly what happens on a retry after a failed arm.
    ptr = project / (f".{name}{'_smoke' if smoke else ''}.rundir")
    env["TRAIN_RUNDIR_OUT"] = str(ptr)
    if ptr.exists():
        ptr.unlink()   # a stale pointer from a previous attempt would mislead everything below

    # ---- train ----------------------------------------------------------------
    t0 = time.time()
    cmd = [sys.executable, "scripts/train/train.py",
           "--data", str(data), "--device", "0", "--name", name]
    if smoke:
        cmd.append("--smoke")
    rc = run(cmd, env)
    train_s = time.time() - t0
    if rc != 0:
        print(f"[{arm}] train.py FAILED rc={rc}", flush=True)
        return rc
    if not ptr.is_file():
        print(f"[{arm}] train.py exited 0 but wrote no run-dir pointer at {ptr} -- "
              f"cannot locate the weights", flush=True)
        return 2
    run_dir = Path(ptr.read_text().strip())
    print(f"\n[{arm}] training finished in {train_s/3600:.2f} h -> {run_dir}", flush=True)

    # ---- hyperparameter parity with v2 ---------------------------------------
    # Reported, never fatal: a genuine, recorded change to training.yaml should not
    # abort a finished run. It MUST be visible, though, or a schedule change gets
    # written up as a label effect.
    parity_cmd = [sys.executable, "scripts/train/check_args_parity.py", str(run_dir)]
    if smoke:
        parity_cmd.append("--smoke")
    parity_ok = run(parity_cmd, env) == 0
    if not parity_ok:
        print(f"[{arm}] !! HYPERPARAMETERS DIFFER FROM v2 -- see the diff above", flush=True)

    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        print(f"[{arm}] no best.pt at {best}", flush=True)
        return 3

    # ---- evaluate on the FROZEN test split -----------------------------------
    # split="test" makes ultralytics read the `test:` key. Defaulting to val here
    # would report a number measured on the wrong split with no sign of it.
    rc = run([sys.executable, "scripts/train/evaluate.py",
              "--weights", str(best), "--data", str(data),
              "--split", "test", "--device", "0", "--tag", "fp32"], env)
    if rc != 0:
        print(f"[{arm}] evaluate.py FAILED rc={rc}", flush=True)
        return rc

    summary_path = run_dir / "eval_fp32_test_summary.json"
    csv_path = run_dir / "eval_fp32_test_per_class.csv"
    if not summary_path.is_file():
        print(f"[{arm}] evaluate.py wrote no summary at {summary_path}", flush=True)
        return 4
    ev = json.loads(summary_path.read_text())

    out = {
        "arm": arm,
        "label": label,
        "smoke": smoke,
        "data": str(data),
        "run_dir": str(run_dir),
        "weights": str(best),
        "train_hours": round(train_s / 3600, 3),
        "args_parity_with_v2": parity_ok,
        "test": {
            "mAP50": round(float(ev["map50"]), 4),
            "mAP50_95": round(float(ev["map50_95"]), 4),
            "precision": mean_present(csv_path, "precision"),
            "recall": mean_present(csv_path, "recall"),
        },
        "false_positives_vs_background": ev.get("false_positives_vs_background"),
        "ground_truth_instances": ev.get("ground_truth_instances"),
        "classes_absent_from_split": ev.get("classes_absent_from_split"),
        "criterion_verdict": ev.get("verdict"),
        "eval_summary": str(summary_path),
        "per_class_csv": str(csv_path),
    }
    (project / result_name).write_text(json.dumps(out, indent=2), encoding="utf-8")

    t = out["test"]
    print(f"\n[{arm}] TEST mAP50={t['mAP50']}  mAP50-95={t['mAP50_95']}  "
          f"P={t['precision']}  R={t['recall']}", flush=True)
    print(f"[{arm}] -> {project / result_name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Diff a run's resolved hyperparameters against the v2 run that anchors the thesis.

WHY THIS EXISTS
---------------
The v3 experiment's whole claim is that the arms differ in their TRAINING LABELS and
in nothing else, and that they stay comparable to v2's published mAP50 of 0.7118. That
claim rests on a chain nobody looks at: config/training.yaml -> train.py's flag
defaults -> ultralytics' own defaults for every key neither of those sets.

Each link can move silently.

  * `config/training.yaml` says `batch: 16` and `patience: 20`. Neither reaches the
    run: train.py's `--batch` defaults to 32 and `--patience 0` maps to 100000. Read
    the config alone and you would conclude v2 trained at batch 16.
  * Most of the set -- workers, deterministic, close_mosaic, lr0, mosaic, erasing --
    is specified NOWHERE in this repo. It is ultralytics 8.4.118's defaults. A version
    bump that changes one of those changes the experiment with no diff to review,
    which is the entire reason requirements-train.txt pins exactly.

A one-second diff against the real v2 args.yaml turns all of that from an assumption
into a check. Run it on the SMOKE run's args.yaml, before the six real GPU-hours.

Usage
-----
    python3 scripts/train/check_args_parity.py <run_dir_or_args.yaml>
    python3 scripts/train/check_args_parity.py <...> --smoke     # allow epochs/fraction
    python3 scripts/train/check_args_parity.py <...> --reference <other args.yaml>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_REFERENCE = HERE / "v2_args_reference.yaml"

# Keys that MUST differ between any two runs -- they identify the run, they do not
# configure it. A difference here is not a finding.
IDENTITY = {"data", "name", "project", "save_dir"}

# Additionally allowed to differ for a smoke run, which is 2 epochs on 2% on purpose.
SMOKE_ONLY = {"epochs", "fraction", "time"}

# Keys whose value is a path or a device string and so legitimately varies with the
# pod, not with the experiment.
ENVIRONMENTAL = {"device", "workers"}


def load(path: Path) -> dict:
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise SystemExit(f"{path}: not a mapping")
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a run directory, or the args.yaml inside one")
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE),
                    help="args.yaml to compare against (default: the tracked v2 record)")
    ap.add_argument("--smoke", action="store_true",
                    help="also tolerate epochs/fraction/time differing")
    ap.add_argument("--strict-env", action="store_true",
                    help="also flag device and workers, which normally vary with the pod")
    args = ap.parse_args()

    run_path = Path(args.run)
    if run_path.is_dir():
        run_path = run_path / "args.yaml"
    if not run_path.is_file():
        raise SystemExit(f"no args.yaml at {run_path}")
    ref_path = Path(args.reference)
    if not ref_path.is_file():
        raise SystemExit(f"reference not found: {ref_path}")

    got, ref = load(run_path), load(ref_path)

    allowed = set(IDENTITY)
    if args.smoke:
        allowed |= SMOKE_ONLY
    if not args.strict_env:
        allowed |= ENVIRONMENTAL

    # Keys present in one file only. This is almost always an ultralytics version
    # bump adding or renaming an argument -- reported, but not a parity failure on
    # its own, because the pin check in pre-flight is the real guard for that.
    only_got = sorted(set(got) - set(ref))
    only_ref = sorted(set(ref) - set(got))

    diffs = []
    for k in sorted(set(got) & set(ref)):
        if got[k] != ref[k]:
            diffs.append((k, ref[k], got[k]))

    real = [(k, a, b) for k, a, b in diffs if k not in allowed]
    expected = [(k, a, b) for k, a, b in diffs if k in allowed]

    print(f"run       : {run_path}")
    print(f"reference : {ref_path}")
    print(f"keys      : {len(got)} in run, {len(ref)} in reference, "
          f"{len(set(got) & set(ref))} shared\n")

    if expected:
        print("differing as expected (identity / environment / smoke):")
        for k, a, b in expected:
            print(f"  {k:<16} reference={a!r}  run={b!r}")
        print()

    if only_got or only_ref:
        print("keys present in only one file (ultralytics version drift):")
        for k in only_got:
            print(f"  + {k:<16} only in the run        = {got[k]!r}")
        for k in only_ref:
            print(f"  - {k:<16} only in the reference  = {ref[k]!r}")
        print()

    if real:
        print(f"PARITY FAILED — {len(real)} hyperparameter(s) differ from v2:")
        for k, a, b in real:
            print(f"  {k:<16} v2={a!r}  ->  this run={b!r}")
        print("\nThe arms are only comparable to v2's 0.7118 if these match. Either")
        print("restore the value or record the change in docs/DECISIONS.md and say so")
        print("in the write-up -- do not let it pass as a label effect.")
        return 1

    print("PARITY OK — every shared hyperparameter matches the v2 run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

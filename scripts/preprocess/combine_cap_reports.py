"""
scripts/preprocess/combine_cap_reports.py
──────────────────────────────────────────
Set operations (union / intersect) over 2+ cap_per_class.py-shaped reports'
per-class "selected" lists. Sits next to cap_per_class.py, its only sibling
that produces cap reports -- report/decision-only, same posture as every
other Stage 5.3-5.9 script: never touches dataset/processed/, dataset/merged/,
or any other pipeline artifact, only reads existing reports and writes a new
one in the same shape scripts/build/merge.py already knows how to consume.

Built for the RunPod dedup union-coverage plan (docs/RUNPOD_DEDUP_PLAN.md) --
full context there, summary here:

  --op union: combine two (or more) --hard-cap presets' real selections (e.g.
  4500 and 9000) into one pool worth deduping once. Whether the smaller cap's
  selection is actually a subset of the larger one depends on
  cap_per_class.py's INSTANCE_TARGET/priority-source budget math, which is
  NOT guaranteed nested in general (see docs/DECISIONS.md DEC-068 and the
  RunPod plan's Context section) -- unioning proves full coverage instead of
  assuming it, at zero extra cost when nesting happens to hold.

  --op intersect: after a post-dedup review pass promotes corrected/added
  labels (which can, rarely, shift what a fresh cap_per_class.py run
  selects), intersect the FRESH selection against the ORIGINAL union that
  was actually deduped. Every image in the result was both (a) selected by
  the current selection logic and (b) actually checked by the dedup run --
  coverage guaranteed by construction, not by a promise that review wouldn't
  touch anything. Put the fresh/post-review report FIRST in --inputs for
  --op intersect: the printed dropped-count is relative to that first input.

Non-clobbering guarantee, mechanically enforced (not just documented): refuses
to write to dataset/reports/cap_report.json or any cap_report_hardcap*.json,
even with --force -- the same norm DEC-068/DEC-066 already established for
cap_per_class.py's and dedup.py's own non-default outputs.

Validation is provenance-only (seed, instance_target, class-key-set agreement
across inputs), not a re-scan of dataset/processed/ -- the realistic failure
mode is a config/code edit slipping in between two cap_per_class.py runs in
the same session, which these cheap checks catch; scripts/build/merge.py
already reports missing_images/missing_labels for anything that doesn't
survive to a real merge.

Usage:
    python3 scripts/preprocess/combine_cap_reports.py --op union \\
        --inputs dataset/reports/cap_report.json dataset/reports/cap_report_hardcap9000.json \\
        --output dataset/reports/cap_report_dedup_union.json

    python3 scripts/preprocess/combine_cap_reports.py --op intersect \\
        --inputs dataset/reports/cap_report.json dataset/reports/cap_report_dedup_union.json \\
        --output dataset/reports/cap_report_4500_covered.json

    python3 scripts/preprocess/combine_cap_reports.py --op union --inputs ... --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Leaf script, not a shared library module — needs the repo root on
# sys.path to import sibling utils regardless of how it's invoked.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.config_loader import get_inactive_processed_source_keys  # noqa: E402
from scripts.utils.file_utils import discover_processed_sources, ensure_dir, reports_dir  # noqa: E402

# Filenames merge.py / cap_per_class.py already treat as canonical -- refused as an --output
# target unconditionally (see module docstring's non-clobbering guarantee).
PROTECTED_OUTPUT_NAME = "cap_report.json"
PROTECTED_OUTPUT_PREFIX = "cap_report_hardcap"


def is_protected_output_name(name: str) -> bool:
    if name == PROTECTED_OUTPUT_NAME:
        return True
    return name.startswith(PROTECTED_OUTPUT_PREFIX) and name.endswith(".json")


def load_cap_report(path: Path) -> dict[str, Any]:
    """Read + minimally validate one cap report (cap_per_class.py's or this script's own output)."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e
    if "classes" not in data:
        raise ValueError(f"{path} has no top-level 'classes' key -- not a valid cap report.")
    return data


def report_selected_sets(data: dict[str, Any]) -> dict[str, set[str]]:
    """{class_name: {"<source>/<filename>", ...}} — the raw 'selected' entries as a set, per class."""
    return {name: set(entry["selected"]) for name, entry in data["classes"].items()}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_provenance() -> tuple[str | None, bool | None]:
    """(commit_hash, is_dirty) — best-effort; (None, None) if git isn't available here."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=REPO_ROOT,
        ).stdout
        return commit, bool(status.strip())
    except Exception:
        return None, None


def combine(op: str, input_paths: list[Path]) -> dict[str, Any]:
    if len(input_paths) < 2:
        raise ValueError(f"--op {op} needs at least 2 --inputs, got {len(input_paths)}.")

    reports = [load_cap_report(p) for p in input_paths]

    seeds = {r.get("seed") for r in reports}
    if len(seeds) > 1:
        raise ValueError(f"Inputs disagree on 'seed': {seeds} — different code state, refusing to combine.")
    instance_targets = {r.get("instance_target") for r in reports}
    if len(instance_targets) > 1:
        raise ValueError(
            f"Inputs disagree on 'instance_target': {instance_targets} — different code state, "
            f"refusing to combine."
        )
    class_key_sets = [frozenset(r["classes"].keys()) for r in reports]
    if len(set(class_key_sets)) > 1:
        raise ValueError(
            "Inputs' class-key sets differ — likely built against different classes.yaml, refusing "
            "to combine."
        )

    per_report_selected = [report_selected_sets(r) for r in reports]
    class_names = sorted(class_key_sets[0])
    flat_per_input = [set().union(*s.values()) if s else set() for s in per_report_selected]

    combined_classes: dict[str, Any] = {}
    dropped_by_class: dict[str, int] = {}
    result_total = 0

    for class_name in class_names:
        sets_for_class = [s.get(class_name, set()) for s in per_report_selected]
        if op == "union":
            result = set().union(*sets_for_class)
        else:  # intersect
            result = set(sets_for_class[0])
            for s in sets_for_class[1:]:
                result &= s
            dropped = len(sets_for_class[0] - result)
            if dropped:
                dropped_by_class[class_name] = dropped

        combined_classes[class_name] = {"selected": sorted(result), "final_images": len(result)}
        result_total += len(result)

    # Pairwise coverage diagnostics, informative regardless of op — computed on each input's
    # FULL selected pool (unioned across classes), the granularity merge.py actually operates
    # at (a single (source, filename) union, per-class provenance already discarded by then).
    coverage_pairwise: dict[str, int] = {}
    for i, path in enumerate(input_paths):
        only_here = set(flat_per_input[i])
        for j, other in enumerate(flat_per_input):
            if i != j:
                only_here -= other
        coverage_pairwise[f"{path.name}_only"] = len(only_here)
    coverage_pairwise["intersection_all"] = len(set.intersection(*flat_per_input)) if flat_per_input else 0
    coverage_pairwise["union_all"] = len(set().union(*flat_per_input)) if flat_per_input else 0

    if op == "union":
        referenced_sources = {item.split("/", 1)[0] for cls in combined_classes.values() for item in cls["selected"]}
        active_sources = set(discover_processed_sources()) - get_inactive_processed_source_keys()
        stale_sources = referenced_sources - active_sources
        if stale_sources:
            print(
                f"  WARNING: union references {len(stale_sources)} source(s) no longer active "
                f"(benched/failed/blocked since these inputs were generated): {sorted(stale_sources)}"
            )

    if op == "intersect":
        if dropped_by_class:
            total_dropped = sum(dropped_by_class.values())
            print(
                f"  {op}: {total_dropped} (source, filename) selection(s) from {input_paths[0].name} "
                f"dropped — not covered by the other input(s). Per class:"
            )
            for name, cnt in sorted(dropped_by_class.items(), key=lambda kv: -kv[1]):
                print(f"    {name:18s} {cnt}")
        else:
            print(f"  {op}: 0 dropped — {input_paths[0].name}'s selection was fully covered.")

    commit, dirty = git_provenance()
    return {
        "classes": combined_classes,
        "combine_op": op,
        "hard_cap_preset": None,  # deliberate — a combined report is never mistaken for a real preset run
        "seed": seeds.pop(),
        "instance_target": instance_targets.pop(),
        "inputs": [
            {
                "path": str(p.resolve()),
                "sha256": sha256_of(p),
                "mtime_iso": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "hard_cap_preset": r.get("hard_cap_preset"),
                "selected_total": len(flat_per_input[i]),
            }
            for i, (p, r) in enumerate(zip(input_paths, reports))
        ],
        "coverage": {
            "result_total": result_total,
            "pairwise": coverage_pairwise,
            "dropped_by_class": dropped_by_class,
        },
        "git_commit": commit,
        "git_dirty": dirty,
        "generated_at": datetime.now().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--op", required=True, choices=("union", "intersect"),
        help="'union': combine selections into one pool worth deduping. 'intersect': cut a fresh "
             "selection back to only images an earlier dedup run actually covered.",
    )
    parser.add_argument(
        "--inputs", required=True, nargs="+", type=Path, metavar="PATH",
        help="2+ cap-report JSON paths. For --op intersect, put the fresh/post-review report "
             "FIRST — the printed dropped-count is relative to that first input.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, metavar="PATH",
        help="Output path (default: dataset/reports/cap_report_combined.json). Refused if it "
             "would overwrite cap_report.json or any cap_report_hardcap*.json, even with --force.",
    )
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing --output.")
    parser.add_argument("--dry-run", action="store_true", help="Print the coverage diff, write nothing.")
    args = parser.parse_args()

    print("=" * 60)
    print("combine_cap_reports.py")
    print("=" * 60)

    if len(args.inputs) < 2:
        parser.error(f"--inputs needs at least 2 paths, got {len(args.inputs)}.")

    output_path = (args.output or reports_dir() / "cap_report_combined.json").resolve()
    if is_protected_output_name(output_path.name):
        parser.error(
            f"--output {output_path} would overwrite a canonical/preset cap report — refused, "
            f"even with --force. Pick a different filename."
        )
    if not args.dry_run and output_path.is_file() and not args.force:
        parser.error(f"--output {output_path} already exists — pass --force to overwrite.")

    print(f"--op {args.op}, {len(args.inputs)} inputs:")
    for p in args.inputs:
        print(f"    {p.resolve()}")

    result = combine(args.op, args.inputs)

    print(
        f"\n{args.op}: {result['coverage']['result_total']} total (source, filename) selections "
        f"across {len(result['classes'])} classes."
    )
    for label, count in result["coverage"]["pairwise"].items():
        print(f"  {label}: {count}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    ensure_dir(reports_dir())
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()

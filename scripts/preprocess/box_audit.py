"""
scripts/preprocess/box_audit.py
─────────────────────────────────
Stage 5.3 Box Audit: reads every source under dataset/processed/ (DEC-046's
intermediate schema — flat images/+labels/, canonical class ids, no source
knowledge needed) and reports box-shape/size sanity plus global class
balance. Read-only — never touches dataset/processed/, only writes to
dataset/reports/.

Sources are auto-discovered (any dataset/processed/<name>/ with both an
images/ and labels/ subdir), not hardcoded — this is what makes one script
cover both of PLAN.md's "native_unspecified sources" and "project_dependent
sources" queued items (Roboflow's per-project bbox_mode was never actually
distinct at the config level, and by Stage 5.3 every source already speaks
the same intermediate schema regardless of its raw format). traffico_y1 has
no processed directory (never pulled — blocked on Roboflow's side, DEC-032/
037) so it's naturally skipped, not specially handled.

Two kinds of shape/size flags, both grouped per (source, canonical class)
rather than one global magic number — a box that's unusually large/oddly
shaped for "Pole" (naturally tall and thin) is normal for "Escalator" and
vice versa:

  1. large_area_outlier — box covers an unusually large fraction of the
     frame relative to its own (source, class) group. This is the
     elevator_status_s4lrk heuristic from docs/OPEN_QUESTIONS.md #3: real
     person-detection boxes vs. near-full-frame "elevator state" pseudo-
     labels. Verified empirically before picking a method (not assumed):
     elevator_status_s4lrk's real area-fraction distribution tops out at
     0.595 with no bimodal gap (see docs/DECISIONS.md DEC-057), so a fixed
     "near 1.0" threshold would catch nothing — a per-group statistical
     outlier test is used instead.
  2. shape_outlier — unusually elongated box (max(w,h)/min(w,h)) relative
     to its own group — the DEC-031 "boxes cling to object shape rather
     than a clean axis-aligned rectangle" (Stairs/Elevator) symptom shows
     up as elongation, not necessarily area.

Outlier method: Tukey's fences (Q3 + 1.5*IQR), the standard textbook
convention for flagging distribution outliers — not a project-specific
invented number (docs/OPEN_QUESTIONS.md's warning about inventing
thresholds is about *this project's own* parameters, e.g. a dedup
similarity cutoff; 1.5*IQR is a generic statistical convention applied
the same way regardless of what the data actually is). Only applied to
groups with >= MIN_GROUP_SIZE boxes — Tukey fences on a handful of points
aren't meaningful.

This script does not delete, move, or "fix" anything. It flags. The
flagged lists are for the student to review in
notebooks/fiftyone_review_processed.ipynb — same posture as
run_mistakenness.py (Stage 5.5) and final_merge_curation.py (Stage 5.7).

Usage:
    python3 scripts/preprocess/box_audit.py
    python3 scripts/preprocess/box_audit.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

# Leaf script, not a shared library module — needs the repo root on
# sys.path to import sibling utils regardless of how it's invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.bbox_utils import validate_bbox  # noqa: E402
from scripts.utils.config_loader import get_canonical_names  # noqa: E402
from scripts.utils.file_utils import (  # noqa: E402
    IMAGE_EXTENSIONS, discover_processed_sources, ensure_dir, processed_dir, reports_dir,
)

MIN_GROUP_SIZE = 20  # below this, Tukey fences aren't meaningful — report stats only, don't flag
TINY_AREA_FRACTION = 1e-4  # absolute backstop for degenerate near-zero boxes, any source/class

# bbox_utils.validate_bbox()'s strict [0,1] bounds check (epsilon=0.0) is designed for
# pre-write conversion (DEC-046 converters call it on freshly-computed floats). This
# script instead parses already-written, 6-decimal-rounded text — round-tripping
# cx±w/2 through that rounding reintroduces ~5e-7 of floating-point noise per corner,
# which (checked directly against real crowdhuman/open_images/roboflow output before
# picking this number, not guessed) was enough to make validate_bbox(epsilon=0.0) flag
# over 100,000 boxes as "invalid" that are not actually defective. BOUNDS_EPSILON is
# three orders of magnitude larger than that noise floor and orders of magnitude
# smaller than any real annotation defect this project has actually seen (DEC-057's
# genuine crowdhuman defects violated bounds by whole box-widths, not fractions of a
# pixel).
BOUNDS_EPSILON = 1e-4


def parse_labels(source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Parse every label file for one source into a flat list of box records:
    {source, label_path (relative to labels dir), class_id, cx, cy, w, h,
    area_fraction, elongation}. Also returns file-pairing stats.
    """
    labels_dir = processed_dir(source) / "labels"
    images_dir = processed_dir(source) / "images"

    label_stems = {p.stem for p in labels_dir.glob("*.txt")}
    image_stems = {p.stem for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}

    file_stats = {
        "images_total": len(image_stems),
        "labels_total": len(label_stems),
        "images_without_label": sorted(image_stems - label_stems)[:20],
        "images_without_label_count": len(image_stems - label_stems),
        "labels_without_image": sorted(label_stems - image_stems)[:20],
        "labels_without_image_count": len(label_stems - image_stems),
        "empty_label_files": 0,
    }

    boxes: list[dict[str, Any]] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        lines = [ln for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            file_stats["empty_label_files"] += 1
            continue
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])
            reason = validate_bbox(cx, cy, w, h, epsilon=BOUNDS_EPSILON)
            elongation = (max(w, h) / min(w, h)) if min(w, h) > 0 else float("inf")
            boxes.append({
                "label_path": str(label_path.relative_to(labels_dir)),
                "class_id": class_id,
                "cx": cx, "cy": cy, "w": w, "h": h,
                "area_fraction": w * h,
                "elongation": elongation,
                "invalid_reason": reason,
            })

    return boxes, file_stats


def tukey_upper_fence(values: list[float]) -> float | None:
    """Q3 + 1.5*IQR. None if there aren't enough points for a meaningful fence."""
    if len(values) < MIN_GROUP_SIZE:
        return None
    sv = sorted(values)
    q1 = statistics.quantiles(sv, n=4)[0]
    q3 = statistics.quantiles(sv, n=4)[2]
    iqr = q3 - q1
    return q3 + 1.5 * iqr


def audit_source(source: str, canonical_names: list[str]) -> dict[str, Any]:
    boxes, file_stats = parse_labels(source)

    by_group: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for b in boxes:
        by_group.setdefault((source, b["class_id"]), []).append(b)

    per_class_stats: dict[str, Any] = {}
    flagged: list[dict[str, Any]] = []
    invalid_found = 0

    for (_, class_id), group_boxes in sorted(by_group.items()):
        class_name = canonical_names[class_id] if 0 <= class_id < len(canonical_names) else f"unknown_id_{class_id}"
        areas = [b["area_fraction"] for b in group_boxes]
        elongs = [b["elongation"] for b in group_boxes if b["elongation"] != float("inf")]

        per_class_stats[class_name] = {
            "class_id": class_id,
            "instances": len(group_boxes),
            "images": len({b["label_path"] for b in group_boxes}),
            "area_fraction": {
                "min": min(areas), "max": max(areas),
                "mean": statistics.mean(areas), "median": statistics.median(areas),
            },
            "elongation": {
                "min": min(elongs) if elongs else None, "max": max(elongs) if elongs else None,
                "mean": statistics.mean(elongs) if elongs else None,
                "median": statistics.median(elongs) if elongs else None,
            },
        }

        area_fence = tukey_upper_fence(areas)
        elong_fence = tukey_upper_fence(elongs) if elongs else None
        per_class_stats[class_name]["area_fraction"]["tukey_upper_fence"] = area_fence
        per_class_stats[class_name]["elongation"]["tukey_upper_fence"] = elong_fence

        for b in group_boxes:
            reasons = []
            if b["invalid_reason"] is not None:
                reasons.append(f"invalid: {b['invalid_reason']}")
                invalid_found += 1
            if b["area_fraction"] < TINY_AREA_FRACTION:
                reasons.append("tiny_box")
            if area_fence is not None and b["area_fraction"] > area_fence:
                reasons.append(f"large_area_outlier (>{area_fence:.4f})")
            if elong_fence is not None and b["elongation"] > elong_fence:
                reasons.append(f"shape_outlier (elongation>{elong_fence:.2f})")
            if reasons:
                flagged.append({
                    "label_path": b["label_path"],
                    "class": class_name,
                    "cx": b["cx"], "cy": b["cy"], "w": b["w"], "h": b["h"],
                    "reasons": reasons,
                })

    return {
        "file_stats": file_stats,
        "boxes_total": len(boxes),
        "boxes_invalid": invalid_found,
        "boxes_flagged": len(flagged),
        "per_class_stats": per_class_stats,
        "flagged_full": flagged,  # not written to the main report as-is — see run()
    }


FLAGGED_SAMPLE_SIZE = 300  # per source, in the main report — full lists would make crowdhuman's
# ~68k flagged entries alone balloon the report to tens of MB for marginal review value; reason
# counts + a bounded sample cover "is this heuristic finding something real" without that cost.
# elevator_status_s4lrk gets its own full-list file regardless (see below) since docs/
# OPEN_QUESTIONS.md #3 specifically asks for a reviewable flagged list for that source.


def run(dry_run: bool = False) -> dict[str, Any]:
    canonical_names = get_canonical_names()
    sources = discover_processed_sources()

    report: dict[str, Any] = {"sources": {}, "global_class_balance": {}}
    global_balance: dict[str, dict[str, int]] = {
        name: {"images": 0, "instances": 0} for name in canonical_names
    }
    full_flagged_by_source: dict[str, list[dict[str, Any]]] = {}

    for source in sources:
        print(f"\n[{source}]")
        result = audit_source(source, canonical_names)
        full_flagged = result.pop("flagged_full")
        full_flagged_by_source[source] = full_flagged

        reason_counts: dict[str, int] = {}
        for entry in full_flagged:
            for r in entry["reasons"]:
                key = r.split(" (")[0].split(": ")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
        result["flagged_reason_counts"] = reason_counts
        result["flagged_sample"] = full_flagged[:FLAGGED_SAMPLE_SIZE]

        report["sources"][source] = result

        fs = result["file_stats"]
        print(f"  images={fs['images_total']} labels={fs['labels_total']} "
              f"empty_label_files={fs['empty_label_files']} "
              f"images_without_label={fs['images_without_label_count']} "
              f"labels_without_image={fs['labels_without_image_count']}")
        print(f"  boxes_total={result['boxes_total']} boxes_invalid={result['boxes_invalid']} "
              f"boxes_flagged={result['boxes_flagged']} reasons={reason_counts}")

        for class_name, stats in sorted(result["per_class_stats"].items()):
            global_balance[class_name]["images"] += stats["images"]
            global_balance[class_name]["instances"] += stats["instances"]

    report["global_class_balance"] = global_balance
    report["_full_flagged_by_source"] = full_flagged_by_source  # stripped before writing; see main()
    print("\n" + "=" * 60)
    print("Global class balance (images non-exclusive across sources, instances = total boxes):")
    for name in canonical_names:
        b = global_balance[name]
        print(f"  {name:18s} images={b['images']:6d}  instances={b['instances']:7d}")

    if dry_run:
        report.pop("_full_flagged_by_source")
        return report

    ensure_dir(reports_dir())

    # Convenience extract for the specific elevator_status_s4lrk heuristic
    # named in docs/OPEN_QUESTIONS.md #3 — full list (not the report's
    # bounded sample), since this is the one source the student was asked
    # to actually review box-by-box.
    elevator_key = "roboflow_elevator_status_s4lrk"
    if elevator_key in full_flagged_by_source:
        elevator_flagged = full_flagged_by_source[elevator_key]
        elevator_path = reports_dir() / "elevator_status_s4lrk_flagged.json"
        with elevator_path.open("w", encoding="utf-8") as fh:
            json.dump(elevator_flagged, fh, indent=2)
        print(f"elevator_status_s4lrk: {len(elevator_flagged)} boxes flagged -> {elevator_path}")

    report.pop("_full_flagged_by_source")
    report_path = reports_dir() / "box_audit_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nFull report written to {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print without writing reports.")
    args = parser.parse_args()

    print("=" * 60)
    print("box_audit.py — Stage 5.3")
    print("=" * 60)

    run(dry_run=args.dry_run)

    if args.dry_run:
        print("\n--dry-run: no report files written.")


if __name__ == "__main__":
    main()

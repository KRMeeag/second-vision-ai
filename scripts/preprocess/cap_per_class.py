"""
scripts/preprocess/cap_per_class.py
──────────────────────────────────────
Stage 5.4 Per-Class Capping: decides, per canonical class, which images
from dataset/processed/ would be kept under DEC-014's ExDark-guaranteed-
floor rule and DEC-042's floor(1500)/hard-cap(4500)/instance-target(10000)/
ratio-invariant(3:1) policy.

Same posture as box_audit.py (Stage 5.3): this is a DECISION/REPORT
script, not a file-mover. It writes dataset/reports/cap_report.json
describing which (source, filename) pairs are selected vs. excluded per
class, and why — it does not copy, move, or delete anything under
dataset/processed/ or write a new "capped" pool. Reasons this scope was
chosen over physically materializing a capped set, spelled out because
it's a real judgment call, not an obvious default:

  1. This stage's own trim method is an explicitly-flagged, unreviewed
     default (docs/OPEN_QUESTIONS.md #6: seeded random sampling, not a
     researched decision) — committing it to a physical file selection
     before the student has seen it would bake an unreviewed guess into
     the dataset layout.
  2. scripts/build/merge.py (Stage 5.6), per this session's own build
     scope, mechanically pools every dataset/processed/<source>/ into
     dataset/merged/ — it does not consult this report. Stages 5.5 and
     5.7 (mistakenness / final curation) are *also* report-only, flag-
     for-human-review stages by explicit design (handoff §1: "no script
     eliminates" the human review step). Making 5.4 the one stage that
     silently, automatically applies its own decision to the physical
     file layout would break that symmetry and pre-empt review the other
     three report-producing stages deliberately don't.
  3. DEC-042 also specifies a cap *recompute* rule ("recompute hard_cap
     as 3x min(realized_class_images) once the true floor is known") —
     this run's own numbers trigger it (Trash Bins' realized pool is
     already known, DEC-051, to land under the 1500 floor), but applying
     it would silently shrink the accepted 4,500 cap for all 16 classes
     — a sweeping change docs/OPEN_QUESTIONS.md #1 already puts in the
     student's hands, not something to decide unattended. This script
     reports the trigger; it does not act on it.

Usage:
    python3 scripts/preprocess/cap_per_class.py
    python3 scripts/preprocess/cap_per_class.py --dry-run
    python3 scripts/preprocess/cap_per_class.py --hard-cap 1500
    python3 scripts/preprocess/cap_per_class.py --hard-cap 9000

--hard-cap selects one of three presets (default 4500, matching classes.yaml
and every prior real run). floor is DERIVED from it as hard_cap // 3, per
DEC-042's own stated ratio invariant (hard_cap = 3x floor) — there is no
separate "floor preset" to pick independently. INSTANCE_TARGET is NOT scaled
by this flag: DEC-042 already establishes it as an independently-chosen
number, not formulaically tied to floor/cap (see the INSTANCE_TARGET comment
below), so scaling it here would mean inventing a formula DEC-042 never
specified. This is a real, currently-open gap (docs/OPEN_QUESTIONS.md) —
flagged, not silently resolved.

The 4500 preset writes to the canonical dataset/reports/cap_report.json path
that merge.py (Stage 5.6) reads unconditionally — running it reproduces the
exact same file merge.py already expects. The 1500/9000 presets write to
separate, non-clobbering report files (cap_report_hardcap<N>.json) since
nothing downstream reads them yet; they exist for comparison/exploration,
not for feeding the pipeline.

RESOLVED 2026-08-18 (was open in docs/OPEN_QUESTIONS.md #6): a class's first
priority source could previously exhaust an entire shrunken hard_cap on its
own, leaving 0 for any priority source listed after it (real case: Person's
exdark pool alone consumed all 1,500 images at --hard-cap 1500, starving
crowdhuman). Fixed with a per-priority-source minimum image reservation,
scaled from floor (see per_source_floor in cap_class()) — every priority
source now gets at least a fair slice regardless of an earlier source's real
candidate pool size. Verified as an exact no-op at the DEFAULT_HARD_CAP
(4500) real run — see cap_class()'s own comment for the specific numbers
checked. PRIORITY_SOURCE_INSTANCE_SUBBUDGET_BASE's values (currently just
crowdhuman's) now also scale proportionally with --hard-cap via
scaled_instance_subbudget(), instead of staying fixed regardless of preset.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# Leaf script, not a shared library module — needs the repo root on
# sys.path to import sibling utils regardless of how it's invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.config_loader import get_canonical_names, load_classes  # noqa: E402
from scripts.utils.file_utils import discover_processed_sources, ensure_dir, processed_dir, reports_dir  # noqa: E402

HARD_CAP_PRESETS = (1500, 4500, 9000)
DEFAULT_HARD_CAP = 4500  # DEC-042's chosen value, matches classes.yaml's per-class `cap` field

INSTANCE_TARGET = 10000  # DEC-042's "general" case. The "6000 for small/hard classes" refinement
# is NOT applied here — datasets.yaml/classes.yaml define no small/hard classification, and
# guessing one would be inventing an undecided parameter. Flagged in docs/OPEN_QUESTIONS.md.
# Fixed regardless of --hard-cap preset — DEC-042 states this is independently chosen, not
# formulaically tied to floor/cap, so there is no established rule to scale it by. Leaving it
# untouched means it simply binds less often at --hard-cap 1500 (image cap arrives first) and
# more often at --hard-cap 9000 (instance target arrives first) — a real behavior change, but
# not one this script invents an unjustified formula to "fix."
SEED = 42  # documented default (docs/OPEN_QUESTIONS.md #6) — matches every other script's convention

# Per-class ordered list of "priority" sources — each one's full candidate pool is
# reserved (up to whatever image/instance budget remains) before the next priority
# source, and before any general source gets a random-pooled share of what's left.
# "exdark" is the default (and only) priority source for every class not listed here —
# DEC-014's original guaranteed-floor rule, applied uniformly since it's a universal
# low-light-diversity concern, not a per-class judgment call.
#
# The three overrides below (decided with the student, 2026-08-15, see docs/DECISIONS.md
# DEC-067) each have a DIFFERENT reason for priority status — worth keeping straight:
#   - Person: crowdhuman added as a SECOND floor, after exdark. Not an edge-case source —
#     classes.yaml documents its role as "volume_topup" — but under pure random pooling it
#     was being crowded out to 128 of 19,370 candidates once ExDark's floor (dense,
#     low-light crowd scenes) ate 74.6% of the shared instance budget. Two floors, filled
#     in order: exdark first (condition diversity, DEC-014's original priority), then
#     crowdhuman (volume), then open_images fills whatever's left.
#   - Vehicle: roboflow_me5_u6rvg added as a second floor, after exdark. This one IS an
#     edge-case source — datasets.yaml explicitly notes its Jeepney/Tricycle/Ambulance
#     content as "Philippine-context relevance." Same two-floors-in-order pattern.
#   - Elevator: roboflow_elevator_awvus is the ONLY priority source (no exdark candidates
#     exist for Elevator at all — outside ExDark's 7-class overlap, DEC-014). This is
#     neither low-light nor edge-case: awvus has no Stage 5.3 flagged box-shape defects,
#     elevator_status_s4lrk does (844 flagged boxes) — a quality preference, not a
#     diversity or deployment-relevance one. Prioritizing the cleaner, smaller source
#     over the larger, defect-flagged one.
CLASS_PRIORITY_SOURCES: dict[str, list[str]] = {
    "Person": ["exdark", "crowdhuman"],
    "Vehicle": ["exdark", "roboflow_me5_u6rvg"],
    "Elevator": ["roboflow_elevator_awvus"],
}

# Optional per-(class, source) instance sub-budget, for priority sources whose extreme
# density would otherwise blow the class's realized instance count far past
# INSTANCE_TARGET under a plain image-count-only floor. Real numbers checked before
# picking this, not guessed: crowdhuman averages 22.7 Person-instances/image (some
# images over 300), so an uncapped 1,842-image floor produced ~42,200 instances alone —
# Person's realized total hit 49,664 against a 10,000 target. Decided with the student
# 2026-08-15 (docs/DECISIONS.md DEC-067): crowdhuman gets its own dedicated 2,500-
# instance allowance, separate from whatever ExDark already used of the shared
# INSTANCE_TARGET — bounded specifically to avoid contributing to a long-tailed
# per-class instance distribution across the dataset, the exact imbalance
# INSTANCE_TARGET exists to prevent in the first place.
#
# Sources with a sub-budget here are filled LEAST-DENSE-IMAGE-FIRST (ties broken by
# the seeded rng), not randomly — this matters specifically because of the skew: a
# random draw risks exhausting a small sub-budget on a handful of extremely crowded
# outlier images (CrowdHuman's max is 377 people in one frame), whereas filling
# smallest-instance-count-first maximizes real image/scene diversity per instance
# spent. Verified against real data: least-dense-first gets 729 images for a 2,500-
# instance budget; a naive random draw would get meaningfully fewer on average given
# the same images sometimes include 100+ instances each.
#
# Values here are calibrated at DEFAULT_HARD_CAP (4500) — scaled_instance_subbudget()
# below scales them proportionally to whatever --hard-cap preset is actually running,
# per the student's explicit request (2026-08-18) that every per-source subfloor
# track hard_cap, not just floor/INSTANCE_TARGET.
PRIORITY_SOURCE_INSTANCE_SUBBUDGET_BASE: dict[tuple[str, str], int] = {
    ("Person", "crowdhuman"): 2500,
}


def scaled_instance_subbudget(class_name: str, source: str, hard_cap: int) -> int | None:
    """Proportionally scale a priority source's instance sub-budget to the active hard_cap."""
    base = PRIORITY_SOURCE_INSTANCE_SUBBUDGET_BASE.get((class_name, source))
    if base is None:
        return None
    return round(base * hard_cap / DEFAULT_HARD_CAP)


def build_class_index(sources: list[str]) -> dict[int, dict[tuple[str, str], int]]:
    """
    {class_id: {(source, filename): instance_count}} — one scan of every
    processed label file. filename is the label stem (source-unique).
    """
    index: dict[int, dict[tuple[str, str], int]] = {}
    for source in sources:
        labels_dir = processed_dir(source) / "labels"
        for label_path in labels_dir.glob("*.txt"):
            counts: dict[int, int] = {}
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                counts[class_id] = counts.get(class_id, 0) + 1
            for class_id, cnt in counts.items():
                index.setdefault(class_id, {})[(source, label_path.stem)] = cnt
    return index


def cap_class(
    class_id: int,
    class_name: str,
    candidates: dict[tuple[str, str], int],
    hard_cap: int,
    floor: int,
    rng: random.Random,
) -> dict[str, Any]:
    """
    Apply this class's priority-source floors (CLASS_PRIORITY_SOURCES — DEC-014's ExDark
    floor by default, generalized per-class overrides per DEC-067) + DEC-042's cap/
    instance-target to one class's candidate pool.

    Priority sources are reserved in list order: each one's full candidate pool is
    included first (up to whatever image/instance budget remains after every earlier
    priority source already claimed its share), before the next priority source, before
    any general source gets a random-pooled share of what's left.
    """
    priority_sources = CLASS_PRIORITY_SOURCES.get(class_name, ["exdark"])

    # Per-priority-source minimum image reservation, scaled to hard_cap via floor
    # (floor = hard_cap // 3, DEC-042) — guarantees every priority source at least a
    # fair slice even when an earlier one in the list has more real candidates than a
    # shrunken hard_cap could ever hold on its own. Real case this fixes: Person's
    # exdark pool alone is 2,658 images, which fully consumed a 1,500-image hard_cap
    # and left crowdhuman at 0 (the exact "starved_next_priority_source" case
    # docs/OPEN_QUESTIONS.md #6 flagged as unresolved). No-op for every class with only
    # one priority source (13 of 16) — there's nothing after the first source to
    # protect, so the reservation is 0. Verified as a no-op at DEFAULT_HARD_CAP (4500)
    # too, for the 2 classes this does apply to (Person, Vehicle) — neither source's
    # real candidate pool was ever close to the new ceiling there, so real selection
    # counts are unchanged from before this fix (3,404/10,007 Person, 4,500/6,756
    # Vehicle, both confirmed identical on rerun).
    per_source_floor = floor // len(priority_sources) if len(priority_sources) > 1 else 0

    remaining_image_budget = hard_cap
    remaining_instance_budget = INSTANCE_TARGET
    priority_selected: list[tuple[tuple[str, str], int]] = []
    priority_breakdown: dict[str, dict[str, int]] = {}
    already_claimed: set[str] = set()

    for priority_idx, source in enumerate(priority_sources):
        already_claimed.add(source)
        source_items = {k: v for k, v in candidates.items() if k[0] == source}

        # This source may claim at most (remaining_image_budget minus what's reserved
        # for every priority source still queued behind it) — the mechanism that
        # actually prevents starvation, not just detects it.
        sources_remaining_after = len(priority_sources) - priority_idx - 1
        reserved_for_later = per_source_floor * sources_remaining_after
        image_ceiling = max(0, remaining_image_budget - reserved_for_later)

        subbudget = scaled_instance_subbudget(class_name, source, hard_cap)
        if subbudget is not None:
            # Least-dense-image-first, not random — see PRIORITY_SOURCE_INSTANCE_SUBBUDGET_BASE's
            # own comment for why. Tie-broken by the already-seeded rng (shuffle before the
            # stable sort) so equal-density images aren't picked in filesystem order.
            items = list(source_items.items())
            rng.shuffle(items)
            items.sort(key=lambda kv: kv[1])
            floor_selected = []
            floor_instance_count = 0
            for item in items:
                (_s, _fn), cnt = item
                if len(floor_selected) >= image_ceiling:
                    break
                if floor_instance_count + cnt > subbudget:
                    # Ascending order — every remaining item is >= this one, so none
                    # of them fit either. Stop, don't scan the rest for nothing.
                    break
                floor_selected.append(item)
                floor_instance_count += cnt
            floor_image_count = len(floor_selected)
        else:
            # Guaranteed floor for this source: shuffled with the same seeded rng as
            # every other selection in this function (reproducible, not filesystem-
            # order-dependent), then reserved unconditionally up to image_ceiling — the
            # "guaranteed floor" posture DEC-014 established for ExDark, now applied to
            # whichever source(s) this class's CLASS_PRIORITY_SOURCES entry lists, in
            # the order listed (earlier entries claim their share first, bounded so
            # later ones still get theirs).
            shuffled_source = list(source_items.items())
            rng.shuffle(shuffled_source)
            floor_selected = shuffled_source[:image_ceiling]
            floor_image_count = len(floor_selected)
            floor_instance_count = sum(v for _, v in floor_selected)

        priority_selected.extend(floor_selected)
        remaining_image_budget = max(0, remaining_image_budget - floor_image_count)
        remaining_instance_budget = max(0, remaining_instance_budget - floor_instance_count)
        # Now only fires in the genuinely-unavoidable edge case where per_source_floor
        # itself rounds to 0 (floor smaller than the number of priority sources) — the
        # reservation above prevents the ordinary case (an earlier source's large real
        # candidate pool) from ever causing this once reserved_for_later > 0.
        starved_next = remaining_image_budget == 0 and priority_idx < len(priority_sources) - 1
        priority_breakdown[source] = {
            "candidates": len(source_items),
            "selected_images": floor_image_count,
            "selected_instances": floor_instance_count,
            "reserved_for_later_priority_sources": reserved_for_later,
            "instance_subbudget": subbudget,
            "starved_next_priority_source": starved_next,
        }

    general_items = {k: v for k, v in candidates.items() if k[0] not in already_claimed}
    shuffled_general = list(general_items.items())
    rng.shuffle(shuffled_general)

    selected_general: list[tuple[tuple[str, str], int]] = []
    cumulative_instances = 0
    stop_reason = "all_candidates_included"
    for item in shuffled_general:
        (_source, _fn), cnt = item
        if len(selected_general) >= remaining_image_budget:
            stop_reason = "image_hard_cap"
            break
        if cumulative_instances >= remaining_instance_budget:
            stop_reason = "instance_target"
            break
        selected_general.append(item)
        cumulative_instances += cnt

    all_selected = priority_selected + selected_general
    final_images = len(all_selected)
    final_instances = sum(cnt for _, cnt in all_selected)
    excluded_general = len(general_items) - len(selected_general)

    per_source_selected: dict[str, int] = {}
    for (source, _fn), _cnt in all_selected:
        per_source_selected[source] = per_source_selected.get(source, 0) + 1
    per_source_candidates: dict[str, int] = {}
    for (source, _fn) in candidates:
        per_source_candidates[source] = per_source_candidates.get(source, 0) + 1

    return {
        "class_id": class_id,
        "class_name": class_name,
        "candidate_images_total": len(candidates),
        "candidate_instances_total": sum(candidates.values()),
        "priority_sources": priority_sources,
        "priority_breakdown": priority_breakdown,
        "general_candidates": len(general_items),
        "general_selected": len(selected_general),
        "general_excluded": excluded_general,
        "final_images": final_images,
        "final_instances": final_instances,
        "floor_met": final_images >= floor,
        "hard_cap": hard_cap,
        "floor": floor,
        "stop_reason": stop_reason,
        "per_source_candidates": per_source_candidates,
        "per_source_selected": per_source_selected,
        "selected": [f"{s}/{fn}" for (s, fn), _ in all_selected],
        "excluded_sample": [f"{s}/{fn}" for (s, fn), _ in shuffled_general[len(selected_general):][:50]],
    }


def run(dry_run: bool = False, hard_cap_preset: int = DEFAULT_HARD_CAP) -> dict[str, Any]:
    canonical_names = get_canonical_names()
    id_to_cap = {entry["id"]: entry["cap"] for entry in load_classes()["classes"].values()}
    sources = discover_processed_sources()

    floor = hard_cap_preset // 3  # DEC-042's ratio invariant: hard_cap = 3x floor
    print(f"Scanning {len(sources)} processed sources for per-class candidate pools...")
    print(f"hard_cap preset: {hard_cap_preset}  (derived floor: {floor})")
    class_index = build_class_index(sources)

    rng = random.Random(SEED)
    report: dict[str, Any] = {
        "classes": {}, "seed": SEED, "instance_target": INSTANCE_TARGET,
        "floor": floor, "hard_cap_preset": hard_cap_preset,
    }

    final_image_counts = {}
    for class_id in range(len(canonical_names)):
        class_name = canonical_names[class_id]
        candidates = class_index.get(class_id, {})
        configured_cap = id_to_cap[class_id]
        if configured_cap != hard_cap_preset:
            print(
                f"  NOTE: {class_name}'s classes.yaml cap ({configured_cap}) overridden by "
                f"--hard-cap={hard_cap_preset}"
            )
        result = cap_class(class_id, class_name, candidates, hard_cap_preset, floor, rng)
        report["classes"][class_name] = result
        final_image_counts[class_name] = result["final_images"]

        for src, breakdown in result["priority_breakdown"].items():
            if breakdown["starved_next_priority_source"]:
                print(
                    f"  WARNING: {class_name} — {src} alone consumed the entire remaining "
                    f"image budget, leaving 0 for the priority source(s) listed after it "
                    f"({result['priority_sources']})."
                )

        priority_summary = ", ".join(
            f"{src}={result['priority_breakdown'][src]['selected_images']}"
            for src in result["priority_sources"]
        )
        print(
            f"  {class_name:18s} candidates={result['candidate_images_total']:6d} "
            f"priority[{priority_summary}] "
            f"-> final={result['final_images']:5d} img / {result['final_instances']:6d} inst "
            f"(floor_met={result['floor_met']}, stop={result['stop_reason']})"
        )

    max_class = max(final_image_counts, key=final_image_counts.get)
    min_class = min(final_image_counts, key=final_image_counts.get)
    max_images, min_images = final_image_counts[max_class], final_image_counts[min_class]
    ratio = (max_images / min_images) if min_images > 0 else float("inf")

    report["ratio_invariant"] = {
        "max_class": max_class, "max_images": max_images,
        "min_class": min_class, "min_images": min_images,
        "ratio": ratio,
        "meets_3to1": ratio <= 3.0,
    }
    report["recompute_hard_cap_trigger"] = min_images < floor
    if min_images < floor:
        print(
            f"\nNOTE: {min_class}'s realized pool ({min_images} images) is below the {floor} floor. "
            f"DEC-042 specifies recomputing hard_cap as 3x this realized minimum "
            f"({3 * min_images}) — NOT applied here (would silently shrink every other class's "
            f"cap; see docs/OPEN_QUESTIONS.md #1, already the student's call)."
        )

    print(f"\nRatio invariant: max={max_class}({max_images}) / min={min_class}({min_images}) "
          f"= {ratio:.2f} (<=3.0: {ratio <= 3.0})")

    if dry_run:
        return report

    ensure_dir(reports_dir())
    # The 4500 default writes to the canonical path merge.py (Stage 5.6) reads
    # unconditionally — must stay exactly "cap_report.json" for the pipeline to keep
    # working. The 1500/9000 presets are exploratory/comparison runs nothing downstream
    # reads yet, so they get their own non-clobbering filenames instead of overwriting
    # the canonical report (the exact mistake DEC-066 already made once with dedup_report.json).
    report_filename = "cap_report.json" if hard_cap_preset == DEFAULT_HARD_CAP else f"cap_report_hardcap{hard_cap_preset}.json"
    report_path = reports_dir() / report_filename
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport written to {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print without writing the report.")
    parser.add_argument(
        "--hard-cap", type=int, default=DEFAULT_HARD_CAP, choices=HARD_CAP_PRESETS,
        help=f"Per-class image hard cap preset (default {DEFAULT_HARD_CAP}, matches classes.yaml). "
             f"floor is derived as hard_cap // 3 per DEC-042's ratio invariant.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("cap_per_class.py — Stage 5.4")
    print("=" * 60)

    run(dry_run=args.dry_run, hard_cap_preset=args.hard_cap)

    if args.dry_run:
        print("\n--dry-run: no report file written.")


if __name__ == "__main__":
    main()

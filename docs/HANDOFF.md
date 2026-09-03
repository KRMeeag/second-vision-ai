# Handoff — 2026-09-04

State at the end of the review phase, written for a fresh session. Read
`docs/DECISIONS.md` DEC-095 through DEC-100 for the reasoning behind any of this.

## Where the project is

The class schema is **13 classes**, just reduced from 16. All source reviews that
were going to happen before training have happened. The cascade has **not** been
re-run, so everything derived is stale.

```
Person  Vehicle  Motorcycle  Pole  Animals  Shelf  Doors
Chairs  Tables  Tricycle  Potholes  Trash Bins  Bicycle
```

## The immediate next step

**Run the cascade.** Nothing else is blocked on a decision.

```
python3 scripts/preprocess/cap_per_class.py --dry-run   # inspect first
python3 scripts/preprocess/cap_per_class.py
python3 scripts/build/merge.py
python3 scripts/preprocess/dedup.py
python3 scripts/build/split.py
python3 scripts/build/generate_yaml.py
```

`dataset/merged/` and `dataset/final/` currently hold 16-class, pre-review data and
must be regenerated before any training run. Expect per-class counts to move a lot:
reviews added boxes and removed images, and three classes are gone.

## What was just done (2026-09-04)

- **Reviews promoted.** `promote_reviews.py --all` copied 4,592 files from
  `labels_reviewed/` to `labels/` across 7 sources. This matters because
  `cap_per_class.py` reads `labels/` only (`cap_per_class.py:220`) — review work is
  invisible to the pipeline until promoted. `cv_project_hovyc` and
  `trashcan_detection_pihfn` were already promoted back in DEC-087.
- **Three classes dropped** (DEC-100). 53,206 of 124,045 label files rewritten,
  23,078 boxes removed, 16,529 files now empty. Verified: no out-of-range class id
  remains anywhere, `load_classes()` validates at nc=13.
- **Four sources benched** — `elevator_awvus`, `stair_gaptw`, `wtf_dwvgm`,
  `crosswalk_detector_lz3hc`. Each was 100% a dropped class.
- **dlsu label patches** (see below).

## Reviews completed

All written back, all with an exclusions JSON in `dataset/reports/`:

| source | images | notes |
|---|---|---|
| `dlsu_d_vehicle_type_detection` | 6,357 | quick pass; student intends to revisit |
| `revised_pedestrian_obstacle` | 2,088 | 267 exclusions |
| `cv_project_hovyc` | 1,040 | promoted DEC-087 |
| `pothole_voxrl` | 665 | |
| `door_detection_zqt59` | 607 | |
| `trashcan_detection_pihfn` | 559 | promoted DEC-087 |
| `roitrikee` | 444 | 48 exclusions |

Never reviewed, still active: `dataset_ninja_road_damage_detector` (1,331) and
`dataset_ninja_pothole_detection` (665). Both feed Potholes, which is over floor.

## Traps that will bite

**dlsu's `labels/` is patched and does NOT match its `native_class_filter`.**
3,372 boxes were folded in post-conversion — Electric Bike -> Tricycle, and
Sedan/SUV/Hatchback/Pickup -> Vehicle, and Bicycle -> Bicycle — for pool images
only, to fix false negatives without widening the pool. **Re-running
`yolo_to_intermediate.py` on dlsu deletes all of them silently** (it clears output
dirs first). Recovery command is in a comment directly under that filter in
`config/datasets.yaml`. `merge_native_class.py` is idempotent and backs up first.

**FiftyOne caches Sample objects per process** (DEC-097). Any code reading a live
review dataset must call `dataset.reload()` first, or it sees a stale snapshot.
`dataset.export()` and the aggregation calls read MongoDB directly and are always
accurate; iteration is not.

**The review notebook is v40** and its cells are heavily interdependent. Search by
cell id — `dupgt_code`, `dupgt_hide_code`, `autoaccept_adjust_code`,
`p0review3build`, `p0review2source` — not by description. **Do not edit the file
while it is open in the student's IDE**; tell them to close it first, or their
buffer overwrites the changes on save. Bump the version header in the same script
as any cell edit.

**Never `git commit` unless explicitly asked.** Nothing in this session was
committed. Seven untracked scripts under `scripts/preprocess/` would be lost to a
`git clean`: `promote_reviews.py`, `drop_classes.py`, `merge_native_class.py`,
`snapshot_state.py`, `deaugment_sources.py`, `build_dedup_keeplist.py`,
`apply_dedup_keeplist.py`.

## The safety net

`dataset/backups/pre_class_drop_20260904_023304/` — 4.1 GB, the complete 16-class
state. Config, reports, every `labels/` and `labels_reviewed/`, all referenced
images including `*_augmented_aside/`, and all 18 FiftyOne review datasets.
Verified self-contained: 43,700 of 43,700 image references resolve inside it,
`labels_reviewed/` byte-identical, FiftyOne round-trip exact.

`dataset/processed/` is gitignored, so for hand review this backup is the only copy.
Do not delete it, or the per-directory `*_bak_*` dirs beside each `labels/`.

## Working style the student asked for

- Give **options with pros and cons** for anything that needs a decision, rather
  than acting on a recommendation.
- Cite decisions with a summary or excerpt, never a bare `DEC-XXX`.
- Name notebook cells by searchable id, not by description.
- Write code rather than describing it, then explain what was written and the flow.

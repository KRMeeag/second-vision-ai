# Stage 5.3–5.9 Scripts — Implementation Reference

This document exists so you can scrutinize *how* each script works, not just that it ran. It's a function-by-function breakdown of every script built during the unattended Stage 5.3–5.9 session, plus the shared helpers they all lean on.

**This is not the place for *why* a threshold/ratio/method was chosen** — that's `docs/DECISIONS.md` (DEC-057 through DEC-066) and `docs/OPEN_QUESTIONS.md`. This doc is about *how the code does what it does*, so you can read a function here, then go open the actual file if you want to check a line yourself.

All 8 scripts share the same shape: a `run(dry_run=False, ...)` function that does the real work and returns a stats dict, a thin `main()` that parses CLI args and calls `run()`, and a `report = {...}` dict written to `dataset/reports/<name>.json` at the end. None of them mutate `dataset/raw/` or `dataset/processed/`. Most are flag-and-report only (Stage 5.3, 5.5, 5.7); `merge.py`, `split.py`, and `generate_yaml.py` are the ones that actually write to `dataset/merged/` and `dataset/final/`.

---

## Pipeline flow, end to end

```
dataset/processed/<source>/{images,labels}/     (16 sources, DEC-046 schema, canonical class ids)
        │
        ▼
5.3  box_audit.py            reads processed/, writes box_audit_report.json (+ elevator flagged list)
        │  (report only — nothing downstream depends on this file existing)
        ▼
5.4  cap_per_class.py        reads processed/, writes cap_report.json (which images to keep per class)
        │
        ▼
5.5  run_mistakenness.py     reads processed/ + cap_report.json, writes mistakenness_report.json
        │  (report only — human review loop, blocked on CVAT/Label Studio choice)
        ▼
5.6  merge.py                reads processed/ + cap_report.json, WRITES dataset/merged/
        │
        ▼
5.6  dedup.py                reads dataset/merged/, writes dedup_report.json
        │
        ▼
5.7  final_merge_curation.py reads dataset/merged/, writes final_merge_curation_report.json
        │  (report only, same as 5.5)
        ▼
5.8  split.py                reads dataset/merged/ + dedup_report.json, WRITES dataset/final/{train,val,test}/
        │
        ▼
5.9  generate_yaml.py        reads dataset/final/, WRITES dataset/final/data.yaml
```

Three things worth noticing about this shape before reading further:

1. **Only three scripts write files that matter downstream**: `merge.py`, `split.py`, `generate_yaml.py`. Everything else produces a JSON report and touches nothing else.
2. **`cap_per_class.py` doesn't move anything** — it's a *decision*, and `merge.py` is the script that actually *applies* that decision by reading `cap_report.json`. If you want to change which images get selected, you re-run `cap_per_class.py` then re-run `merge.py` (and everything after it).
3. **5.5 and 5.7 don't block 5.6/5.8/5.9.** They produce ranked reports for you to act on, but the pipeline's file-moving scripts don't wait for that review to happen — see `docs/DECISIONS.md` DEC-058's reasoning if you want the full argument for why.

---

## Shared utilities these scripts lean on

Two files got new helper functions this session, used across most of the 8 scripts below. Knowing what these do makes the per-script breakdowns shorter, since I won't re-explain them each time.

### `scripts/utils/file_utils.py`

- **`IMAGE_EXTENSIONS`** — module-level tuple `(".jpg", ".jpeg", ".png", ".bmp", ".webp")`. The single source of truth for "what counts as an image file" everywhere in this codebase.
- **`list_images(directory, recursive=True) -> list[Path]`** — every file under `directory` whose extension is in `IMAGE_EXTENSIONS`, sorted. Used instead of a bare `Path.iterdir()` so a stray non-image file (a `.DS_Store`, for instance — which already exists elsewhere under this project's `dataset/` tree) can't accidentally get treated as an image.
- **`build_stem_index(source) -> dict[str, Path]`** — for one source's `dataset/processed/<source>/images/`, returns `{filename_stem: full_path}`. Lets a script look up "give me the image for this label file" by dictionary lookup instead of re-globbing the directory for every single file (an O(N×M) pattern that was slow enough to matter — see `docs/DECISIONS.md` DEC-061).
- **`discover_processed_sources() -> list[str]`** — lists every `dataset/processed/<name>/` that has both an `images/` and a `labels/` subdirectory. This is what lets `box_audit.py` and `cap_per_class.py` operate on "however many sources currently exist" instead of a hardcoded list.
- **`prefixed_filename(source, filename) -> str`** — `f"{source}__{filename}"`. This is the naming scheme that makes `dataset/merged/`'s pooled filenames traceable back to their original source (and lets `split.py` recover the source by splitting on the first `"__"`).
- **`safe_copy(src, dst, overwrite=False)`** — a copy wrapper most of these scripts use instead of raw `shutil.copy`, so overwrite behavior is explicit at every call site.

### `scripts/utils/bbox_utils.py`

- **`validate_bbox(cx, cy, w, h, epsilon=0.0) -> str | None`** — returns `None` if the box is well-formed, or a string reason if not (non-positive width/height, or a corner outside `[0, 1]`). The `epsilon` parameter exists because `box_audit.py` re-parses already-rounded label text, which reintroduces tiny floating-point noise (~5e-7 per corner) that a strict `epsilon=0.0` check would misflag as invalid — see the function's own docstring for the measured numbers.
- **`clip_bbox(cx, cy, w, h) -> (cx, cy, w, h)`** — clamps a box into `[0, 1]`, corner-by-corner independently. This was a real bug fix this session: the old version only clamped one side of each axis, so a box sitting entirely outside the frame (e.g. both `x1` and `x2` past the right edge) came out with negative width instead of being pulled back to a valid (possibly zero-width) box. All 5 converter scripts now re-check `w > 0 and h > 0` *after* calling this, in case a box collapses to nothing after clipping.

---

## Stage 5.3 — `scripts/preprocess/box_audit.py`

**What it does:** reads every processed source's labels, computes shape/size statistics per `(source, class)` group, and flags boxes that are statistical outliers within their own group. Doesn't touch `dataset/processed/` — read-only.

**Why per-group, not one global threshold:** a box that's 80% of the frame is normal for "Escalator" and bizarre for "Pole." Flagging is always relative to a box's own `(source, class)` peers, never a single fixed number across the whole dataset.

**Constants:**
- `MIN_GROUP_SIZE = 20` — a `(source, class)` group smaller than this doesn't get outlier-flagged (not enough points for a statistically meaningful fence); it still gets reported, just without a `tukey_upper_fence`.
- `TINY_AREA_FRACTION = 1e-4` — an absolute backstop; a box below this area fraction gets flagged as `tiny_box` regardless of group statistics.
- `BOUNDS_EPSILON = 1e-4` — passed to `validate_bbox()`, see above.
- `FLAGGED_SAMPLE_SIZE = 300` — the main report only keeps the first 300 flagged entries *per source* (crowdhuman alone would otherwise balloon the report to tens of MB); the full list per source is kept in memory and used to write `elevator_status_s4lrk_flagged.json` separately, in full.

**Functions:**

| Function | What it does |
|---|---|
| `parse_labels(source)` | Reads every `.txt` label file for one source. For each box, computes `area_fraction = w*h` and `elongation = max(w,h)/min(w,h)`, and runs it through `validate_bbox()`. Also computes file-pairing stats (images with no label, labels with no image, empty label files). Returns `(list of box records, file_stats dict)`. |
| `tukey_upper_fence(values)` | The statistical core. Computes Q1 and Q3 via `statistics.quantiles(sv, n=4)`, then returns `Q3 + 1.5*(Q3-Q1)` — the standard "Tukey's fence" outlier threshold. Returns `None` if there are fewer than `MIN_GROUP_SIZE` values. |
| `audit_source(source, canonical_names)` | Orchestrates one source: calls `parse_labels`, groups boxes by `(source, class_id)`, computes per-group area/elongation stats and their Tukey fences, then walks every box again and flags it if: `validate_bbox` found a defect, `area_fraction < TINY_AREA_FRACTION`, `area_fraction > area_fence`, or `elongation > elong_fence`. A box can collect multiple reasons at once. |
| `run(dry_run)` | Top-level loop: calls `discover_processed_sources()`, runs `audit_source` on each, accumulates a `global_class_balance` (image/instance counts per canonical class, summed across all sources), extracts the `elevator_status_s4lrk` source's *full* flagged list into its own file, and writes the bounded/sampled main report. |
| `main()` | argparse wrapper (`--dry-run`), calls `run()`. |

**Outputs:** `dataset/reports/box_audit_report.json` (per-source stats + a 300-entry sample of flagged boxes each), `dataset/reports/elevator_status_s4lrk_flagged.json` (the full flagged list for that one specific source, since it was the one you were asked to actually review box-by-box).

**Worth scrutinizing:** the `MIN_GROUP_SIZE`/`TINY_AREA_FRACTION`/`FLAGGED_SAMPLE_SIZE` constants are all fixed numbers chosen without a formal derivation — reasonable defaults, but not researched the way, say, the dedup threshold was (see `docs/DECISIONS.md` DEC-057 for the reasoning behind each).

---

## Stage 5.4 — `scripts/preprocess/cap_per_class.py`

**What it does:** decides, per canonical class, which `(source, filename)` pairs would be kept under the project's floor/cap/instance-target policy. Writes a *decision*, not a file move — `dataset/processed/` is untouched, and no new "capped" directory is created. `merge.py` (Stage 5.6) is the script that actually reads this decision and copies files.

**Constants:**
- `HARD_CAP_PRESETS = (1500, 4500, 9000)`, `DEFAULT_HARD_CAP = 4500` — `--hard-cap` (DEC-068, 2026-08-17) selects one of these; every class's image cap for that run. `FLOOR` is no longer a fixed constant — it's *derived* each run as `hard_cap // 3` (DEC-042's own stated ratio invariant, hard_cap = 3× floor), computed once in `run()` and passed into `cap_class()`.
- `INSTANCE_TARGET = 10000` — total box count ceiling per class (not image count — a single image can contribute multiple instances). Deliberately **not** scaled by `--hard-cap` — DEC-042 states it's independently chosen, and no scaling formula was ever specified, so none is invented here. One real consequence: at `--hard-cap 9000` it binds far more often (more image budget lets dense classes reach the instance ceiling first); at `--hard-cap 1500` it almost never binds (image ceiling arrives first).
- `SEED = 42` — used for every random shuffle in this script, so re-running with the same inputs gives the same output.
- `CLASS_PRIORITY_SOURCES: dict[str, list[str]]` — per-class ordered list of "priority" sources (DEC-067, generalized from DEC-014's ExDark-only floor). Every class not listed defaults to `["exdark"]`. Three real overrides: `Person: ["exdark", "crowdhuman"]`, `Vehicle: ["exdark", "roboflow_me5_u6rvg"]`, `Elevator: ["roboflow_elevator_awvus"]` (no ExDark candidates exist for Elevator at all).
- `PRIORITY_SOURCE_INSTANCE_SUBBUDGET: dict[tuple[str, str], int]` — optional per-(class, source) instance ceiling for a priority source, currently just `{("Person", "crowdhuman"): 2500}`. Exists because CrowdHuman is exceptionally dense (mean ~22.7 Person-instances/image, some images 300+) — without this, an uncapped image-count floor for crowdhuman blew Person's realized instances to 49,664 against a 10,000 target.

**Functions:**

| Function | What it does |
|---|---|
| `build_class_index(sources)` | One pass over every source's label files, building `{class_id: {(source, filename): instance_count}}` — for every class, every candidate image and how many boxes of that class it has. This is the full candidate pool before any capping logic runs. |
| `cap_class(class_id, class_name, candidates, hard_cap, floor, rng)` | The actual capping logic, applied to one class's candidate pool. See the walkthrough below — this is the function worth reading closely if you want to check the selection logic yourself. |
| `run(dry_run, hard_cap_preset)` | Loads `config/classes.yaml`'s per-class `cap` values (printing a `NOTE` whenever `--hard-cap` overrides them — currently every class is `4500`, so this only prints when a non-default preset is used), derives `floor = hard_cap_preset // 3`, calls `build_class_index`, then calls `cap_class` once per canonical class using a single shared `random.Random(SEED)` instance (meaning the *order* classes are processed in affects what gets drawn for each — canonical id order, 0 to 15). Also computes the "ratio invariant" (`max class size / min class size <= 3.0`) and checks whether DEC-042's cap-recompute trigger fired, without applying it. |
| `main()` | argparse wrapper (`--dry-run`, `--hard-cap {1500,4500,9000}`), calls `run()`. |

**`cap_class()` walkthrough** (this is the one function actually worth stepping through line by line if you want to verify the logic):

1. Look up this class's `priority_sources` list (`CLASS_PRIORITY_SOURCES.get(class_name, ["exdark"])`).
2. **Priority sources, reserved in list order** — for each one in turn: if `PRIORITY_SOURCE_INSTANCE_SUBBUDGET` has an entry for `(class_name, source)`, shuffle that source's candidates with the shared `rng`, stable-sort ascending by instance count (least-dense-image-first — maximizes scene diversity per instance spent, verified against real CrowdHuman density data before picking this order), then greedily take items until the next one would exceed the sub-budget or the remaining image budget. Otherwise (no sub-budget), shuffle and take up to whatever image budget remains, unconditionally — the original DEC-014 "guaranteed floor" behavior, now applied per-class. After each priority source, `remaining_image_budget`/`remaining_instance_budget` are reduced by what it claimed, and a `starved_next_priority_source` flag is set `True` if this source's claim left zero budget for a priority source still queued behind it (DEC-068 — detects, e.g., ExDark alone exhausting a shrunken `--hard-cap 1500` budget and starving crowdhuman; does not prevent it).
3. **General fill:** every remaining candidate not from an already-claimed priority source gets shuffled with the same `rng`, then walked one at a time into the selection until **either** the remaining image budget or the remaining instance budget runs out — whichever hits first, recorded as `stop_reason` (`"image_hard_cap"`, `"instance_target"`, or `"all_candidates_included"` if the shuffle ran out before either budget did).
4. Final selection = every priority source's claim + whatever got picked from the general fill. The function also records a per-priority-source breakdown, per-source candidate/selected counts, and a 50-entry sample of what got excluded.

**Outputs:** `dataset/reports/cap_report.json` at the `--hard-cap 4500` default (the canonical path `merge.py` reads unconditionally) — per class, `selected` (full list of `"source/filename"` strings) and `excluded_sample` (first 50), plus the ratio-invariant and cap-recompute-trigger findings. The 1500/9000 presets write to separate `dataset/reports/cap_report_hardcap<N>.json` files instead, so a comparison run can never silently clobber the canonical report.

**Worth scrutinizing:** the *general* fill's trim method is still a flat "seeded random until budget runs out," not ranked by any quality signal (e.g. `box_audit.py`'s flagged-box status) — `docs/OPEN_QUESTIONS.md` #6, a documented, deliberate default, not an oversight. Also worth your own judgment: whether `INSTANCE_TARGET` staying fixed across all three `--hard-cap` presets is the right call, and how a shrunken hard_cap should be split fairly across multiple priority sources when the first one alone can exhaust it (both flagged in `docs/OPEN_QUESTIONS.md`, neither resolved).

---

## Stage 5.5 — `scripts/curate/run_mistakenness.py`

**What it does:** runs a pretrained (COCO) YOLOv8n model over Stage 5.4's capped selection and computes FiftyOne Brain's "mistakenness" score for each image — a measure of how much a model's predictions disagree with your ground-truth boxes. High mistakenness usually means either a genuinely hard/ambiguous image, or a labeling mistake (missing box, wrong class, etc.) worth a second look.

**Why only 7 of 16 classes:** mistakenness needs something to compare your ground truth against, and the only "something" available (no trained model of this project's own exists yet) is a COCO-pretrained model. Only classes with an unambiguous COCO equivalent can be scored this way. The `COCO_CROSSWALK` dict is the mapping, and it's derived from `config/classes.yaml`'s own already-decided `native_class` fields — not invented for this script.

**Module-level constants (also imported directly by `final_merge_curation.py` — see below):**
- `COCO_CROSSWALK: dict[str, list[str]]` — canonical class key → list of COCO class names, e.g. `"vehicle": ["car", "bus", "truck"]`.
- `CANONICAL_KEY_TO_NAME` — lowercase key → the actual canonical class name string, e.g. `"vehicle" → "Vehicle"`.
- `NO_COCO_ANALOG` — the 9 classes this script can't score.
- `ELIGIBLE_CANONICAL_IDS` — the numeric class ids for the 7 eligible classes, computed once at import time via `get_class_id()`.

**Functions:**

| Function | What it does |
|---|---|
| `yolo_to_fo_bbox(cx, cy, w, h)` | Converts YOLO's center-based box format to FiftyOne's top-left-based `[x, y, w, h]` format. A one-line coordinate conversion, but every FiftyOne-using script needs it. |
| `load_union_images(cap_report_path, limit)` | Reads `cap_report.json`, takes the union of the 7 eligible classes' `selected` lists (an image selected for *any* of those 7 classes gets scored — not just images selected specifically for, say, "Person"). |
| `load_eligible_ground_truth(source, filename, canonical_names)` | For one image, reads its label file and keeps only the boxes belonging to one of the 7 eligible classes (a Pole box on an otherwise-eligible image is silently excluded from what gets compared — there's nothing to compare it against). |
| `run(dry_run, limit)` | The main orchestration: builds the image list, loads the `yolov8n.pt` model, asserts every COCO class name in the crosswalk actually exists in the model's own `model.names` (a real check, not assumed), runs batched inference (`BATCH = 16`), builds a scratch FiftyOne dataset with both `ground_truth` and `predictions` fields per sample, calls `fiftyone.brain.compute_mistakenness()`, then reads the scores back out and sorts descending. The FiftyOne dataset is wrapped in `try/finally` so `dataset.delete()` always runs, even if something raises mid-computation. |
| `main()` | argparse wrapper (`--dry-run`, `--limit N` for a smoke test), calls `run()`. |

**Outputs:** `dataset/reports/mistakenness_report.json` — every scored image, ranked by mistakenness descending, with its ground-truth class list and box counts for both ground truth and predictions.

**Worth scrutinizing:** a `mistakenness` value of exactly `-1.0` is FiftyOne's own sentinel for "no matched prediction/ground-truth pair could be found" — it is *not* a real score on the `[0, 1]` scale and should be filtered out (or treated separately) before drawing conclusions from the ranking, e.g. `if r["mistakenness"] != -1.0`.

---

## Stage 5.6 (merge half) — `scripts/build/merge.py`

**What it does:** reads Stage 5.4's `cap_report.json`, takes the union of every class's `selected` images, and physically copies them (image + full label file) into `dataset/merged/`, with filenames prefixed by source (`prefixed_filename()`) so two sources' files can never collide.

**Important nuance:** an image selected because of one class (say, "Chairs") might also contain a valid box for another class (say, "Tables") that wasn't itself capped-selected for this image. `merge.py` copies the image's **entire original label file**, not just the box(es) that caused its selection — so a class's real post-merge count can end up slightly higher than its own `cap_report.json` figure. This is why the script recomputes true per-class counts from the merged labels themselves at the end, rather than trusting `cap_report.json`'s pre-merge numbers.

**Functions:**

| Function | What it does |
|---|---|
| `load_selected_union(cap_report_path)` | Reads `cap_report.json`, returns the set of every `(source, filename)` pair appearing in *any* class's `selected` list. |
| `run(dry_run)` | The main flow — see walkthrough below. |
| `main()` | argparse wrapper (`--dry-run`), calls `run()`. |

**`run()` walkthrough:**

1. Load the selected union.
2. **Clear `dataset/merged/images/` and `dataset/merged/labels/` before writing** (`shutil.rmtree` then recreate) — this matters: without it, re-running `merge.py` after `cap_report.json` changes would leave the *previous* run's now-deselected files sitting there as silent orphans. This was a real bug found and fixed mid-session (see `docs/DECISIONS.md` DEC-050/061).
3. Build a `{source: {stem: image_path}}` index via `build_stem_index()` for every source that appears in the selection (one directory listing per source, not one glob per file — a real performance fix, see DEC-061).
4. For each `(source, filename)` in the union: look up its image and label file. If either is missing, record it in `missing_images`/`missing_labels` and skip (doesn't crash). Otherwise copy both, with the image renamed via `prefixed_filename()` and the label renamed to match.
5. After copying, **re-scan `dataset/merged/labels/*.txt`** to compute the real, post-merge per-class image/instance counts — this is the "supersedes `cap_report.json`'s pre-merge estimate" step mentioned above. Out-of-range class ids found during this re-scan are tracked in `unknown_class_ids_in_merged_labels` rather than crashing the script.

**Outputs:** `dataset/merged/images/`, `dataset/merged/labels/` (the actual pooled dataset), `dataset/reports/merge_report.json` (counts, missing-file lists, real post-merge per-class counts).

**Worth scrutinizing:** step 2 (the `rmtree`) means every real run of `merge.py` fully regenerates `dataset/merged/` from scratch — there's no incremental/partial-merge mode. If you only want to add a handful of newly-corrected images later, this script will still redo the *entire* pool, not just the delta (this is by design — see `docs/DECISIONS.md` DEC-061 — but worth knowing before you run it against a very large future pool).

---

## Stage 5.6 (dedup half) — `scripts/preprocess/dedup.py`

**What it does:** runs two independent duplicate checks against `dataset/merged/`: an exact-duplicate check (byte-identical files, via filehash) across the *entire* merged pool, and a near-duplicate check (embedding-distance-based) across a *sample* of it.

**Why a sample for near-duplicates but not exact-duplicates:** exact-duplicate detection is a filehash comparison — cheap, no model, no degradation risk, so it runs on all 51,529 images. Near-duplicate detection needs to compute a neural embedding for every image, which is measurably expensive on this machine — an unbounded full-pool run was measured degrading from ~89 img/s to under 10 img/s within 90 seconds (see the module docstring for the full diagnosis). `NEAR_DUP_SAMPLE_SIZE = 6000` is a deliberate scope bound, not laziness.

**Constants:**
- `NEAR_DUPLICATE_THRESHOLD = 0.2` — FiftyOne Brain's own documented default for `compute_near_duplicates`, not invented for this project.
- `NEAR_DUP_SAMPLE_SIZE = 6000`, `NEAR_DUP_SEED = 42`.
- `EMBEDDING_NUM_WORKERS = 4` — pinned because FiftyOne's own default worker count over-subscribed this machine's single MPS (Apple GPU) device, causing the throughput collapse mentioned above.

**Functions:**

| Function | What it does |
|---|---|
| `stratified_sample(images, sample_size)` | Groups images by source (recovered from the `source__filename` prefix), then takes a proportional, seeded-random slice from each source — so the 6,000-image sample doesn't accidentally over-represent whichever source happens to sort first alphabetically. |
| `run(dry_run, limit)` | The main flow — see walkthrough below. |
| `main()` | argparse wrapper (`--dry-run`, `--limit N` — note: `--limit` caps *both* checks to the same N images, useful for a fast smoke test, but **it still writes to the real `dataset/reports/dedup_report.json` path** — there's no separate smoke-test output location, so running `--limit` against a machine that already has a real full-scale report will overwrite it). |

**`run()` walkthrough:**

1. List every image in `dataset/merged/images/` via `list_images()` (extension-filtered, so a stray `.DS_Store` can't sneak in).
2. Build the near-duplicate sample via `stratified_sample()` (or reuse the same limited list if `--limit` was passed).
3. Build **two** separate FiftyOne datasets: one with every image (for the exact-duplicate check), one with just the sample (for the near-duplicate check). Both are created inside the same `try` block, and both get `.delete()`'d in a `finally`, `None`-guarded so a failure creating either one doesn't leak the other.
4. `fiftyone.brain.compute_exact_duplicates(dataset)` — returns groups of `{keep_id: [duplicate_ids]}`. Translated back to filenames via a `.get(id, id)` fallback lookup (defensive against an id FiftyOne might return that isn't in the local `id_to_name` map — unlikely, but both this and the near-dup lookup below degrade the same way if it ever happens).
5. `fiftyone.brain.compute_near_duplicates(near_dataset, threshold=0.2, model=mobilenet-v2-imagenet-torch, num_workers=4)` — returns a `neighbors_map` of `{keep_id: [(duplicate_id, distance), ...]}`.
6. Both results get written into one combined report.

**Outputs:** `dataset/reports/dedup_report.json` — `images_checked` (full pool size), `near_duplicate_sample_size`, `exact_duplicates` (full groups), `near_duplicates` (sample-only groups, each with a `distance` value per duplicate).

**Worth scrutinizing — a real, documented nuance in the near-duplicate report:** some `distance` values in `near_duplicates` exceed the `0.2` threshold (values like 5.19 were observed). This isn't a bug: FiftyOne's `neighbors_map` reports the distance to the nearest *surviving unique* neighbor via a separate post-hoc query — not necessarily the specific neighbor that originally caused the point to be flagged as a duplicate (which may itself have already been removed as someone else's duplicate). The *flagging itself* is threshold-correct; the *specific distance number shown* isn't guaranteed to be ≤0.2. If you're reading this report and see a large distance, don't assume the flag is wrong — read `docs/DECISIONS.md` DEC-062 for the full trace through FiftyOne's own source that confirmed this.

---

## Stage 5.7 — `scripts/curate/final_merge_curation.py`

**What it does:** the same mistakenness computation as Stage 5.5, run again — this time against `dataset/merged/` (the post-cap, post-merge pool) instead of each source's raw `dataset/processed/`. Catches whether the same under-annotation/hard-image patterns Stage 5.5 found still show up after merging (they do — see the top-ranked results in `final_merge_curation_report.json`).

**Relationship to `run_mistakenness.py`:** this script imports `CANONICAL_KEY_TO_NAME`, `COCO_CROSSWALK`, `ELIGIBLE_CANONICAL_IDS`, `NO_COCO_ANALOG`, and `yolo_to_fo_bbox` directly from `run_mistakenness.py` — same crosswalk, same 7 eligible classes, no separate copy of *that* logic. **Worth knowing:** the batched-inference loop itself (the `for i in range(0, len(paths), BATCH): ...` block) is a separate, hand-copied version of the same loop in `run_mistakenness.py`, not a shared function — the two have already drifted cosmetically (one iterates `COCO_CROSSWALK.values()`, the other `COCO_CROSSWALK.items()`, functionally identical but a real DRY gap flagged and deliberately left as a follow-up in `docs/DECISIONS.md` DEC-066, not fixed this session).

**Functions:**

| Function | What it does |
|---|---|
| `load_eligible_ground_truth(label_path, canonical_names)` | Same idea as `run_mistakenness.py`'s version, but reads directly from a merged-pool label path instead of looking up by `(source, filename)` — the merged pool's flat, source-prefixed naming makes this simpler. |
| `run(dry_run, limit)` | Scans every label file under `dataset/merged/labels/`, keeps images with at least one eligible-class box, builds an image index via `list_images()` (extension-filtered), runs the same batched YOLO inference + `compute_mistakenness()` pattern as Stage 5.5. |
| `main()` | argparse wrapper (`--dry-run`, `--limit N`), calls `run()`. |

**Outputs:** `dataset/reports/final_merge_curation_report.json` — same shape as `mistakenness_report.json`, plus `missing_images` (eligible labels with no matching image file — tracked, not silently dropped) and `eligible_labels_found` (so a `--dry-run` preview count and the real run's actual scored count can be compared and any gap explained).

**Worth scrutinizing:** since this reuses the same 7-class COCO crosswalk as Stage 5.5, it has the same blind spot — 9 canonical classes still can't be scored this way, for the same reason.

---

## Stage 5.8 — `scripts/build/split.py`

**What it does:** partitions `dataset/merged/` into `dataset/final/{train,val,test}/`, source-stratified (each of the 16 sources is split at the same ratio, independently, then pooled — so no single source can land entirely in one split by chance), with duplicate-aware grouping so a near/exact-duplicate pair from Stage 5.6 can't end up straddling train and val (or any other pair of splits).

**Constants:** `TRAIN_RATIO = 0.75`, `VAL_RATIO = 0.125`, `TEST_RATIO = 0.125` (the remainder, not applied independently — see `assign_splits()`), `SEED = 42`.

**Functions:**

| Function | What it does |
|---|---|
| `load_duplicate_groups(merged_image_count)` | Reads `dedup_report.json`. Returns `None` (duplicate-aware grouping skipped) if the report doesn't exist, or if its `images_checked` count is less than the current merged pool size (meaning the report is stale/partial). Otherwise returns every group from both `exact_duplicates` and `near_duplicates` as flat filename lists. |
| `assign_splits(filenames, duplicate_groups)` | The actual split logic. See walkthrough below — this is the one worth reading closely. |
| `run(dry_run)` | Orchestrates: lists merged images, loads duplicate groups, calls `assign_splits`, verifies the assignment is complete and non-overlapping, checks for cross-split leakage (and **raises** if any is found in a real run — see below), clears and rewrites `dataset/final/{split}/`, copies files, and computes per-split per-class/per-source distribution stats from the copied files themselves. |
| `main()` | argparse wrapper (`--dry-run`), calls `run()`. |

**`assign_splits()` walkthrough** (the core algorithm — union-find over duplicate groups, then a per-source stratified split):

1. Recover each filename's source from its `source__filename` prefix.
2. **Union-find over the duplicate groups**: `parent[f] = f` initially for every file; `find(x)` walks up parent pointers with path compression; `union(a, b)` links two files' groups together. For every duplicate group reported by `dedup_report.json`, every member gets `union`'d with the group's first member. This step exists specifically because a file can appear in *more than one* reported group (an exact-duplicate group and a separate near-duplicate group, since those two checks have different coverage) — a naive "last group wins" approach would silently break an earlier link when a later, overlapping group gets processed. Union-find merges all of these into one connected component per actual duplicate cluster, regardless of how many separate groups originally reported pieces of it.
3. `group_of = {f: find(f) for f in filenames}` — every file now maps to its cluster's representative.
4. Build `reps_by_source`: one representative per source, per cluster (so a whole cluster of duplicates counts as *one* unit when deciding train/val/test proportions, not N separate units).
5. For each source, independently: shuffle its representative list with `random.Random(SEED)`, then slice it into `train`/`val`/`test` by `round(n * ratio)` — the first `n_train` representatives (post-shuffle) go to train, the next `n_val` to val, the rest to test.
6. Every filename gets the split of its cluster's representative — so an entire duplicate cluster always lands in exactly one split.

**Outputs:** `dataset/final/{train,val,test}/{images,labels}/` (the actual files), `dataset/reports/split_report.json` (ratios used, per-split counts, per-split-per-class counts, per-split-per-source counts, and `cross_split_duplicate_leakage` — should always be `[]` in a successful run, since a non-empty leakage list now causes the script to `raise` before writing anything).

**Worth scrutinizing:**
- The `round(n * ratio)` split-size math can, in principle, round a small source's val/test allocation down to zero if it has very few duplicate-group representatives (≤7 or so). This doesn't currently happen — the smallest real source (`crowdhuman`, 128 images) still gets non-zero val/test — but it's a latent edge case if a much smaller source is ever added.
- Duplicate-aware grouping only covers what `dedup_report.json` actually checked — exact-duplicates cover the full pool, but near-duplicates only cover a 6,000-image sample (Stage 5.6). A near-duplicate pair outside that sample was never checked and could theoretically still straddle two splits; this is a known, documented scope limit, not a bug in `split.py` itself.
- The leakage check is a **hard gate** for a real run (`raise RuntimeError`), not just a printed warning — if you ever see this script fail with a leakage error, don't work around it by re-running; investigate `dataset/reports/split_report.json`'s `cross_split_duplicate_leakage` field first.

---

## Stage 5.9 — `scripts/build/generate_yaml.py`

**What it does:** writes `dataset/final/data.yaml`, the file `ultralytics` (the YOLOv8 training library) actually reads to find your dataset. Mechanical — every value it writes is already dictated by `config/classes.yaml`'s schema and `dataset/final/`'s real, on-disk split contents.

**Functions:**

| Function | What it does |
|---|---|
| `run(dry_run)` | Counts real files in each of `dataset/final/{train,val,test}/images/` and `.../labels/`. **Raises `FileNotFoundError` if any split is empty** and this isn't a dry run — refuses to write a yaml that points at nothing. Builds the `data_yaml` dict (see below) and writes it with a short explanatory header comment. |
| `main()` | argparse wrapper (`--dry-run`), calls `run()`. |

**The one subtlety actually worth understanding here:** `data_yaml` deliberately has **no `path` key**. This isn't an oversight — it's the fix for a real bug found during code review. `ultralytics`'s dataset-loading code (`check_det_dataset()`) resolves an *explicit* `path` value relative to whatever directory the training process happens to be running from (its CWD) — never relative to the yaml file's own location. Since training runs on RunPod, not this machine, and the invocation directory there isn't guaranteed to be `dataset/final/`, an explicit `path: "."` would have silently pointed `train`/`val`/`test` at the wrong place. Omitting `path` entirely lets `ultralytics` fall through to its own correct fallback: it resolves paths relative to the yaml file's own directory instead. This was verified directly against the real installed `ultralytics` package (not assumed) — see `docs/DECISIONS.md` DEC-066 for the exact verification steps, including calling `check_det_dataset()` from an unrelated directory to confirm it still resolves correctly.

**Outputs:** `dataset/final/data.yaml` — `train`/`val`/`test` (relative paths), `nc` (16), `names` (id → name dict, matching `config/classes.yaml`'s `names:` field order exactly).

**Worth scrutinizing:** this script's `nc`/`names` come from `config/classes.yaml`'s `names:` field specifically — that file *also* has a separate `classes:` block with per-class metadata (native_class, cap, etc.) that happens to be ordered differently (grouped by category, not by numeric id). If you ever edit `classes.yaml`, make sure you're checking `names:` when verifying id order, not `classes:` — this exact confusion nearly caused a false alarm during this session's own verification pass.

---

## If you want to re-run any of this

Every script supports `--dry-run` (prints what it would do without writing anything) and the ML-heavy ones (`dedup.py`, `run_mistakenness.py`, `final_merge_curation.py`) support `--limit N` for a fast smoke test on a subset. **One caveat proven the hard way this session:** `--limit` still writes to the *real* report path — there is no separate smoke-test output location. Running `--limit 50` against a machine that already has a real full-scale report will silently overwrite it with the smoke-test result. If you want to test a code change safely, either check the report's own count field afterward (e.g. `images_checked`) before trusting it, or re-run at full scale immediately after testing.

# RunPod Dedup Plan — Union-Coverage for 4500-now / 9000-later

- **Date approved:** 2026-08-21
- **Status:** Approved, implementation in progress. Not yet run for real (no
  RunPod execution, no real `cap_per_class.py --hard-cap 9000` run, no fresh
  `--hard-cap 4500` run) as of this writing.
- **Related:** DEC-062 (dedup.py's original design), DEC-068/069 (`--hard-cap`
  presets, `INSTANCE_TARGET` scaling decision), DEC-072 (killed local
  full-scale attempt), DEC-086/087 (`audit_status` gating, priority-source
  reordering — the config this plan's nesting measurement was taken against),
  DEC-088 (`hide_duplicates` notebook toggle this plan's phase 2 relies on).

This document is the verbatim plan approved via Claude Code's plan mode. It's
saved here (not just left in `~/.claude/plans/`, which is local, ephemeral
Claude Code state outside the repo) so it's part of the project's permanent
record. Once executed for real, a proper DEC entry should be added to
`docs/DECISIONS.md` per the Documentation section below, and this file's
Status line updated.

---

## Context

The pipeline order is `cap_per_class.py` (decides which images per class, up to a
`--hard-cap` ceiling) → `merge.py` (physically builds `dataset/merged/`) → `dedup.py`
(GPU-heavy exact+near-duplicate detection over the whole merged pool) → `split.py`
(duplicate-aware train/val/test split). Dedup is expensive — a full local run was
killed after ~9 hours (DEC-072) — so the plan is to rent a RunPod GPU instance for it.

The student is training at hard_cap=4500 today but sees a real chance they'll need
hard_cap=9000 later, once training shows more data is needed. Their goal: pay for
RunPod GPU time once, and have that one dedup result stay valid for both cap values —
never re-run GPU dedup on RunPod again just because the cap changed.

**Why this isn't just "run dedup at 9000, done":** `dedup_report.json` is a snapshot of
whatever pool it was computed against (bare filenames, no other pool-identity marker).
`merge.py` fully rebuilds `dataset/merged/` from scratch every run — there's no
incremental mode. If the true cap=4500 selection ever contains even one image that
wasn't part of whatever pool got deduped, that image was never checked, silently.

**What the investigation found:** whether cap=4500's selection is a strict subset of
cap=9000's selection depends on `cap_per_class.py`'s internal budget math — specifically
whether `INSTANCE_TARGET` (fixed at 10,000, deliberately *not* scaled by `--hard-cap`,
per DEC-068) causes priority sources to starve the general pool differently at
different caps. A Plan agent actually replicated `cap_class()`'s logic in-process
against the live working tree rather than reasoning about it abstractly. Result: **under
today's config (post-DEC-087), 4500 is a strict subset of 9000** — every class either
has an empty general pool at both caps, or has priority-source instance consumption
that doesn't change with hard_cap. But this is a property of the current
`CLASS_PRIORITY_SOURCES` config, not a structural guarantee — a stale pre-DEC-087
report already showed a 17-image violation for Person before today's reordering
happened to close it, and the same agent showed a concrete one-config-edit-away
mechanism that would reopen it.

**Decision: build the coverage guarantee anyway, via a union of the two real cap
selections, not just the raw 9000 selection.** Today this costs *nothing* extra — the
union is byte-identical to the 9000 selection. It's insurance against exactly the
config drift demonstrated above, computed and proven fresh every time rather than
assumed.

**Bonus finding, unrelated to 9000 but must be fixed first:** while tracing this,
`merge.py`'s exclusion mechanism (`load_excluded_pairs()`) turned out to be a total
no-op due to a `.txt`-extension mismatch against `cap_report.json`'s extension-less
entries — verified directly against the real `cv_project_hovyc_excluded.json` (71
real, hand-reviewed exclusions, confirmed 0 matches against the current cap report's
key format). This has apparently never worked. It must be fixed before any of the
real cap/merge runs below.

**A real tension the first draft of this plan got wrong, now resolved:** the
student's actual workflow does a *second* round of visual review after dedup,
specifically so duplicate images don't waste review time — and that review can
promote corrected/added labels, which can (rarely, but demonstrated: DEC-087's
`cv_project_hovyc`) shift what `cap_per_class.py` selects. A blanket "nothing may
change after dedup" rule directly conflicts with that established, sensible order.
The resolution (detailed in the sequence below): do that review while the *deduped
union pool* is still the live `dataset/merged/`, and afterward mechanically intersect
whatever `cap_per_class.py` selects against what was actually deduped — so coverage
is guaranteed by construction, not by a fragile promise not to touch anything.

**First action once this plan is approved, before step 1 below:** save this plan
verbatim to `docs/RUNPOD_DEDUP_PLAN.md`. The file at `~/.claude/plans/` is ephemeral
Claude Code state, not part of the repo — the student asked for this documented
properly.

---

## Prerequisite fixes (land before step 1 — all small, non-GPU, no RunPod involved)

### Fix A — `merge.py`: exclusion matching is a no-op
`scripts/build/merge.py`'s `load_excluded_pairs()` reads `excluded_filenames` entries
with their `.txt` extension intact, but `load_selected_union()`'s entries are
extension-less stems (`label_path.stem`). The set difference between them never
matches. Fix: strip the extension when building the excluded set —
`excluded.add((source, Path(filename).stem))`. No notebook change needed; the
notebook's own internal representation is self-consistent, this is purely a
consumption-side bug in `merge.py`.

### Fix B — `split.py`: pool-count mismatch has zero margin at the 9000 cutover
`split.py` counts the live merged pool via raw `img_dir.iterdir() if p.is_file()`;
`dedup.py` counts via `list_images(..., recursive=False)` (extension-filtered). They
normally agree, but a single stray non-image file (macOS `.DS_Store` already exists
elsewhere under `dataset/` — Finder has visited this tree) would desync them. This
matters more than it sounds: at the future 9000-cutover step in this plan, the
dedup report's `images_checked` and the live pool count can be numerically **equal**,
so the staleness gate (`images_checked < merged_image_count`) has *zero margin* — one
stray file flips it and silently disables duplicate-aware splitting with no error.
Fix: make `split.py` use the same `list_images()` helper dedup.py uses.

### Fix C — `split.py`: coverage note will misdescribe a full-scale run
`split.py`'s `duplicate_group_coverage_note` unconditionally states near-duplicate
groups only cover a stratified sample. After this plan's `--full-scale` run, that's
false and would be baked into `split_report.json` as a wrong claim. Fix: gate the
note on `near_duplicate_sample_size >= images_checked` (full-scale sets sample size
equal to the full pool).

### Fix D — `split.py`: make the superset-reuse case observable, not silent
The plan deliberately relies on `split.py` tolerating a `dedup_report.json` that
describes more images than the currently-live pool (verified safe: the union-find
code filters absent filenames via `present = [f for f in group if f in source_of]`,
no crash) — but right now this is silent, and the existing success-path print would
read as nonsense (e.g. "covers the full pool (65379/50032 images)"). Add an explicit
note when `images_checked > merged_image_count`, and fix the coverage print to handle
this case sensibly.

### Fix E — `dedup.py`: `--limit` must not overwrite the canonical report
Currently `--limit N` still writes to the real `dataset/reports/dedup_report.json`
path — the same footgun `cap_per_class.py` already solved for its own hardcap presets
(DEC-068: non-default presets write to `cap_report_hardcap<N>.json`, never clobbering
the canonical file). Apply the same pattern: `--limit` writes to
`dedup_report_limit<N>.json` instead. This matters specifically because the RunPod
smoke-test steps below use `--limit` repeatedly on a box that will otherwise have no
record of which report is "real."

### Fix F — `dedup.py`: expose `--num-workers`, don't hand-edit
`EMBEDDING_NUM_WORKERS = 4` is a hardcoded module constant, tuned specifically to fix
an MPS-oversubscription problem on the student's Mac — CUDA-irrelevant, and the value
needs to be swept during the RunPod smoke test anyway. Add
`--num-workers INT` (default: current constant), thread it into both
`compute_near_duplicates(num_workers=...)` and `compute_exact_duplicates(dataset,
num_workers=...)` (the latter currently isn't passed `num_workers` at all and
defaults to 1 — a free fix for several GB of filehashing). Record the value used in
the report output (`"embedding_num_workers": num_workers`) for reproducibility.

---

## New capability: `merge.py --cap-report PATH`

`merge.py` currently only reads the hardcoded `dataset/reports/cap_report.json`, with
no override. Add:

```python
parser.add_argument(
    "--cap-report", type=Path, default=None, metavar="PATH",
    help="Alternate cap report to merge (default: dataset/reports/cap_report.json). "
         "Must have cap_per_class.py's {'classes': {<class>: {'selected': [...]}}} shape.",
)
```

- `run(dry_run: bool = False, cap_report_path: Path | None = None)`; resolves to
  `cap_report_path or reports_dir() / "cap_report.json"`. Omitting the flag reproduces
  today's behavior exactly (backward compatible).
- Print the resolved absolute path unconditionally — this script will be pointed at
  several different reports over the course of this plan, and `merge_report.json`
  currently gives no clue which one produced the pool on disk. Add
  `"cap_report_path"`, `"cap_report_hard_cap_preset"`, `"cap_report_union_kind"` to
  `merge_report.json`'s output.
- Hard failures: report file missing, invalid JSON, missing `"classes"` key, any
  `selected` entry without a `/` separator (currently would raise an opaque
  `ValueError` from the bare `.split("/", 1)`).
- Do **not** add an incremental/`--no-clean` mode. The existing `rmtree`-then-rebuild
  behavior is what keeps "which pool is currently on disk" unambiguous across the
  several merges this plan does — keep it destructive.

---

## New script: `scripts/preprocess/combine_cap_reports.py`

Sits next to `cap_per_class.py`, its only sibling producing cap reports. Does set
operations on 2+ cap reports' per-class `selected` lists — **union**, to build the
pool that gets deduped, and **intersect**, to safely cut a post-review selection back
down to only images that were actually deduped (see the sequence below for why both
are needed).

```
usage: combine_cap_reports.py --op {union,intersect} --inputs PATH [PATH ...]
                               [--output PATH] [--force] [--dry-run]

--op       required. "union" (for building the dedup target pool) or "intersect"
           (for safely cutting a fresh selection back to dedup-covered images only)
--inputs   nargs="+", required, 2+ cap-report JSON paths
--output   default: dataset/reports/cap_report_combined.json
--force    allow overwriting an existing --output
--dry-run  print the diff, write nothing
```

**Hard failures:** fewer than 2 inputs; any input missing/unreadable/lacking
`"classes"`; inputs disagree on `seed` or `instance_target` (different code state);
inputs' class-key sets differ (different `classes.yaml`); `--output` resolves to
`cap_report.json` or any `cap_report_hardcap*.json` — refuse **always, even with
`--force`** (mechanically enforces the existing DEC-068/DEC-066 "never clobber the
canonical report" norm); `--output` exists without `--force`.

**Warnings, not failures:** union's source set doesn't match currently-active sources
(catches a source getting benched mid-process); for `--op intersect` specifically,
**always print the per-class dropped-image count** — this is the visible signal that
review revealed something the dedup run didn't cover. Not an error; a number worth
reading and deciding about.

**Validation scope — provenance only, not a re-scan.** The realistic failure mode is
a config/code edit slipping in between runs — the seed/instance_target/class-set
checks catch that cheaply. Do not re-scan `dataset/processed/` to verify every stem
still exists (slow, and `merge.py` already reports `missing_images`/`missing_labels`).

**Output shape** (same `{"classes": {...}}` shape `merge.py` already reads, plus
provenance so "this pool is fully covered by that dedup report" is a checkable claim):

```json
{
  "classes": {"<name>": {"selected": [...], "final_images": N}},
  "combine_op": "union" | "intersect",
  "hard_cap_preset": null,
  "seed": 42, "instance_target": 10000,
  "inputs": [{"path", "sha256", "mtime_iso", "hard_cap_preset", "selected_total"}],
  "coverage": {"result_total": N, "pairwise": {...}, "dropped_by_class": {...}},
  "git_commit": "<rev-parse HEAD>", "git_dirty": bool,
  "generated_at": "<iso8601>"
}
```

`hard_cap_preset: null` is deliberate — nothing downstream should mistake this for a
real preset run.

---

## Execution sequence

**Phase 1 — build and dedup the union pool.**

1. Land prerequisite fixes A-F.
2. `cap_per_class.py --hard-cap 4500` → canonical `dataset/reports/cap_report.json`.
   Needed regardless — the on-disk one predates today's DEC-087 priority reordering.
3. `cap_per_class.py --hard-cap 9000` → `dataset/reports/cap_report_hardcap9000.json`.
   First-ever real run at this preset (previously `--dry-run`-only per DEC-068/069).
4. `combine_cap_reports.py --op union --inputs dataset/reports/cap_report.json
   dataset/reports/cap_report_hardcap9000.json --output
   dataset/reports/cap_report_dedup_union.json`.
5. `merge.py --cap-report dataset/reports/cap_report_dedup_union.json` → builds
   `dataset/merged/` as the union pool. Verify `merged_count` matches the union
   report's total; investigate any `missing_images`/`missing_labels`.
6. **RunPod**: upload the union pool, smoke-test, run
   `dedup.py --full-scale --num-workers <N>` (see below) → authoritative
   `dedup_report.json`. Download it back (+ optionally embeddings). Tear the pod down
   — nothing past this point needs a GPU again for this cap decision.

**Phase 2 — post-dedup visual review, against the still-live union pool.**

7. Locally, using `fiftyone_review_processed.ipynb`: for whichever sources still need
   the second review pass, set `restrict_to_merged=True` (scoped to whatever's
   currently in `dataset/merged/` — at this point, the union pool, the broadest
   relevant scope for either cap value) and `hide_duplicates=True` (now usable —
   `dedup_report.json`'s `images_checked` matches the current, still-union, merged
   pool count exactly). Review normally: label corrections/additions go to
   `labels_reviewed/` and get promoted to `labels/` as usual; new exclusions go to
   `*_excluded.json` as usual. **Do not re-run `cap_per_class.py` or `merge.py`
   during this phase** — that would replace the live union pool before review scope
   needs it gone, and `restrict_to_merged`/`hide_duplicates` need it to stay put.

**Phase 3 — cut back to the real training pool, coverage guaranteed by construction.**

8. Re-run `cap_per_class.py --hard-cap 4500` (and `--hard-cap 9000`, cheap to do now
   for symmetry even if 9000 isn't being adopted yet) fresh, picking up any label
   changes from phase 2.
9. `combine_cap_reports.py --op intersect --inputs dataset/reports/cap_report.json
   dataset/reports/cap_report_dedup_union.json --output
   dataset/reports/cap_report_4500_covered.json`. This intersects the *fresh*
   post-review 4500 selection against the *original* deduped union — every image in
   the output was both (a) actually selected by the current selection logic and (b)
   actually checked by the RunPod dedup run. **Read the printed dropped-count.** Zero
   is the expected, common case (most review work is bbox corrections that don't
   shift `cap_per_class.py`'s output at all). A nonzero count means review revealed
   real new content that the dedup run never saw — worth a note in the DEC entry
   either way; if the count is large enough to matter, that's the trigger for
   deciding whether a small supplementary dedup pass is worth it, not something to
   silently absorb.
10. `merge.py --cap-report dataset/reports/cap_report_4500_covered.json` → the real
    training pool: true 4500 selection, review corrections included, dedup-coverage
    guaranteed.
11. `final_merge_curation.py` (Stage 5.7, report-only, unchanged — fits here
    naturally, right before commit-to-split, matching the student's original stated
    order).
12. `split.py` → final pool + `dedup_report.json`. Confirm `split_report.json` shows
    `duplicate_aware: true` and (post Fix-C) doesn't claim sampling.
13. **Later, if 9000 is adopted for real:** repeat 9 with
    `cap_report_hardcap9000.json` in place of `cap_report.json` →
    `cap_report_9000_covered.json`, then `merge.py --cap-report
    cap_report_9000_covered.json`, then `split.py` again. No RunPod re-run. (This is
    the step Fix B protects — check the live pool for stray non-image files before
    trusting the exact-equality case there.)

---

## RunPod execution specifics

**FiftyOne on the pod is a library dependency, not the visual app.** `dedup.py` calls
`fiftyone.brain.compute_exact_duplicates()` / `compute_near_duplicates()` directly —
that's the Stage-5.6 design (DEC-062: use FiftyOne Brain's own built-in
duplicate-detection algorithms rather than a custom implementation). Those functions
need a `fo.Dataset` object, which needs FiftyOne's backing database running, purely
to hold data in memory — an architectural property of the library, unrelated to any
UI. `dedup.py` never calls `fo.launch_app()`. **No visual inspection happens on
RunPod, and none is needed to.** All review (including phase 2 above) happens
locally, in the existing notebook, using only the small `dedup_report.json` brought
back. The pod is up only for the embedding computation and torn down right after.

**Cost is dominated by upload, not GPU time.** A measured extrapolation of the
near-duplicate search at full union-pool scale (~65k points) puts the search itself
under a minute; the whole run is plausibly 15-30 minutes of wall clock. Spend
planning effort on the transfer and environment, not on GPU tier.

**Upload out:**
- `dataset/merged/images/` only (not `labels/`, not `dataset/processed/`) — tens of
  thousands of files, several GB. Stream, don't gzip (JPEGs don't compress):
  `tar -cf - dataset/merged/images | ssh <pod> 'tar -xf - -C /workspace/second-vision-ai'`,
  or `runpodctl send`. Start this first, it's the long pole.
- Repo tree minus `dataset/`: `AGENTS.md` **must** be present —
  `get_repo_root()` requires both `config/` and `AGENTS.md` to resolve paths, and a
  partial upload fails with a confusing `RuntimeError` otherwise. Full `scripts/`
  tree (no `__init__.py` files anywhere — namespace packages, directory structure
  itself matters), `config/`, `requirements.txt`.
- Don't `pip install -r requirements.txt` on the pod — it pulls roboflow/ultralytics/
  opencv and can downgrade the pod's CUDA-matched torch build. Install `fiftyone`
  only, then assert `torch.cuda.is_available()` before running anything. FiftyOne
  needs its bundled MongoDB (may need `fiftyone-db-ubuntu2204` / a writable
  `FIFTYONE_DATABASE_DIR` on a minimal container) and outbound internet for the zoo
  model download.
- Pod sizing: prioritize **≥8 vCPU / ≥32 GB RAM / ≥60 GB disk over GPU class** — the
  bottlenecks are JPEG decode in the DataLoader and the CPU-side duplicate search,
  not the embedding forward pass. A cheaper GPU with more vCPUs beats a flagship GPU
  with few.

**Smoke-test sequence (so a partial report can never be mistaken for the real one):**
1. `dedup.py --dry-run` — writes nothing, prints the pool count. Cross-check against
   `find dataset/merged/images -type f | wc -l` on the pod — catches a silently
   truncated upload before spending any GPU time.
2. `--limit 500 --num-workers 8` — proves CUDA + MongoDB + zoo-model download work,
   gives a first throughput number. (Post Fix-E, this writes to
   `dedup_report_limit500.json`, not the canonical path.)
3. Sweep `--limit 2000` across a couple of `--num-workers` values to pick one.
4. Real run: `dedup.py --full-scale --num-workers <best>`, under `tmux`, log to file.
5. Before downloading, confirm both `images_checked` and `near_duplicate_sample_size`
   equal the full union-pool count — either being smaller means a smoke-test
   artifact, not the real run.

**Use `--full-scale`.** The 6,000-image near-dup sample was a measured MPS
workaround, not a methodological choice (dedup.py's own docstring says so), and
renting a GPU while keeping that workaround wastes the rental. More importantly: a
~9% sample finds a small fraction of cross-pool near-duplicate pairs — for the
leakage-prevention purpose `split.py` uses this report for, a sampled check is not
much protection. Exact-duplicate detection was already always full-pool (cheap
filehash); this closes the other half of the gap that's been open since DEC-062.

**Download back:** just `dataset/reports/dedup_report.json` (a few MB). Optionally
also export embeddings (`compute_near_duplicates(embeddings="<field>")` stores them;
pull via `dataset.values(...)`, ~150-200 MB at this scale) — cheap to add now, and
since dedup.py's own docstring already flags the 0.2 threshold as possibly
miscalibrated for this embedding space, having embeddings on hand means re-tuning it
later never needs a second RunPod trip.

---

## Verification

- After step 4: union report's coverage numbers are sane (compare against the
  in-process measurement this plan was designed around: ~50k for 4500, ~65k for
  9000, 0 dropped from 4500→9000 today).
- After step 5: `merged_count` in `merge_report.json` equals the union report's
  total, `missing_images`/`missing_labels` empty or explainable.
- After step 6: `dedup_report.json`'s `images_checked` matches the union pool size
  exactly; `embedding_num_workers` recorded; confirmed not a `--limit` artifact.
- After step 7: spot check that `hide_duplicates` actually suppressed images in the
  App (the skip-count print should be nonzero if any duplicates existed for that
  source).
- After step 9: the printed dropped-count is understood and, if nonzero, written
  down — not silently ignored either way.
- After step 12: `split_report.json` shows `duplicate_aware: true`; the Fix-D note
  appears in the run log if the pool size differs from `images_checked` (confirms
  the superset-reuse path executed, not a silent skip).
- Step 13, whenever it happens: same checks, watching specifically that Fix B is in
  place before trusting the exact-equality case at that cutover.

## Documentation

Once run for real, log a DEC entry covering: the union-coverage design and why (with
the measured nesting-holds-today-but-is-fragile finding stated plainly, not
overstated as a live bug); Fixes A-F with the real numbers each one changed (e.g. the
71 previously-non-excluded `cv_project_hovyc` images); the phase-2/3 resolution for
ongoing review work and any nonzero drop-count from step 9; the RunPod run's actual
throughput/cost/instance type for future reference; the `--full-scale` decision and
its consequence for `split_report.json`'s coverage claims. Update `TASKS.md`'s table
to match, same pattern as every other DEC entry this session.

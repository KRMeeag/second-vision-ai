# OPEN_QUESTIONS.md — Second Vision AI

> **Last updated:** 2026-08-26
>
> **Concurrent-session note (2026-08-20):** this repo has a deliberate split-agent setup (see the architectural handoff doc), and two sessions editing `docs/DECISIONS.md` concurrently produced a real DEC-number collision this session (see DEC-083's Consequences) — a live example, not a hypothetical. If you're picking this file up fresh, don't assume you have the complete picture; check `docs/DECISIONS.md`'s actual latest entry and `git log` before trusting this file's own "newly surfaced" section as exhaustive.
>
> Every question here blocks something specific — either a script that can't be written correctly without the answer, or a stage that can't be marked done. When a question gets answered, record the answer as a new entry in `docs/DECISIONS.md` (it's a real decision) and delete or mark the entry here as resolved. Don't let this file and `DECISIONS.md` disagree — `DECISIONS.md` is the one that future sessions trust.
>
> This list reflects what's known as of Stage 5.1/5.2's completion (all 5 sources acquired and converted — see DEC-056). If an autonomous agent session works through Stage 5.3–5.9 before this is next read, it will likely append its own findings here — check the bottom of the file for a "newly surfaced" section before assuming this is the complete list.

---

## Data-scope questions

### 1. Trash Bins — reopen the secondary-source search? (DEC-039, DEC-051, DEC-058, DEC-069)
**Partially resolved 2026-08-18 (DEC-069)**: student supplied 2 candidate Roboflow projects; checked both via SDK before acting on either.
- `eyecue/trashcan-detection-pihfn` — clean, added. Single class `Trashbin`, no ambiguity. Real result after a full `cap_per_class.py`+`merge.py` rerun: Trash Bins grew from **1,104 → 1,663 images**.
- `ronits-workspace-e52mh/nyb` — real concerns, deliberately NOT added: its class list (`bin_elevated`/`bin_caged`/`bin_ground` alongside `z_action`/`z_no_action`/`trap_object`) looks like it's from an unrelated pest/wildlife camera-trap context, and versions are non-monotonic in size. Needs the student to look at real images on the project page before deciding — see `config/datasets.yaml`'s comment for the full writeup.
- License could not be verified for either (no SDK field, `WebFetch` 403 on both project pages) — student should confirm the license shown on each page directly before treating either as final.

**New finding, changes the remaining decision**: with Trash Bins healthier, **Doors (1,337 images) is now the actual ratio-invariant minimum**, not Trash Bins. Ratio improved from 4.08 to **3.37**, still above 3:1. DEC-042's recompute rule (`3 × min(realized_class_images)`) would now mean **4,011** (not the earlier 3,312), keyed on Doors.

**Likely resolved 2026-08-18 (DEC-071, student-sourced)**: found and added a second Doors source, `nathaly-espinoza/door-detection-zqt59` — SDK-checked first, real classes are `door`(1908)/`knob`(376)/`hinged`(1739)/`lever`(1239), pulled+converted with `native_class_filter: "door"` only → **4,493 images, 4,888 boxes**, sitting in `dataset/processed/roboflow_door_detection_zqt59/`. Not yet merged (pending the cap_per_class.py→merge.py rerun, deliberately deferred — see item #7). Once merged, Doors' candidate pool grows from 1,337 to potentially ~5,800 images, which should resolve the ratio-invariant bottleneck without needing (b) or (c) below — **but not yet confirmed with real post-merge numbers**.

**Fully resolved 2026-08-18 (DEC-077)** — real `cap_per_class.py` + `merge.py` rerun confirmed Doors did NOT stay short: candidates jumped 1,337 → 5,830, selection hit the hard cap at 4,500 images. **Ratio invariant now met**: max=Vehicle(4,500) / min=Trash Bins(1,663) = **2.71** (≤ 3.0) — neither fallback ((a) accept ratio, (b) recompute per DEC-042) was needed.

**Resolved 2026-08-18 (DEC-073), all three sub-items:**
- (i) `ronits-workspace-e52mh/nyb` — student decided to drop it, not worth pursuing. Trash Bins is no longer the bottleneck and the mixed pest/wildlife-camera-trap class list was reason enough on its own. Comment in `config/datasets.yaml` updated to record the closure.
- (ii) `door_detection_zqt59`'s `hinged` class — student confirmed it's door-hinge hardware, not an additional door-type label. Correctly dropped by the existing `native_class_filter: "door"`; every hinged-only instance's image still has a real `door` box, so no images were lost, only the hinge boxes. No code or re-pull needed.
- (iii) License — student confirmed both `eyecue/trashcan-detection-pihfn` and `door_detection_zqt59` are **CC BY 4.0**. Recorded as a `license:` field on both entries in `config/datasets.yaml`.

### 2. traffico_y1 — formally close it out?
**Resolved 2026-08-18 (DEC-069).** Student confirmed: closed. `audit_status: pending` → `benched` in `config/datasets.yaml`, same pattern as `jeep_hozhs`/DEC-037. Nothing further to do.

---

## Stage 5.3 (Box Audit) — items already queued in `docs/PLAN.md`

### 3. elevator_status_s4lrk — isolate real detection boxes from classification-style ones
**Heuristic built and run 2026-08-13 (DEC-057)** — `scripts/preprocess/box_audit.py`, flagged list at `dataset/reports/elevator_status_s4lrk_flagged.json` (844 boxes).

Correction to this entry's original framing: checked the real area-fraction distribution before picking a threshold, and there's no "near-full-frame" cluster to separate out — max area fraction across all 6,786 real boxes is 0.595, with no bimodal gap anywhere. A fixed area threshold would have flagged nothing. The actual defect is **shape**, not size: 842 of 844 flagged boxes are extreme-elongation outliers (several are literal hairline slivers along a frame edge, e.g. w=0.61, h=0.00003) — this matches DEC-031's original description of this source ("boxes cling to object shape rather than a clean axis-aligned rectangle") much better than the "state pseudo-label" framing did. `roboflow_stairs_i2yia` shows the same signature (232/233 flagged boxes are shape outliers) — worth reviewing alongside elevator.

**Superseded 2026-08-20 (DEC-082).** The premise of this item ("isolate the usable subset") turned out too optimistic — direct visual review found the defect is worse than shape/elongation, and worse than the flagged-outlier rate suggested (Tukey's fences only catch statistical outliers *within a source's own distribution*, so a source that's uniformly bad across most images produces few flags, by construction). `elevator_status_s4lrk` and `stairs_i2yia` are both now `audit_status: failed`, fully excluded rather than partially salvaged. No "usable subset" extraction needed — moot.

**Still needs you**: the flagged list is real and grounded, but deciding what to do with each flagged box (relabel, drop, or it's actually fine) is visual review — same as before, just with the correct heuristic behind it now.

**How, concretely (added 2026-08-18)**: open `notebooks/fiftyone_review_processed.ipynb`, set `source_key = "roboflow_elevator_status_s4lrk"` and `flagged_report_path` to `dataset/reports/elevator_status_s4lrk_flagged.json` (both already the notebook's defaults). It loads only the 750 images with a flagged box, and marks the specific flagged detection(s) — not just the image — so you can tell which box in a multi-box image is the problem one. Same notebook, same mechanism works for `roboflow_stairs_i2yia` too (the other source `box_audit.py` flagged with the same shape-defect signature) — swap `source_key`/`flagged_report_path` accordingly.

### 4. Near-exhaustive review for smaller Roboflow pools (D-018r)
Not a script gap — this is genuinely your visual-review time. Tooling confirmed ready 2026-08-18: `notebooks/fiftyone_review_processed.ipynb` browses `dataset/processed/<source>/`, `dataset/merged/`, and `dataset/final/<split>/` (set `source_key` accordingly, `flagged_report_path=None` for a full-pool pass) — use whichever stage you actually want to eyeball.

**Review approach resolved 2026-08-18 (DEC-076)**, after checking the real scale (30,072 images / 41,793 instances across the 12 currently-merged small Roboflow pools — cap only cut this group ~23%, nowhere near crowdhuman's 96%): every pool gets a full pass, no source skipped, but **pace varies by a risk ranking** (`box_audit.py --pool merged`'s per-source flagged-box rate, cross-checked against DEC-031's Stairs/Elevator box-shape finding) — slow/thorough on the highest-flag-rate and DEC-031-known-issue sources, faster skim on the rest. No hard cutoff between tiers. Full ranked table in DEC-076. `door_detection_zqt59` needs this ranking added once it merges (not yet in `dataset/merged/`).

**Status: method decided, not yet executed** — `docs/PLAN.md`'s Stage 5.3 row for this stays "Todo" until the student has actually gone through all 12/13 pools.

---

## Tooling decisions

### 5. CVAT or Label Studio for Stage 5.5's correction loop?
Still an open checkbox in `TASKS.md` Phase 1 ("Verify CVAT/Label Studio integration via FiftyOne"). This **blocks `scripts/curate/reimport_corrections.py`** specifically — that script's whole job is parsing one tool's export format back into the pipeline, so it can't be written correctly without knowing which tool. (`run_mistakenness.py`, the other half of Stage 5.5, doesn't need this — computing mistakenness scores is tool-agnostic, and was built and run this session — DEC-060.)

**Confirmed genuinely blocked, not built 2026-08-13**: checked before skipping, not just deferred on sight — `reimport_corrections.py`'s entire job is parsing one specific tool's export format (CVAT XML vs. Label Studio JSON are structurally different), so writing it against a guessed tool would mean either guessing wrong and rewriting later, or building something tool-agnostic-in-name-only that doesn't actually parse anything real. No new information surfaced this session that would resolve this — still needs the student's actual tool choice.

**Resolved 2026-08-18** — confirmed FiftyOne's own built-in in-App annotation, not CVAT/Label Studio. Verified via `docs.voxel51.com`: it's a stable, native App feature (not a plugin — checked specifically, since the student separately heard "FiftyOne annotation plugin" from another source and asked whether it's the same thing. It isn't: `voxel51/fiftyone-plugins`' actual `annotation` plugin is a UI wrapper around `dataset.annotate()`, i.e. orchestrating the CVAT/Label-Studio external-backend round-trip — the opposite of what's built here) that fully supports creating/editing/deleting bounding boxes directly in the App, plus per-*label* tagging (used by DEC-079's `accept` mechanism) and per-*sample* tagging (DEC-078's `exclude` mechanism). `reimport_corrections.py` is **not needed as originally scoped** — corrections happen in-place on the FiftyOne dataset; `notebooks/fiftyone_review_processed.ipynb`'s write-back cells (DEC-072/074/078/079) are the simpler replacement this entry anticipated.

---

## Algorithm/policy specifics not yet pinned down

### 6. `cap_per_class.py`'s trim method when sources overlap the cap
**Revised with the student 2026-08-15 (DEC-067)**, superseding DEC-058's plain-random default for the classes below. `scripts/preprocess/cap_per_class.py` now supports per-class ordered "priority sources" (`CLASS_PRIORITY_SOURCES`) — every class not listed still uses the original DEC-058 behavior (ExDark floor, then pure random pooling among everything else). Three classes got real overrides after going through `config/classes.yaml`/`datasets.yaml` source-by-source with the student:

- **Person**: `["exdark", "crowdhuman"]` — crowdhuman (documented `role: volume_topup`) was being crowded out to 128/19,370 candidates by ExDark's instance-heavy floor; now gets a dedicated 2,500-instance sub-budget (`PRIORITY_SOURCE_INSTANCE_SUBBUDGET`), filled least-dense-image-first to avoid CrowdHuman's extreme-crowd outliers (some images have 300+ people) blowing the instance count — real result: 729 crowdhuman images, Person's total instances landed at 10,007 (essentially back at the original ~10,000 target, not the 49,664 an uncapped floor produced).
- **Vehicle**: `["exdark", "roboflow_me5_u6rvg"]` — me5_u6rvg is a documented edge-case source (Jeepney/Tricycle/Ambulance, "Philippine-context relevance" per `datasets.yaml`). Real result: 3,188 images (up from 1,649 under random pooling).
- **Elevator**: `["roboflow_elevator_awvus"]` — favors the source with no Stage 5.3 flagged box-shape defects over the larger, defect-flagged `elevator_status_s4lrk`. Real result: all 1,777 awvus candidates included.

**Explicitly reviewed and left as pure random pooling** (no edge-case source found in either project's documentation): Pole (`pole_detection_z76mb` vs `utility_poles_44tzx`), Stairs (`escalator_stairs` vs `stairs_i2yia`).

**Still not decided, still using the default**: whether audit-flag status from `box_audit_report.json` should feed into the *general* (non-priority) random fill for any class, not just source-level priority. Also still not applied: DEC-042's "6,000 instance target for small/hard classes" refinement — no config defines which classes qualify.

**Not yet cascaded**: `dataset/reports/cap_report.json` reflects this new logic (run for real 2026-08-15), but `dataset/merged/` and everything downstream (`dedup_report.json`, `final_merge_curation_report.json`, `dataset/final/`) still reflect the *previous* cap decision — re-running `merge.py` onward is pending the student's go-ahead.

**Update 2026-08-17 (DEC-068)**: `cap_per_class.py` now takes `--hard-cap {1500,4500,9000}` (default 4500, real run confirmed a no-op vs. DEC-067's numbers). `FLOOR` is derived from it (`hard_cap // 3`) instead of being fixed. Two things this made concrete but did NOT resolve, still genuinely open:

- **Should `INSTANCE_TARGET` (10,000) scale with the hard_cap preset?** Student's call (2026-08-18): **keep it fixed at 10,000 for now.** Asked for literature that might argue against a fixed instance target while image count grows — researched honestly, no paper says that exact thing, but three real sources are relevant: Gupta/Dollár/Girshick's original LVIS paper (CVPR 2019) chose *image*-level Repeat Factor Sampling specifically because raw instance counts alone are a noisy balancing signal; Chang et al. (ICML 2021, arXiv:2104.05702) found *both* image-level and object-level control are necessary, neither alone is sufficient — which this project's design already does (hard_cap bounds images, INSTANCE_TARGET bounds instances); and this project's own DEC-042 anchor paper (arXiv:2403.07113) found sampling/reweighting rebalancing that helps two-stage detectors does **not** reliably transfer to one-stage detectors like YOLOv5/YOLOv8 — suggesting further hand-tuning instance-sampling knobs may have limited payoff for this model family specifically, and leaning on Ultralytics' built-in mosaic/mixup augmentation may matter more.
- **How should a shrunken hard_cap be split fairly across multiple priority sources?** **Resolved 2026-08-18 (DEC-069)** — root-caused, not just patched: the real cause was ExDark's own unbounded *image* floor (not crowdhuman's instance subbudget) consuming the whole shrunken budget. Fixed with a general per-priority-source minimum image reservation, scaled from `floor`. Verified as an exact no-op at the default `--hard-cap 4500`; verified it fixes the `--hard-cap 1500` starvation case for real (crowdhuman/me5_u6rvg now get a guaranteed minimum instead of 0). `PRIORITY_SOURCE_INSTANCE_SUBBUDGET` (crowdhuman's) also now scales proportionally with `--hard-cap`, per the student's literal request.

4500 (the default) has been run for real with this fix; 1500/9000 remain `--dry-run`-verified only.

### 7. Stage 5.6 dedup — detection method and threshold
**Built and run 2026-08-14 (DEC-062)** — `scripts/preprocess/dedup.py` uses FiftyOne Brain's own two built-in checks rather than an invented approach: `compute_exact_duplicates()` (filehash, no threshold) against the **full** 51,529-image merged pool, and `compute_near_duplicates(threshold=0.2)` — 0.2 being FiftyOne Brain's own documented default (its docstring: "[0.1, 0.25] works well for the default setup"), not a number invented for this project — against a **seeded (42), per-source-proportional stratified sample of 6,000 images**, not the full pool (see below for why).

**Real results:** exact-duplicates — **1,893 groups / 2,143 duplicate files** across the full pool. Near-duplicates — **520 groups / 818 flagged files** on the 6,000-image sample. Full detail (every group, filenames, distances) in `dataset/reports/dedup_report.json`.

**Why the near-duplicate check is sampled, not full-pool:** measured directly, not assumed — an unbounded run against all ~51.5k images degraded from ~89 img/s to under 10 img/s within 90 seconds once FiftyOne's default `num_workers` over-subscribed this machine's single MPS device. Pinning `num_workers=4` (matching an isolated clean benchmark of ~28 img/s) helped but didn't fully restore that throughput at full scale (real run averaged ~9-13 img/s) for reasons not fully diagnosed — so the 6,000-image bound was kept as a deliberate, documented scope limit, not full coverage. Exact-duplicate detection has no such limit (filehash, no embedding model) and covers every image.

**Caveat also worth your eyes:** some flagged near-duplicate pairs report `distance` values well above 0.2 (e.g. 5.19, 6.05). Verified via FiftyOne Brain's own source that this is expected — `neighbors_map`'s reported distance is to the nearest *surviving unique* neighbor (a separate post-hoc query), not necessarily the original neighbor that triggered the threshold-based flag. The flagging itself is still threshold-correct; only the reported distance number can be misleading if read as "how close a duplicate this is."

**Embedding model alternatives (researched 2026-08-18, answering "what are the other choices"):** checked the real installed FiftyOne Zoo model list rather than guess. Realistic alternatives to `mobilenet-v2-imagenet-torch`: the `resnet*-imagenet-torch` family (classic, well-validated for similarity/dedup, slower than mobilenet-v2 but faster than the options below), `clip-vit-base32-torch`/`open-clip-torch` (semantically richer — CLIP embeddings capture higher-level content, could catch semantically-similar-but-pixel-different images mobilenet-v2 might miss, or miss visually-identical-but-differently-cropped ones depending on what you actually want "near-duplicate" to mean), and the `dinov2-*` family (strong recent self-supervised embeddings, generally the best quality option, but larger/slower). No single "correct" answer — the tradeoff is speed vs. how semantic vs. how purely-visual you want the similarity notion to be.

**Full-scale attempt (2026-08-18):** student explicitly asked to run the near-duplicate check at full scale, locally, accepting "over an hour." Added `dedup.py --full-scale`. **Result: projected ~5 hours, then ~9 hours after the student's laptop slept mid-run** (process survived the sleep, but post-resume throughput never recovered) — running at ~1-4 img/s throughout, worse even than the already-documented partial-degradation case (~9-13 img/s) that originally motivated the 6,000-sample bound.

**Killed 2026-08-18 (DEC-072), student's call.** Reasoning (the student's own, not just accepted): a follow-up merge+dedup cycle was already needed regardless once `door_detection_zqt59` (item #1) is folded in, so the in-flight run was already validating a pool known to be incomplete — no point burning more hours on it. **Current state: only the stale Aug 14 6,000-sample result exists on disk** (`images_checked: 51529`) — no full-scale result was ever saved (no incremental checkpointing, so the killed run produced nothing).

**Decide, once curation (item #1's merge, item #3's review) is finished and ready for a real final pass:** run the full-scale near-duplicate check locally again, or move to RunPod. RunPod would likely be meaningfully faster (this machine's MPS backend already needed `num_workers` tuning once to avoid over-subscription, and a dedicated cloud GPU wouldn't share that or the sleep risk) but no benchmark exists yet — recommended approach is a small `--limit` smoke test on the rented instance first, not committing to the full run blind. `split.py`/`generate_yaml.py` reruns are blocked on this — not because the scripts crash without it (`split.py` degrades gracefully, printing a warning and skipping duplicate-aware grouping if `dedup_report.json` is missing or stale relative to the current merged pool size), but because skipping it means losing train/eval leakage protection, which AGENTS.md's Dataset Curation Rules explicitly calls out as a hard requirement.

### 8. Stage 5.8 split — exact ratio within the approved range
**Resolved 2026-08-18** — student chose **70/15/15** (was 75/12.5/12.5), matching Ultralytics Academy's own documented baseline ("Split the Dataset Correctly," docs.ultralytics.com). Also asked whether this whole stage even matters given Ultralytics might auto-split — checked: `ultralytics.data.utils.autosplit()` exists but is optional/manual-invoke only, standard training requires pre-split directories referenced in `data.yaml`. `split.py` is necessary, not redundant with anything automatic. Ratio constant updated in `scripts/build/split.py`; **not yet re-run** — blocked on item #7's dedup resolution (`split.py` needs a fresh `dedup_report.json`).

**Previous real result (2026-08-14, DEC-063, now stale post-DEC-069's fresh merge)**: train=38,691, val=6,383, test=6,455 images at the old 75/12.5/12.5 ratio and the old 51,529-image merged pool. Zero cross-split duplicate leakage (a real union-find bug was caught and fixed in getting there — see DEC-063). Will be superseded once item #7 resolves and `split.py`/`generate_yaml.py` re-run.

---

## Newly surfaced (2026-08-20, DEC-082)

### 9. Escalator — replacement class, or renumber to 15?
**Resolved 2026-08-20 (DEC-083).** Escalator's slot (id 6) reused for **Shelf** (Open Images, "Shelf" — 22,899 boxes/6,011 images), chosen over Cart (rejected — mostly horse-drawn carts on visual inspection) and Countertop (rejected — mostly residential kitchen counters) after browsing all three in `fiftyone_preview.ipynb`'s new ad-hoc candidate section. A suitable replacement was found, so the renumber-to-15-classes fallback was never triggered — `classes.yaml`/`config_loader.py` still show `nc: 16`.

### 10. Stairs — is `stair_gaptw` alone (1,564 images) enough?
**Resolved 2026-08-20 (DEC-083), better than expected.** `revised_pedestrian_obstacle` (a new multi-class Roboflow source added for other reasons — see DEC-083) turned out to have a `stairs` native class too — real, unplanned bonus. Stairs' active total is now **2,294 images** (`stair_gaptw` 1,564 + `revised_pedestrian_obstacle` 730), comfortably clear of the 1,500 floor with real margin. No backup source search needed.

---

## Newly surfaced (2026-08-21, DEC-085)

### 11. `dataset_ninja_road_damage_detector` — is it actually an active source, or a dormant fallback?
**Resolved 2026-08-21 (DEC-086).** Student's call: "we need that fallback now" — option (a), made active. `audit_status: approved` set explicitly in `config/datasets.yaml`, matching its sibling `dataset_ninja_pothole_detection`.

Implementing this surfaced a bigger, separate problem: `cap_per_class.py` had **no `audit_status` awareness at all** — every `dataset/processed/<source>/` directory was scanned unconditionally regardless of status, meaning any source benched/failed *after* its first pull (all 7 from DEC-082/083) would have silently stayed eligible for the next real cap/merge run, contradicting what those entries claimed. Found, fixed (`get_inactive_processed_source_keys()` in `config_loader.py`, wired into `cap_per_class.py`), and verified via `--dry-run`: Potholes now correctly shows 2,661 candidates (`pothole_vhmow` excluded, `dataset_ninja_road_damage_detector` included), matching the number this item was originally about. Full writeup in DEC-086 — confirmed the last *real* run (DEC-077, 2026-08-18) predates this risk and isn't affected.

---

## Newly surfaced (2026-08-25/26, DEC-091 / DEC-092 / DEC-093)

### 12. `roboflow_revised_pedestrian_obstacle` — are its 1,888 near-duplicate flags genuine, or false positives?
**Deferred by the student mid-review 2026-08-26 ("ask me that again later, im focused on fixing up the labels") — genuinely open, not forgotten.** This source was never included in `BULK_FALSE_POSITIVE_SOURCES` during DEC-090's threshold review, and has **0** entries on the duplicate side of `dataset/reports/near_duplicate_false_positives.json` (it appears once, but only as the *kept* side of another group, which clears nothing for it). Real numbers: 1,888 near-duplicate `duplicate`-role entries plus 1 exact, against 3,960 images in the 5,500-cap selection — a **47.7%** flag rate, currently all treated as genuine.

Interim action taken (not a resolution): the student rebuilt the review dataset with `hide_duplicates = True`, so ~2,071 images are under review instead of 3,960. That's a review-time decision only — it does not record any judgment about whether the flags are correct.

Worth deciding with DEC-090's own empirical finding in mind: false-positive-ness consolidates cleanly by source (a source's flags tend to be *mostly* genuine or *mostly* false positives, rarely mixed), so a look at ~10 flagged groups should settle it rather than a full pass.

### 13. `split.py` has no awareness of `near_duplicate_false_positives.json`
**Open, known, and pre-existing** — flagged in `fiftyone_near_dup_inspection.ipynb`'s own write-back cell comment at the time it was built ("does not touch... `split.py`'s duplicate grouping. Deciding whether/how those specific pairs should stop being treated as duplicates downstream is a separate follow-up, not made here"), never resolved since.

Verified directly 2026-08-26: `near_duplicate_false_positives.json` is read **only** by the two notebooks — nothing under `scripts/` references it. `split.py` reads `dataset/reports/dedup_report.json` directly and forces every duplicate group into a single split to prevent train/val leakage. So all 3,829 pairs the student cleared as false positives are **still grouped as duplicates at split time**.

Effect is conservative, not dangerous — no leakage risk — but large forced groups could distort the 70/15/15 ratios (DEC-071). Needs a decision before `split.py` runs for real: either have `split.py` subtract the false-positive pairs, or accept the over-grouping deliberately.

### 14. `crowdhuman` and `open_images` — a few duplicate boxes of unknown origin
**Flagged, not investigated.** A full scan of every processed source (2026-08-26) found 6 duplicate `(class, bbox)` lines across `crowdhuman`'s 19,370 label files, and 3 across `open_images`' 31,011. **Not** the DEC-093 write-back bug — neither source has ever been through this notebook's write-back (no `labels_reviewed/` exists for either), so the cause is different and unidentified. Most plausible guess for `open_images` is the cross-folder box merge (DEC-052); no hypothesis for `crowdhuman`. Well under 0.1% of files each, so left alone deliberately rather than fixed blind.

### 15. DEC-092's tag-loss mechanism — RESOLVED 2026-08-26 (DEC-097)
**Answer: there was no tag loss.** FiftyOne caches `Sample` objects by id inside the Python process, so `for sample in dataset` returns the cached object rather than what the App (a separate process) has since written to MongoDB. The kernel was reading a snapshot from before the tags existed; the tags were in the database the whole time. Proven in a controlled probe — kernel sees 1 box/no tags, DB holds 2 boxes/`exclude`, iteration returns the stale 1, `dataset.reload()` returns the correct 2. It also explains why `dataset.export()` (the backup) was always accurate: export queries MongoDB directly. Fixed by calling `dataset.reload()` first in every notebook cell that reads the live dataset (v29). The original entry is kept below for the record.

#### Original entry
Not a decision to make, but recorded here because it's an unresolved *risk* affecting how review sessions should be run. See DEC-092 for the three disproven hypotheses and the standing operating rule (back up immediately before write-back; recover from the static export rather than re-running write-back live if anything looks inconsistent).

### 17. Most Roboflow sources ship pre-augmented — de-augment, and what does that do to the floors?
**Discovered 2026-08-26, mid-review, while investigating why duplicates were still visible after `hide_duplicates = True`.**

**PARTIALLY RESOLVED 2026-08-26 (DEC-094).** Options B and C were both applied for real: 22,254 augmented files moved aside via the new `scripts/preprocess/deaugment_sources.py` (reversible, `--revert`), and `split.py` now unions augmented-sibling groups into its leakage-prevention grouping. `cap_per_class.py`/`merge.py` re-run — `dataset/merged/` is now 44,606 images (was 67,109). The "don't re-run GPU dedup at a higher threshold" question is also settled (measured 46.6% recall; filename matching is exact and free).

**AUDITED 2026-08-26 (DEC-095).** A follow-up per-source naming audit confirmed the uniform regex was the right rule — sources sort cleanly into not-Roboflow / Roboflow-with-augmentation-off / Roboflow-with-augmentation-on, and the first two had nothing to collapse. It also proved Roboflow augments **only the train split** (so size-1 groups are val/test, not misses), and surfaced a separate leakage class now fixed in `split.py`: **173 base names / 346 files are the same photo under two source keys**. Two residual failure modes are not fixable from filenames — see #18.

**STILL OPEN — the floor decision.** Real post-merge counts put four classes under DEC-042's 1,500 floor: **Stairs 1,417 · Trash Bins 1,339 · Elevator 1,338 · Pedestrian Lane 1,193**, with the ratio invariant at 4.61. Provisionally accepted: Stairs and Trash Bins staying low. Pedestrian Lane is under consideration for benching. **Elevator was surfaced by the corrected analysis and has not been ruled on.** Also still open: which Roboflow sources to drop outright. The original analysis below is kept intact as the reasoning behind DEC-094.

**The finding, verified not inferred.** Roboflow exports use `<original>_<ext>.rf.<hash>.<ext>` naming, so augmented copies of one photo share a base name. Proof from base image `100` in `revised_pedestrian_obstacle`, which exists as 7 files: two of them carry boxes `14 0.462500 0.495312 0.482812 0.793750` and `14 0.537500 0.495312 0.482812 0.793750` — identical width/height with the x-center mirrored exactly around 0.5, i.e. a **horizontal flip**. Others share identical box geometry with different md5 (brightness/noise variants).

Files vs. distinct base images, **active** sources only: `dlsu_d_vehicle_type_detection` 21,142→8,731 · `door_detection_zqt59` 4,493→724 (**6.2x**) · `revised_pedestrian_obstacle` 4,323→2,100 · `elevator_awvus` 1,777→755 · `elevator_status_0iq4p` 1,461→606 · `stair_gaptw` 1,564→967 · `wtf_dwvgm` 1,326→475 · `roitrikee` 665→483 · `trashcan_detection_pihfn` 559→215. Whole active pool: **97,933 → 75,679** (22,254 redundant files). Clean at 1.00x: `pothole_voxrl`, `cv_project_hovyc`, `crosswalk_detector_lz3hc`, and every non-Roboflow source.

**Why the existing dedup run doesn't cover this, and why re-running won't help.** Measured recall on `revised_pedestrian_obstacle`: of 3,539 base-sibling pairs that actually exist, dedup@0.2 grouped only 1,648 — **46.6%**. `mobilenet-v2-imagenet-torch` embeddings are not flip-invariant, so a flipped copy is genuinely distant in that feature space; no threshold catches flips reliably without also flagging masses of unrelated images (the student already hand-reviewed 3,829 false positives at 0.2). Base-name matching gives 100% recall and 100% precision instantly. **Decision taken: do not re-run GPU dedup for this** — it would cost ~8 hours, mostly upload, for a strictly worse signal than a regex. The existing report also stays valid after de-augmentation (report covering more images than the pool is the documented, tolerated superset case).

**Three harms, in severity order.** (1) **Train/val/test leakage** — augmented siblings of one photo can land in different splits, inflating apparent performance; `split.py`'s duplicate grouping only catches what the embedding threshold flagged, so it misses ~53% of siblings. (2) **Inflated counts corrupt the cap/floor policy** — a class "clearing" 1,500 on augmented copies does not have 1,500 distinct scenes. (3) **Multiplied review effort** — reviewing `door_detection_zqt59`'s 4,493 files shows only 724 distinct photos.

**Options laid out for the student (B and C chosen, not yet applied):**
- **A — de-augment at source** (fork + regenerate on Roboflow with augmentation off, as DEC-090 did for `crosswalk_detector_lz3hc`). Gold standard, but 9 sources to redo, and re-export changes filenames, invalidating completed review work on `trashcan_detection_pihfn` (already promoted) and `revised_pedestrian_obstacle` (in progress).
- **B — keep one file per base name in `dataset/processed/`** (move the rest aside, reversible). Cheap, deterministic, flows naturally through cap→merge→split. Cost: the kept variant may itself be a transformed copy (harmless for training — correctly-transformed boxes are valid data), and it exposes the floor problem below.
- **C — make `split.py` group by base name.** Fixes the leakage completely, zero data change, zero re-review, no floor impact. **Treated as mandatory regardless of A/B.**

**The blocking question — de-augmenting drops four classes below the 1,500 floor:** Pedestrian Lane 2,947→**1,201**, Stairs 2,371→**1,441**, Elevator 3,191→**1,340**, Trash Bins 1,683→**1,339**. (Tricycle 3,798→1,788 is thin but clears.) The student has provisionally accepted Stairs and Trash Bins staying low, and is separately weighing benching Pedestrian Lane; **Elevator was newly revealed by the corrected analysis and has not been ruled on.** Framing worth preserving: B does not *cause* this shortfall, it *reveals* one that already existed — if the floor exists to guarantee enough genuinely distinct data, those classes were never really clearing it.

**Related, recorded here because it shapes the drop decision:** 11 of 16 classes survive dropping Roboflow entirely (Open Images / ExDark / CrowdHuman / Dataset Ninja cover them). The five with **no non-Roboflow fallback at all** are exactly Doors, Stairs, Elevator, Tricycle, and Pedestrian Lane. The only Roboflow sources droppable at zero cost to any other class are `wtf_dwvgm` (475 distinct) and `crosswalk_detector_lz3hc` (202), both Pedestrian-Lane-exclusive.

### 16. Downstream reports are stale after 2026-08-26's deduplication
Not a question so much as a queued action. Removing 554 (`cv_project_hovyc`) + 420 (`trashcan_detection_pihfn`) duplicate boxes from those sources' live `labels/` changed their real instance counts, and `cap_class()` budgets against instance counts (`INSTANCE_TARGET`) — so `cap_report_hardcap5500.json`, `box_audit_report.json`, and `merged_box_audit_report.json` all now slightly overcount for those sources. Image *selection* is very likely unaffected (deduplication only removes redundant boxes inside already-selected images), but this hasn't been verified. Separately, promoting `roboflow_pothole_voxrl` will turn it from a 1-class into a 10-class source, which *will* change candidate pools. **The cap → merge cascade must be re-run after promotions and before `split.py`** — self-corrects naturally at that point, no special handling needed.

### 18. Four sources still overstate their distinct-photo count, and filenames cannot fix it
**Surfaced 2026-08-26 by the DEC-095 naming audit.** Two augmentation patterns survive de-augmentation because the transform is baked into the base name itself, not into Roboflow's export suffix:

- **Export → re-upload → re-export.** `door_detection_zqt59` is the clear case: its *valid* and *test* splits carry group sizes 5 and 7, which train-only augmentation cannot produce, and 72.4% of its bases end in `_JPG` — the fossil of a previous export. Pass 2 treated pass 1's augmented copies as fresh originals, giving 4,493 files from 724 "photos" that are themselves partly duplicates.
- **Pre-upload augmentation.** Bases like `IMG-20210825-WA0006flip_output_output` read as their own image. Measured: `escalator_stairs` 158/2,663 bases, `revised_pedestrian_obstacle` 56/2,100, `stairs_i2yia` 29/1,372.

**Impact is on counting, not leakage** — embedding dedup plus `split.py`'s sibling and cross-source grouping still prevent these from straddling splits. But DEC-042's 1,500 floor is measured against a mild overcount for these four sources, which matters because three of them feed classes already at or near the floor (Stairs, Doors, Pedestrian Lane).

**PARTIALLY RESOLVED 2026-08-26 (DEC-096).** `door_detection_zqt59` is fixed: re-pinned v3 -> v1 (removing a 2.60x Roboflow layer) and then filename-de-augmented (1,601 -> 654), which is legitimate for this source because all 474 of its multi-file bases are distinctive 16-hex/`Door####` names with zero camera-roll names. Its pre-upload augmentation was confirmed from v1's own splits: `valid` holds 251 base-groups of size 3 and `test` 147, and Roboflow never augments val/test. `revised_pedestrian_obstacle` is fixed by fork (1.80x -> 1.00x). **Still open for `escalator_stairs` and `stairs_i2yia`** (both currently inactive), whose `_flip_output_output` chains remain unmeasured.

**Not yet decided:** whether to measure the true counts (perceptual hashing would catch both modes, at the cost of a new pipeline stage and threshold to tune) or to re-export `door_detection_zqt59` with augmentation off (DEC-094's option A, still available). Neither is triggered unless a class's floor decision turns on the exact number.

---

## Framing note for whoever reads this next

Stages 5.5 (Model-Assisted Curation) and 5.7 (Final Pre-Split Curation Gate) can have their **scripts** built ahead of time, but the **stages themselves** cannot complete without the student manually reviewing flagged samples in an annotation tool. Don't read "all scripts built" as "pipeline ready to run end-to-end unattended" — several stages have a human-labor step baked into their design, not just a missing script.

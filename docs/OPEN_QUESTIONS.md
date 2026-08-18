# OPEN_QUESTIONS.md — Second Vision AI

> **Last updated:** 2026-08-18
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

**Still to decide, once real post-merge numbers exist:** if Doors is still short after the merge for some reason, fall back to (a) accept whatever ratio results, or (b) recompute every class's cap per DEC-042's rule. Also still open, not blocking: (i) is `ronits-workspace-e52mh/nyb` worth a closer look, or should it be dropped entirely — student hasn't looked yet; (ii) `door_detection_zqt59`'s `hinged` class (1,739 instances) was left out as ambiguous — worth the student's own visual check, could add more Doors instances if it turns out to be additional real doors rather than a hardware sub-part; (iii) license unverified for both `eyecue/trashcan-detection-pihfn` and `door_detection_zqt59` (no SDK field, WebFetch blocked on both project pages) — student should confirm directly before treating the dataset as final for any external use.

### 2. traffico_y1 — formally close it out?
**Resolved 2026-08-18 (DEC-069).** Student confirmed: closed. `audit_status: pending` → `benched` in `config/datasets.yaml`, same pattern as `jeep_hozhs`/DEC-037. Nothing further to do.

---

## Stage 5.3 (Box Audit) — items already queued in `docs/PLAN.md`

### 3. elevator_status_s4lrk — isolate real detection boxes from classification-style ones
**Heuristic built and run 2026-08-13 (DEC-057)** — `scripts/preprocess/box_audit.py`, flagged list at `dataset/reports/elevator_status_s4lrk_flagged.json` (844 boxes).

Correction to this entry's original framing: checked the real area-fraction distribution before picking a threshold, and there's no "near-full-frame" cluster to separate out — max area fraction across all 6,786 real boxes is 0.595, with no bimodal gap anywhere. A fixed area threshold would have flagged nothing. The actual defect is **shape**, not size: 842 of 844 flagged boxes are extreme-elongation outliers (several are literal hairline slivers along a frame edge, e.g. w=0.61, h=0.00003) — this matches DEC-031's original description of this source ("boxes cling to object shape rather than a clean axis-aligned rectangle") much better than the "state pseudo-label" framing did. `roboflow_stairs_i2yia` shows the same signature (232/233 flagged boxes are shape outliers) — worth reviewing alongside elevator.

**Still needs you**: the flagged list is real and grounded, but deciding what to do with each flagged box (relabel, drop, or it's actually fine) is visual review — same as before, just with the correct heuristic behind it now.

**How, concretely (added 2026-08-18)**: open `notebooks/fiftyone_review_processed.ipynb`, set `source_key = "roboflow_elevator_status_s4lrk"` and `flagged_report_path` to `dataset/reports/elevator_status_s4lrk_flagged.json` (both already the notebook's defaults). It loads only the 750 images with a flagged box, and marks the specific flagged detection(s) — not just the image — so you can tell which box in a multi-box image is the problem one. Same notebook, same mechanism works for `roboflow_stairs_i2yia` too (the other source `box_audit.py` flagged with the same shape-defect signature) — swap `source_key`/`flagged_report_path` accordingly.

### 4. Near-exhaustive review for smaller Roboflow pools (D-018r)
Not a script gap — this is genuinely your visual-review time. **Resolved 2026-08-18**: `notebooks/fiftyone_review_processed.ipynb` now browses `dataset/processed/<source>/`, `dataset/merged/`, and `dataset/final/<split>/` (set `source_key` accordingly) — use whichever stage you actually want to eyeball.

---

## Tooling decisions

### 5. CVAT or Label Studio for Stage 5.5's correction loop?
Still an open checkbox in `TASKS.md` Phase 1 ("Verify CVAT/Label Studio integration via FiftyOne"). This **blocks `scripts/curate/reimport_corrections.py`** specifically — that script's whole job is parsing one tool's export format back into the pipeline, so it can't be written correctly without knowing which tool. (`run_mistakenness.py`, the other half of Stage 5.5, doesn't need this — computing mistakenness scores is tool-agnostic, and was built and run this session — DEC-060.)

**Confirmed genuinely blocked, not built 2026-08-13**: checked before skipping, not just deferred on sight — `reimport_corrections.py`'s entire job is parsing one specific tool's export format (CVAT XML vs. Label Studio JSON are structurally different), so writing it against a guessed tool would mean either guessing wrong and rewriting later, or building something tool-agnostic-in-name-only that doesn't actually parse anything real. No new information surfaced this session that would resolve this — still needs the student's actual tool choice.

**Recommendation (researched 2026-08-18, not yet confirmed by student)**: use FiftyOne's own built-in in-app annotation, not CVAT. Verified via `docs.voxel51.com`: it's a stable feature (since v1.13.0, not experimental) that fully supports creating/editing/deleting bounding boxes directly in the App — drag to create, drag corners to resize, autosave. It's explicitly designed for ad-hoc, single-person editing, which is exactly this situation; CVAT's real strength (team-coordinated annotation projects) doesn't apply here. Practical consequence if confirmed: **`reimport_corrections.py` may not be needed as originally scoped at all** — corrections happen in-place on the FiftyOne dataset, no external export/reimport round-trip. What would still be needed instead is a much simpler script: write the edited FiftyOne dataset's labels back out to the intermediate-schema files.

**Decide:** confirm the FiftyOne-built-in recommendation (in which case `reimport_corrections.py` gets rescoped, not built as originally planned), or still prefer CVAT/Label Studio for some reason not yet surfaced (e.g., wanting the correction work to live outside FiftyOne for some other reason).

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

## Framing note for whoever reads this next

Stages 5.5 (Model-Assisted Curation) and 5.7 (Final Pre-Split Curation Gate) can have their **scripts** built ahead of time, but the **stages themselves** cannot complete without the student manually reviewing flagged samples in an annotation tool. Don't read "all scripts built" as "pipeline ready to run end-to-end unattended" — several stages have a human-labor step baked into their design, not just a missing script.

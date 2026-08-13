# OPEN_QUESTIONS.md — Second Vision AI

> **Last updated:** 2026-08-13
>
> Every question here blocks something specific — either a script that can't be written correctly without the answer, or a stage that can't be marked done. When a question gets answered, record the answer as a new entry in `docs/DECISIONS.md` (it's a real decision) and delete or mark the entry here as resolved. Don't let this file and `DECISIONS.md` disagree — `DECISIONS.md` is the one that future sessions trust.
>
> This list reflects what's known as of Stage 5.1/5.2's completion (all 5 sources acquired and converted — see DEC-056). If an autonomous agent session works through Stage 5.3–5.9 before this is next read, it will likely append its own findings here — check the bottom of the file for a "newly surfaced" section before assuming this is the complete list.

---

## Data-scope questions

### 1. Trash Bins — reopen the secondary-source search? (DEC-039, DEC-051)
Open Images yielded only 1,106 raw images for Trash Bins ("Waste container") — about 18% of the 6,075-image request, and already below the 4,500 final-cap target before any audit/dedup shrinkage. DEC-039 benched searching for a secondary source on the assumption the primary would be sufficient; DEC-051 flagged that assumption as questionable once the real yield came in low.

**Decide:** accept ~1,000-ish final Trash Bins images as sufficient, or spend time finding a secondary source (Roboflow Universe search, another Open Images-adjacent class, etc.)?

### 2. traffico_y1 — formally close it out?
Blocked since it was first considered (zero generated Roboflow versions on the source project's side — not something fixable from our end). `me5_u6rvg` was reactivated specifically to replace it as Vehicle's secondary (DEC-037), and turned out to carry real Tricycle/Van/Truck content too (DEC-053) — likely already covers what `traffico_y1` would have added.

**Decide:** this is probably just a "confirm and mark `benched`/`abandoned` in `datasets.yaml`" formality rather than a real open question — but worth a explicit yes since it's been sitting as `pending` with a null `pinned_version`.

---

## Stage 5.3 (Box Audit) — items already queued in `docs/PLAN.md`

### 3. elevator_status_s4lrk — isolate real detection boxes from classification-style ones
The existing audit note already confirms genuine detection-style samples exist (real bounding boxes on people inside elevators), alongside what look like near-full-frame "elevator state" labels that aren't real object boxes. A heuristic (e.g., flag boxes above some area-fraction threshold as suspect) can narrow this down automatically, but the actual threshold and what to do with flagged images is a judgment call — visual review needed either way.

**Not purely a question** — an overnight agent can build the flagging heuristic and produce a candidate list; you still need to look at the flagged samples.

### 4. Near-exhaustive review for smaller Roboflow pools (D-018r)
Not a script gap — this is genuinely your visual-review time, using `notebooks/fiftyone_review_processed.ipynb` (built 2026-08-13) once each project's converted output exists. No blocking question, just noting it here so it doesn't get lost as a "someone will get to it eventually" item.

---

## Tooling decisions

### 5. CVAT or Label Studio for Stage 5.5's correction loop?
Still an open checkbox in `TASKS.md` Phase 1 ("Verify CVAT/Label Studio integration via FiftyOne"). This **blocks `scripts/curate/reimport_corrections.py`** specifically — that script's whole job is parsing one tool's export format back into the pipeline, so it can't be written correctly without knowing which tool. (`run_mistakenness.py`, the other half of Stage 5.5, doesn't need this — computing mistakenness scores is tool-agnostic.)

**Decide:** CVAT, Label Studio, or something else? Worth a quick look at FiftyOne's integration quality/docs for each before deciding, not just picking arbitrarily.

---

## Algorithm/policy specifics not yet pinned down

### 6. `cap_per_class.py`'s trim method when sources overlap the cap
DEC-014 already defines the *priority order* — ExDark's guaranteed floor is reserved first (for the 6 overlapping classes), then Primary/Secondary volume fills the remaining budget up to `classes.yaml`'s cap (DEC-042: uniform 4,500). What's **not** yet specified: when Primary + Secondary combined exceed the remaining budget after the floor, how are candidates trimmed down to fit — random sample (reuse the existing `SEED = 42` convention), or ranked by something (audit-flag status, box quality, recency)?

**Decide:** random-with-seed is the simplest and matches how every acquire script already samples — probably the right default unless you want quality-ranked trimming specifically.

### 7. Stage 5.6 dedup — detection method and threshold
`docs/PLAN.md` calls for "cross-source dedup (highest risk pairs flagged)" but doesn't specify *how* — FiftyOne Brain's built-in near-duplicate/uniqueness tooling (exact-match + perceptual-hash-style similarity) is the natural fit given FiftyOne's already the project's chosen tool, but the similarity threshold that separates "same photo, different source" from "just visually similar" needs an actual decision, not an invented default — this project already got burned once by an unvalidated invented number (the original 1.35 buffer factor, corrected by DEC-042's literature-grounded replacement). Worth 10 minutes of the same kind of grounding before picking a number.

**Decide:** approach + threshold. If you want, I can research FiftyOne Brain's recommended defaults and propose a specific number with reasoning, same way the per-class cap got decided — just say so.

### 8. Stage 5.8 split — exact ratio within the approved range
`docs/PLAN.md`'s Split Strategy table already gives ranges (train 70-80%, val 10-15%, test 10-15%) and the val split's dual role as Hailo calibration data is already documented. Any single point within those ranges is already-approved territory — this is a low-stakes "pick one" (e.g., 75/12.5/12.5), not a real blocker. Flagged only so whichever script picks a number, it's a deliberate choice recorded somewhere (even just a code comment), not silently arbitrary.

---

## Framing note for whoever reads this next

Stages 5.5 (Model-Assisted Curation) and 5.7 (Final Pre-Split Curation Gate) can have their **scripts** built ahead of time, but the **stages themselves** cannot complete without the student manually reviewing flagged samples in an annotation tool. Don't read "all scripts built" as "pipeline ready to run end-to-end unattended" — several stages have a human-labor step baked into their design, not just a missing script.

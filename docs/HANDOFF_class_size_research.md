# HANDOFF — Research Task: Citation-Backed Justification for Per-Class Image Count Range

> **Purpose of this document:** self-contained brief for a *separate* conversation/session/tool focused on literature research — not dataset engineering. Everything needed to pick this up cold is below; you should not need the original conversation that produced it.

---

## The question to answer

**Find peer-reviewed or otherwise citable research, published 2020 or later, that can defend a per-class training image count in the range of ~1500–3500 images for fine-tuning a pretrained object detector (YOLOv8s) on custom classes.**

This is not a search for a study that proves "N images is the correct/optimal number" — no such precise universal number exists in the literature, and claiming otherwise would be indefensible. What's needed is enough credible, recent evidence that a range in this ballpark is a **reasonable, defensible choice** — precedent, sample-efficiency/learning-curve evidence, or methodological guidance from a real study, not just framework documentation.

## Why this matters

This dataset (Second Vision — a YOLOv8s object detector for an assistive-navigation smart glass, 15 classes) is being sized in this range. This is a **thesis project**, and the anticipated defense-panel question is:

> "Why did you choose to train your detection model with a range of 1500 to 3500 images [per class]?"

The current answer leans on Ultralytics' own framework documentation (see below), which is practitioner guidance, not peer-reviewed research. A panelist may draw exactly that distinction. The goal of this research task is to find something that survives that follow-up.

---

## What's already established (don't re-derive this)

All of the following is recorded in this repo's `docs/DECISIONS.md` (DEC-025 through DEC-028) and is considered settled — the research task is to *strengthen the citation backing*, not revisit these conclusions:

1. **Ultralytics' own guidance**: "≥1500 images per class" and "≥10,000 instances per class" recommended, per their "Tips for Best Training Results" docs (originally a YOLOv5 doc, carried forward and still cited for YOLOv8). Source: https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/ — **this is framework documentation, not a peer-reviewed study.** No specific research citation is given on that page for where the 1500/10,000 figures come from.

2. **Transfer learning reduces data needs for classes that overlap COCO's own 80 pretrained classes.** Roboflow's own transfer-learning documentation demonstrates a working detector from as few as ~364 images / ~4,900 labels (BCCD dataset), reasoning that a COCO-pretrained backbone "does not need to learn basic visual features again... it adapts its existing knowledge." Source: https://blog.roboflow.com/yolo-training/

3. **COCO itself has a well-documented long-tail class imbalance** (Oksuz et al., "Imbalance Problems in Object Detection: A Review," arXiv:1909.00169; COCO-LT/COCO-MLT long-tail benchmark variants). This literature addresses imbalance *within COCO's own pretraining distribution* — it does not directly prescribe a custom fine-tuning dataset size, and should not be over-extended to justify one.

4. **Two dataset-specific papers behind this project's edge-case sources**, both directly on-point because they're the actual datasets in use:
   - Loh & Chan (2019), *Computer Vision and Image Understanding*, vol. 178, pp. 30-42 (the ExDark dataset's source paper): standard benchmarks including COCO contain "less than 2% low-light images." https://arxiv.org/pdf/1805.11227
   - Shao et al. (2018), arXiv:1805.00123 (CrowdHuman's source paper): crowd/occlusion scenarios are "still under-represented in current human detection benchmarks," COCO included. https://arxiv.org/abs/1805.00123

5. **Resulting project framework** (DEC-027, DEC-028): per-class targets are split into a *general-condition* budget (can lean toward the lower end of the range for COCO-overlapping classes, per point 2) and an *edge-case* budget (ExDark/CrowdHuman — NOT discounted, per point 4, since COCO's pretrained prior barely covers these conditions). This produced a working range of roughly 1500 images (floor, for classes with no COCO analogue) up to ~2500–3500 (Person, which uniquely has both edge-case sources).

**Note both 2019 papers (point 4) predate the "2020 onwards" requirement** — they're kept in the brief as background/context for *why* the edge-case split exists, not as the answer to the research question itself. Don't present them as satisfying the "2020+" ask.

---

## Suggested research directions

None of these have been checked yet — they're starting points, not confirmed leads:

- **Few-shot / low-shot object detection** — an active subfield since ~2019-2020 specifically about how much labeled data a detector needs via transfer learning. Survey papers here likely discuss sample-efficiency curves directly relevant to this question.
- **"Sample-efficient transfer learning object detection"** / **"data-efficient fine-tuning"** as search terms.
- **Domain-specific case studies using YOLOv5/v7/v8, published 2020+**, that fine-tuned on custom datasets of a comparable size (hundreds to low-thousands of images per class) — e.g. agricultural pest/crop detection, wildlife camera-trap detection, PPE/safety-equipment detection, pothole/road-damage detection (there may even be a paper behind the `road-damage-detector` / RDD2022 dataset already in this project's pipeline — worth checking if it discusses dataset sizing rationale). These function as **precedent citations**: "prior published work achieved acceptable performance fine-tuning YOLO at a similar per-class scale."
- **Learning-curve / dataset-size-vs-mAP studies** — papers that plot detection performance against training set size and show a diminishing-returns knee. If one exists showing the knee falls in the low-thousands-per-class range, that's strong, defensible, precise evidence for this exact argument.
- Check whether Ultralytics' own GitHub (issues/discussions) has ever cited internal experiments or external research behind the 1500/10,000 numbers in point 1 — even informal maintainer commentary with data behind it would strengthen that citation.

## What counts as "good enough"

Not required: a study proving 1500–3500 is optimal for *this* task specifically (15 custom classes, assistive navigation, YOLOv8s) — that study doesn't exist and won't.

Sufficient: 2–3 credible, post-2020 sources that collectively show (a) transfer-learning detection performance plateaus in a similar low-thousands-per-class range in comparable settings, and/or (b) precedent of published work successfully fine-tuning a YOLO-family model at this scale. Enough to say in a defense: "this range is consistent with published sample-efficiency findings and comparable prior work," not just "the framework docs said so."

---

## When this is resolved

Bring the findings back to this repo and log them as a new decision in `docs/DECISIONS.md` (next available number — check the file for the current highest `DEC-XXX`), following the existing template at the bottom of that file. Cross-reference DEC-027 and DEC-028. Update or delete this handoff file once superseded.

## Repo pointers (for context, not required reading to do the research)

- `docs/DECISIONS.md` — DEC-002 (class schema), DEC-014 (ExDark guaranteed floor), DEC-019 (model-assisted curation), DEC-025/026/027/028 (this thread's full reasoning chain)
- `config/classes.yaml` — current per-class caps (mostly 5000, some uncapped)
- `AGENTS.md` — project ground rules and communication style

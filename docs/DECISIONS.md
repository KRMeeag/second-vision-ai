# DECISIONS.md — Second Vision AI Decision Log

> This document records key technical decisions made during the project, including context, rationale, and alternatives considered. Entries are ordered chronologically.

---

## Decision Format

Each entry follows this structure:

- **Date:** When the decision was made
- **Status:** `Accepted` | `Superseded` | `Deprecated`
- **Context:** What prompted the decision
- **Decision:** What was decided
- **Rationale:** Why this option was chosen
- **Alternatives Considered:** What else was evaluated
- **Consequences:** Expected impact and tradeoffs

---

## DEC-001: Repository Scope — Training Pipeline Only

- **Date:** 2026-07-31
- **Status:** Accepted

### Context

The Second Vision project spans hardware (RPi5, ESP32, motors), application logic (threading, serial, TTS), and AI model development (dataset, training, export). A decision was needed on whether to maintain a monorepo or separate repositories.

### Decision

Maintain **two separate repositories**:

- `second-vision-ai` — Dataset curation, training, ONNX export, Hailo compilation support
- `second-vision` — Raspberry Pi runtime, embedded code, application logic

### Rationale

- Training and runtime have fundamentally different dependency trees (Ultralytics/PyTorch vs. HailoRT/GStreamer)
- Training may occur on cloud GPUs or Colab; runtime runs on RPi5
- Separation enforces a clean interface: this repo produces `.onnx` files; the production repo consumes `.hef` files
- Independent version histories improve traceability for the thesis

### Alternatives Considered

- **Monorepo**: Rejected — dependency conflicts and unclear responsibility boundaries
- **Three repos** (data, training, production): Rejected — over-segmentation for a thesis project

**A composite object gets boxed by its PARTS, and no IoU threshold can see that.** A Philippine tricycle is a motorcycle plus a sidecar cabin, so `yolov8n` boxes one half of it — a strict subset of the `Tricycle` box, scoring low IoU and passing untouched.

Three successive attempts to threshold this all failed, and the failure is instructive. IoU >= 0.5 left 685 of `roitrikee`'s 1,207 alias-eligible predictions overlapping a tricycle. Adding an 80%-containment test caught 241 more and still left 190. Measured across every band, the survivors were spread **26 / 36 / 20 / 23 / 12 / 19 / 18** from containment 0.1 to 0.8 — flat, no valley anywhere, unlike the clean bimodal split same-class matching enjoys. Every candidate cutoff both left overlaps behind and started eating real objects.

The clean question is not *how much* they overlap but *whether* they do: **190** of the 444 survivors touched a Tricycle box, **254** touched none. On a source labelling only `Tricycle`, an alias-class box overlapping a tricycle is a mis-parse of that tricycle; one touching no tricycle is a separate object. That split needs no threshold and leaves nothing behind — verified at 0 remaining overlaps, with all 867 `Person` predictions untouched.

Note this made the containment helper dead code: **containment > 0 and IoU > 0 are the same predicate**, since both reduce to a non-empty intersection. The implementation is one `_iou` call, not two metrics.

**Which sources aliases may run on is computed, not configured.** Aliases are enabled for a composite class only when the dataset's own `ground_truth` contains none of its alias classes. `dlsu_d_vehicle_type_detection` labels `Motorcycle` (426) and `Vehicle` (426) alongside `Tricycle`, with **112 + 30** of those real boxes inside a Tricycle box — an any-overlap rule there would be actively destructive, and it switches itself off. `roitrikee` and `augmented_tricycle` label only `Tricycle`, so it switches on. A hand-kept source list would have had to be right about this and would go stale as sources are reviewed.

### Consequences

- The handoff artifact between repos is clearly defined: ONNX model files
- Hailo compilation may need documentation in both repos (build steps here, runtime config there)

---

## DEC-002: 15-Class Canonical Schema

- **Date:** 2026-07-31
- **Status:** Accepted (class list partially superseded — see below)

### Context

The project needed a fixed set of object detection classes that are (1) grounded in user research, (2) meaningful for assistive navigation, and (3) feasible to train with available public datasets.

### Decision

Adopt the following **15-class schema** as the authoritative class list:

```
0: Person          5: Stairs          10: Wet Floor Sign
1: Vehicle         6: Escalator       11: Potholes
2: Two Wheeler     7: Doors           12: Trash Bins
3: Cart            8: Chairs          13: Elevator
4: Animals         9: Tables          14: Pedestrian Lane
```

> **Superseded (partial):** ID 3 (`Cart`) was dropped and replaced with `Pole` per DEC-021. ID 10 (`Wet Floor Sign`) was dropped and replaced with `Tricycle` per DEC-015. ID 2 (`Two Wheeler`) was split into `Motorcycle` (ID 2, slot reused) and `Bicycle` (new ID 15) per DEC-038 — `nc` is now 16, not 15. The remaining 12 entries are still authoritative — see `config/classes.yaml` for the current canonical list.

### Rationale

- Each class was justified by survey data (Figures B.6–B.11 from user research)
- Classes were selected to complement depth estimation — detecting things depth sensors *cannot* adequately describe
- The count (15) is manageable for YOLOv8n and Hailo-8 deployment
- Classes cover the primary user environments: schools, malls, streets, workplaces

### Alternatives Considered

- **Fewer classes (8-10)**: Rejected — would omit safety-critical items like Wet Floor Sign and Potholes
- **More classes (20+)**: Rejected — increases training complexity and annotation burden without clear user benefit
- **COCO-80 subset**: Rejected — COCO classes don't include domain-specific items (escalators, wet floor signs, pedestrian lanes)

### Consequences

- All dataset converters must map source labels into this schema
- Unmapped source classes are discarded, not force-mapped
- Hailo runtime will see these as classes 1–15 (background injected as class 0)

---

## DEC-003: YOLOv8 as Detection Architecture

- **Date:** 2026-07-31
- **Status:** Accepted

### Context

An object detection architecture was needed that balances accuracy, inference speed on Hailo-8, and compilation compatibility with the Hailo Dataflow Compiler.

### Decision

Use **YOLOv8** (Ultralytics) as the detection architecture, starting with **YOLOv8n** (nano variant).

### Rationale

- YOLOv8 is officially supported by the Hailo Model Zoo — compilation paths are documented and tested
- YOLOv8n provides the best latency-accuracy tradeoff for real-time assistive use
- Ultralytics provides clean ONNX export with minimal post-processing complexity
- Extensive community support and training documentation
- Transfer learning from COCO pretrained weights is straightforward

### Alternatives Considered

- **YOLOv5**: Rejected — older architecture, YOLOv8 offers better accuracy at similar speed
- **YOLOv9/v10**: Rejected — newer architectures with uncertain Hailo DFC compatibility; deployment risk too high
- **EfficientDet**: Rejected — less Hailo Model Zoo support; more complex export pipeline
- **SSD**: Rejected — lower accuracy for similar inference cost

### Consequences

- Training, export, and compilation workflows are well-documented for this architecture
- Upgrading to YOLOv8s or YOLOv8m is straightforward if nano proves insufficient
- Must avoid custom head modifications that could break ONNX/Hailo compatibility

---

## DEC-004: Validation Split as Hailo Calibration Dataset

- **Date:** 2026-07-31
- **Status:** Accepted

### Context

Hailo DFC requires a calibration dataset during INT8 quantization. A decision was needed on which data to use for calibration.

### Decision

Use the **validation split** as the Hailo calibration dataset, following the official Hailo example workflow.

### Rationale

- The validation split represents unseen images while remaining domain-representative
- Using the training set for calibration risks overfitting the quantization to training data
- The test set should remain a pure holdout — using it for calibration would compromise evaluation integrity
- This approach is documented in Hailo's official examples and model zoo

### Alternatives Considered

- **Training split**: Rejected — risk of quantization overfitting
- **Separate calibration split**: Rejected — reduces already-limited data; adds split management complexity
- **Test split**: Rejected — contaminates the holdout evaluation set

### Consequences

- The validation split must be large enough (~500-1000 images minimum) and representative of deployment conditions
- Validation split quality directly impacts both model evaluation accuracy and Hailo compilation quality
- Extra care needed to ensure the val split covers all 15 classes adequately

---

## DEC-005: Multi-Source Dataset Strategy

- **Date:** 2026-07-31
- **Status:** Superseded (by DEC-012 through DEC-023, 2026-08-04)

### Context

No single public dataset contains all 15 target classes with sufficient quantity and diversity. A strategy for dataset construction was needed.

### Decision

Combine **multiple public sources** (COCO, Open Images, Mapillary, Roboflow) with custom-collected images, unified under a single class mapping.

> **Superseded:** The finalized source list (HANDOFF v3 + DEC-024, 2026-08-06) replaced COCO and Mapillary with Open Images V7, CrowdHuman, ExDark, and Dataset Ninja. Objects365 was also dropped (DEC-024). Custom image collection was dropped. See DEC-013, DEC-014, DEC-016, DEC-017, DEC-018r, DEC-024 for the full revised source strategy.

### Rationale

- COCO provides strong baselines for common objects (person, vehicle, chair)
- Open Images offers scale and diversity for many classes
- Mapillary contributes street-level perspective matching deployment conditions
- Roboflow fills gaps for niche classes (escalators, potholes, wet floor signs)
- Custom images ensure deployment-representative data

### Alternatives Considered

- **Single dataset (COCO only)**: Rejected — missing many target classes entirely
- **Custom dataset only**: Rejected — prohibitive annotation effort for a thesis timeline
- **Synthetic data**: Rejected — domain gap risk too high for an assistive safety application

### Consequences

- Each source requires a dedicated converter script (`scripts/convert/`)
- Careful deduplication is needed when sources share underlying images
- Class distributions will be imbalanced across sources — requires monitoring and potential resampling
- Annotation quality may vary across sources — validation scripts must catch inconsistencies

---

## DEC-012: Canonical Bounding-Box Semantics

- **Date:** 2026-08-04
- **Status:** Accepted

### Context

Different source datasets use different bounding-box conventions. CrowdHuman provides three box types (fbox/hbox/vbox), Objects365 uses native boxes of uncertain modality, and ExDark uses `l,t,w,h` absolute coordinates. A project-wide policy was needed before any conversion work began.

### Decision

All exported YOLO labels represent the **visible extent** of objects, not estimated/hidden (amodal) extent.

- **CrowdHuman**: use `vbox` field only. Discard `fbox`/`hbox` from training export; retain as metadata for future ablation. Entries tagged `mask` or with `extra.ignore == 1`: exclude entirely.
- **Objects365**: native boxes retained after manual visual audit (~100-200 samples/class). Labeled `bbox_mode: native_unspecified` pending audit.
- **Roboflow/Dataset Ninja projects**: each audited independently (~50-100 samples), labeled `bbox_mode: project_dependent`.
- **ExDark**: native `l,t,w,h` boxes, treated as `native_unspecified` pending same visual audit as Objects365.

### Rationale

Visible-extent boxes are the appropriate semantic for obstacle detection in assistive navigation — the user needs to know where the object *is*, not where it might extend behind another object.

### Consequences

- `box_audit.py` must sample and verify box modality for every `native_unspecified` and `project_dependent` source before data enters the training pipeline.

---

## DEC-013: Mapillary Vistas Shelved

- **Date:** 2026-08-04
- **Status:** Deferred

### Context

Mapillary Vistas was originally planned as a primary data source. Investigation revealed that the Dataset Ninja "mapillary-vistas-dataset" link is a re-hosted mirror of Mapillary Vistas itself, not a separate dataset.

### Decision

Mapillary Vistas is **not part of the current pipeline**. Shelved for potential future use.

### Rationale

- The Dataset Ninja mirror does not add distinct data
- Mapillary Vistas licensing adds complexity
- The Potholes use case (original reason for considering Mapillary) is now covered by dedicated Dataset Ninja pothole datasets (DEC-016)

### Consequences

- All references to Mapillary as a data source must be removed from documentation
- If revisited later, it should be logged as a new decision

---

## DEC-014: ExDark as Cross-Cutting Low-Light Augmentation Layer

- **Date:** 2026-08-04
- **Status:** Accepted

### Context

The ExDark (Exclusively Dark Image Dataset) contains ~7,363 low-light images across 12 object classes. Rather than treating it as a primary or secondary source for individual classes, a strategy for leveraging its unique condition-diversity value was needed.

### Decision

ExDark is a **cross-cutting low-light augmentation layer** that injects low-light image variants into **six classes**: Person, Vehicle, Two Wheeler, Chairs, Tables, Animals — the classes present in both ExDark's 12-class taxonomy and our canonical list.

> **2026-08-13 correction:** DEC-038 later split "Two Wheeler" into Motorcycle + Bicycle. ExDark's Motorbike/Bicycle native classes both still map to canonical classes (Motorcycle and Bicycle respectively — see `datasets.yaml`'s `exdark_to_canonical_class_map`), so the real overlap is now **seven** classes, not six: `animals, bicycle, chairs, motorcycle, person, tables, vehicle`. Verified directly against the current config before this count fed into `cap_per_class.py`'s design.

ExDark images are subject to a **guaranteed floor** rule in `cap_per_class.py`: available ExDark images are reserved FIRST before Primary/Secondary volume fills the cap, preventing them from being crowded out.

Use `wraphex/ExDark2Yolo` parser as the basis for `convert/exdark_to_intermediate.py`.

### Rationale

- ExDark's value is **condition diversity** (low-light), not volume (~600-900 images per eligible class)
- Without a guaranteed floor, Primary source volume (Open Images) would easily fill the 5000 cap before ExDark images are considered
- Low-light conditions are directly relevant to the assistive-navigation deployment scenario

### Consequences

- `cap_per_class.py` implements 3-tier fill logic with ExDark floor reservation
- ExDark class mappings (People→Person, Car→Vehicle, Motorbike/Bicycle→Two Wheeler, Dog/Cat→Animals) are documented in `config/datasets.yaml`

---

## DEC-015: Wet Floor Sign Dropped, Tricycle Added

- **Date:** 2026-08-04
- **Status:** Accepted

### Context

No viable public dataset source was found for the Wet Floor Sign class. The Philippine deployment context motivated adding Tricycle as a detection class.

### Decision

- **Wet Floor Sign** (formerly ID 10) is **dropped** from the canonical schema.
- **Tricycle** is **added at ID 10** with `status: possible` — included in the pipeline but may be removed if source quality is insufficient.

### Rationale

- Wet Floor Sign: exhaustive search of Roboflow Universe, Dataset Ninja, and public benchmarks found no usable annotated dataset
- Tricycle: common motorized three-wheeler in Philippine streets; multiple Roboflow datasets available (augmented-tricycle, traffico-y1)
- ID 10 slot reuse avoids schema reordering that would ripple through all other class references

### Consequences

- `nc` remains 15 (14 confirmed + 1 possible)
- Tricycle sources require Stage 5.3 box audit review (traffico-y1 in particular — see DEC-017)
- If Tricycle is ultimately dropped, ID 10 becomes unused and `nc` reduces to 14

---

## DEC-016: Potholes Source Finalized as Dataset Ninja

- **Date:** 2026-08-04
- **Status:** Accepted

### Context

The original plan designated Mapillary Vistas and Roboflow as Potholes sources. After Mapillary was shelved (DEC-013), dedicated pothole datasets were identified on Dataset Ninja.

### Decision

Potholes (ID 11) sourced from two Dataset Ninja datasets:
- **Primary**: `pothole-detection` — 665 images, single-class (pothole), PASCAL VOC format
- **Secondary**: `road-damage-detector` (RDD2022) — 47,420 images, 7 classes total. Class-filtered at conversion time to extract only `pothole`-tagged instances. Role: `volume_topup`.

Cap of **5000** applies.

### Rationale

- `pothole-detection` is low friction — single class, clean annotations, standard VOC format
- `road-damage-detector` provides volume but requires filtering (same pattern as Objects365 multi-class extraction)
- In practice, the filtered pothole count from RDD2022 will likely be the binding constraint, not the 5000 cap

### Consequences

- `voc_to_intermediate.py` converter needed for the primary source
- `coco_style_to_intermediate.py` handles RDD2022 with `native_class_filter: pothole`
- Dedup check needed between the two Dataset Ninja sets (possible shared source imagery)

---

## DEC-017: Multi-Provider / Multi-Class Roboflow Sources

- **Date:** 2026-08-04
- **Status:** Accepted

### Context

Several canonical classes draw from multiple Roboflow projects, and some Roboflow projects span multiple canonical classes. The config schema needed to support this many-to-many relationship.

### Decision

`config/classes.yaml` supports a **list of providers per class**, each with an optional `native_class_filter` field. Affected classes:
- **Stairs**: 3 dedicated projects + filtered slice of escalator-stairs
- **Escalator**: filtered slice of escalator-stairs
- **Elevator**: 2 projects (one flagged for audit)
- **Pedestrian Lane**: pedestrian-and-animal-crossing (filtered)
- **Tricycle**: augmented-tricycle + traffico-y1 (filtered)
- **Vehicle secondary**: Jeepney projects + traffico-y1

**Mandatory audits before pipeline commitment:**
- `elevator-status-s4lrk` — possible classification-vs-detection mismatch
- `traffico-y1` — possible cross-class overlap across Vehicle/Two-Wheeler/Tricycle

### Rationale

- Several niche classes have no single dominant dataset — aggregation is necessary
- The `native_class_filter` pattern mirrors Objects365 multi-class handling
- Audit gates prevent bad data from entering the pipeline silently

### Consequences

- `acquire_roboflow.py` must iterate over a list of `(project_id, native_class_filter)` pairs per canonical class
- Dedup checks across providers within the same canonical class are essential (e.g., Stairs' 4 sources)

---

## DEC-018r: Local Pull-and-Resolve for Roboflow-Native Sources

- **Date:** 2026-08-06 (revised; original 2026-08-04)
- **Status:** Accepted (supersedes prior fork-and-fix approach)

### Context

Roboflow Universe projects are community-contributed and may contain annotation errors, missing labels, or inconsistent class naming. The original D-018 proposed forking each project into our Roboflow workspace and correcting annotations using Roboflow's native UI before acquisition. This created two parallel correction workflows — one cloud-based (Roboflow UI) and one local (FiftyOne/CVAT for large-scale sources).

### Decision

Roboflow Universe projects are **pulled locally** via a pinned dataset version (Python SDK: `project.version(N).download(format)`) rather than forked and edited on Roboflow's platform. The pinned `workspace/project/version` triplet is recorded in `config/datasets.yaml` as the provenance anchor (replacing the former `forked_project_id`).

Roboflow-native pools then flow through the **same curation pipeline** as large-scale sources (Stage 5.3 box audit, Stage 5.5 mistakenness) rather than a separate cloud-editing workflow. Given their smaller size, Roboflow pools may be reviewed **near-exhaustively** at Stage 5.3 (box audit) and Stage 5.5 (mistakenness) rather than sampled.

Native class remapping (e.g., renaming the pole dataset's verbose class label to `"pole"`) happens during conversion (Stage 5.2) rather than in a Roboflow fork.

### Rationale

- Avoids maintaining two parallel correction workflows (Roboflow UI-based vs. FiftyOne/CVAT-based)
- Keeps all annotation correction infrastructure and audit logs local and under repo version control rather than split across a third-party platform
- Pinned versions provide reproducible provenance without depending on Roboflow workspace state
- Smaller Roboflow pools benefit from the same mistakenness ranking — near-exhaustive review is practical at their scale

### Alternatives Considered

- **Fork-and-Fix on Roboflow (original D-018)**: Rejected — required maintaining two correction workflows and split audit logs between cloud and local
- **No version pinning**: Rejected — Roboflow Universe projects can be updated by their owners at any time; unpinned downloads are not reproducible

### Consequences

- `config/datasets.yaml` has `pinned_version` and `download_format` fields per Roboflow project (replacing `forked_project_id` / `fork_date` / `corrections`)
- `acquire_roboflow.py` reads pinned version from `datasets.yaml` and downloads via SDK
- Stage 5.0 is **eliminated** as a separate stage — Roboflow audit/correction happens at Stages 5.3 and 5.5 alongside all other sources
- Native class remapping is handled in conversion scripts (Stage 5.2), not on the Roboflow platform

---

## DEC-019: Model-Assisted Curation for Large-Scale Sources

- **Date:** 2026-08-04
- **Status:** Accepted

### Context

Large-scale sources (Open Images, CrowdHuman, ExDark, RDD2022) contain too many images for manual review. A scalable approach to annotation quality was needed.

### Decision

After per-class capping (Stage 5.4), use FiftyOne's `compute_mistakenness()` to rank samples by predicted-vs-ground-truth disagreement, surfacing both wrong labels and missing annotations. Only the **flagged subset** is sent to CVAT/Label Studio (via FiftyOne's annotation integrations) for manual correction, then re-imported to overwrite into `dataset/curated/`.

### Rationale

- The capped per-class pool (≤5000 images) is the first point where review is tractable
- `compute_mistakenness()` uses a pretrained model's predictions as a signal — high disagreement between prediction and ground truth flags likely annotation errors
- Sending only flagged samples to CVAT/Label Studio keeps manual effort focused
- Separating pre-correction (`dataset/processed/`) and post-correction (`dataset/curated/`) states maintains auditability

### Consequences

- FiftyOne is a hard dependency (`requirements.txt`)
- `scripts/curate/run_mistakenness.py` runs a pretrained YOLOv8 model for predictions
- `scripts/curate/reimport_corrections.py` pulls fixes back from CVAT/Label Studio
- Curation logs written to `dataset/reports/curation_log_<source>.json`

---

## DEC-020: Final Pre-Split Curation Gate

- **Date:** 2026-08-04
- **Status:** Accepted

### Context

Per-source curation (DEC-019) happens on individual per-class pools. Cross-source issues (e.g., inconsistent class conventions between near-duplicate images from different sources) are invisible at the per-source stage.

### Decision

After merge + cross-source dedup (Stage 5.6), run one final FiftyOne mistakenness pass on the **fully merged pool** before `split.py` executes. Corrections at this stage are fixed in place — they do not re-enter capping/merging. This is the last correction checkpoint.

### Rationale

- Merging creates new adjacencies between images from different sources that may reveal inconsistencies
- A final gate ensures the training data has been reviewed at every aggregation level: per-source, per-class-capped, and merged
- The "fix in place, no re-entry" rule prevents infinite correction loops

### Consequences

- `scripts/curate/final_merge_curation.py` implements this gate
- Split proceeds immediately after — no further corrections allowed
- Any issues found after splitting should be logged for the next training iteration, not patched retroactively

---

## DEC-021: Cart Dropped, Pole Added

- **Date:** 2026-08-04
- **Status:** Accepted

### Context

Cart (ID 3) was included in the original 15-class schema (DEC-002) but no viable public dataset source was identified during the dataset acquisition planning phase. Poles (utility poles, street poles) are common obstacles in Philippine street environments and a Roboflow dataset was identified.

### Decision

- **Cart** (formerly ID 3) is **dropped** from the canonical schema.
- **Pole** is **added at ID 3**.
- Source: Roboflow project `test-j2maq/pole-detection-z76mb` (~3,100 images, single class).

### Rationale

- Cart: no annotated dataset found across Roboflow Universe, Dataset Ninja, Objects365, or other public sources
- Pole: utility poles and street poles are common physical obstacles in Philippine urban/suburban environments; visually distinct enough for reliable detection
- ID 3 slot reuse avoids schema reordering
- Per D-018r, the Roboflow source is pulled locally and the native class label remapped to "pole" during conversion (Stage 5.2)

### Consequences

- `nc` remains 15
- Pole is treated as an uncapped, Roboflow-only class (similar tier to Stairs, Doors, Escalator)
- No ExDark cross-cutting applies (ExDark does not contain pole annotations)
- `config/datasets.yaml` includes the `pole-detection-z76mb` project entry

---

## DEC-022: YOLOv8n → YOLOv8s Upgrade

- **Date:** 2026-08-04
- **Status:** Accepted (supersedes DEC-003 variant selection)

### Context

DEC-003 established YOLOv8n (nano) as the baseline architecture, noting that upgrading to YOLOv8s was straightforward if needed. During dataset planning, the larger and more diverse dataset (Open Images + multi-source aggregation) motivated reconsidering the model capacity.

### Decision

Upgrade the training target from **YOLOv8n** (nano) to **YOLOv8s** (small).

### Rationale

- The finalized dataset is substantially larger and more diverse than originally planned — YOLOv8s has more capacity to leverage this
- YOLOv8s remains well within Hailo-8's computational budget (26 TOPS)
- YOLOv8s is supported in the Hailo Model Zoo with documented compilation paths
- The accuracy improvement from n→s is meaningful for an assistive safety application

### Consequences

- `config/training.yaml` baseline changes from `yolov8n.pt` to `yolov8s.pt`
- ONNX export and Hailo DFC compilation paths remain identical
- Inference latency increases slightly but remains within real-time requirements
- DEC-003's core rationale (YOLOv8 family, Hailo compatibility) is unchanged

---

## DEC-023: Pole Source Designated

- **Date:** 2026-08-04
- **Status:** Accepted

### Context

After adding Pole to the canonical schema at ID 3 (DEC-021), a source dataset needed to be designated.

### Decision

Use Roboflow project **`test-j2maq/pole-detection-z76mb`** as the primary (and only) source for the Pole class.

- ~3,100 images, single class
- Native class label in original project: `"Utility pole detection - v1 2023-05-17 3:44pm"` — remapped to `"pole"` during conversion (Stage 5.2)
- Pinned at version 2 in `datasets.yaml`
- No cap (naturally volume-limited)
- No ExDark cross-cutting
- Standard D-018r local pull-and-resolve workflow applies

### Rationale

- Single-class dataset reduces class-filtering complexity
- Volume (~3,100) is adequate for an obstacle class
- Per DEC-018r, annotation quality is reviewed through the same FiftyOne/CVAT pipeline as all other sources

### Consequences

- `pole-detection-z76mb` added to `config/datasets.yaml` under `roboflow_projects`
- `classes.yaml` pole entry references this project
- Native class remapping handled in conversion scripts (Stage 5.2)

---

## DEC-024: Objects365 Dropped, Open Images V7 as Primary

- **Date:** 2026-08-06
- **Status:** Accepted

### Context

Objects365 was originally designated as the primary source for 7 high-volume classes (Person, Vehicle, Two Wheeler, Animals, Chairs, Tables, Trash Bins). During implementation planning, Objects365 proved infeasible — the dataset is ~712 GB total, requires complex category-indexed annotation JSON parsing to filter needed images, and the download infrastructure is unreliable.

Open Images V7 was already in the pipeline as a secondary source for Two Wheeler. It covers all the same classes with a more accessible download mechanism via FiftyOne Zoo.

### Decision

**Objects365 is dropped entirely.** Open Images V7 replaces it as the primary source for all 7 classes:

| Class | Old Primary | New Primary | OI Native Class |
|-------|-----------|-------------|----------------|
| Person | Objects365 "Person" | Open Images | "Person" |
| Vehicle | Objects365 ["Car","Bus","Truck"] | Open Images | ["Car","Bus","Truck"] |
| Two Wheeler | Objects365 "Motorcycle" | Open Images | ["Motorcycle","Bicycle"] |
| Animals | Objects365 ["Dog","Cat","Horse"] | Open Images | ["Dog","Cat"] |
| Chairs | Objects365 "Chair" | Open Images | "Chair" |
| Tables | Objects365 "Table" | Open Images | "Table" |
| Trash Bins | Objects365 "Trash bin Can" | Open Images | "Waste container" |

### Rationale

- Objects365 download is ~712 GB total with unreliable infrastructure — even class-filtered downloads require parsing the full annotation JSON first
- Open Images V7 via FiftyOne Zoo provides native class-filtered downloads — no bulk download needed
- Open Images has sufficient volume and diversity for all 7 classes
- FiftyOne is already a hard dependency (DEC-019) — using it for acquisition unifies the toolchain
- Two Wheeler simplifies from dual-source (Objects365 primary + OI secondary) to single-source (OI only)

### Alternatives Considered

- **Keep Objects365 with selective download**: Rejected — even selective download requires parsing the full annotation JSON (~several GB), and the download servers are unreliable
- **COCO 2017 as replacement**: Rejected — already superseded in DEC-005; fewer classes and less diversity than Open Images

### Consequences

- `acquire_objects365.py` is no longer needed — replaced by expanded `acquire_openimages.py`
- `dataset/raw/objects365/` directory removed
- Animals class drops Horse (not available in ExDark for low-light augmentation consistency; verify OI availability)
- Trash Bins native class name is "Waste container" in OI — must verify before writing acquire script
- Open Images native class names for all 7 classes must be verified against OI class hierarchy (replaces old Blocker #5 scope)
- DEC-005 supersession note updated to reference Open Images instead of Objects365

---

## DEC-025: Narrow Phase 1 Utils Scope — Defer to FiftyOne Native Capabilities

- **Date:** 2026-08-06
- **Status:** Accepted

### Context

PLAN.md originally scoped three Phase 1 utility modules — `file_utils.py`, `bbox_utils.py`, `image_utils.py` — as generic dataset-tooling helpers. Review against FiftyOne's actual capabilities (FiftyOne is already a hard dependency per DEC-019) found significant overlap: FiftyOne ships native importers for several of the exact formats this project's sources use, and FiftyOne Brain provides deduplication tooling that is more capable than a hand-rolled equivalent.

### Decision

- **`file_utils.py`**: Unchanged, kept in full scope. No FiftyOne overlap — this project's `raw/ → processed/ → curated/ → merged/ → final/` directory layout and source-prefixed merge-filename policy are repo-specific and FiftyOne has no opinion on them.
- **`bbox_utils.py`**: Narrowed. FiftyOne's built-in importers (`fo.types.VOCDetectionDataset`, `COCODetectionDataset`, `YOLOv5Dataset`) can load `pothole-detection` (pascal_voc), `road-damage-detector` (coco_style), and all 14 Roboflow projects (yolov8 format) directly, with FiftyOne handling box-coordinate math internally — no custom conversion code needed for those sources. Scope is now limited to: (1) raw box parsing for the two non-standard-format sources, ExDark (native `l,t,w,h` absolute) and CrowdHuman (`.odgt` fbox/hbox/vbox, DEC-012's vbox-only rule), and (2) a lightweight `validate_bbox()` / `clip_bbox()` guard, since FiftyOne does not enforce box sanity (in-range coordinates, positive width/height) automatically.
- **`image_utils.py`**: Dropped as a standalone module. Corrupt-image detection uses `dataset.compute_metadata(skip_failures=True)` directly. Deduplication (Stage 5.6 `dedup.py`, and DEC-020's final curation gate) uses FiftyOne Brain's exact/near-duplicate detection (embedding-based similarity) instead of hand-rolled perceptual hashing.

### Rationale

- Avoids reimplementing already-solved, better-tested functionality
- Keeps the toolchain unified around FiftyOne, echoing DEC-019's own stated rationale for adopting it as a hard dependency
- Hand-rolled perceptual hashing is strictly weaker than FiftyOne Brain's embedding-based near-duplicate search for the same task
- Less custom code to maintain, consistent with AGENTS.md's engineering priorities (avoid unnecessary complexity, favor maintainability and practicality over reinventing tooling)

### Alternatives Considered

- **Keep the original full three-module plan**: Rejected — duplicates functionality FiftyOne/FiftyOne Brain already provide, for no accuracy or maintainability benefit

### Consequences

- `PLAN.md` and `TASKS.md` Phase 1 task lists updated to reflect the narrowed scope
- `scripts/convert/` implementations for VOC/COCO-style/YOLO sources should use FiftyOne's native `from_dir(dataset_type=...)` importers rather than custom parsers
- `scripts/preprocess/dedup.py` (Stage 5.6) must call FiftyOne Brain rather than a custom hashing module
- `bbox_utils.py` scope is limited to ExDark + CrowdHuman parsers and the validate/clip guard

---

## DEC-026: Local Machine Scoped to Dataset Pipeline; Training on RunPod

- **Date:** 2026-08-07
- **Status:** Accepted

### Context

The local development machine lacks sufficient GPU compute for YOLOv8s training at the scale this project's dataset requires.

### Decision

The local machine (this repo, as run by the student) is scoped to **dataset acquisition, conversion, curation, and final split/YAML generation only** — Stages 5.1–5.9, ending at `dataset/final/` + a generated `data.yaml`. **Model training (PLAN.md Phase 3) runs on RunPod**, a rented cloud GPU environment. The curated final dataset is uploaded to RunPod before training begins.

### Rationale

- YOLOv8s training at meaningful epoch counts and dataset scale requires GPU compute beyond the local machine's capacity
- RunPod provides on-demand rented GPU compute, in the same spirit as the cloud-notebook training environments (e.g. Colab) referenced in the Hailo retraining documentation this project follows
- Keeps local tooling lightweight: heavy training-scale CUDA/PyTorch setup isn't required locally. `ultralytics` remains a local dependency regardless, for Stage 5.5's model-assisted curation (mistakenness scoring runs inference with a pretrained checkpoint — not full training)

### Alternatives Considered

- **Train locally at reduced scale/epochs**: Rejected — compromises model quality for a safety-relevant assistive-navigation application; AGENTS.md prioritizes correctness over local convenience

### Consequences

- `notebooks/train_yolov8.ipynb` (Phase 3) executes on RunPod, not locally
- A dataset packaging/upload step (`dataset/final/` + `data.yaml` → RunPod) is needed between Stage 5.9 and Phase 3 — not yet built; added as a task under Phase 3 in PLAN.md
- Whether Hailo DFC compilation (Phase 5) also needs to run on RunPod or another GPU-equipped environment (AGENTS.md requires GPU-accelerated compilation for production) is **not yet decided** — revisit once training is underway
- `config/training.yaml`'s `device` field is correctly left unset already — RunPod will auto-detect its own GPU

---

## DEC-027: Bounded Acquisition Pulls, Not Unbounded Class-Filtered Download

- **Date:** 2026-08-07
- **Status:** Accepted

### Context

Stage 5.1 acquisition scripts were on track to rely solely on FiftyOne Zoo's `classes=[...]` filter to scope downloads, with no explicit per-class sample ceiling. For high-volume classes (Person, Vehicle, etc.), Open Images' class-filtered pool alone can still run into the thousands of images — a real problem given the local machine's memory/disk constraints (the same constraint behind DEC-026 moving training to RunPod).

### Decision

Acquisition scripts bound each per-class pull using FiftyOne Zoo's `max_samples` parameter (with `shuffle=True` and a fixed `seed` for reproducibility), sized as a modest buffer above that class's intended contribution to the final cap — not the full class-filtered pool, and not the bare final target with zero buffer for cleaning-stage rejects. If a class ends up short after box audit (5.3) / mistakenness curation (5.5) rejects some fraction, acquisition is re-run for that class with a higher `max_samples` to top up, rather than pulling everything up front.

### Rationale

- Matches the local machine's actual resource constraints (DEC-026)
- Avoids downloading/holding thousands of images that would just be discarded during cleaning
- FiftyOne Zoo already exposes `max_samples`/`shuffle`/`seed` for exactly this purpose — no custom tooling needed
- Iterative top-up (pull small, clean, expand if short) matches how the student actually wants to work without sacrificing reproducibility, as long as seed is fixed

### Alternatives Considered

- **Unbounded class-filtered pull, cap only at Stage 5.4 (`cap_per_class.py`)**: Rejected as the sole strategy — still risks large raw downloads before any cleaning happens. Stage 5.4 remains necessary regardless, but for a different reason (see Consequences).

### Consequences

- Stage 5.4 (`cap_per_class.py`) is still required — it merges *multiple sources per class* (primary + secondary + lowlight-augment) under DEC-014's tiered fill logic and enforces the joint cap. That's a different concern than bounding a single source's raw pull size, so this decision doesn't replace it.
- `acquire_openimages.py` (and other acquire scripts) must accept/set a per-class `max_samples` value, informed by `config/classes.yaml`'s per-class cap and that class's expected secondary/lowlight contribution — not just the class's raw availability in the source.
- Exact re-run/top-up behavior against FiftyOne Zoo's local caching needs to be verified in practice once the script is written.

---

## DEC-028: Edge-Case Coverage Exempt from Transfer-Learning Volume Discount

- **Date:** 2026-08-07
- **Status:** Accepted

### Context

DEC-027 reasoned that per-class acquisition volume for classes overlapping COCO's pretrained 80 classes (Person, Vehicle, Two Wheeler, Chairs, Tables, Animals) can lean toward the lower end of the 1500–5000 range, since YOLOv8s' COCO-pretrained backbone already has strong relevant visual features for those classes. That reasoning doesn't automatically extend to the low-light (ExDark, DEC-014) and crowd-density/occlusion (CrowdHuman) supplementary sources already built into the pipeline for exactly these classes.

### Decision

Per-class volume targets for COCO-overlapping classes are split into two budget lines, not one undifferentiated number:

- **General-condition volume** (ordinary Open Images pulls) — may lean toward the lower end of the range per DEC-027's transfer-learning reasoning.
- **Edge-case volume** (ExDark low-light floor per DEC-014; CrowdHuman crowd/occlusion slice for Person) — **not** discounted by the same reasoning; sized and protected on its own merits.

For Person specifically (the only class with both edge-case sources), the total target moves up slightly from DEC-027's initial ballpark to roughly **2500–3500 images**, explicitly composed of general OI + a real CrowdHuman allocation + ExDark's guaranteed floor. Vehicle, Two Wheeler, Chairs, Tables, and Animals keep DEC-027's lower-to-middle guidance, but must still preserve ExDark's guaranteed floor rather than letting ordinary volume crowd it out.

### Rationale

- Loh & Chan (2019), *Computer Vision and Image Understanding*, vol. 178, pp. 30-42 — the ExDark dataset's source paper — report that standard benchmarks including COCO contain "less than 2% low-light images," meaning the COCO-pretrained backbone has minimal relevant prior for low-light conditions specifically, unlike its strong prior for ordinary daylight object shapes.
- Shao et al. (2018), arXiv:1805.00123 — CrowdHuman's source paper — motivate the entire dataset by showing crowd scenarios and occlusion are "still under-represented in current human detection benchmarks," including COCO.
- The deployment use case (assistive navigation, incl. indoor/low-light environments and crowded public spaces) makes both conditions directly deployment-relevant, not incidental diversity.

### Alternatives Considered

- **Uniform per-class volume target with no condition-based split**: Rejected — risks under-provisioning exactly the conditions (dark environments, crowded spaces) most relevant to real-world assistive-navigation use, purely because they're a minority share of a class's total available pull.

### Consequences

- Exact per-class numeric splits (general OI vs. CrowdHuman vs. ExDark, per class) are still TBD — to be set when the acquire scripts are actually written.
- `docs/PROJECT.md` / future `docs/experiments.md` should note edge-case composition (% low-light, % crowd-dense) per class alongside raw counts, not just total image counts, since total count alone doesn't capture this distinction.

---

## DEC-029: Coffee Table Added for Tables; Blocker #5 Resolved

- **Date:** 2026-08-08
- **Status:** Accepted

### Context

TASKS.md Blocker #5 required verifying Open Images V7's native class name strings for Trash Bins and Tables before `acquire_openimages.py` could be written. Verified directly against the live OI V7 class hierarchy via `fiftyone.utils.openimages.get_classes()` (601 total classes) rather than guesswork.

### Decision

- **Blocker #5 resolved**: `"Waste container"` (Trash Bins) and `"Table"` (Tables) both confirmed as exact OI V7 class names.
- **Tables' Open Images primary source expanded** from `native_class: "Table"` to `native_class: ["Table", "Coffee table"]` — `"Coffee table"` confirmed to exist as a separate OI class. Matches the existing multi-class list pattern already used for Vehicle (`["Car","Bus","Truck"]`) and Two Wheeler (`["Motorcycle","Bicycle"]`).

### Rationale

Coffee tables are common in mall/lounge/waiting-area environments, which are part of this project's deployment context (README.md/PROJECT.md cite malls as a target environment). A detector trained only on dining-style "Table" would miss this common furniture form factor.

### Alternatives Considered

- **Also add `"Kitchen & dining room table"`** (confirmed to exist in OI's hierarchy too): Deferred — not requested. Can be added the same way later if needed.

### Consequences

- `config/classes.yaml` (`tables.primary.native_class`) and `config/datasets.yaml` (`open_images.native_classes_needed`) updated.
- `acquire_openimages.py` (not yet written) must pass both native class strings for the Tables class.
- TASKS.md Blocker #5 and the corresponding Stage 5.1 checklist item marked resolved.

---

## DEC-030: Raw Acquisition Buffer Multiplier (~1.3-1.4x), 3500 as General Final-Target Ceiling

- **Date:** 2026-08-08
- **Status:** Accepted (multiplier is a starting estimate, expected to be recalibrated)

### Context

DEC-027 established that acquisition pulls should be bounded via `max_samples` rather than unbounded, but didn't set a specific buffer size. Pulling exactly the final per-class target at acquisition time leaves no room for Stage 5.3 (box audit) and Stage 5.5 (mistakenness) rejecting some fraction of samples. Separately, 3500 (the top of DEC-028's Person range) is adopted as a general final-target ceiling for merged per-class pools wherever source volume allows it, rather than maintaining a bespoke target per class.

### Decision

Raw acquisition pulls (`max_samples`) are sized at roughly **1.3-1.4x** the intended final per-class/per-source target, for sources with enough volume to support it (Open Images, for all 7 primary classes). Smaller Roboflow-only niche-class sources remain uncapped/take-what's-available and are not subject to this multiplier — they're volume-constrained regardless.

### Rationale

Estimated compounding attrition through box audit (~5-10%, Open Images boxes are pre-verified/generally clean) and mistakenness curation (~5-15%, some flagged samples corrected rather than dropped) plus minor cross-source dedup loss lands around 20-25% total — a ~1.3-1.4x raw buffer should net close to the final target after cleaning.

### Alternatives Considered

- **Pull exactly the final target, top up later if short**: Rejected as the default — creates a slower iterate-acquire-check-repeat loop. A buffer aims to get it right in one pass, accepting some wasted pull if attrition is lower than estimated.

### Consequences

- **This multiplier is an unvalidated starting estimate**, not derived from real data — no class has been through box audit or mistakenness yet. It should be recalibrated using actual attrition numbers once the first class completes Stage 5.3 + 5.5, and that recalibration should be logged as a follow-up decision.
- Applies per-source (e.g. Open Images' share of a class's target), not to the class's overall merged total, since CrowdHuman/ExDark shares have their own separate volume logic (DEC-014's guaranteed floor, DEC-028's edge-case budget).

---

## DEC-031: Box-Shape Finding (Stairs, Elevator) — Polygon-Derived Annotations, Not True Bboxes

- **Date:** 2026-08-08
- **Status:** Accepted

### Context

Manual pre-audit (ahead of `box_audit.py`, Stage 5.3, being built) of `stairs_lusiz`, `stairs_hsatv`, and `elevator_status_s4lrk` found that although all three are labeled as object-detection format, their ground-truth boxes wrap tightly around the object's silhouette rather than sitting as a simple axis-aligned rectangle — consistent with annotations originally drawn as a polygon (or an unusually tight fit) and exported in bbox-shaped format regardless.

### Decision

- `stairs_lusiz` and `stairs_hsatv`: `audit_status: failed` (new status value — see below). Not discarded; relabeling is possible if the volume is worth recovering, just not prioritized right now.
- `stairs_i2yia` and `escalator_stairs` (Stage 5.1's other two Stairs sources, both `pending`, no box-shape issue observed) are prioritized over relabeling the two failed sources.
- `elevator_status_s4lrk`: stays `pending`, not failed — it has a genuine usable subset (people-in-elevator images with real detection-style boxes) alongside the same box-shape concern. Needs the real Stage 5.3 pass to separate usable from not, not a blanket accept/reject.
- **New `audit_status` value added**: `failed`, distinct from `rejected`. `failed` = didn't pass this initial look but is a relabel candidate, not permanently excluded. `datasets.yaml`'s schema comment updated accordingly.

### Rationale

Diagonal/irregular-silhouette objects (staircases especially) are the most likely to expose this pattern, since a true axis-aligned box around a diagonal shape should include visible empty space the annotations don't show. Recording this now — rather than only in `box_audit.py`'s eventual per-sample audit — flags it as a heuristic that script should specifically check for.

### Consequences

- `docs/preprocessing.md` holds the supporting detail (per-source findings log) and a note for `box_audit.py`'s future design.
- `config/datasets.yaml` and `config/classes.yaml` updated with `audit_status`/`audit_note` reflecting all of the above.

---

## DEC-032: Traffico-y1 Class List Resolved; Jeepney Secondary Consolidated to `jeep_hozhs`

- **Date:** 2026-08-08
- **Status:** Accepted (Jeepney-secondary-source choice superseded by DEC-037 — see below)

### Context

`traffico_y1` was flagged as a priority audit (full class list unknown, possible overlap with Vehicle/Two Wheeler/Tricycle) since DEC-017. Separately, Vehicle had two candidate Jeepney-focused Roboflow secondary sources (`me5_u6rvg`, `jeep_hozhs`) with unclear relative value.

### Decision

- **traffico-y1's full native class list**: `Jeepney`, `Motorcycle`, `null`, `Tricycle`, `tricycle`. Mapped as: `Jeepney` → `vehicle`, `Motorcycle` → `two_wheeler`, `{Tricycle, tricycle}` → `tricycle` (case-variant pair, both map to the same canonical class). `null` left unresolved (likely an empty/background class artifact, not a real label to map).
- Two Wheeler gains a new `secondary_providers` entry for traffico-y1's Motorcycle slice — previously Two Wheeler had no Roboflow secondary at all, only Open Images + ExDark.
- **`me5_u6rvg` deprioritized** (`audit_status: failed`) in favor of `jeep_hozhs`: substantial image overlap between the two, and the ~7k images `me5_u6rvg` offers isn't needed given Vehicle already draws from Open Images, ExDark, and traffico-y1's Jeepney slice. Not deleted, in case `jeep_hozhs` alone proves insufficient later.

> **Superseded (partial):** The `me5_u6rvg`-vs-`jeep_hozhs` choice was reversed by DEC-037 (2026-08-08) once traffico-y1 turned out to be permanently blocked and forking was declined — `me5_u6rvg` is active again, `jeep_hozhs` is benched. The traffico-y1 class-list resolution above is unaffected and still authoritative.

### Rationale

Resolves TASKS.md Open Blocker #2 (traffico-y1 audit) directly from the actual downloaded class list rather than guesswork. The Jeepney source consolidation avoids redundant near-duplicate volume that would just add cleaning-stage work for no representational benefit.

### Consequences

- `config/datasets.yaml` (`traffico_y1`, `me5_u6rvg`) and `config/classes.yaml` (`vehicle`, `two_wheeler`, `tricycle`) updated.
- TASKS.md Open Blocker #2 marked resolved.
- Dedup between traffico-y1's Jeepney slice and `jeep_hozhs` still needed at Stage 5.6 (both are Jeepney-focused, from different projects — possible image overlap not yet checked).

---

## DEC-033: Philippine-Context Representation as a Second Edge-Case Axis (extends DEC-028)

- **Date:** 2026-08-08
- **Status:** Accepted

### Context

DEC-028 established that a class's edge-case volume (conditions the COCO-pretrained backbone has little/no prior for, like ExDark's low-light slice) shouldn't be discounted by the transfer-learning volume reasoning that applies to ordinary-condition data. Reviewing Vehicle's sourcing (DEC-032) surfaced a second, distinct kind of edge case: **Jeepneys are a Philippine-specific vehicle type with no Open Images or COCO analogue at all** — unlike ordinary cars/buses/trucks, which Vehicle's Open Images primary source already covers well via COCO-style pretraining overlap.

### Decision

Extend DEC-028's framework: a class's edge-case budget can include a **geographic/cultural-representativeness axis**, not just a lighting-condition axis, whenever part of a class's real-world deployment distribution (Philippine streets, per this project's deployment context) has no representation in generically-sourced Western/global data. This axis is **not** subject to the transfer-learning volume discount, for the same underlying reason as DEC-028's low-light/crowd-density exemptions: the pretrained backbone has no relevant prior to lean on.

Concretely: Vehicle's Jeepney slice (traffico-y1 + `jeep_hozhs`) and Two Wheeler's Motorcycle-via-traffico-y1 slice are budgeted on their own terms, not folded into "Vehicle already overlaps COCO so needs less data" reasoning. Tricycle was already treated this way by construction (DEC-002/015 — it has no COCO analogue at all, was never subject to the discount in the first place).

### Rationale

Same logic as DEC-028, applied to a different kind of domain gap: COCO/Open Images are generically Western-sourced datasets and don't meaningfully feature Philippine-specific vehicle types. A detector that's only seen generic sedans/buses/trucks has no head start on recognizing a jeepney, regardless of how strong its general "vehicle" prior is.

### Consequences

- Vehicle and Two Wheeler's per-class volume targets (still TBD — deferred pending the student's own Open Images exploration, per prior conversation) should, when set, budget the Jeepney/Motorcycle-via-traffico-y1 slices separately from the Open Images general-condition slice.
- This framework should be checked against any future source that similarly fills a Philippine-specific gap (e.g. tricycle sources already fit this pattern).

---

## DEC-034: Potholes — Combine Two Pure-Pothole Sources, Deprioritize RDD2022, Dataset Ninja Currently Blocked

- **Date:** 2026-08-08
- **Status:** Accepted

### Context

Potholes was originally sourced as Dataset Ninja `pothole-detection` (primary, 665 images, single-class) + `road-damage-detector`/RDD2022 (secondary volume_topup, 47,420 images across 7 classes, requires filtering). The student found an additional Roboflow candidate (`pothole-vhmow`, pure pothole images, "a bit lower quality") and proposed combining it with the existing Dataset Ninja primary rather than picking one exclusively — reasoning that both are small, single-purpose sources and their combined volume will stay under 3500 regardless.

Separately, while verifying Dataset Ninja's actual download mechanism (see prior turn's research: it requires the `dataset-tools`/`supervisely` packages, delivers in Supervisely's native format rather than Pascal VOC/COCO as `datasets.yaml` previously assumed, and needs a local `sly.convert.to_pascal_voc()` step before FiftyOne can ingest it), an actual download attempt was made for `pothole-detection`. It failed: the resolved download link is Dropbox-hosted and returned "Link Temporarily Disabled" (a Dropbox free-tier rate limit), not real data. Installing `dataset-tools` also required `brew install libmagic` (a system-level dependency, not pip-installable) before it would even import, and downgraded `xmltodict` below FiftyOne's stated requirement (`fiftyone.utils.voc` still imported successfully in this environment, but the version conflict is real, not cosmetic).

### Decision

- **Potholes now has two co-primary sources**: Dataset Ninja `pothole-detection` + Roboflow `pothole-vhmow` (new `pothole_vhmow` entry in `datasets.yaml`), both "pure pothole" (no other classes to filter out).
- **`road-damage-detector` (RDD2022) is deprioritized** to fallback-only status — kept in config but not actively pursued, per the student's reasoning that it mixes potholes with crack/repair classes requiring filtering, which the two pure-pothole sources avoid entirely.
- **Dataset Ninja `pothole-detection` is marked `audit_status: blocked`**, not rejected — same "shelved, not discarded" treatment as the failed Stairs sources (DEC-031). Retry later; Dropbox rate limits are often temporary. Roboflow's `pothole-vhmow` is prioritized as the currently-working path.
- `datasets.yaml`'s `annotation_format` for both `dataset_ninja_*` entries corrected: native delivery is Supervisely format, not directly Pascal VOC/COCO as previously documented.

### Rationale

Combining two small, clean, single-purpose sources avoids RDD2022's filtering complexity for no volume benefit (665 + pothole-vhmow's size comfortably covers the sub-3500 target the student is aiming for). The Dataset Ninja blocker is treated as temporary infrastructure unavailability, not a quality judgment — but three independent friction points in one attempt (dependency conflict, missing system library, broken download link) are enough real-world evidence to deprioritize actively pursuing it right now in favor of the working Roboflow path.

### Consequences

- `dataset-tools`/`supervisely` are installed locally (plus `libmagic` via Homebrew) but **not yet added to `requirements.txt`** — deliberately withheld until Dataset Ninja's `pothole-detection` is confirmed actually retrievable; no point pinning a dependency chain for a source that isn't currently working.
- `pothole_vhmow` needs a pinned version before acquisition, same as every other unpinned Roboflow project (TASKS.md Blocker: version pinning).
- If Dataset Ninja's link comes back later, `dataset_ninja_pothole_detection`'s `audit_status` should flip back to matching the rest of the pipeline's audit flow — not a new decision, just an update.
- `utility-poles-44tzx` (the other Roboflow link the student found, containing both poles and potholes) is not yet added anywhere — still an open option for either Potholes or Pole, pending the student's call.

---

## DEC-035: `utility-poles-44tzx` Added as Dual Secondary (Pole + Potholes)

- **Date:** 2026-08-08
- **Status:** Accepted

### Context

While reviewing Roboflow pothole candidates (DEC-034), the student found `mahindra-university-ogyod/utility-poles-44tzx`, which contains both poles and potholes in the same project.

### Decision

Add `utility_poles_44tzx` as a secondary source for **both** Pole and Potholes — Pole previously had only one source (`pole-detection-z76mb`, DEC-023), so this also gives it a second, diversity-role provider for the first time.

### Rationale

No cost to including it for both classes it actually contains — a single acquisition pull serves two canonical classes.

### Consequences

- `config/datasets.yaml` and `config/classes.yaml` updated (`pole.secondary_providers`, `potholes.secondary_providers`).
- Native class label strings not yet verified — needs confirmation during Stage 5.3 box audit, same as `escalator_stairs`' multi-class handling.
- Needs a pinned version before acquisition, same as every other unpinned Roboflow project.

---

## DEC-036: Source Splits Discarded on Acquisition — One Final Split at Stage 5.8

- **Date:** 2026-08-08
- **Status:** Accepted

### Context

Every source (Open Images, Roboflow projects, etc.) ships with its own train/valid/test split. PLAN.md's Stage 5.8 already specifies a "source-stratified split" and notes the validation split must retain "proportional source representation per class" for Hailo calibration purposes — which implies a single, deliberate final split computed after merge, not inherited from sources — but this hadn't been stated as an explicit rule acquire/convert scripts need to follow.

### Decision

Every source's pre-existing train/valid/test assignment is **discarded at acquisition time**. All pulled images for a given (source, class) are treated as one flat pool. The only train/val/test assignment that matters is the one `scripts/build/split.py` (Stage 5.8, not yet built) computes on the fully merged, deduped, and curated pool.

### Rationale

- Sources use inconsistent split ratios — concatenating their splits directly would produce an incoherent final split.
- Stage 5.6 cross-source dedup happens after acquisition; a near-duplicate pair spanning two sources needs to land in the same final split, which is only guaranteed if splitting happens once, globally, after merge.
- The Hailo-calibration representativeness requirement (val split proportional per class per source) can only be engineered deliberately across the whole merged pool.

### Consequences

- Acquire scripts (Roboflow, Open Images, etc.) should pull from *all* of a source's available splits, not just one — e.g. `acquire_openimages.py` should pull train+validation+test from Open Images, not just the `validation` split `explore_dataset.py` uses for a quick preview. Restricting to one split at acquisition time would just shrink the available pool for no benefit, since the split label gets discarded anyway.
- Resolves the `jeep_hozhs` version-pinning question — differences in a candidate version's split proportions are irrelevant to source-selection, since splits are rebuilt from scratch regardless.
- `scripts/build/split.py`'s design (Stage 5.8, unbuilt) is the one place split ratios and stratification logic actually live.

---

## DEC-037: `jeep_hozhs` Benched, `me5_u6rvg` Reactivated (Multi-Class); New `audit_status: benched`

- **Date:** 2026-08-08
- **Status:** Accepted (supersedes the Jeepney-source part of DEC-032)

### Context

`traffico_y1` remains blocked — the API confirms zero generated versions despite 8,807 raw images existing in the project (per direct GUI check, 2026-08-08). Forking it into the student's own workspace was identified as a viable fix (DEC-018r-compatible — see the prior turn's discussion of why forking here doesn't reverse DEC-018r), but the student declined due to uncertainty over whether forking consumes Roboflow account credits. That uncertainty wasn't resolved either way — the decision is to avoid the risk rather than confirm the mechanism.

Separately, the student reviewed `me5_u6rvg` (deprioritized by DEC-032) directly and found it has the same multi-class shape as `traffico_y1`: Ambulance, Cars, Jeepney, Motorcycle, null — not a plain single-class Vehicle source as it was configured.

### Decision

- **`jeep_hozhs` is benched** (`audit_status: benched`, new status value — see below). Nothing wrong with the source; it's being swapped out because `me5_u6rvg` covers the same Jeepney need plus additional useful classes.
- **`me5_u6rvg` is reactivated** (`audit_status: pending`, `pinned_version: 1`, its only available version, 7,243 images total). Restructured with a `native_class_filter` dict, same pattern as `traffico_y1`: `Ambulance`/`Cars`/`Jeepney` → `vehicle`, `Motorcycle` → `two_wheeler`, `null` unresolved.
- **New `audit_status: benched`** added to the schema, distinct from `failed`. `failed` = a quality problem was found (e.g. DEC-031's box-shape issue). `benched` = nothing wrong with the source, just not the currently-chosen option for strategic/logistics reasons. Keeping them distinct matters because revisiting a `failed` source means fixing a problem, while revisiting a `benched` one just means reconsidering a choice.

### Rationale

`traffico_y1` being genuinely blocked (not just unpinned) removes it as Vehicle's near-term Jeepney source. `me5_u6rvg` already contains the same Jeepney content `jeep_hozhs` was chosen for, plus Ambulance/Cars/Motorcycle the project can use — better fit than continuing to wait on a blocked source or accept the credit-consumption uncertainty of forking it.

### Consequences

- `me5_u6rvg`'s Vehicle-specific image count is **not** 7,243 — that figure is the total across all 5 of its classes. The actual Ambulance+Cars+Jeepney subset is unknown until the project is acquired and class-filtered. Don't cite 7,243 as Vehicle volume.
- Two Wheeler now has two candidate Roboflow secondaries for Motorcycle: `me5_u6rvg` (active) and `traffico_y1` (blocked, kept for whenever/if it's unblocked).
- If `traffico_y1` ever becomes downloadable (owner generates a version, or the credit-consumption question gets resolved favorably), it becomes a second Jeepney/Motorcycle/Tricycle source rather than a replacement for `me5_u6rvg` — no need to re-bench anything at that point.
- `docs/DECISIONS.md`'s DEC-032 entry annotated with a partial-supersession note pointing here, following the same pattern used for DEC-002.

---

## DEC-038: "Two Wheeler" Split into Motorcycle and Bicycle

- **Date:** 2026-08-08
- **Status:** Accepted (supersedes the Two Wheeler framing of DEC-002; carries forward the Motorcycle-specific parts of DEC-032/DEC-037 under a renamed key)

### Context

The original schema (DEC-002) merged Motorcycle and Bicycle into a single `Two Wheeler` class (ID 2). Both primary sources (Open Images, ExDark) have always labeled Motorcycle/Motorbike and Bicycle as separate native classes — the merge was a labeling-time choice in this project's schema, not a sourcing constraint.

Discussing whether to keep them merged surfaced two independent arguments for splitting, raised from different angles: motorcycles are fast and carry materially higher kinetic-collision risk than bicycles (the student's original motivation); bicycles are also close to silent, so unlike a motorcycle, the user has no independent auditory cue to rely on if the device doesn't flag it. The only concrete downstream behavior difference identified is TTS wording ("motorcycle" vs. "bicycle"), but that distinction directly serves the device's core purpose — communicating detected hazards to a visually impaired user — which the student identified as the deciding factor regardless of the schema-churn cost.

### Decision

- **`Two Wheeler` (ID 2) is renamed to `Motorcycle`**, slot reused. `native_class` narrowed to Open Images `"Motorcycle"` / ExDark `"Motorbike"` only. Existing Roboflow secondary mappings (`me5_u6rvg`, `traffico_y1` — both already filtered to `"Motorcycle"` specifically) carry over unchanged under the new key.
- **`Bicycle` is added as a new class at ID 15** (appended at the end, not inserted mid-schema) — sourced from Open Images `"Bicycle"` and ExDark `"Bicycle"`. No new acquisition source needed; both were already being pulled and discarded under the old merged mapping.
- `nc`: 15 → **16**.

### Rationale

- The two arguments are complementary, not competing: a merged "two-wheeler" warning blurs exactly the two things that matter for how the user should react — how urgently to react (motorcycle: fast, high consequence) and whether they'd otherwise know it's there at all (bicycle: silent, low self-detectability).
- ID slot reuse for Motorcycle (keeps ID 2) plus appending Bicycle at the end (ID 15) follows the same low-ripple pattern as DEC-015/DEC-021 — avoids renumbering IDs 3–14.
- Zero incremental acquisition cost: both sources already label these as distinct native classes, so this is a config/mapping change, not a new-source-identification problem.

### Alternatives Considered

- **Keep merged `Two Wheeler`**: Rejected — the student judged the hazard/urgency distinction (and the TTS wording it drives) important enough to the device's core use case to justify the schema churn.
- **Insert `Bicycle` in schema order near Vehicle/Motorcycle** (e.g., new ID 3, shifting `Pole`…`Pedestrian Lane` down by one): Rejected — would ripple through every other class's ID for no benefit; appending at the end achieves the same outcome with a much smaller blast radius.

### Consequences

- `nc` changes from 15 to 16 everywhere referenced: `config/classes.yaml`, `scripts/utils/config_loader.py` (`EXPECTED_NC`, `CANONICAL_NAMES`), `AGENTS.md`, `README.md`, `docs/PROJECT.md`.
- `config/datasets.yaml`: `exdark_to_canonical_class_map`'s `Motorbike`/`Bicycle` entries split (were both `two_wheeler`, now `motorcycle` and `bicycle` respectively); `traffico_y1` and `me5_u6rvg`'s `canonical_classes`/`native_class_filter` keys renamed `two_wheeler` → `motorcycle`.
- Any future acquisition/conversion script must key off `motorcycle`/`bicycle`, not the retired `two_wheeler` key.
- Hailo runtime mapping shifts accordingly: Bicycle becomes runtime ID 16 (after the +1 background-class offset).

---

## DEC-039: Trash Bins Secondary Source Benched — Open Images Primary Judged Sufficient

- **Date:** 2026-08-08
- **Status:** Accepted

### Context

Blocker #1 (open since Phase 0) — no secondary Roboflow project had ever been identified for Trash Bins; `classes.yaml`'s `secondary_providers` entry was left as a `"TBD"` placeholder. Rather than keep searching for a project with no clear evidence it's needed, the student chose to move on and revisit only if warranted later.

### Decision

Bench the search for a Trash Bins secondary source. Open Images (`"Waste container"`, already verified per DEC-029) remains the sole source for now. The `"TBD"` placeholder in `classes.yaml` is marked `audit_status: benched`, not deleted — the slot stays reserved for later.

### Rationale

Same benched-vs-failed distinction established in DEC-037: nothing is wrong with the idea of a secondary source, it's just not currently justified. No evidence yet that the Open Images primary is insufficient, and continuing to search preemptively delays other pipeline work — consistent with DEC-027's bounded-effort philosophy.

### Alternatives Considered

- **Keep searching for a Roboflow secondary now**: Rejected — no evidence of insufficiency yet; premature effort.
- **Delete the secondary_providers entry entirely**: Rejected — keeping the benched placeholder preserves the reminder that a secondary was considered, without implying it's an open TODO blocking other work.

### Consequences

- Blocker #1 marked benched (not resolved) in `TASKS.md`'s Open Blockers table.
- Revisit trigger: post-acquisition/curation evaluation (Stage 5.1 or later) shows Open Images `"Waste container"` volume/diversity is insufficient for the target range.
- If revisited, the same Roboflow-search workflow used for other classes applies — no new process needed.

---

## DEC-040: `acquire_openimages.py` Buffer Factor Pinned to 1.35, Per-Class Export Layout

- **Date:** 2026-08-10
- **Status:** Accepted

### Context

Building `scripts/acquire/acquire_openimages.py` (Stage 5.1, highest-priority acquisition script — primary source for 8 of 16 classes) required turning two already-decided but still-abstract policies into concrete code: DEC-027's "bounded via `max_samples`, not unbounded downloads" and DEC-030's "~1.3–1.4x raw acquisition buffer over the post-merge cap" (a range, not a pinned number). A concrete `max_samples` value has to be computed per class before any FiftyOne Zoo call can be made.

### Decision

- `BUFFER_FACTOR = 1.35` (the midpoint of DEC-030's 1.3–1.4x range), applied as `max_samples = ceil(classes.yaml cap * BUFFER_FACTOR)` for every class whose `primary.source` is `open_images`.
- Each canonical class is pulled as a separate FiftyOne Zoo dataset (filtered to that class's native Open Images label(s)) and exported independently to `dataset/raw/open_images/<class_key>/` as `images/` + a COCO-style `labels.json`, matching `datasets.yaml`'s `annotation_format: coco_style` for this source.
- No cross-class dedup at this stage — an image containing both a Chair and a Table may be exported into both `chairs/` and `tables/`. Per DEC-036/Stage 5.6, dedup happens later at merge time, not acquisition.
- Native Open Images class names are NOT remapped to canonical class IDs here — `labels.json` keeps every detection FiftyOne exports for that image (any class present in-frame, not just the filtered target), same as raw source data. Remapping is explicitly Stage 5.2's job.

### Rationale

1.35 is a defensible midpoint rather than either boundary of DEC-030's range, and pinning one number (vs. leaving it as a runtime flag) keeps the script's behavior reproducible and simple to reason about for a first acquisition run — it can be revisited via a code change (with a note here) if Stage 5.3/5.6 evaluation shows the buffer was too tight or too generous. Per-class export directories were chosen over one merged Open Images pull because it keeps Stage 5.1's output directly traceable to the per-class targets in `classes.yaml`, at the cost of some duplicate raw storage for multi-class images — an acceptable tradeoff since raw storage is disposable and dedup is already a planned later stage.

### Alternatives Considered

- **Expose buffer factor as a CLI flag instead of a pinned constant**: Rejected for the first run — adds a decision surface with no evidence yet that the default needs tuning per class. Can be added later if a specific class's volume proves wrong.
- **One merged Open Images pull across all 8 classes, split into per-class subsets in code**: Rejected — more complex to write and reason about than 8 independent zoo pulls, for a dataset size where the simplicity is worth more than the minor download-time savings.
- **Remap to canonical class names/IDs during export**: Rejected — would duplicate Stage 5.2's job and violate the stage boundary the pipeline is designed around (PLAN.md: acquisition vs. conversion are separate stages).

### Consequences

- `scripts/acquire/acquire_openimages.py` is built, smoke-tested end-to-end (a real 3-image pull + COCO export verified manually, then cleaned up — no leftover data or FiftyOne datasets from testing).
- Running it for real (`python3 scripts/acquire/acquire_openimages.py`) will pull up to 6,750 images per class (8 classes) — a substantial download; the one-time ~4.8GB Open Images metadata cache is now already warmed on this machine from testing, so subsequent pulls (this real run, and any future re-runs) won't repeat that download.
- `TASKS.md` Stage 5.1 checklist updated; `acquire_crowdhuman.py`, `acquire_exdark.py`, `acquire_datasetninja.py`, `acquire_roboflow.py` remain unbuilt.

---

## DEC-041: CrowdHuman, ExDark, and Dataset Ninja Sources — Manual Download, Scripts Handle Parsing Only

- **Date:** 2026-08-10
- **Status:** Accepted

### Context

`acquire_openimages.py` (DEC-040) and the Roboflow tooling both build on real SDKs/APIs (FiftyOne Zoo, Roboflow SDK) with auth, resumability, and versioning built in. The three remaining unbuilt acquisition scripts — CrowdHuman, ExDark, and Dataset Ninja (`pothole-detection`, `road-damage-detector`) — were reconsidered before writing any code, checking each source's actual distribution mechanism directly rather than assuming one:

- **CrowdHuman**: 7 separate files (3 train zips, 1 val zip, 1 test zip, 2 `.odgt` annotation files) mirrored across Baidu Drive and Google Drive — no API. The Baidu test-zip link is additionally gated behind a fetch code.
- **ExDark**: a single ~1.5GB static archive linked from the GitHub README — not an API.
- **Dataset Ninja**: `dtools.download()` (the `dataset-tools` SDK) is a thin wrapper around what turned out to be a Dropbox-hosted archive — already observed failing for real this session (rate-limited Dropbox link returned an HTML "link disabled" page instead of data, for `pothole-detection`). The Dataset Ninja website also exposes a plain "Download" button per dataset that bypasses the SDK entirely.

### Decision

- CrowdHuman, ExDark, and both Dataset Ninja sources are downloaded **manually** by the student, not by a script. For Dataset Ninja specifically, download directly from the website's per-dataset "Download" button rather than via `dtools.download()`.
- CrowdHuman: only `CrowdHuman_train01.zip`, `CrowdHuman_train02.zip`, `CrowdHuman_train03.zip`, `CrowdHuman_val.zip`, `annotation_train.odgt`, and `annotation_val.odgt` need to be pulled. `CrowdHuman_test.zip` is skipped entirely (avoiding its Baidu fetch-code gate) since DEC-036 already discards every source's original split and rebuilds one stratified split at Stage 5.8 — a held-out test set from the source would just get pooled back in anyway.
- `acquire_crowdhuman.py`, `acquire_exdark.py`, and `acquire_datasetninja.py` are rescoped: each assumes its source's raw files are already present under `dataset/raw/<source>/` (placed there manually by the student), and its job starts at parsing/validating/converting annotations — not fetching anything. `bbox_utils.py` already contains the ExDark- and CrowdHuman-specific box-conversion logic these scripts will use.

### Rationale

A script only pays for itself when it saves real, repeated effort or buys reproducibility/resumability an API already gives for free. None of these three cases qualify — each is a single, unchanging static archive pulled exactly once, so scripting the download would mean writing and maintaining file-host-specific logic (Google Drive's large-file confirmation flow, Baidu Drive's fetch-code, etc.) purely to automate a one-time click. Worse, doing so risks adding a fragile dependency layer on top of an already-fragile host without adding real robustness — exactly what happened when `dataset-tools` was installed for Dataset Ninja (DEC-034's audit note): it required a system-level `libmagic` fix, downgraded `xmltodict` below FiftyOne's stated floor, and still failed on a rate-limited host it couldn't do anything about.

### Alternatives Considered

- **Script all five acquisition sources uniformly, including these three, for pipeline consistency**: Rejected — consistency for its own sake isn't worth added fragile, single-purpose download code. The acquisition scripts stay more valuable doing something genuinely hard to do by hand (parsing/converting inconsistent annotation formats), not download automation for its own sake.
- **Keep using `dtools.download()` for Dataset Ninja, on the theory the Dropbox rate limit was temporary**: Rejected — the website's own direct download button reaches the same content without the SDK dependency, so there's no reason to keep routing through `dataset-tools` for this.

### Consequences

- `TASKS.md`'s Stage 5.1 checklist entries for `acquire_crowdhuman.py`, `acquire_exdark.py`, and `acquire_datasetninja.py` updated to reflect the narrowed scope.
- `config/datasets.yaml`'s `download_strategy`/`download_command` fields for `crowdhuman`, `exdark`, `dataset_ninja_pothole_detection`, and `dataset_ninja_road_damage_detector` updated to state the manual boundary explicitly, including exact files needed and target `dataset/raw/<source>/` paths.
- The student needs to manually download and place files before the corresponding acquire script can be run for each of these three sources — this is now a prerequisite step, not something `acquire_*.py` will do on its own.

---

## DEC-042: Per-Class Image Ceiling Set at 4,500 (Floor 1,500, 3:1 Ratio Invariant)

- **Date:** 2026-08-10
- **Status:** Accepted
- **Related:** DEC-002 (class schema), DEC-025–DEC-028 (sizing reasoning chain). Supersedes DEC-030's unvalidated "3,500 post-merge ceiling" with a literature-grounded floor/ratio derivation instead; DEC-030's 1.3–1.4x raw-acquisition buffer concept still stands, now applied against this decision's 4,500 hard cap rather than the old 3,500/5,000 figures.

### Context

`config/classes.yaml` currently leaves several classes uncapped (`cap: null`: Pole, Stairs, Escalator, Doors, Tricycle, Elevator, Pedestrian Lane) while the eight Open Images-primary classes are capped at a round `5000`. Neither number was derived from anything — DEC-030 flagged its own 3,500 post-merge ceiling as an "unvalidated estimate." Before writing the classes that consume this cap (Stage 5.2 conversion, Stage 5.4 `cap_per_class.py`), the per-class ceiling needed an actual basis, since YOLOv8 provides no built-in class-balancing mechanism to fall back on: classification uses plain BCE, Distribution Focal Loss operates on box regression (not classification), `fl_gamma` defaults to 0, and there is no built-in class weighting. YOLOv8's architectural improvements (anchor-free detection, task-aligned assignment) address foreground–background imbalance, not foreground–foreground imbalance across our 16 classes.

Left uncapped, collection would grow the seven Open Images-backed classes past 6,750 while the nine source-constrained classes cap at roughly 1,300–3,100 images — exceeding a 6:1 max-to-min ratio.

### Decision

Per-class training set is bounded at **1,500 images (floor)** and **4,500 images (hard cap)**. No class in `config/classes.yaml` may be uncapped. The image cap is a guardrail; the operative stop condition is instance count.

The ceiling is relative, not absolute: 4,500 is not independently derived, it is 3× the 1,500 floor, from a max-to-min ratio target of ~3:1. If the realized floor for any class lands below 1,500, the ceiling recomputes as 3× the realized floor — **the invariant is the ratio, not the value.**

Enforcement rule:

```
floor              = 1500 images
hard_cap           = 4500 images
instance_target    = 10000 instances   # 6000 for small/hard classes
stop_condition     = whichever of {hard_cap, instance_target} binds first
ratio_invariant    = max(class_images) / min(class_images) <= 3.0
uncapped_allowed   = false
```

Recompute `hard_cap` as `3 × min(realized_class_images)` once Stage 5.2 filtering closes and the true floor is known.

Expected distribution: most classes land 1,500–3,000; a few 3,000–4,500; none above. Dense classes (Person via CrowdHuman, 20+ instances/image) hit the instance target well before the image cap; sparse classes (~1 instance/image) will never approach it. If many classes are pressing against 4,500, that's a signal instance-based capping isn't being applied correctly — not a sizing outcome to accept.

### Rationale

**Why cap at all:** the standard algorithmic remedies for class imbalance are unreliable for this architecture — benchmarking on foreground–foreground imbalance (arXiv:2403.07113) found sampling and loss weighting counterproductive in one-stage detectors, often reducing overall mAP. Dataset composition at collection time is therefore the only reliable lever available.

**Why 3:1:** heuristic, and documented as such. Ratio is the correct variable to constrain once every class clears absolute sufficiency, and all measured absolute thresholds sit at or below our 1,500 floor: ~6,000 instances for small hard objects (Rabbi et al. 2020), ~210–325 labels (Song et al. 2025), ~500 images (Apeinans et al. 2024). Past that point, relative spread is what distinguishes a good distribution from a bad one. The specific ratio value is not validated — published long-tail work operates at 100:1 and above, and the 1:1–5:1 band is uncharacterized for YOLOv8-class detectors. Recorded as a design constraint, with empirical characterization left as future work.

**Why not higher:**
- Post-knee returns measured at 0.03–0.11% AP per added label (Song et al. 2025).
- Head-class volume does not rescue system performance: YOLO-OD (Wang et al. 2024) carried 22,318 car and 53,756 person instances and still reported 42.02 mAP50 on YOLOv8-s, bottlenecked by its ~1,200–1,700-instance classes.
- Past a source's variety, additional images are near-duplicates, raising train/val leakage risk — Rohe et al. 2024 observed a performance *drop* going 300→600 images on a non-diverse pool.
- Capacity is fixed by the deployment target (YOLOv8s train, YOLOv8n on Hailo-8).

### Alternatives Considered

- **Leave classes uncapped, cap only where sourcing is naturally volume-constrained**: Rejected — this is the status quo being replaced; produces the >6:1 ratio blowout described in Context.
- **Algorithmic class-balancing (sampling/loss reweighting) instead of dataset-level capping**: Rejected — arXiv:2403.07113's benchmarking found these counterproductive for one-stage detectors like YOLOv8, often reducing overall mAP rather than helping.
- **Adopt long-tail-literature ratios (100:1 or looser)**: Rejected — that literature targets a different regime (extreme long-tail with many classes below absolute sufficiency); every class here already clears the cited absolute-sufficiency thresholds at the 1,500 floor, so the long-tail band doesn't characterize our situation.
- **Keep a higher round cap (~5,000–6,750, the prior working numbers)**: Rejected on diminishing-returns evidence (Song et al. 2025's post-knee AP/label figures), near-duplicate/leakage risk at low source diversity (Rohe et al. 2024), and YOLO-OD's demonstration that raw head-class volume alone doesn't fix tail-class-bottlenecked performance (Wang et al. 2024).

### Consequences

- `config/classes.yaml`'s `cap` field needs updating for all 16 classes: the seven currently `null` (Pole, Stairs, Escalator, Doors, Tricycle, Elevator, Pedestrian Lane) become `4500`; the eight currently `5000` (Person, Vehicle, Motorcycle, Bicycle, Animals, Chairs, Tables, Trash Bins) and Potholes' `5000` drop to `4500`. The schema comment documenting `cap` (`null = uncapped`) needs updating to reflect `uncapped_allowed: false`. **Not yet applied as of this entry** — recorded first per the student's request, config edit still pending.
- `scripts/acquire/acquire_openimages.py` (DEC-040) needs **no code change** — `compute_max_samples()` already reads `cap` directly from `classes.yaml` and multiplies by DEC-030's buffer factor, so once the config is updated the raw per-class pull target recalculates automatically from `ceil(5000 × 1.35) = 6750` to `ceil(4500 × 1.35) ≈ 6075`.
- Stage 5.2 (box audit) and Stage 5.4 (`cap_per_class.py`, not yet built) must implement the instance-based `stop_condition` and `ratio_invariant` check described above — image-count capping alone is not sufficient enforcement of this decision.
- **Open dependency**: per-class instances-per-image must be measured on the merged post-filter set before `hard_cap` can be recomputed as `3 × min(realized_class_images)`. Until then, the instance-based stop condition is estimated from published source characteristics rather than this project's own data.
- **Applied 2026-08-11**: `config/classes.yaml`'s `cap` field updated for all 16 classes — the seven previously `null` and the nine previously `5000` are now uniformly `4500`; the schema comment at the top of the per-class config block updated to state `uncapped_allowed: false`. `python3 scripts/utils/config_loader.py` re-validated clean (`nc=16`). Confirmed `acquire_openimages.py` needed no code change: `--dry-run` shows `max_samples` recalculated automatically from `ceil(4500 × 1.35) = 6075` per class.

---

## DEC-043: `acquire_openimages.py` Filters `IsDepiction` and `IsGroupOf` Detections

- **Date:** 2026-08-11
- **Status:** Accepted
- **Related:** DEC-040 (script this modifies)

### Context

The student's own scratch exploration (`fiftyone_test.ipynb`, not part of the tracked pipeline) surfaced `IsDepiction` — an Open Images per-detection attribute distinguishing real photographed objects from depictions (drawings, icons, statues, etc.) — while experimenting with FiftyOne's `ViewField` API directly. Separately, the student noted their already-deployed on-device algorithm determines "a group of a class" by counting multiple individual instance detections within an area of interest at inference time — which depends on training data carrying individual per-instance boxes. Open Images' `IsGroupOf` attribute marks the opposite: a single box drawn around a *cluster* of same-class instances rather than one. Keeping those would teach the model that a group of objects is one object, directly conflicting with the on-device counting logic.

### Decision

`pull_class()` in `scripts/acquire/acquire_openimages.py` now filters both attributes out before export:
```python
view = dataset.filter_labels(
    LABEL_FIELD,
    (F("IsDepiction") == False) & (F("IsGroupOf") == False),
)
view = view.match(F(f"{LABEL_FIELD}.detections").length() > 0)
```
`filter_labels` prunes non-matching detections from each sample's label list without removing samples; the follow-up `.match()` then drops any sample left with zero detections after that pruning (an image whose only relevant detection was a depiction/group-of box is otherwise useless for training). `actual_count`, the zero-check, and the export call all now operate on this filtered `view` rather than the raw `dataset`; `dataset.delete()` (cleanup of the underlying persisted FiftyOne dataset, not the view) is unaffected.

Geolocation-based filtering was also investigated as a possibility (raised by the student from FiftyOne's docs) and **rejected** — checked directly against `fiftyone/utils/openimages.py`'s importer source, which contains no latitude/longitude/geo handling at all. Open Images carries no per-image geotag data via FiftyOne's loader, so there is nothing to filter on. The Philippine-context representation need that prompted the question is already served by DEC-033's dedicated-source strategy (Roboflow Jeepney/Tricycle projects), not a lever open_images acquisition has available.

### Rationale

Both filters directly serve data-quality/training-signal correctness, not just volume: depictions are non-photographic content that don't represent real-world detection targets, and group-of boxes contradict the individual-instance labeling this project's downstream on-device logic requires. Applying both at acquisition time (Stage 5.1) rather than later in the pipeline keeps `dataset/raw/` free of images that would need re-filtering or re-export downstream.

### Alternatives Considered

- **Filter only `IsDepiction`, leave `IsGroupOf` for a later stage**: Rejected once the student explained the on-device group-counting algorithm — the conflict with group-of boxes is a correctness issue for this project specifically, not a nice-to-have, so there's no reason to defer it.
- **Geolocation filtering for Philippine-context representation**: Rejected — no geo metadata exists on Open Images samples via FiftyOne's loader to filter on; the need is already met a different way (DEC-033).

### Consequences

- Verified for real, not just read through: ran the edited `pull_class()` against a real 30-image Chair pull (seed 42) and compared against an unfiltered baseline pull of the same 30 images. Baseline: 463 total detections, 16 flagged `IsDepiction`, 15 flagged `IsGroupOf`. Filtered export: 29 images (1 fully emptied by the filter, correctly dropped), 432 annotations — `463 − 432 = 31`, exactly matching `16 + 15` with no overlap between the two flags in this sample. Test artifacts cleaned up afterward (export directory removed, FiftyOne test datasets deleted).
- `max_samples` (the raw pull request size, per DEC-030/DEC-042) is unchanged — it's a pre-filter request size, so some fraction of requested samples will now always be dropped post-filter. Not currently a problem (buffered well above the per-class cap already), but worth remembering if a class's actual yield ever looks short.
- No `IsOccluded`/`IsTruncated`/`IsInside` filtering added — not raised by the student, left as a possible future refinement, not decided here.

---

## DEC-044: `acquire_openimages.py` Corrected to Pull All Three Open Images Splits

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-036 (the rule this corrects an implementation gap against), DEC-040 (script this modifies)

### Context

While starting `acquire_roboflow.py`, re-checked DEC-036's Consequences section, which explicitly states acquire scripts "should pull from *all* of a source's available splits... e.g. `acquire_openimages.py` should pull train+validation+test from Open Images, not just the validation split." `acquire_openimages.py` as built (DEC-040) never followed this — it hardcoded `ZOO_SPLIT = "train"` and only ever pulled that one split. Confirmed directly (`OpenImagesV7Dataset().supported_splits`) that `train`, `test`, and `validation` are all available.

### Decision

`ZOO_SPLIT = "train"` replaced with `ZOO_SPLITS = ("train", "validation", "test")`; `pull_class()` now passes `splits=list(ZOO_SPLITS)` instead of `split=ZOO_SPLIT`. Tested directly first and found that FiftyOne applies `max_samples` **per split**, not as a combined total, when `splits` (plural) is used (a 5/5/5 test came back as 15 samples, tagged by split) — so `pull_class()` now divides the buffered total across the three splits: `per_split_max_samples = ceil(max_samples / 3)`.

### Rationale

DEC-036 already decided this; this is a correctness fix bringing the implementation in line with a standing decision, not a new judgment call. Splitting the buffer evenly across all three rather than pulling only `train` maximizes the available pool per class at no cost, since the split label is discarded anyway.

### Consequences

- Verified for real: a 15-sample pull correctly returned 5 from each split (visible via the Zoo loader's own per-split download log), 15 total after the DEC-043 quality filter, matching the DEC-030/042 buffered-total math (`total = per_split × 3`, not `per_split` alone). Test artifacts cleaned up.
- `compute_max_samples()` still returns the same buffered *total* per class (unchanged) — the per-split division happens inside `pull_class()`, not exposed as a separate config value.
- Actual yield per class may still land under the buffered target if validation/test are too small for a given class (both are much smaller than `train` in Open Images) — expected and already handled by the existing per-class report, not a new failure mode.

---

## DEC-045: `acquire_roboflow.py` Built — Eligibility Rule and Split Handling

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-018r (Local Pull-and-Resolve), DEC-036 (split handling)

### Context

11 of `config/datasets.yaml`'s 16 `roboflow_projects` entries are ready to pull (pinned version, `audit_status: pending`); the other 5 are `failed`, `benched`, or `blocked` (including `traffico_y1`, which is `pending` but still has `pinned_version: null`). Needed a single, correct eligibility rule rather than hardcoding which 11 to pull.

### Decision

- Eligibility rule: `audit_status == "pending" AND pinned_version is not None`. Verified this selects exactly the intended 11 and skips the intended 5, with each skip reported with its reason (not silently dropped).
- Each eligible project downloads via `project.version(pinned_version).download(download_format, location=..., overwrite=True)` to `dataset/raw/roboflow_projects/<project_key>/`, reusing `parse_workspace_project()` from `list_roboflow_versions.py` rather than re-parsing URLs.
- Unlike `acquire_openimages.py`, there's no `max_samples` bounding here — Roboflow doesn't support partial/bounded version downloads, and a pinned version is already a fixed, curated size. Whatever `train/valid/test` folders the SDK produces are kept as-downloaded; DEC-036 already discards that split assignment at Stage 5.8, not at acquisition.

### Rationale

A config-driven eligibility rule (vs. a hardcoded list) means `datasets.yaml` stays the single source of truth — reclassifying a project's `audit_status` later automatically changes what this script pulls next run, no script edit needed.

### Consequences

- Verified for real, not just read through: `--dry-run` confirmed the 11/5 split matches the manual audit exactly; a real pull of the smallest eligible project (`cv_project_hovyc`, Doors) completed successfully — 1,341 images, correct `yolov8` folder structure, report written to `dataset/reports/acquire_roboflow_report.json`. This data is real Stage 5.1 output, not a test artifact — left in place.
- Environment note: `roboflow` (and `dataset_tools`) had dropped out of the `second-vision` conda env since the 2026-08-08 session (only `fiftyone` persisted) — reinstalled `roboflow` per `requirements.txt`'s existing pin. Same numpy/opencv downgrade as documented before (2.4.6→2.3.5, 4.14→4.10.0.84); reverified fiftyone+roboflow+numpy+opencv import together cleanly.
- Remaining 10 eligible projects not pulled yet — left for the student to run when ready (`python3 scripts/acquire/acquire_roboflow.py`), same posture as `acquire_openimages.py`'s full 8-class run.

---

## DEC-046: Stage 5.2 "Intermediate Schema" Defined; `acquire_exdark.py` Built

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-036 (split handling), DEC-041 (parse-only scope), DEC-012 (vbox/visible-extent policy)

### Context

The student placed ExDark's manually-downloaded archive under `dataset/raw/exdark/` and surfaced a third-party reference script (`wraphex/ExDark2Yolo`, github.com/wraphex/ExDark2Yolo) — already cited as `parser_reference` in `datasets.yaml`, so a legitimate resource, not a random find. But `datasets.yaml` also says to "adapt into shared intermediate schema rather than direct-to-YOLO," and that schema was never actually defined anywhere in the project (checked: the phrase appears exactly once, with no spec). Needed a concrete definition to write the converter at all. Separately, the reference script as published can't be run unmodified against this project:

1. **Bakes in its own train/test/val split** — conflicts with DEC-036 (source splits discarded at acquisition; one split computed at Stage 5.8).
2. **Emits all 12 ExDark classes with its own positional class ids** — this project only maps 8 to canonical classes (`exdark_to_canonical_class_map`); Boat/Bottle/Bus/Cup have no canonical slot.
3. **No box clamping** — a known issue in the reference script; `bbox_utils.py` (built DEC-025, unused until now) already covers this via `validate_bbox`/`clip_bbox`.

Verified directly against the placed data before writing anything: all 7,363 images present (matches `datasets.yaml`'s documented total exactly), zero image/annotation filename mismatches when looked up directly as `annotation_dir / (image_filename + ".txt")` — simpler than the reference script's derive-image-path-from-annotation-name approach, so that part wasn't ported.

### Decision

- **Stage 5.2 "intermediate schema" defined**: a flat pool (no split subfolders) of `images/` + `labels/`, one YOLO-format `.txt` per image, using **canonical class ids** (`config/classes.yaml`'s `id` field per class key — not `config_loader.CANONICAL_NAMES`' title-case strings, which don't match `datasets.yaml`'s lowercase class-key map values). This is deliberately "the final training format minus the split" — no new format invented, since 3 of the 5 sources already deliver close to this natively (Roboflow) or need conversion to it anyway (ExDark, CrowdHuman).
- Per DEC-041, this lives in `scripts/acquire/acquire_exdark.py` itself (not a separate `exdark_to_intermediate.py` — `bbox_utils.py`'s docstring referencing that name was stale, corrected in the same edit).
- Iterates **all 12** ExDark native class folders, not just the 8 mapped ones — an image filed under e.g. `Boat/` can still contain a co-annotated canonical object (a Person, say); skipping non-canonical folders outright would silently lose those. Non-canonical boxes are filtered per-line, not per-folder.
- Images left with zero canonical boxes after filtering are dropped entirely (same "drop now-empty samples" pattern as `acquire_openimages.py`'s DEC-043 filter).
- Every converted box passes through `bbox_utils.validate_bbox`/`clip_bbox` — genuinely fixes the reference script's documented clamping gap, not just ported logic.

### Rationale

Defining the intermediate schema as "final format minus split" avoids inventing a bespoke schema with its own parser/writer to maintain, while still satisfying DEC-036 (flat, unsplit) and the multi-source-remapping need (canonical ids baked in at conversion time, not deferred). Reusing the reference script's parsing *logic* (bbGt header skip, field extraction) while rejecting its split/class-id/clamping choices matches `datasets.yaml`'s own instruction to use it "as conversion basis," not verbatim.

### Consequences

- Verified for real at full scale (not a sample): 6,042 images converted, 18,366 boxes kept, 41 boxes clipped (bbGt boxes that ran slightly past image edges — the exact defect the reference script couldn't catch), 0 invalid, 5,344 non-canonical boxes correctly dropped. Cross-checked output on disk: 6,042 images and 6,042 labels present, class ids used are exactly `{0,1,2,4,8,9,15}` (Person/Vehicle/Motorcycle/Animals/Chairs/Tables/Bicycle — the 7 canonical classes ExDark actually maps to), per-class box counts sum exactly to 18,366.
- This intermediate-schema definition now governs the remaining Stage 5.2-equivalent work: `acquire_crowdhuman.py` (DEC-041, still unbuilt) should follow the same shape.
- Output lives at `dataset/processed/exdark/` — genuinely usable data now, not a test artifact.

---

## DEC-047: CrowdHuman Mirror Switched to HuggingFace After Rejecting a Pseudo-Labeled Kaggle Alternative

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-012 (vbox-only policy), DEC-028 (CrowdHuman's crowd/occlusion edge-case role, exempt from volume discount), DEC-041 (manual download scope)

### Context

DEC-041's original CrowdHuman mirrors (Baidu Drive, Google Drive) are dead — Baidu's download doesn't progress, the Google Drive link no longer resolves (student-confirmed 2026-08-13). Two alternatives were evaluated before picking a replacement:

1. `kaggle.com/datasets/menhari/crowd-human-crowd-detection` — the student's first find. Its own description states: *"Bounding boxes generated for humans using a pre-trained YOLOv8 model."* This is not CrowdHuman's ground truth — it's pseudo-labels from a detector's own predictions. That directly undermines DEC-028's reason for including CrowdHuman at all: COCO-pretrained backbones have a weak prior specifically on crowded/occluded scenes, so CrowdHuman's genuine human-annotated labels were meant to correct that blind spot. Training on a YOLOv8 model's own predictions for exactly the scenario it's weakest at would reinforce the blind spot, not fix it. It also collapses DEC-012's vbox/fbox/hbox choice to a single generic auto-detected box — there's no occlusion-aware annotation to select from anymore.
2. `huggingface.co/datasets/sshao0516/CrowdHuman` — checked via the dataset card (WebFetch): same official filenames DEC-041 already specified (`CrowdHuman_train01/02/03.zip`, `CrowdHuman_val.zip`, `annotation_train.odgt`, `annotation_val.odgt`), genuine human-annotated odgt ground truth with all three box types, matching known official CrowdHuman statistics (15,000 train / 4,370 val / 470K instances). A mirror of the real archive, not a repackaging.

### Decision

Reject the Kaggle pseudo-labeled version. Switch `datasets.yaml`'s `crowdhuman.download_url` to the HuggingFace mirror. No change to DEC-041's manual-download scope or DEC-012's vbox-extraction logic — same files, same format, just a working host.

### Rationale

Label provenance matters more than convenience here specifically because CrowdHuman's whole role in this project (DEC-028) depends on it being genuine, independent ground truth for a condition COCO-pretrained models handle poorly. A source that can't offer that isn't a faster path to the same outcome — it's a different, less useful dataset that happens to share CrowdHuman's name and images.

### Alternatives Considered

- **Accept the Kaggle version, demote its role to ordinary bonus Person volume**: Rejected — would leave the actual crowd/occlusion edge-case gap uncovered while still consuming acquisition effort, worse than just fixing the mirror.
- **Keep retrying the dead Baidu/Google Drive links**: Rejected once a working genuine mirror was found — no reason to keep fighting a dead host.

### Consequences

- `config/datasets.yaml`'s `crowdhuman` block updated: `download_url`, `license` (now states CC-BY-NC-4.0 + citation requirement, per the HF dataset card), and `download_strategy` (explains the mirror switch and why Kaggle was rejected). Re-validated clean via `config_loader.py`.
- **Not yet verified against real bytes on disk** — WebFetch's dataset-card summary is a strong signal (specific, consistent with already-known official CrowdHuman stats) but not proof; same discipline as ExDark applies once the student places the files under `dataset/raw/crowdhuman/`.
- `acquire_crowdhuman.py` (DEC-041, still unbuilt) is unaffected in design — same odgt parsing, same vbox-only extraction, same intermediate-schema output shape (DEC-046) once files are actually in place.

---

## DEC-048: CrowdHuman Reverts to Scripted Download (via `huggingface_hub`) — Partially Supersedes DEC-041; `acquire_crowdhuman.py` Built

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-041 (original manual-download scope, partially superseded), DEC-047 (HF mirror switch), DEC-012 (vbox-only policy), DEC-046 (intermediate schema)

### Context

DEC-041 put CrowdHuman in the same "manual download, parse-only" bucket as ExDark and Dataset Ninja, reasoning that scripting a one-time static-archive download wasn't worth maintaining fragile host-specific logic (Google Drive's confirm-token flow, Baidu's fetch-code gate). DEC-047 moved CrowdHuman to a HuggingFace mirror after the original hosts died. That changes the calculus DEC-041 was built on: checked directly via HF's public API (`gated: false, private: false`) before deciding anything — this repo needs no login or token. `huggingface_hub` is a real, maintained SDK, not host-specific scraping — the same category as the Roboflow SDK already used under DEC-018r's "Local Pull-and-Resolve" pattern, not the category DEC-041 was avoiding.

### Decision

- CrowdHuman moves back to scripted download+parse (like `acquire_openimages.py`/`acquire_roboflow.py`), not manual-placement-then-parse. `acquire_crowdhuman.py` pulls `CrowdHuman_train01/02/03.zip`, `CrowdHuman_val.zip`, `annotation_train.odgt`, `annotation_val.odgt` via `huggingface_hub.hf_hub_download()`; skips `test.zip` (DEC-036 — held-out test data gets pooled and re-split at Stage 5.8 regardless).
- `huggingface_hub>=0.24.0` added to `requirements.txt`.
- ExDark and Dataset Ninja are **unaffected** — neither has a comparable public SDK (ExDark is a single static GitHub-linked archive; Dataset Ninja's own SDK was already rejected in DEC-041/DEC-034 for being an unreliable Dropbox wrapper). DEC-041's core reasoning still holds for those two.
- Output follows DEC-046's intermediate schema: flat `images/` + `labels/` pool to `dataset/processed/crowdhuman/`, canonical Person id (`get_class_id("Person")`), `vbox` only per DEC-012.

### Rationale

DEC-041's own stated test — "does scripting save real effort or buy reproducibility/resumability an API already gives for free" — now answers differently for this specific source. At 14.2GB total across 4 files, resumable/checksummed transfer matters more here than for any other single source pulled this session; a manual browser download of that size is meaningfully more failure-prone than a purpose-built client.

### Consequences

- Verified for real before writing the full script: downloaded both real annotation files (`annotation_train.odgt` 80MB/15,000 entries, `annotation_val.odgt` 23MB/4,370 entries — both match official CrowdHuman counts exactly) via `hf_hub_download()`, confirming the SDK path actually works unauthenticated. Scanned the full real `annotation_train.odgt` for `tag`/`extra.ignore` distributions before writing the filter — confirmed `datasets.yaml`'s pre-existing filtering note (exclude `tag: mask` or `extra.ignore == 1`) was already exactly correct.
- `--dry-run` (parse-only, no download) run against the real annotation files: 19,370 images total, 439,046 person boxes kept after filtering.
- Box-conversion pipeline (`convert()`) verified against a **real** annotation entry (7 real vbox values) paired with a synthetic image sized to bound them, since testing against a real image would require downloading at least one full multi-GB zip. All 7 boxes converted correctly, sane normalized coordinates, class id 0 (Person).
- **One assumption NOT yet verified against real bytes, flagged explicitly in the script**: each zip's internal layout is assumed to be a top-level `Images/` folder with files named `<ID>.jpg` (CrowdHuman's well-documented standard release structure). Confirming this requires downloading a multi-GB zip, which wasn't done without the student's go-ahead. `extract_images()` raises loudly (`KeyError` naming the exact problem) if this assumption is wrong, rather than silently producing an empty or corrupt image pool.
- The two real annotation files are already correctly placed under `dataset/raw/crowdhuman/` (fetched via the sanctioned SDK path, not test debris — left in place). The four image zips (~14GB total) are not downloaded yet — left for the student to trigger via `python3 scripts/acquire/acquire_crowdhuman.py`, same posture as the full `acquire_openimages.py`/`acquire_roboflow.py` runs.
- `datasets.yaml`'s `crowdhuman.download_strategy` (written under DEC-047, before this reversal) still says "MANUAL DOWNLOAD... not scripted" — now stale, needs updating to reflect this decision.

---

## DEC-049: `acquire_datasetninja.py` Built — Real Supervisely Format Parsed Directly, No SDK

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-034 (Potholes sourcing strategy), DEC-041 (manual download scope), DEC-046 (intermediate schema), DEC-025 (custom-converter scope — this is the case that scope didn't cover)

### Context

Student manually placed both Dataset Ninja sources — `pothole-detection` (co-primary) and `road-damage-detector` (deprioritized fallback) — under `dataset/raw/`, resolving the 2026-08-08 Dropbox blocker (the rate limit had cleared). Verified the actual format directly rather than trusting `datasets.yaml`'s prior assumption: both are genuine Supervisely project exports (`ds/img/`, `ds/ann/*.json`, `meta.json`), confirming the existing audit_note's suspicion (Pascal VOC was wrong). Annotation JSON gives image dimensions directly (`size.width/height`, no need to open the image) and boxes as a corner pair (`points.exterior: [[x1,y1],[x2,y2]]`) — a third distinct raw box format this project now handles (ExDark/CrowdHuman are top-left+w/h; this is two corners).

### Decision

- Parse the Supervisely JSON directly (~20 lines) rather than using the `supervisely`/`dataset_tools` SDK's `sly.convert.to_pascal_voc()` path `datasets.yaml` originally planned. That SDK already caused real friction this project hit before (system-level `libmagic` dependency, `xmltodict` downgrade) for a format simple enough not to need it.
- Promoted `bbox_utils._xyxy_to_yolo` to public `xyxy_to_yolo` (was private, used only inside `clip_bbox`) — this is now a second genuine caller, not just internal plumbing.
- Verified point ordering isn't guaranteed top-left-first across the real data (1 of 1,740 real boxes in `pothole-detection` was a degenerate zero-width pair, not simply reordered) — rather than adding special-case order-fixing, confirmed `validate_bbox`'s existing non-positive-width/height check catches it correctly on its own.
- `road-damage-detector` has 4 native classes; only `"pothole"` maps to canonical (DEC-034) — same per-line filtering pattern as `acquire_exdark.py`.
- Output follows DEC-046's intermediate schema, one processed dir per source (not merged — that's Stage 5.6).

### Consequences

- Verified for real at full scale, both sources: `pothole-detection` — 665/665 images converted, 1,739 boxes kept, 1 correctly dropped as invalid (matches the point-ordering finding exactly). `road-damage-detector` — 1,331/3,321 images converted (rest dropped, no pothole-class object present), 2,657 pothole boxes kept, 4,342 crack-class boxes correctly dropped. Cross-checked on-disk output: image/label counts match exactly, and every label across both sources uses class id 11 (Potholes) exclusively — zero crack-class leakage.
- **Real finding, not assumed**: `road-damage-detector`'s Dataset Ninja-hosted export is 3,321 images / 4 classes — a curated subset, not the full academic RDD2022 (47,420 images / 7 classes). `datasets.yaml` updated to state both figures, not silently overwritten.
- `config/datasets.yaml`'s `dataset_ninja_pothole_detection` block updated: `audit_status: blocked` → `approved`, `annotation_format`/`bbox_mode` corrected to match the verified real format, `dataset_tools` SDK references removed from the active path (kept only as history in the audit_note).
- This is the third distinct raw box format handled this session (ExDark/CrowdHuman top-left+w/h via `xywh_abs_to_yolo`; Dataset Ninja corner-pair via `xyxy_to_yolo`) — `bbox_utils.py` now covers both without needing a fourth.
- Stage 5.1's acquisition scripts are now **all built**: `acquire_openimages.py`, `acquire_roboflow.py`, `acquire_exdark.py`, `acquire_crowdhuman.py`, `acquire_datasetninja.py`. Remaining Stage 5.1 work is running the large not-yet-triggered pulls (full `acquire_openimages.py`/`acquire_roboflow.py`/`acquire_crowdhuman.py` runs), not building anything new.

---

## DEC-050: `acquire_openimages.py` Reruns Now Clean Stale Exports; `fiftyone_preview.ipynb` Regression Fixed

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-040 (script this modifies), DEC-044 (the split fix whose rename broke the notebook)

### Context

Student asked three concrete questions before running the full `acquire_openimages.py` for real: (1) is `max_samples` actually a hard cap, (2) does re-running after a code change clean up the previous export, (3) an import error in `fiftyone_preview.ipynb`. All three checked directly rather than answered from memory.

1. **`max_samples` enforcement**: read FiftyOne's actual Open Images downloader source (`fiftyone/utils/openimages.py`) rather than trusting prior empirical tests alone — confirmed `valid_ids = valid_ids[:max_samples]` / `target_ids = target_ids[:max_samples]`, a literal Python slice on the shuffled candidate id list, applied before any download happens. Not a soft heuristic; cannot pull more than requested regardless of a class's true size in Open Images.
2. **Rerun behavior**: tested directly — pulled Chairs unfiltered (30 images) into a directory, then re-pulled the same seed with full filtering (29 images, DEC-043's filter drops 1) into the *same* directory. `labels.json` correctly reflected only the new 29 — but the image *files* on disk stayed at 30. The dropped image's file was orphaned: physically present, unreferenced. FiftyOne's own "will be merged with existing files" message on re-export means exactly that — files merge, they don't get replaced.
3. **Notebook import error**: `fiftyone_preview.ipynb` (built before DEC-044) imported `ZOO_SPLIT` (singular) and used `split=ZOO_SPLIT` — both removed when DEC-044 renamed it to `ZOO_SPLITS` (plural) and switched to `splits=[...]`. The notebook was never updated when that fix landed — a real regression, not a fabricated question.

### Decision

- `pull_class()` now clears `export_dir` (`shutil.rmtree` if it exists) immediately before `ensure_dir` + `view.export()`, so every run is a clean, reproducible snapshot of exactly what that run pulled — no manual cleanup needed between filter/logic changes.
- `fiftyone_preview.ipynb` updated: imports `ZOO_SPLITS`, uses `splits=list(ZOO_SPLITS)`, and divides `max_samples` by `len(ZOO_SPLITS)` before passing it to the Zoo loader — same per-split-division fix DEC-044 applied to the real script, now actually present in the notebook too.

### Rationale

A script that leaves orphaned files on rerun is a correctness trap specifically for the workflow this student described — iterating on filter logic and re-running to check the result. Silent staleness there could mean training on images the current code would no longer have chosen, without any signal that happened. Clearing the directory removes the failure mode entirely rather than documenting it as a gotcha to remember.

### Consequences

- Verified the fix directly: pulled 89 images (run 1), then re-pulled the same class with a much smaller request (15) into the same directory — file count dropped to exactly 15, no accumulation. Cleaned up test artifacts and FiftyOne test datasets afterward.
- Verified the notebook fix directly (equivalent code run outside Jupyter, since notebooks can't be exec'd from bash): import succeeds, `6075 total (~2025/split)` printed correctly, a small-scale real pull (5/split × 3 = 15) confirmed the split mechanism works from the notebook's exact code path.
- General lesson surfaced, not just this one instance: renaming/removing a script's public constants (module-level names imported elsewhere) needs a check for other consumers — `fiftyone_preview.ipynb` imports directly from `acquire_openimages.py` by design (DEC-040's rationale: avoid re-typing native class names/typos), so it's coupled to that script's public names and will need the same check for any future rename.

---

## DEC-051: Full Real Acquisition Runs — Roboflow and Open Images Both Complete

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-040 (acquire_openimages.py), DEC-045 (acquire_roboflow.py), DEC-050 (rerun-cleanliness fix applied just before this run)

### Context

Student authorized running the large, previously-deliberately-unrun acquisition pulls for real: full `acquire_openimages.py` (8 classes) and, once that looked stable, the remaining 10 Roboflow projects — explicitly OK'd running them concurrently after asking whether that would cause problems on my end. Checked before agreeing: the two scripts are fully independent processes (FiftyOne's local dataset machinery vs. plain HTTP+zip extraction), write to non-overlapping directories (`dataset/raw/open_images/` vs `dataset/raw/roboflow_projects/`), and share no mutable state — no correctness risk from running together, only shared network bandwidth.

### Decision

Ran both in parallel as background processes. Added per-class error isolation to `acquire_openimages.py`'s main loop first (try/except around each `pull_class()` call, matching the pattern `acquire_roboflow.py` already had) — a single class's transient failure should not silently abort the remaining classes during an unattended-ish multi-class run.

### Consequences

- **Roboflow: complete, verified.** All 11 eligible projects pulled, 0 errors, 2.6GB total. Cross-checked every project's actual image count against `datasets.yaml`'s documented estimates from the earlier version-pinning research — 10 matched exactly, 1 (`elevator_status_s4lrk`) came in at 3,611 vs. a documented 3,656; verified this is real (file count on disk matches the report exactly, split-by-split) and not a script bug — the original figure was just a slightly-off manual GUI estimate from weeks earlier.
- **Open Images: complete, verified.** All 8 classes succeeded, 0 errors, 9.7GB total. Spot-checked on-disk file counts against the report for 3 classes — exact match. Yield varied by class, same "val/test splits smaller than train" effect already seen and explained for Chairs during preview-notebook testing: person 5,870, vehicle 5,962, animals 5,827, tables 4,436 landed close to the 6,075 request; motorcycle 2,611, bicycle 2,999, chairs 3,236 landed meaningfully lower; **trash_bins came in at only 1,106** (~18% of the request) — Open Images simply doesn't have many "Waste container" instances across all three splits combined for this class specifically.
- **Flagged, not treated as an error**: Trash Bins' low yield is worth the student's attention against DEC-039 (which benched searching for a secondary Trash Bins source, on the reasoning that the Open Images primary was "judged sufficient for now"). 1,106 raw images is well under the 4,500 final-cap target, and Stage 5.2-5.7 curation/dedup will only shrink that further — DEC-039's revisit trigger ("post-acquisition evaluation shows the primary alone is insufficient") may now actually be met. Not acted on here — this is real acquisition data now on disk, a technical success, not a blocking error; left for the student to decide whether to reopen DEC-039.
- General note for future large runs: the per-class error isolation added to `acquire_openimages.py` here is a real robustness improvement independent of this specific run, not a one-off hack — worth keeping.

---

## DEC-052: `openimages_to_intermediate.py` Built — Cross-Class-Folder Boxes Merged, Not Dropped

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-040 (acquire_openimages.py), DEC-046 (intermediate schema defined), DEC-043 (IsDepiction/IsGroupOf filter)

### Context

Stage 5.2 needed a converter for Open Images' 8 per-class COCO-style exports (`dataset/raw/open_images/<class_key>/`) into DEC-046's intermediate schema. Two real findings surfaced while inspecting the actual exported `labels.json` files, not assumed from the acquire script's docstring:

1. **Each class-folder's export is not pre-filtered to its own class.** `acquire_openimages.py`'s `classes=native_classes` argument only selects which *images* to pull; every other object FiftyOne's `ground_truth` field has for those images is exported too. The `animals/` folder's `labels.json` has 213 distinct category names in it, not just Dog/Cat — only 7,202 of its 17,330 annotations are actually Dog/Cat. This didn't match the acquire script's own docstring claim of "filtered to that class's native label(s)."
2. **The same photo can be independently pulled under multiple class-folders.** Verified directly via pairwise filename-set intersection across all 8 folders: 1,578 of `chairs/`'s 3,236 images are also in `tables/`; 706 also in `person/`; `person/` ∩ `tables/` = 541; `person/` ∩ `vehicle/` = 510; etc. (full pairwise matrix in `dataset/reports/openimages_to_intermediate_report.json`'s generation logs). A naive per-folder-independent conversion into a shared flat pool would have the second folder's write silently clobber the first's label file on filename collision, discarding real ground-truth boxes for whichever class processed first.

### Decision

- The converter filters each class-folder's annotations to that folder's own configured `native_classes` (same "drop what doesn't map" pattern already used by `acquire_exdark.py`) — non-native-class annotations are dropped, not miscounted or attributed elsewhere.
- Where the same filename appears in more than one class-folder, **all valid canonical boxes from every folder it appeared in are merged into a single label file** for that image, rather than picking one folder and discarding the rest. The photo genuinely contains all of those objects — dropping a Table box from a photo that also has Chairs (or vice versa) would silently throw away correct, already-existing ground truth for no reason. Explicitly discussed and confirmed with the student, who connected this correctly to the later manual-correction stage: merging preserves boxes that already exist; it doesn't create the ones that don't (that's Stage 5.3/5.5's job, not this converter's).
- A per-class **final image + instance count audit** was added to the script and its report (`final_per_class_counts`): for each class, counts both how many images in the merged pool carry at least one box of that class (non-exclusive — an image with both Chair and Table boxes counts toward both), and the total instance (box) count for that class. These are the numbers each class will actually draw on once merged (Stage 5.6), distinct from — and always ≤ — the raw per-folder pull counts, per the student's explicit request (image counts, then instance counts added as a follow-up ask).

### Rationale

Filtering to native_classes per folder matches the established pattern for every other source this project has converted (ExDark, Dataset Ninja) — a box only counts for the class it was actually pulled to represent. Merging on cross-folder collision was chosen over "last write wins" because the overlap is large enough to matter (chairs∩tables alone is ~49% of the smaller folder) and the alternative has no upside — it only silently loses real labeled data for zero benefit.

### Consequences

- Real run: 26,715 unique images (down from 32,047 raw per-folder pulls — the difference is exactly the cross-folder duplication), 67,760 boxes kept, 81 clipped, 0 invalid, 2,758 images had boxes merged from more than one class-folder.
- Final per-class counts (images non-exclusive / instances = total boxes): person 4,678 images / 14,346 instances, vehicle 5,580 / 12,543, animals 5,705 / 7,202, chairs 2,930 / 12,992, motorcycle 2,565 / 4,792, bicycle 2,887 / 5,890, tables 4,349 / 8,074, trash_bins 1,104 / 1,921. Image counts are lower than the raw per-folder pull counts reported in DEC-051 for every class — expected, since DEC-051's numbers were single-folder pull counts before any cross-folder merge or native-class filtering was applied.
- Output at `dataset/processed/open_images/images/` + `.../labels/`, matching DEC-046's schema exactly — ready to be picked up by Stage 5.6 merge alongside ExDark, Dataset Ninja, and (once run) CrowdHuman.
- Remaining Stage 5.2 work: `yolo_to_intermediate.py` for Roboflow's native per-project YOLOv8 exports — not yet built.
- `docs/PLAN.md`'s Stage 5.2 table was also corrected in this session: it had never been updated to reflect that CrowdHuman/ExDark/Dataset Ninja conversion was folded into their acquire scripts (DEC-041, DEC-046, DEC-049), so it still listed those 3 as separate `⬜ Todo` converter scripts. Fixed to show only the 2 converters actually still needed.

---

## DEC-053: Roboflow Real-Format Audit Before Building `yolo_to_intermediate.py` — 4 Real Mismatches Found

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-031 (box-shape audit precedent), DEC-032/DEC-037 (traffico_y1 forking declined), DEC-036 (splits discarded), DEC-052 (same verify-before-build pattern applied to Open Images)

### Context

Before writing `yolo_to_intermediate.py`, checked every Roboflow project's actual downloaded `data.yaml` against what `config/datasets.yaml` assumed for it (`canonical_class(es)`/`native_class_filter`) — same rigor as DEC-052. Found real mismatches in 4 of the 10 on-disk projects:

1. **`escalator_stairs`**: configured filter keys were `stairs`/`escalator`; real class names are `downstair`/`upstair`/`Escalator` (different names, different case) — the filter as configured would have matched nothing.
2. **`cv_project_hovyc`**: configured as a single-class `doors` project with no filter; real export has 5 classes (`Door` + 4 `Exit *` signage classes). Without a filter, all 5 would have been blanket-mapped to Doors, mislabeling exit signage.
3. **`me5_u6rvg`**: exported `data.yaml`'s `names` are placeholder digit strings (`"0"`–`"7"`), not real class names — an export-side bug on Roboflow's part, not something re-downloading fixes. Real names recovered via `roboflow.Project.classes` (the live per-class instance-count API) and matched to indices by exact instance count: index 2=Motorcycle(709), 3=Tricycle(610), 4=Van(872), 5=Ambulance(254), 6=Cars(722), 7=Truck(974) — 6 unique exact matches, high confidence. Indices 0+1 (3244+1149=4393) are both very likely "Jeepney" (API total 4394, off by 1 — plausible version/split discrepancy, not a blocker). This also revealed real Tricycle/Van/Truck content the config never accounted for.
4. **`pedestrian_and_animal_crossing`**: the only version ever generated (v1, already downloaded) exports just 1 class, `people` — bounding boxes on pedestrians, not on the crosswalk/lane marking `config/datasets.yaml`'s `pedestrian_lane` mapping actually needs. Querying the project's live class list (not just the downloaded export) via the API surfaced 3 more classes never included in any generated version, one of them literally named `==============================` (2,610 instances) — almost certainly the real crosswalk-marking annotations, mislabeled with a garbage name by whoever created it, sitting unused because it was never selected into a generated version.

### Decision

- **`escalator_stairs`, `cv_project_hovyc`**: fixed mechanically — same "drop what doesn't map" convention already used by every other source (ExDark's non-canonical folders, Open Images' non-native categories). `native_class_filter` corrected to match real names; `cv_project_hovyc` gets an explicit filter now (`Door` only) instead of a blanket single-class mapping.
- **`me5_u6rvg`**: presented the recovered mapping and its Tricycle/Van/Truck discovery to the student with three options (full mapping / conservative original scope / skip). Student chose the full recovered mapping — added a new `native_class_index_override` field to `datasets.yaml` (index → recovered real name) that `yolo_to_intermediate.py` applies before the normal name→canonical `native_class_filter` step, since this project's `data.yaml` can never supply real names on its own.
- **`pedestrian_and_animal_crossing`**: presented three options (drop the source / fork to get the real class / keep `people` as a loose proxy). Student chose forking — explicitly accepting the same credit-consumption uncertainty that led to declining an identical fork for `traffico_y1` (DEC-032/037). Executed via the Roboflow SDK, which — unlike the Roboflow Universe *website's* "Fork" button — turned out to be scriptable (`Workspace.fork_project`, `Project.generate_version`), not a manual-only action like ExDark/CrowdHuman's downloads:
  1. `ws.fork_project(url=".../pedestrian-and-animal-crossing")` → forked into the student's own workspace (`kenth`) as `pedestrian-and-animal-crossing-vcjuo`. Confirmed all 4 original classes carried over, including the `==============================` one.
  2. `project.generate_version(settings={"augmentation": {}, "preprocessing": {"auto-orient": True}})` → new version 1, 3,536 images (train 2,981 / valid 369 / test 186) — includes every annotated image from the source project, not just the 1,071 in the original narrow export.
  3. Downloaded to `dataset/raw/roboflow_projects/pedestrian_and_animal_crossing_v2/`.

### Rationale

Verifying real per-project format before writing a converter is the same discipline DEC-052 just applied to Open Images — a converter built on a documented assumption instead of the real file would have either silently mislabeled data (`cv_project_hovyc`) or silently produced zero output (`escalator_stairs`, `pedestrian_and_animal_crossing` as originally downloaded). The SDK's `fork_project`/`generate_version` methods being real and scriptable — discovered by inspecting the installed `roboflow` package directly rather than assuming forking is UI-only — changes the calculus from DEC-032/037's original "manual, uncertain-cost, not worth it" framing to "scriptable, one-time cost the student explicitly accepted this time."

### Consequences

- `config/datasets.yaml` updated for all 4 projects with corrected/expanded `native_class_filter`, the new `native_class_index_override` field (me5_u6rvg only), and audit notes explaining what changed and why.
- `pedestrian_and_animal_crossing`'s original narrow download (`.../pedestrian_and_animal_crossing/`, 1,071 images, `people`-only) is superseded by the forked-and-regenerated `.../pedestrian_and_animal_crossing_v2/` (3,536 images, real classes). The old directory is left in place for now rather than deleted immediately — cheap to remove once the converter is confirmed working against the new one.
- A new Roboflow project (`kenth/pedestrian-and-animal-crossing-vcjuo`) now exists permanently in the student's own Roboflow account as a side effect of forking — expected and accepted, not an accident.
- `yolo_to_intermediate.py` needs to handle two general-purpose mechanisms beyond plain name-based filtering: (1) an optional `native_class_index_override` for projects whose exported names aren't usable as-is, applied before the normal filter step; (2) reading each project's real `data.yaml` `names` list directly rather than trusting `datasets.yaml`'s documented class list.
- **Real conversion run, 10 of 11 projects**: `augmented_tricycle` 3,021 img/3,915 boxes, `cv_project_hovyc` 1,337/1,634 (29 Exit-signage boxes correctly dropped, spot-checked — only class id 7/doors present in output), `elevator_awvus` 1,777/2,611, `elevator_status_s4lrk` 3,587/6,786, `escalator_stairs` 8,684/11,816, `me5_u6rvg` 7,243/8,534 (recovered mapping applied, 0 non-canonical drops — confirms every one of the 8 placeholder indices was real Vehicle/Motorcycle/Tricycle content, none wasted), `pole_detection_z76mb` 3,100/3,897, `pothole_vhmow` 871/2,189, `stairs_i2yia` 1,559/1,876, `utility_poles_44tzx` 5,089/6,436. `pedestrian_and_animal_crossing` still shows 0/0 (all 1,507 `people` boxes correctly dropped as non-canonical) — expected, its converted output is pending the forked `_v2` download.
- **Forked download stalled mid-transfer** (49MB into ~probably-larger zip, no progress for 40+ seconds) — diagnosed as the student's current network being shared/slow (their own explanation, matches the symptom: process still alive, just no bytes moving), not a script or Roboflow-side failure. Left running in the background rather than killed/retried, since retrying wouldn't fix a slow network. `pedestrian_and_animal_crossing`'s conversion is the one remaining piece of Stage 5.2.

---

## DEC-054: Notebooks Moved to `notebooks/`, Each Clearly Labeled by Pipeline Stage; New `fiftyone_review_processed.ipynb`

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-046 (intermediate schema), DEC-050 (fiftyone_preview.ipynb built)

### Context

Student tried pointing `fiftyone_explore.ipynb` at `dataset/processed/dataset_ninja_pothole_detection` (converted intermediate-schema output) and hit `ValueError: Data directory '.../data/' does not exist`. Root cause: that notebook was built for the *raw* COCO-style layout (`data/` + `labels.json`, from `acquire_openimages.py`'s exports), not the intermediate schema (flat `images/`+`labels/`, plain YOLO `.txt`, canonical ids) that every processed source now uses (ExDark, both Dataset Ninja sources, Open Images, 10 Roboflow projects). The three existing notebooks (`fiftyone_test.ipynb`, `fiftyone_explore.ipynb`, `fiftyone_preview.ipynb`) also lived loose at the repo root with no indication of which pipeline stage each was for — confirmed with the student that the intermediate-schema data genuinely is meant to be browsed (it's what Stage 5.3 Box Audit / Stage 5.5 Curation operate on), not just a pass-through.

### Decision

- Moved all 3 existing notebooks into a new `notebooks/` folder (`git mv` for the tracked `fiftyone_test.ipynb`, plain move for the two untracked ones).
- Added a markdown cell at the top of every notebook stating: what it's for, when to use it (relative to pipeline stage), what it expects on disk, and what it's explicitly *not* for (pointing at the others). Added `notebooks/README.md` with the same information as an at-a-glance table.
- Fixed path robustness in `fiftyone_explore.ipynb` and `fiftyone_preview.ipynb`: both previously assumed the notebook's cwd was the repo root (`Path.cwd()` used directly, or a bare relative `export_dir` string) — true when they lived at the repo root, not guaranteed once moved. Both now walk up from `Path.cwd()` to find the repo root (same pattern as `config_loader.get_repo_root()`), so they work regardless of where Jupyter's kernel actually launches from.
- Built `notebooks/fiftyone_review_processed.ipynb` — the piece that was actually missing. Per the student's explicit choice: builds the FiftyOne dataset manually by iterating `images/`+`labels/` and constructing `fo.Detection` objects directly (label = `config_loader.get_canonical_names()[class_id]`, bounding box converted from YOLO center-based to FiftyOne's top-left-based convention), rather than using FiftyOne's `YOLOv5Dataset` importer — that importer requires a `dataset.yaml` + per-split folder structure the intermediate schema deliberately doesn't have (splits aren't computed until Stage 5.8, DEC-036).

### Rationale

Fighting FiftyOne's opinionated YOLOv5 importer into accepting a schema it wasn't designed for (fake dataset.yaml, fake single-split folder) would add indirection for no real benefit — a ~30-line manual loop over a schema this project already fully controls (DEC-046) is simpler and easier to trust. Per-notebook labeling was chosen over renaming the files outright, since the existing names are already referenced by their current spelling throughout `docs/DECISIONS.md`'s history — moving folders and adding labels gets the same clarity without creating stale references.

### Consequences

- Verified for real before considering this done: ran the new notebook's build logic (minus the interactive App launch) against `dataset_ninja_pothole_detection` — 50 images loaded, 142 detections, `distinct(...) == ['Potholes']` (correct single-class output, matches DEC-049's conversion result).
- Any future script rename that changes `acquire_openimages.py`'s public constants needs to check `notebooks/fiftyone_preview.ipynb` too — unchanged risk from DEC-050, just relocated.
- Notebook paths in this file's own history (DEC-050, TASKS.md's completed-table entries) refer to the old root-level filenames — left as-is, since they're accurate for the date they describe; only new references should use the `notebooks/` prefix.

---

## DEC-055: `pedestrian_and_animal_crossing` Fork Completed; Roboflow Stage 5.2 Fully Done; CrowdHuman Pulled For Real

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-053 (fork initiated, findings), DEC-047/048 (CrowdHuman mirror + scripted download)

### Context

The forked-project download (DEC-053) failed once with a genuine dropped connection (`BrokenPipeError`/`ChunkedEncodingError` — confirmed from the traceback, not assumed) on the student's earlier slow/shared network. Student confirmed their connection was stable again and authorized both a retry and starting CrowdHuman's real ~14GB pull in parallel.

Retried cleanly (removed the partial zip first, `overwrite=True` on a fresh directory). The exported `data.yaml` revealed one more thing worth noting: Roboflow's YOLO export sanitizes special characters in class names — the live API's `"=============================="` became `"------------------------------"` (dashes) in the actual downloaded `data.yaml`. Caught by checking the real file rather than assuming the API name would survive export unchanged.

Separately, a real bug surfaced while converting the fixed project: running `yolo_to_intermediate.py --projects <one>` overwrote `yolo_to_intermediate_report.json` wholesale, discarding the other 10 projects' results from the earlier full run (the on-disk converted images/labels for those 10 were untouched — only the report was affected).

### Decision

- Moved the forked download into the canonical `dataset/raw/roboflow_projects/pedestrian_and_animal_crossing/` path (replacing the old people-only export). `config/datasets.yaml` updated: `original_url` now points at the forked project (`kenth/pedestrian-and-animal-crossing-vcjuo`) so future `acquire_roboflow.py` reruns reproduce this data, not the original narrow one; a new `forked_from_url` field preserves the true source project for attribution. `native_class_filter` corrected to the real sanitized name (`"------------------------------"`, dashes).
- Fixed `yolo_to_intermediate.py`'s report-writing: now merges into any existing report (`existing.update(all_stats)`) instead of overwriting it, so a `--projects` subset run can't silently discard other projects' results again. Verified: reran a single-project subset, confirmed all 11 keys survived.
- Ran the full `yolo_to_intermediate.py` batch once more (idempotent — reprocesses everything, same merge-safe report either way) to get one complete, trustworthy report.
- Started `acquire_crowdhuman.py` for real (download + extract + parse + convert, all in one script run) in parallel with the fork retry — independent directories, no shared state, same safe-to-parallelize reasoning as DEC-051.

### Consequences

- **Roboflow Stage 5.2 is now fully done, all 11 projects**: `pedestrian_and_animal_crossing` converts to 2,158 images / 2,610 boxes (matches the API's `"=============================="` instance count, 2,610, exactly — confirms the recovered class was captured completely, not partially). Full-batch total across all 11 projects: **38,426 images, 52,304 boxes**.
- CrowdHuman's real pull is in progress (3.1GB in as of the last check, `CrowdHuman_train01.zip` done) — this is Stage 5.1's last remaining piece. Report pending until it completes.

---

## DEC-056: CrowdHuman Pulled For Real — Stage 5.1 and Stage 5.2 Both Now Fully Complete

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-047/048 (mirror + scripted download), DEC-046 (intermediate schema), DEC-052/DEC-055 (Open Images / Roboflow Stage 5.2)

### Context

`acquire_crowdhuman.py`'s real ~14GB image-zip pull ran to completion in the background (download → extract → parse → convert, one script run) once the student's network stabilized. This was the one assumption in the whole pipeline flagged as genuinely unverified against real bytes: `extract_images()`'s `Images/<ID>.jpg` zip-layout guess, called out explicitly in the script's own docstring as "not yet verified... raises loudly if Images/ isn't found rather than silently producing wrong output."

### Decision

No code changes needed — the run simply confirmed the assumption held. Recording the result as its own decision because it closes out both Stage 5.1 (acquisition) and Stage 5.2 (conversion) for every one of the 5 sources, a real milestone worth marking rather than letting the report JSON be the only record.

### Consequences

- **0 errors.** `Images/<ID>.jpg` layout confirmed correct against real bytes across all 4 zips — the flagged assumption is resolved, not just untested.
- **0 images_missing** — every image referenced in both `.odgt` files was actually present in the zips, and the count matches exactly what `--dry-run`'s annotation-only parse had already predicted before any bytes were downloaded (19,370 images, 439,046 boxes) — strong independent confirmation the parsing logic and the real data agree.
- Real numbers: 19,370 images converted, 439,046 boxes kept, 18,917 clipped, 0 invalid. Raw 21GB (`dataset/raw/crowdhuman/`), processed 11GB (`dataset/processed/crowdhuman/`, DEC-046 intermediate schema) — image/label counts verified matching exactly (19,370/19,370) on disk, not just trusted from the report.
- **Stage 5.1 (Acquisition) is now 100% complete** — all 5 sources (Open Images, Roboflow, ExDark, Dataset Ninja ×2, CrowdHuman) fully pulled.
- **Stage 5.2 (Conversion) is now 100% complete** — Open Images (DEC-052) and Roboflow (DEC-055) via dedicated converters; ExDark/CrowdHuman/Dataset Ninja via their acquire scripts directly (DEC-041/046/049). Every source now sits in `dataset/processed/<source>/`, DEC-046's intermediate schema, canonical class ids, ready for Stage 5.3 (Box Audit).
- Total raw pool across all sources, pre-audit/pre-cap/pre-dedup (informational only — real usable counts come after Stage 5.3–5.7): Open Images 26,715 img / 67,760 boxes, Roboflow 38,426 / 52,304, ExDark 6,042 / 18,366, Dataset Ninja (pothole+RDD) 1,996 / 4,396, CrowdHuman 19,370 / 439,046.

---

## DEC-057: `box_audit.py` Built (Stage 5.3); `clip_bbox()` Bug Found and Fixed Across All 5 Converters

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-046 (intermediate schema this audits), DEC-031 (elevator/stairs box-shape concern this heuristic targets), `docs/OPEN_QUESTIONS.md` #3 (elevator_status_s4lrk flagging heuristic), DEC-025 (bbox_utils.py scope)

### Context

Unattended overnight session (`.agents/handoff-2026-08-13-stage-5.3-through-5.9.md`) building Stage 5.3–5.9. `box_audit.py` is source-agnostic by design: every `dataset/processed/<source>/` already speaks DEC-046's intermediate schema (canonical class ids, flat images/+labels/), so one script auditing box shape/size and global class balance covers both of `docs/PLAN.md`'s stale "native_unspecified sources" / "project_dependent sources" line items at once — that Roboflow/bbox_mode distinction stopped mattering once every source converged on the same post-Stage-5.2 format.

Before picking a threshold for the elevator_status_s4lrk "state pseudo-label vs. real detection" heuristic (`docs/OPEN_QUESTIONS.md` #3), checked the real area-fraction distribution rather than guessing a "near full frame" cutoff: max observed area fraction across all 6,786 real boxes is 0.595, with no bimodal gap anywhere in the histogram (0.5–0.6 bucket has only 18 of 6,786). A fixed "near 1.0" threshold — the framing `docs/OPEN_QUESTIONS.md` used — would have caught nothing. The real signal turned out to be elongation, not area: DEC-031's own description of this source ("boxes cling to object shape rather than a clean axis-aligned rectangle") is a shape defect, and the data confirms it — 842 of 844 flagged elevator boxes are extreme-elongation outliers (several literal hairline slivers along a frame edge, e.g. `w=0.612, h=0.00003`), not large-area ones.

While validating the script against real output, found and ran down a real bug: `bbox_utils.validate_bbox()` on freshly-round-tripped label text was flagging over 100,000 boxes across the pipeline as "invalid" — but every one of them turned out to be floating-point noise (~5e-7) from reconstructing corner coordinates out of a 6-decimal-rounded center-form box, not real defects (verified directly: max real violation magnitude across every source was exactly 5e-7, the FP noise floor, except for one source). That one exception was real: `dataset/processed/crowdhuman/` has exactly 5 boxes (of 439,046) with negative width/height and out-of-[0,1] centers. Traced to a genuine bug in `bbox_utils.clip_bbox()`: it clamped only the *lower* corner to `>=0` and the *upper* corner to `<=1` independently per axis, so a box positioned **entirely** outside the frame (e.g. `x1=1.24, x2=1.30`, both past the right edge) clamps to `x1=1.24` (unchanged — `max(0, 1.24)`) and `x2=1.0` (`min(1, 1.30)`), leaving `x1 > x2` — a negative-width box that evades the pre-clip `validate_bbox()` "non-positive width/height" guard because that guard only runs *before* clipping, not after. All 5 real instances were CrowdHuman `vbox` entries that fell entirely outside their image.

### Decision

- `scripts/preprocess/box_audit.py` built: auto-discovers every `dataset/processed/<source>/`, computes per-(source, canonical-class) area-fraction and elongation (`max(w,h)/min(w,h)`) stats, flags outliers via **Tukey's fences (Q3 + 1.5×IQR)** — the standard statistical-outlier convention, not a project-specific invented number, applied per group (not one global magic threshold) so naturally-elongated classes like Pole/Pedestrian Lane aren't penalized for their normal shape. Also flags non-positive/out-of-bounds boxes (epsilon-toleranced at 1e-4, three orders of magnitude above the ~5e-7 text-rounding noise floor and orders of magnitude below any real defect seen) and a `TINY_AREA_FRACTION = 1e-4` absolute backstop. Read-only — writes only to `dataset/reports/`, never touches `dataset/processed/`.
- Global class-balance (images non-exclusive, instances = total boxes) reported per canonical class across **all** sources combined — informational input for Stage 5.4's `cap_per_class.py`.
- `elevator_status_s4lrk`'s full flagged list (844 boxes) written separately to `dataset/reports/elevator_status_s4lrk_flagged.json` for the student's review per `docs/OPEN_QUESTIONS.md` #3. The main report caps each source's dumped flagged-box list at 300 (`FLAGGED_SAMPLE_SIZE`) plus a reason-count breakdown — a full dump would have put ~68,000 CrowdHuman entries alone into the JSON for marginal review value; full counts and stats are preserved regardless of the sample cap.
- **Bug fix, not a data fix**: `bbox_utils.clip_bbox()` corrected to clamp both corners of each axis independently into `[0, 1]` (`min(max(x, 0), 1)` for x1 *and* x2, same for y), so a box entirely outside frame now degenerates to a zero-width/height box at the boundary instead of a negative one. All 5 converters that call `clip_bbox()` (`acquire_crowdhuman.py`, `acquire_exdark.py`, `acquire_datasetninja.py`, `openimages_to_intermediate.py`, `yolo_to_intermediate.py`) got a matching one-line addition: re-check `w<=0 or h<=0` *after* clipping (not just before) and drop if so, since clipping can legitimately produce a degenerate result that clipping itself can't fix.
- Per this session's hard rule ("never touch `dataset/processed/` destructively; if a script needs to modify something there, stop and treat it as a question"): the 5 known-bad boxes already baked into `dataset/processed/crowdhuman/` are **left as-is**, not regenerated by re-running `acquire_crowdhuman.py`. They're now correctly flagged by `box_audit.py`'s `boxes_invalid` count for downstream stages to see. Re-running the acquire script to produce a clean file is the student's call, not made here.

### Rationale

Grounding the elevator heuristic in the real distribution (rather than assuming "near full frame" and picking a threshold like 0.7 or 0.8 that this data would never trip) avoids repeating the exact mistake this project has already been burned by once (the original invented 1.35 buffer factor). Tukey's fences were chosen over a fixed z-score or hand-picked percentage specifically because they're a recognized, parameter-light convention — defensible as "not an invented number" in the same way DEC-042's literature-grounded ratios are, just from statistics rather than domain literature. Fixing `clip_bbox()` now (even though nothing built in Stage 5.3–5.9 calls it again) was judged in-scope because it's a code-correctness fix in a shared utility, not a `dataset/processed/` data mutation — leaving a known, now-understood bug in place for a future re-run to rediscover would be worse than fixing it once it was found.

### Alternatives Considered

- **Fixed "near-full-frame" area-fraction threshold (e.g. 0.7) for the elevator heuristic**: Rejected — checked against real data first and found no box anywhere near that threshold; would have silently flagged zero boxes and looked like a clean pass when the real defect (shape, not size) was sitting unflagged.
- **Regenerate `dataset/processed/crowdhuman/` immediately after fixing `clip_bbox()`, to get a fully clean file**: Rejected for this session — re-running an acquire script to rewrite `dataset/processed/` output is explicitly the "stop and treat it as a question" case this session's hard rules call out, even though the fix itself is safe and the affected count is tiny (5 of 439,046). Left for the student to trigger if they want a byte-clean file.
- **Dump every flagged box into the main JSON report**: Rejected — `crowdhuman` alone would contribute ~68,000 entries; a bounded sample plus full reason-counts and per-class stats preserves everything actually decision-relevant at a fraction of the size.

### Consequences

- Real run against all 16 processed sources: 581,879 total boxes audited, `boxes_invalid` = 5 (all in crowdhuman, all pre-existing and now correctly identified — see Decision), 0 elsewhere. `dataset/reports/box_audit_report.json` (1.3MB) and `dataset/reports/elevator_status_s4lrk_flagged.json` (219KB, 844 entries) written.
- Global class balance (pre-cap, pre-dedup, informational only — real usable counts come after Stage 5.4–5.7): Person 26,706 img/460,852 inst, Vehicle 12,953/22,685, Motorcycle 3,861/6,573, Pole 8,189/10,333, Animals 7,314/9,129, Stairs 5,988/6,476, Escalator 4,255/7,216, Doors 1,337/1,634, Chairs 4,160/15,369, Tables 5,344/9,557, Tricycle 3,495/4,525, Potholes 2,867/6,585, Trash Bins 1,104/1,921, Elevator 5,364/9,397, Pedestrian Lane 2,158/2,610, Bicycle 3,638/7,010.
- `roboflow_stairs_i2yia` shows the same shape-outlier signature as elevator_status_s4lrk (232 of 233 flagged boxes are `shape_outlier`, matching DEC-031's "polygon-derived, not axis-aligned" finding) — worth the student's attention alongside elevator during review, not something this script resolves on its own.
- `docs/OPEN_QUESTIONS.md` #3 updated: the heuristic is built and the flagged list produced, but the framing corrected from "area-fraction outliers" to "shape (elongation) outliers" based on what the real data actually shows. Still needs the student's visual review — this script flags, it doesn't decide.
- No files under `dataset/raw/` or `dataset/processed/` were modified. `traffico_y1` correctly has no processed directory and was silently skipped by auto-discovery — no special-case code needed for it.
- `docs/PLAN.md`'s Stage 5.3 table can now mark `box_audit.py` built and both the native_unspecified/project_dependent audit-run rows done (one script covered both, per Context).

---

## DEC-058: `cap_per_class.py` Built (Stage 5.4) — Decision/Report Only, Not Wired Into merge.py

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-014 (ExDark guaranteed floor, corrected to 7 classes), DEC-042 (floor/hard-cap/instance-target/ratio-invariant policy), `docs/OPEN_QUESTIONS.md` #1 (Trash Bins shortfall), #6 (trim method default)

### Context

Unattended overnight session, Stage 5.4. Before writing anything, resolved a real architectural ambiguity the handoff didn't spell out: does `cap_per_class.py` physically materialize its selection (copy files somewhere), or just decide and report? Checked every relevant doc rather than guessing: `docs/PLAN.md`'s Stage 5.4 row ends in "Review `dataset/reports/cap_report.json`" (a human-review checkpoint, not an automatic gate); this session's own Stage 5.6 scope call for `merge.py` says, verbatim, "pool every `dataset/processed/<source>/` into `dataset/merged/`" — mechanical, no mention of consulting a cap decision; and Stages 5.5/5.7 are both explicitly designed as flag-for-human-review stages with no script that "eliminates" the review step (handoff §1). Treating 5.4 as the one stage that silently, automatically commits its own decision to a physical file selection would break that symmetry and bake in a trim method already flagged as an unreviewed default (`docs/OPEN_QUESTIONS.md` #6) before the student has seen it.

Also surfaced: this run's own numbers trigger DEC-042's cap-recompute rule. Trash Bins' realized candidate pool is 1,104 images — under the 1,500 floor, same shortfall DEC-051 already flagged. DEC-042 says recompute `hard_cap` as `3 × min(realized_class_images)` once the true floor is known; doing so here would drop every class's cap from 4,500 to ~3,312, a sweeping change `docs/OPEN_QUESTIONS.md` #1 already puts in the student's hands ("accept ~1,000-ish Trash Bins, or find a secondary source").

### Decision

- `scripts/preprocess/cap_per_class.py` built: scans every `dataset/processed/<source>/` once, builds a per-class candidate index, and for each of the 16 canonical classes applies DEC-014's ExDark floor (every ExDark image for that class reserved first, whether or not the class is in the known 7-class overlap — derived from what's actually in the data, not a hardcoded class list) followed by DEC-042's policy: remaining image budget = `hard_cap − exdark_floor` (hard_cap read from `classes.yaml`'s `cap` field per class, not hardcoded), remaining instance budget = `10,000 − exdark_floor_instances`, candidates beyond the floor shuffled with `random.Random(SEED=42)` and accepted until either budget binds — the seeded shuffle order **is** the "seeded random trim" default `docs/OPEN_QUESTIONS.md` #6 asked for.
- The 6,000-instance-target variant for "small/hard classes" (DEC-042's own phrasing) is **not applied** — no config anywhere defines which classes qualify, and guessing would be inventing an undecided parameter. Every class uses the 10,000 general target; documented here and in `docs/OPEN_QUESTIONS.md` as a real gap, not silently dropped.
- **Decision/report only** — writes `dataset/reports/cap_report.json` (per-class selected/excluded (source, filename) pairs, counts, stop reason, ratio invariant) and does **not** copy, move, or delete any files. `dataset/curated/`, `dataset/merged/` untouched. `scripts/build/merge.py` (built later this session, DEC-059) pools directly from `dataset/processed/` per its own literal scope, not from this report — applying the cap to the physical file flow is a follow-up step for the student, not done tonight.
- DEC-042's cap-recompute trigger is detected and reported (`recompute_hard_cap_trigger: true`, with the specific recomputed value shown) but **not applied** — same reasoning as above, this is `docs/OPEN_QUESTIONS.md` #1's call, not an autonomous one.

### Rationale

A script whose own core parameter (the trim method) is an admitted unreviewed default shouldn't be the one stage in this pipeline that silently commits its output to disk without a review step — every other stage with an unresolved judgment call (box_audit's flags, mistakenness scores, dedup pairs, final curation flags) is report-only for the same reason. Deriving the ExDark-floor class set from real per-class data (rather than hardcoding DEC-014's "7 classes" list) means this script keeps working correctly even if a future ExDark class mapping changes, without needing a matching code edit.

### Alternatives Considered

- **Copy selected images into `dataset/curated/<source>/` as a materialized capped pool**: Rejected — `dataset/curated/` is already earmarked by `docs/PLAN.md`'s Stage 5.5 row for CVAT/Label-Studio re-imported corrections specifically; overloading it with an unreviewed cap selection before that stage runs would conflate two different provenances in the same directory.
- **Apply DEC-042's cap-recompute rule automatically since the trigger condition is real and already met**: Rejected — recomputing would silently change every one of the 16 classes' effective cap based on one already-known, already-flagged low-volume class; that's exactly the kind of consequential, discussion-worthy change this session's hard rules say to surface, not decide.
- **Hardcode DEC-014's corrected 7-class ExDark-overlap list**: Rejected in favor of deriving it from which classes actually have ExDark-sourced candidates in the real index — self-verifying, one less hardcoded list to keep in sync with `datasets.yaml`.

### Consequences

- Real run against all 16 classes: Person and Chairs both correctly hit the `instance_target` stop *before* the image hard cap (2,820 img/10,009 inst and 2,935 img/10,017 inst respectively) — exactly the dense-class behavior DEC-042 predicted for Person via CrowdHuman's ~20+ instances/image. Vehicle, Pole, Animals, Stairs, Tables, Elevator hit the `image_hard_cap` stop at exactly 4,500. Motorcycle, Escalator, Doors, Tricycle, Potholes, Trash Bins, Pedestrian Lane, Bicycle include every candidate (`all_candidates_included`) — their natural pool never reaches 4,500.
- Confirmed via spot-check (not just trusted): sampled 5 "selected" entries from Person's list, all 5 have both an image and a label file present on disk at the claimed `dataset/processed/<source>/` path.
- Ratio invariant **not met**: max/min = Vehicle(4,500)/Trash Bins(1,104) = 4.08, above the 3:1 target. Trash Bins and Doors (1,337 images) both land under the 1,500 floor. Both are pre-existing, already-flagged findings (DEC-051, `docs/OPEN_QUESTIONS.md` #1) — this run confirms them with exact final numbers, doesn't newly discover them.
- `docs/OPEN_QUESTIONS.md` #1 and #6 both updated with this run's concrete numbers.
- `dataset/reports/cap_report.json` (3.6MB) — full per-class selected/excluded lists, safe to regenerate any time (deterministic given `SEED=42` and unchanged `dataset/processed/` contents).

---

## DEC-059: `merge.py` Built (Stage 5.6) — Corrects DEC-058: Merges the Capped Selection, Not Raw processed/

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-058 (cap_per_class.py, whose scope assumption this corrects), DEC-052 (precedent for keeping "bonus" cross-class boxes rather than stripping them)

### Context

While researching `merge.py`'s real scope (before writing it), re-checked `README.md`'s directory table rather than relying solely on this session's own handoff document — same "verify against real docs, don't trust one that might be stale" discipline this project has used all session. `README.md` labels `dataset/merged/` as **"Post-cap, post-merge, pre-split"**, an explicit, pre-existing architectural statement that capping happens before merging. `DEC-058` (cap_per_class.py) had missed this — it read the handoff's terser scope-call table ("pool every `dataset/processed/<source>/` into `dataset/merged/`") as "everything, uncapped," and deliberately kept `cap_per_class.py` decision-only on that assumption. This entry corrects that: `merge.py` DOES consume `dataset/reports/cap_report.json`.

### Decision

- `scripts/build/merge.py` built: reads `dataset/reports/cap_report.json`, takes the union of every one of the 16 classes' "selected" (source, filename) pairs, and copies each image + its FULL original (unfiltered) label file into `dataset/merged/images/` + `dataset/merged/labels/`, source-prefixed via `file_utils.prefixed_filename()`.
- An image selected by one class's cap decision but also carrying a valid box for a class that *didn't* select it keeps that box — labels aren't stripped down to "only the boxes that earned this image its spot." Same reasoning as DEC-052's cross-folder Open Images merge: it's already-correct ground truth, dropping it loses real signal for no benefit.
- Consequence of that: a class's real post-merge count can exceed its own `cap_report.json` figure. `merge.py` recomputes true post-merge per-class counts directly from the merged label files and reports those as authoritative, rather than trusting the pre-merge estimate.
- Stage 5.5 (human correction) hasn't run — `dataset/curated/` is still empty, and this merge is explicitly the *capped, pre-correction* pool, not the fully-realized pipeline. Documented in the script's own docstring so a future re-run after Stage 5.5 actually completes isn't mistaken for redundant.
- **Real bug found and fixed before running either script for real**: both `merge.py` and `run_mistakenness.py` (DEC-060) originally resolved each selected file via `img_dir.glob(f"{filename}.*")` called once per file — an O(N×M) trap against source directories this large (open_images alone has 26,715 files). A background test run of `run_mistakenness.py` against its full 22,846-image scope was still running after 4+ minutes without reaching the inference phase; killed it, diagnosed the glob-per-file pattern as the cause, and replaced it in both scripts with a single per-source directory listing (`{stem: path}` dict) built once, then O(1) lookups. `merge.py`'s dry-run then completed in under a second (was previously killed after 2+ minutes with no output).

### Rationale

Trusting `README.md` over the handoff's own terser table where they conflict follows this session's explicit instruction (handoff §0: "if something here conflicts with `DECISIONS.md`, `DECISIONS.md` wins" — extended here to README.md, an equally pre-existing, non-improvised architecture doc, over a same-session handoff's own paraphrase). Keeping "bonus" boxes rather than stripping them avoids re-litigating a tradeoff this project already made explicitly (DEC-052) for the same shape of problem.

### Consequences

- Real run: 51,556 unique images merged (100% of the union — 0 missing images, 0 missing labels). Verified on disk, not just trusted: `dataset/merged/images/` and `dataset/merged/labels/` both contain exactly 51,556 files; spot-checked one merged label file's content against its source `dataset/processed/<source>/labels/` original — byte-for-byte identical.
- Real post-merge per-class counts (images/instances), superseding `cap_report.json`'s pre-merge figures: Person 3,653/13,073 (up from cap's own 2,820 — "bonus" boxes from images selected by other classes), Vehicle 4,608/8,450, Motorcycle 3,861/6,573, Pole 4,500/5,644, Animals 4,541/5,621, Stairs 4,500/4,876, Escalator 4,255/7,216, Doors 1,337/1,634, Chairs 3,440/12,516, Tables 4,672/8,400, Tricycle 3,495/4,525, Potholes 2,867/6,585, Trash Bins 1,104/1,921, Elevator 4,500/7,904, Pedestrian Lane 2,158/2,610, Bicycle 3,638/7,010.
- `dataset/reports/merge_report.json` written — per-source merged counts, any missing-file warnings (none), real post-merge class counts.
- The same glob-per-file performance fix needed applying to `run_mistakenness.py` too (DEC-060) — found via this script's dry-run hanging, so the fix landed in both before either was run for real at scale.

---

## DEC-060: `run_mistakenness.py` Built and Run (Stage 5.5) — COCO-Pretrained Proxy Model, 7/16 Classes

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-028 (CrowdHuman's crowd/occlusion role — relevant to reading this run's results), DEC-058/059 (cap_report.json / merge.py, whose output this stage builds on)

### Context

FiftyOne Brain's `compute_mistakenness` needs both a `ground_truth` and a `predictions` field per sample — this project has no trained model of its own yet (Phase 3, RunPod, not done here), so predictions had to come from somewhere. Used a pretrained, COCO-trained YOLOv8n (`yolov8n.pt`, zero-shot, zero training) — `ultralytics`/`torch` were already declared in `requirements.txt` but not yet installed in this environment; installed them to fulfill an already-decided project dependency, not a new one.

That approach only works for canonical classes with an unambiguous COCO analog. Built the crosswalk from `classes.yaml`'s own already-decided `native_class` fields (not invented): Person→person, Vehicle→car/bus/truck, Motorcycle→motorcycle, Bicycle→bicycle, Animals→dog/cat (not COCO's broader animal set — `classes.yaml`'s own Animals scope is Dog/Cat only), Chairs→chair, Tables→dining table (COCO's only table class, a forced 1:1). The other 9 classes (Pole, Stairs, Escalator, Doors, Tricycle, Potholes, Trash Bins, Elevator, Pedestrian Lane) have no COCO equivalent and are explicitly excluded, not silently skipped.

Scoped to Stage 5.4's *capped* selection (`cap_report.json`'s per-class "selected" lists for the 7 eligible classes), not the raw uncapped `processed/` pool — keeps this bounded to a size that finishes in one unattended run.

**Real performance bug found and fixed before the real run**: both this script and `merge.py` (DEC-059) originally resolved each file via `img_dir.glob(f"{filename}.*")` called once per file — an O(N×M) trap against source directories this large. A background test was still running after 4+ minutes without reaching the inference phase; killed it, diagnosed the pattern, replaced it in both scripts with a single per-source `{stem: path}` index built once. Also found: running two heavy ML workloads concurrently (this script's YOLO inference and a `dedup.py` timing test) on this single machine caused a ~15x throughput collapse from MPS/GPU resource contention — noted as an operational lesson, not re-litigated per-script.

### Decision

- `scripts/curate/run_mistakenness.py` built: for the union of images selected across the 7 eligible classes' capped selections, runs `yolov8n.pt` inference (MPS device, Ultralytics' own default confidence threshold of 0.25 — a tool default, not tuned), builds a FiftyOne dataset with `ground_truth` (original labels, filtered to the 7 eligible classes only, so non-eligible-class boxes in the same image don't pollute the comparison) and `predictions` (YOLO output remapped through the crosswalk), and runs `fob.compute_mistakenness(dataset, "predictions", label_field="ground_truth")`.
- Ranks all scored samples by mistakenness descending, writes the full ranked list (not a truncated sample — this list IS the priority order for review) to `dataset/reports/mistakenness_report.json`. Flag-only, same posture as every other Stage 5.3-5.7 script — no file is moved, corrected, or deleted.
- FiftyOne dataset is scratch (deleted after extracting results) — the JSON report is the persisted artifact, matching this project's established convention for test/computation-only FiftyOne datasets.

### Rationale

A COCO-pretrained proxy model is a legitimate, well-established "model-assisted curation" technique when a project has no model of its own yet — it's not a perfect signal (COCO's label semantics and box conventions differ from this project's), but it's a real, grounded comparison rather than an invented heuristic, and it's honestly scoped to only the classes where the comparison is actually meaningful.

### Consequences

- Real run: 22,846 images scored (matches the union size computed from `cap_report.json`). Report is 5.1MB.
- **Real, useful finding surfaced by the run itself, not by inspection**: the highest-mistakenness samples (score ≈0.97) are almost all `open_images` Chairs/Tables images where the model's prediction count exceeds ground truth (e.g. 3 gt boxes / 7 predicted) — plausible genuine under-annotation in the source data, exactly the kind of thing this stage exists to surface, not something engineered into the test.
- **Second finding, a different failure mode than "mistaken label"**: 3,998 of 22,846 scored samples (17.5%) show `mistakenness == -1.0` — checked against FiftyOne's own docstring for `compute_mistakenness`: per-sample mistakenness is the *maximum mistakenness across matched ground-truth/prediction pairs*; when a sample's detections don't match at all (either the model missed every ground-truth object, or CrowdHuman's crowd density produced far more predictions than matchable ground truth), there's no matched pair to score, and FiftyOne appears to fall back to a `-1.0` sentinel rather than a `[0, 1]` score. This is a *different* kind of flag than the high-score "likely-wrong-label" cases — it clusters on crowded CrowdHuman scenes (matches DEC-028's documented COCO-pretrained blind spot on crowds/occlusion) and total-miss ExDark low-light cases. Recorded here rather than silently treated as "just more low-priority samples" — a `-1.0` score is not comparable to a `0.01` score on the same scale, and the student should know that before sorting by it naively.
- `docs/OPEN_QUESTIONS.md` #5 updated to confirm `run_mistakenness.py` (the tool-agnostic half of Stage 5.5) is done; `reimport_corrections.py` remains genuinely blocked on the CVAT/Label Studio choice, confirmed not silently deferred.

---

## DEC-061: Code Review Pass on the New Stage 5.3-5.9 Scripts — 5 Real Bugs Found and Fixed, Including a DEC-050-Shaped Stale-Output Bug in merge.py

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** DEC-050 (the earlier acquire_openimages.py stale-export bug this repeats), DEC-057/058/059 (scripts most affected), handoff §4.5 (explicitly suggested running code-review on the new scripts)

### Context

Ran `/code-review` (medium effort, 6 parallel angles: reuse, simplification, removed-behavior, cross-file tracing, altitude/conventions, efficiency) against every script built this session, per the handoff's own closing suggestion. Findings ranged from real bugs to legitimate-but-lower-value cleanup; this entry records what was actually fixed and, briefly, what was deliberately left for time-budget reasons.

**Most consequential finding, self-inflicted while fixing a smaller one**: fixing `cap_per_class.py`'s ExDark-floor RNG-bypass (below) changed the exact images selected for several classes (same seed, same policy, but a different draw once ExDark's own selection started consuming `rng` state it hadn't before). Re-ran `cap_per_class.py` and `merge.py` to regenerate `cap_report.json`/`dataset/merged/` against the fixed code — and in doing so, re-discovered **the exact stale-output bug DEC-050 already found and fixed once this session, in a different script**: `merge.py` never cleared `dataset/merged/images|labels/` before writing, so `safe_copy(overwrite=True)` only overwrote filenames present in *both* the old and new selection — files present only in the old run's selection were silently orphaned. Caught by checking real file counts on disk (this project's standing discipline) rather than trusting the script's own tally: `merge.py` reported "Merged: 51529/51529" while `find dataset/merged/images -type f | wc -l` showed 60,119 — 8,590 stale files, and downstream per-class counts were correspondingly (and silently) inflated (e.g. Vehicle's real post-merge count read 6,862 instead of the correct 4,601).

### Decision

Fixed, in order of consequence:

1. **`merge.py` stale-output bug (new)**: `rmtree` both output dirs before writing, same fix pattern as DEC-050's `pull_class()`. Re-ran; disk count now matches the report exactly (51,529 = 51,529).
2. **`cap_per_class.py`'s ExDark-floor selection now uses the seeded `rng`** before slicing, instead of raw dict/filesystem iteration order — was the one selection path in the file not using it, inconsistent with the script's own stated "seeded random trim" policy (harmless today only because ExDark volumes never reach `hard_cap`, but silently non-reproducible if that ever changes).
3. **`merge.py`'s `canonical_names[class_id]` and `split.py`'s same lookup now bounds-checked** — previously an out-of-range class id in a merged label (corrupt data, not seen in practice) would raise `IndexError` *after* files were already copied, leaving a half-written pool with no report explaining why. Now counted separately and reported instead of crashing.
4. **`merge.py`'s `per_source_counts` now only increments on the success path** — previously incremented before the missing-image/missing-label checks, so it silently counted "attempted" rather than "actually merged" (misleading in exactly the run where it would have mattered — this run had 0 missing, so it didn't manifest, but the field's meaning didn't match its name).
5. **`dataset.delete()` for the three FiftyOne-Brain scripts' (`dedup.py`, `run_mistakenness.py`, `final_merge_curation.py`) scratch datasets now runs in a `finally` block** — a mid-computation error previously left a multi-thousand-sample orphaned dataset registered in FiftyOne's backing store.
6. **Consolidated duplicated helpers into `file_utils.py`**, flagged independently by 3+ of the 6 review angles: `build_stem_index(source)` (was duplicated verbatim in `merge.py` and `run_mistakenness.py` — the exact O(N×M)-avoidance fix from DEC-059/060, itself already paid for twice) and `discover_processed_sources()` (was duplicated in `box_audit.py` and `cap_per_class.py`). Both new helpers filter to `IMAGE_EXTENSIONS` (the old copies used a bare `is_file()` check, which a stray non-image file like `.DS_Store` could have silently corrupted).
7. **`bbox_utils.validate_bbox()` gained an `epsilon: float = 0.0` parameter** instead of `box_audit.py` maintaining a separate `audit_validate_bbox()` reimplementation of the same bounds logic — one canonical validator now covers both the strict pre-write case (converters, epsilon=0.0, unchanged behavior) and the epsilon-toleranced post-round-trip case (box_audit.py, epsilon=1e-4).

**Deliberately not fixed**, for remaining-session-time reasons — noted here so they're a documented, findable follow-up rather than silently dropped: the YOLO label-line-parsing loop is still re-implemented independently in 6 files (a `bbox_utils.iter_yolo_label_lines()` helper would consolidate it); report-writing boilerplate (`ensure_dir` + `json.dump` + print) is repeated across 7 new + 8 pre-existing scripts (pre-existing pattern, not newly introduced); `split.py`'s dedup-report staleness check compares image *counts*, not identity, so a same-size-but-different-contents re-merge could theoretically be accepted as "covers the full pool" when it doesn't. None of these affect this run's actual output — they're maintainability/robustness items for a future pass, not correctness gaps in tonight's numbers.

### Rationale

The stale-output bug is the one worth dwelling on: it recurred in a *different* script than DEC-050's original, via a code path (a re-run triggered by fixing a smaller, unrelated bug) that's exactly the scenario an unattended session should expect — fixing one thing and needing to re-run something downstream. The fact that it was caught the same way DEC-050 caught the original instance (checking real `find | wc -l` output against the script's own printed tally, not trusting the printed tally) rather than a different mechanism suggests this "clear output dir before writing" pattern should probably be a `file_utils.py` helper too, not something each script re-implements — flagged here rather than done, given time already spent this pass.

### Consequences

- `dataset/merged/` regenerated clean: 51,529 images (down from the first run's 51,556 — the ExDark-floor RNG fix changed a handful of per-class selections at the margins, not the overall scale), verified file count matches the merge report exactly.
- Real post-merge counts after the fix: Person 3,667 img/12,997 inst, Vehicle 4,601/8,544, Motorcycle 3,861/6,573, Pole 4,500/5,695, Animals 4,547/5,634, Stairs 4,500/4,904, Escalator 4,255/7,216, Doors 1,337/1,634, Chairs 3,419/12,402, Tables 4,668/8,315, Tricycle 3,495/4,525, Potholes 2,867/6,585, Trash Bins 1,104/1,921, Elevator 4,500/7,824, Pedestrian Lane 2,158/2,610, Bicycle 3,638/7,010 — supersede DEC-059's now-stale figures.
- All 10 affected scripts re-verified via `--dry-run`/syntax check after the refactor; `box_audit.py --dry-run` output confirmed byte-identical to its pre-refactor run (the consolidation changed nothing observable, as intended).
- `scripts/utils/file_utils.py` gained two new public functions (`build_stem_index`, `discover_processed_sources`), available to any future Stage 5.3+ script without re-deriving either.

---

## DEC-062: `dedup.py` Built and Run (Stage 5.6) — Exact Duplicates Full-Pool, Near-Duplicates Sampled

- **Date:** 2026-08-13
- **Status:** Accepted
- **Related:** `docs/OPEN_QUESTIONS.md` #7 (dedup method/threshold), DEC-059/061 (merge.py, whose output this reads)

### Context

First attempt at a full-pool (51,529 image) near-duplicate run degraded from ~89 img/s at the start to under 10 img/s within 90 seconds, projecting to an open-ended 1+ hour runtime — killed rather than let run unbounded. Root cause found by comparing against an earlier isolated benchmark that *had* run cleanly at ~28 img/s: that benchmark passed `num_workers=4` explicitly; the real script left it at FiftyOne's default, which appears to over-subscribe this machine's single MPS device with more DataLoader worker processes than it handles well concurrently, degrading over time rather than failing outright.

### Decision

- Pinned `num_workers=4` for the embedding computation (matches the clean benchmark).
- **`compute_exact_duplicates`** (filehash-based, no embedding model, no degradation risk) still runs against the **full** 51,529-image merged pool — real result: 1,893 groups, 2,143 duplicate files (~4.2% of the pool).
- **`compute_near_duplicates`** (embedding-based) runs against a **seeded (42), per-source-proportional stratified sample of 6,000 images**, not the full pool — a deliberate, documented scope bound given the measured (and only partially explained by the `num_workers` fix) throughput ceiling on this hardware, kept as a safety margin against the same degradation pattern recurring unattended. Real result on the sample: 818 images flagged near-duplicate across 520 groups (threshold=0.2, FiftyOne Brain's own default, `mobilenet-v2-imagenet-torch` embeddings on MPS — chosen for measured speed over FiftyOne's silent default; see `dedup.py`'s own module docstring for the full embedding-model rationale and its threshold-transfer caveat).
- **Real nuance surfaced while sanity-checking the output, not assumed correct on faith**: several flagged near-duplicate pairs report a `distance` well above the 0.2 threshold (e.g. 5.19, 6.05). Traced into FiftyOne Brain's actual `DuplicatesMixin.find_duplicates()` source: the reported distance is to the nearest *surviving unique* neighbor, found via a separate post-hoc k=1 query — not necessarily the specific neighbor that caused the original point to be thresholded as a duplicate (which may itself have been removed as someone else's duplicate first). The *flagging itself* (unique vs. duplicate) is threshold-correct; the *specific "kept" pairing and distance shown* in the report is not guaranteed to be ≤0.2. Documented so the report isn't misread.

### Rationale

Exact-duplicate detection has no accuracy/coverage tradeoff to make (it's a hash comparison) — running it on the full pool costs nothing extra, so there was no reason to sample it too. Near-duplicate detection is the one with a real, measured cost, so it's the one that got bounded — consistent with this project's practice of matching verification effort to actual risk/cost rather than applying one blanket policy.

### Alternatives Considered

- **Keep pushing for a full-pool near-duplicate run** (e.g. chunked processing, retry after the `num_workers` fix): the `num_workers` fix alone didn't fully restore the clean benchmark's throughput (real run averaged ~9-13 img/s post-fix, not ~28), suggesting a second, unidentified factor (possibly real content/size variance across the full pool vs. the benchmark's narrower slice). Chasing it further wasn't worth the time against an already-long session; the sampled result is real, grounded, and enough to inform the student's threshold/approach decision (`docs/OPEN_QUESTIONS.md` #7) either way.

### Consequences

- `dataset/reports/dedup_report.json` (671KB) — exact-duplicate groups cover the full pool; near-duplicate groups are explicitly labeled as sample-based, with the sample method and size recorded in the report itself (not just this doc).
- Nothing under `dataset/merged/` was modified — flag-and-report only, same as every other Stage 5.3-5.7 script.
- `docs/OPEN_QUESTIONS.md` #7 updated with real numbers and the distance-interpretation caveat.
- If the student wants a full-pool near-duplicate pass later, re-running `dedup.py` with `NEAR_DUP_SAMPLE_SIZE` raised (or a chunked-processing rewrite) is the natural follow-up — not attempted here.

---

## DEC-063: `split.py` Built and Run (Stage 5.8) — Fixed a Real Union-Find Bug in Duplicate-Group Leakage Prevention

- **Date:** 2026-08-14
- **Status:** Accepted
- **Related:** `docs/OPEN_QUESTIONS.md` #7 (dedup coverage), #8 (split ratio), DEC-062 (`dedup_report.json`'s schema, whose output this reads)

### Context

`split.py`'s own stated design goal is that no exact- or near-duplicate group found by `dedup.py` straddles more than one split (train/val/test) — leakage that would let the model see a near-identical image at train time and again at eval time. A first real dry-run against the full 51,529-image merged pool and the finalized `dedup_report.json` (DEC-062) printed `WARNING: 56 duplicate groups still straddle multiple splits (unexpected — investigate)` — the script's own leakage check catching its own bug, not silently passing.

Investigated with a standalone script rather than guessed: `dedup_report.json`'s exact-duplicate and near-duplicate checks have different coverage (DEC-062), so the same filename can legitimately be reported inside two different groups — one from each check. Confirmed on real data: 223 filenames appear in more than one group; 187 of those have groups with genuinely different membership (not just the same pair reported twice). `assign_splits()`'s original grouping logic did `group_of[f] = rep` per group in a plain loop — a last-write-wins overwrite, not a union-find — so processing a later overlapping group could silently sever an earlier group's link for a shared file, breaking it off from duplicates it was transitively chained to.

### Decision

Replaced the last-write-wins grouping with a real union-find (`parent`/`find`/`union` with path compression) over all duplicate groups before assignment, so overlapping groups merge into one connected component and move as a single unit. Also fixed `load_duplicate_groups()`'s log message and `split_report.json`'s output stats, which previously implied "dedup_report.json covers the full pool" for both duplicate checks — true only for exact-duplicates; near-duplicates only cover DEC-062's 6,000-image sample. Both fixes are correctness/accuracy fixes to already-agreed behavior, not new judgment calls, so made directly rather than flagged as blocked.

Re-ran `split.py` for real after the fix. **Real result:** train=38,691, val=6,383, test=6,455 (74.7% / 12.4% / 12.5%, close to the 75/12.5/12.5 target — DEC-042-style small deviation expected from source-stratification plus duplicate-groups moving as a unit). Zero cross-split leakage warnings. Verified against real files on disk (not just the script's own tally): `find`'s extension-aware count across `dataset/final/` sums to 51,529 (1,645 `.jpeg` + 48,758 `.jpg` + 1,126 `.png`), exactly matching `split_report.json`'s `split_counts` total.

### Rationale

A duplicate-aware split that only *sometimes* keeps duplicate groups together is worse than no duplicate-awareness at all — it creates a false sense of leakage prevention while still leaking on exactly the cases (187 files) where the two dedup checks' independently-discovered groups overlap, which is not a rare edge case at this data's scale. Fixing the union-find was mechanical (a standard, well-understood algorithm) and directly served the script's own already-documented goal — not a new scope decision requiring the student.

### Alternatives Considered

- **Drop near-duplicate groups from split-grouping entirely, keep only exact-duplicates**: would have sidestepped the overlap bug by construction, but throws away real leakage-prevention value from the near-duplicate sample for no reason — the actual bug was fixable directly.
- **Leave the 56-group warning and flag it as a student decision**: rejected — this is a verifiable code defect against the script's own stated contract (verified by tracing the actual overlapping groups), not an ambiguous judgment call like the trim-method or threshold questions elsewhere in this pipeline.

### Consequences

- `scripts/build/split.py` now performs correct transitive duplicate-group merging; `dataset/reports/split_report.json` records `duplicate_group_coverage_note` explicitly distinguishing exact-dup (full-pool) from near-dup (sampled) coverage, so a reader of the report alone (not just this doc) sees the caveat.
- `dataset/final/{train,val,test}/{images,labels}/` populated for real, verified clean on disk.
- Per-class per-split distribution recorded in `split_report.json`; all 16 classes present in all 3 splits with roughly proportional counts (e.g. Trash Bins: train=831/val=132/test=141, still thin per `docs/OPEN_QUESTIONS.md` #1, but proportionally split).
- `docs/OPEN_QUESTIONS.md` #8 updated with the real split counts.

---

## DEC-064: `final_merge_curation.py` Built and Run (Stage 5.7) — Same Method as DEC-060, Reused Against the Merged Pool

- **Date:** 2026-08-14
- **Status:** Accepted
- **Related:** DEC-060 (`run_mistakenness.py`, whose crosswalk/logic this reuses directly), `docs/OPEN_QUESTIONS.md` #5 (CVAT/Label Studio, blocks the actual correction loop this feeds)

### Context

Stage 5.7 is a second mistakenness-scoring checkpoint, identical in method to Stage 5.5 (DEC-060) but run against `dataset/merged/` (post-cap, post-merge, source-prefixed) instead of each source's own `dataset/processed/`. `final_merge_curation.py` imports `COCO_CROSSWALK`, `CANONICAL_KEY_TO_NAME`, `ELIGIBLE_CANONICAL_IDS`, and `yolo_to_fo_bbox` directly from `run_mistakenness.py` rather than duplicating them, so the 7-class COCO-analog scope and crosswalk provenance is identical and doesn't need re-litigating here.

An initial dry-run against the corrected 51,529-image merged pool (post-DEC-061 RNG fix) found 22,824 eligible images — down slightly from an earlier, now-stale dry-run's 22,851 (against the pre-fix 51,556-image pool), consistent with the RNG fix's expected small effect.

### Decision

Ran for real: yolov8n COCO-pretrained proxy inference (batches of 16, MPS) over all 22,824 eligible images, then `fiftyone.brain.compute_mistakenness()` against ground truth. **Real result:** 22,824 images scored and ranked, report at `dataset/reports/final_merge_curation_report.json` (5.4MB). Top-of-ranking pattern matches DEC-060's Stage 5.5 finding almost exactly: highest-mistakenness samples are open_images Chairs/Tables images where the model detects more boxes than ground truth has (e.g. top score 0.9701, `open_images__ada7e0a339ecdbfd`, gt=15 pred=17) — the same plausible-under-annotation signal, now confirmed to persist through merge rather than being an artifact of the pre-merge pool. 17.5% of scores are exactly -1.0 (FiftyOne's no-matched-pair sentinel, not a real [0,1] score) — same proportion as Stage 5.5, but the per-source mix shifted: `open_images` (2,407) and `exdark` (1,177) dominate as before, but `roboflow_me5_u6rvg` now contributes 400 sentinel scores that didn't show up the same way pre-merge, worth the student's attention if reviewing that source specifically.

Verified no leftover FiftyOne datasets after the run (`fo.list_datasets()` returns `[]`), same try/finally posture as every other FiftyOne-using script this session.

### Rationale

Reusing DEC-060's crosswalk/logic directly (import, not copy) means this checkpoint can't silently drift from Stage 5.5's — if the 7-class scope or crosswalk ever needs to change, there's one place to change it, not two to keep in sync by hand.

### Consequences

- `dataset/reports/final_merge_curation_report.json` written — flag-and-report only, `dataset/merged/` untouched.
- Same posture as DEC-060: this ranks candidates for human review, it does not decide or apply anything. The actual correction loop is still blocked on `docs/OPEN_QUESTIONS.md` #5.
- Ran independently of, and concurrently with, `split.py`'s real run (DEC-063) — safe because this is a torch/MPS-bound job and `split.py` is pure file I/O, not two MPS jobs contending for the same device (the resource-contention pattern this session had already learned to avoid).

---

## DEC-065: `generate_yaml.py` Built and Run (Stage 5.9) — Mechanical, No New Decisions

- **Date:** 2026-08-14
- **Status:** Accepted
- **Related:** DEC-063 (`split.py`, whose output this reads), `config/classes.yaml` (schema this must match), `config/training.yaml` (pre-existing `data:` reference this satisfies)

### Context

Last script in the handoff's scope table. Purely mechanical: write `dataset/final/data.yaml` matching `config/classes.yaml`'s existing, already-decided schema (`nc: 16`, `names:` in canonical id order) and `config/training.yaml`'s pre-existing `data: dataset/final/data.yaml` reference — no new judgment call, just executing an already-agreed format against real, now-populated data.

### Decision

Ran for real against `dataset/final/{train,val,test}/` (populated by DEC-063's real `split.py` run). Verified non-empty before writing (script's own built-in check): train=38,691, val=6,383, test=6,455 images+labels, matching. Wrote `dataset/final/data.yaml` with `path: .` (relative — training runs on RunPod, not this machine, per `config/training.yaml`'s own Phase 3 framing) and the class list.

**Verified for real, not assumed**: cross-checked the written `names:` block against `config/classes.yaml`'s own `names:` field (the authoritative 0-indexed mapping, distinct from that file's separate `classes:` metadata block, which is grouped by category and is NOT id-ordered — a real point of possible confusion caught by checking both, not just one) — exact match, all 16 classes, correct order (`Bicycle` at id 15 in both).

### Rationale

Nothing to weigh — this is Stage 5.9 executing a format two other files (`classes.yaml`, `training.yaml`) had already committed to before this script existed.

### Consequences

- `dataset/final/data.yaml` exists and is real-data-verified-correct. `config/training.yaml`'s `data:` reference now resolves to a populated file.
- This is the last script in the handoff's Stage 5.3-5.9 "Build" scope table — every item is now built and run for real except `reimport_corrections.py` (genuinely blocked, `docs/OPEN_QUESTIONS.md` #5).
- **Not a green light to train yet** — `dataset/final/` reflects the *pre-correction* pool (DEC-059/064's framing): Stage 5.5/5.7's flagged samples haven't been reviewed, and `docs/OPEN_QUESTIONS.md` has several open items. `data.yaml` pointing at real data means training is *mechanically possible*, not that the dataset is considered finished.

---

## DEC-066: Second Code Review Pass (post-DEC-061) — 14 Findings, Highest-Severity Ones Fixed and Verified

- **Date:** 2026-08-14
- **Status:** Accepted
- **Related:** DEC-061 (the first code-review pass, which covered box_audit.py/cap_per_class.py/merge.py/run_mistakenness.py as they existed at that point), DEC-062/063/064/065 (the scripts this pass covers)

### Context

Per the handoff's §4.5 suggestion, ran `/code-review` again against the scripts built/finalized since DEC-061: `dedup.py` (heavily redesigned since), `final_merge_curation.py`, `split.py`, `generate_yaml.py`. 14 findings came back. Each was checked against real behavior (not accepted on the reviewer's word alone) before deciding whether to fix.

### Decision

**Fixed, verified against real behavior:**
1. **`generate_yaml.py`'s `path: "."` doesn't do what its own docstring claimed** — the highest-severity finding. Verified directly by reading the actually-installed `ultralytics==8.4.118`'s `check_det_dataset()` source: an explicit `path` value resolves relative to the training process's CWD (or as absolute), never relative to the yaml file itself; that behavior only happens via a fallback (`Path(data["yaml_file"]).parent`) that's skipped whenever `path` is truthy. Fixed by omitting `path` entirely. Verified end-to-end: called the real `check_det_dataset()` against the real `dataset/final/data.yaml` from `/tmp` (a CWD with nothing to do with this project) — `train`/`val`/`test` all resolved correctly to `dataset/final/{split}/images`, all reported as existing. Also made the "verify splits non-empty" check in the same file an actual hard gate (`raise FileNotFoundError`) instead of a print-only warning, matching what its docstring already claimed it did.
2. **`split.py` reintroduced the stale-output bug DEC-050/061 already fixed twice** — no `shutil.rmtree()` before writing `dataset/final/{split}/`. Same fix as those two: clear before write. Also made the cross-split duplicate-leakage check (the entire point of DEC-063's union-find fix) a hard `raise`, not just a print, for the real run (still non-fatal under `--dry-run`).
3. **`split.py`/`merge.py`/`final_merge_curation.py` had inconsistent missing-file and unknown-class-id tracking** — `split.py` didn't track images with no matching label (merge.py already does, for the identical situation) or out-of-range class ids in its per-split stats (again, merge.py already does). `final_merge_curation.py` silently dropped eligible labels with no matching image, with no counter and no explanation for why `--dry-run`'s preview count could differ from the real run's. All three brought to parity: added `missing_labels`/`unknown_class_ids_in_split_labels` to `split_report.json`, `missing_images`/`eligible_labels_found` to `final_merge_curation_report.json`.
4. **`dedup.py`/`final_merge_curation.py` built image lists without the project's own `IMAGE_EXTENSIONS` filter** (`img_dir.iterdir()`/`is_file()` only) — the exact stray-file risk `file_utils.list_images()`/`build_stem_index()` exist to prevent, and `.DS_Store` files already exist elsewhere under this exact `dataset/` tree on this machine. Fixed to use `list_images()` (or an equivalent inline filter for `final_merge_curation.py`'s flat pooled directory, which `build_stem_index()` itself doesn't cover since it's per-source). **Verified this didn't silently corrupt either script's already-completed real run**: re-checked `dataset/merged/images/`'s real extension breakdown (1,645 `.jpeg` + 48,758 `.jpg` + 1,126 `.png` = 51,529, exact match, DEC-063) — no stray files existed when those runs happened, so this is a future-run robustness fix, not a correction to already-produced reports.
5. **`dedup.py`'s two `fo.Dataset()` constructions happened before the `try:` block**, so a failure creating the second would leak the first (a gap in DEC-061's own try/finally fix). Moved both inside `try`, `None`-guarded the `finally` cleanup. Also fixed an inconsistency where the exact-duplicate report used an unguarded `id_to_name[id]` lookup while the structurally-identical near-duplicate report used a defensive `.get(id, id)` — made both degrade the same way.

**An unplanned consequence of verifying fix #4**: a `--limit 50` smoke test of the *already-fixed* `dedup.py`, run to confirm the refactor didn't break the real (non-dry-run) code path, overwrote the real `dataset/reports/dedup_report.json` (671KB, full-scale DEC-062 results) with a 50-image smoke-test result — `dedup.py` always writes to the same report path regardless of `--limit`, a fact I knew but didn't account for before running the smoke test against the real output path. Caught immediately by checking the report's own `images_checked` field (50, not 51,529) rather than assuming the smoke test was harmless. Fixed by re-running `dedup.py` at full scale again; exact-duplicate results reproduced identically (1,893 groups/2,143 files — deterministic, filehash-based), near-duplicate results also reproduced identically given the same seed (818/520, see the re-run's real output in this doc's own edit history / `dataset/reports/dedup_report.json`'s current contents).

**Flagged but deliberately NOT fixed (latent, not currently triggered, or already-accepted scope):**
- **`split.py`'s `round(n * ratio)` split-size math can zero out a small source's val/test allocation** for `n<=7` duplicate-group-representative counts. Checked against real data: the smallest real source (`crowdhuman`, 128 images) still gets non-zero val (16) and test (16) — not triggered today. Left as a documented latent risk rather than speculative-fixed for a case that doesn't exist in this dataset.
- **`split.py`'s `load_duplicate_groups()` staleness check** (`images_checked < merged_image_count`) only catches an undercount, not a same-or-larger count with different file identities. This is a narrower version of a gap DEC-061 already flagged as deliberately unfixed for the analogous merge.py case; left consistent with that existing precedent rather than fixed unilaterally in only one of the two places it applies.
- **`split.py`'s unguarded `int(parts[0])`** during label parsing (a malformed non-numeric class id would raise uncaught) — confirmed this is a pre-existing pattern shared identically by `merge.py`'s equivalent loop, not something newly introduced in `split.py`. Left as-is rather than fixed in one file only, since all labels are generated by this project's own converters (external corruption is the only realistic trigger).
- **`final_merge_curation.py` duplicates ~35 lines of COCO-crosswalk-assertion and batched-inference logic from `run_mistakenness.py`** rather than importing it, despite the module docstring's claim of reusing that logic "directly... rather than duplicating." The two copies have already drifted cosmetically (`for coco_list in COCO_CROSSWALK.values()` vs `for key, coco_list in COCO_CROSSWALK.items()` — functionally identical, since neither loop actually uses the dict key). A real DRY violation and a legitimate maintainability risk, but refactoring it this late in an unattended session, with no time budget left to re-verify both scripts' real output afterward, carries more regression risk than value tonight. Left for the student as a documented follow-up, not silently ignored.

### Rationale

Verified before fixing, not on the reviewer's word: the `path: "."` finding in particular could have looked like reviewer overreach (the docstring sounded confident) — reading the actually-installed library's own source and testing against the real generated file from an unrelated CWD is what turned "plausible claim" into "confirmed bug, confirmed fix." The same verify-first posture caught that fixes #1-#5 didn't require re-running the already-completed `merge.py`/`final_merge_curation.py`/`split.py` outputs (no stray files, no missing labels, no leakage existed in those already-produced reports) — except `dedup.py`, whose report genuinely did need regenerating, for a reason (the smoke-test overwrite) unrelated to the code-review findings themselves.

### Consequences

- `scripts/build/generate_yaml.py`, `scripts/build/split.py`, `scripts/preprocess/dedup.py`, `scripts/curate/final_merge_curation.py` all patched; `dataset/final/data.yaml` regenerated with the `path` fix (train/val/test file counts unchanged — this was a resolution-logic fix, not a data fix).
- `dataset/reports/dedup_report.json` regenerated at full scale after the accidental smoke-test overwrite; real numbers unchanged from DEC-062 (1,893/2,143 exact, 818/520 near, both fully reproducible from the seeded/deterministic design).
- `dataset/reports/split_report.json` and `dataset/reports/final_merge_curation_report.json` were NOT regenerated — their fixes were additive-only (new report fields, defensive filters) and verified not to change already-produced output; re-running either would cost real compute time for zero change in conclusions.
- The DRY duplication between `run_mistakenness.py` and `final_merge_curation.py`, and the two lower-priority latent risks above, are left as open follow-ups — noted here and in the final handoff summary, not silently dropped.

---

## DEC-067: `cap_per_class.py` Generalized Beyond ExDark-Only Floors — Per-Class Priority Sources, Decided With the Student

- **Date:** 2026-08-15
- **Status:** Accepted
- **Related:** DEC-014 (ExDark's original guaranteed-floor rule, which this generalizes), DEC-042 (INSTANCE_TARGET, which this respects for general fill but required a carve-out for one priority source), `docs/OPEN_QUESTIONS.md` #6 (trim method)

### Context

The student asked a pointed question: `cap_per_class.py` only ever gave guaranteed-floor treatment to ExDark (DEC-014's low-light diversity layer) — every other source, regardless of its documented role in `config/classes.yaml` (`primary`/`secondary`/`role: volume_topup`) or actual deployment relevance, competed equally at random for whatever budget was left. Real numbers backed up the concern: for Person, `crowdhuman` had 19,370 candidate images (by far the largest pool, explicitly documented as a `volume_topup` secondary source) but only 128 were selected, because ExDark's floor alone consumed 74.6% of the shared 10,000-instance budget before crowdhuman or `open_images` were considered at all.

Went through `config/classes.yaml` and `config/datasets.yaml` class-by-class with the student to distinguish which competing sources are genuinely edge-case/deployment-relevant (documented as such), which are general-case, and which is really a quality-vs-volume tradeoff rather than either:

- **8 of 16 classes** already use every candidate from every source (`stop_reason: all_candidates_included`) — no competition exists, nothing to decide.
- **Person**: `crowdhuman` isn't an edge-case source, it's a documented `volume_topup` role being starved by ExDark's instance-heavy floor.
- **Vehicle**: `roboflow_me5_u6rvg` IS a documented edge-case source (`datasets.yaml`: "Tricycle presence is a genuine bonus given the class's Philippine-context relevance," Jeepney/Ambulance content).
- **Pole, Stairs**: no documentation found marking either side of either competing pair as edge-case — student confirmed no special priority needed, left as pure random pooling.
- **Elevator**: not edge-case-vs-general at all — `elevator_status_s4lrk` (larger) has Stage 5.3's flagged box-shape defects (844 boxes); `elevator_awvus` (smaller) doesn't. A quality preference, not a diversity one.
- **Animals, Chairs, Tables**: no edge-case-specific source exists for these at all (just ExDark + one general source) — nothing to decide.

### Decision

Generalized the guaranteed-floor mechanism from "always exactly ExDark" to a per-class ordered list, `CLASS_PRIORITY_SOURCES: dict[str, list[str]]` — every class not listed defaults to `["exdark"]` (today's behavior, unchanged), with three overrides:

- **Person**: `["exdark", "crowdhuman"]` — crowdhuman gets a second floor, after ExDark.
- **Vehicle**: `["exdark", "roboflow_me5_u6rvg"]` — me5_u6rvg gets a second floor, after ExDark.
- **Elevator**: `["roboflow_elevator_awvus"]` — awvus is the *only* priority source (no ExDark candidates exist for Elevator at all — outside DEC-014's 7-class overlap).

Priority sources are reserved in list order, each one's full candidate pool included first (up to whatever image/instance budget remains), before the next priority source, before any general source's random-pooled fill — the exact same posture DEC-014 established for ExDark specifically, now applied per-class to whichever source(s) `CLASS_PRIORITY_SOURCES` lists.

**A real complication surfaced immediately, not glossed over**: applying this naively to crowdhuman blew Person's realized instance count to 49,664 (target: 10,000) — crowdhuman's 1,842-image uncapped floor alone contributed ~42,200 instances, because CrowdHuman is exceptionally dense (mean 22.7 Person-instances/image, some images over 300). The student explicitly wanted this bounded ("may contribute to the long tail problem"). Resolved with a second mechanism: `PRIORITY_SOURCE_INSTANCE_SUBBUDGET: dict[tuple[str, str], int]`, currently `{("Person", "crowdhuman"): 2500}` — a dedicated instance allowance for that one (class, source) pair, filled **least-dense-image-first** (not randomly) to maximize image/scene diversity per instance spent rather than risk exhausting a small sub-budget on a handful of extreme-crowd outlier images. Verified against real data before picking the number: least-dense-first at a 2,500-instance budget yields 729 images.

### Rationale

The mechanism (ordered priority list + optional per-source instance sub-budget) generalizes cleanly from DEC-014's original single-source special case without disturbing it — every class not explicitly listed in either dict behaves exactly as before. The sub-budget's least-dense-first fill order is a direct, data-verified response to CrowdHuman's specific density profile, not a general policy — it only activates when a sub-budget is actually configured for that (class, source) pair.

### Consequences (real numbers, run for real 2026-08-15, `dataset/reports/cap_report.json` regenerated)

| Class | Metric | Before | After |
|---|---|---|---|
| Person | crowdhuman images | 128 | **729** |
| Person | total images / instances | 2,819 / 10,036 | 3,404 / 10,007 |
| Vehicle | roboflow_me5_u6rvg images | 1,649 | **3,188** |
| Vehicle | total images / instances | 4,500 / 8,289 | 4,500 / 6,756 |
| Elevator | elevator_awvus images (favored) | 1,491 | **1,777 (all)** |
| Elevator | elevator_status_s4lrk images (deprioritized) | 3,009 | 2,723 |

Person's total instance count landed at 10,007 — essentially back at DEC-042's original ~10,000 target, not the 5x-over blowout an uncapped floor would have produced.

**Not yet cascaded downstream.** `dataset/reports/cap_report.json` is regenerated and real, but `dataset/merged/` (and everything built from it — `dedup_report.json`, `final_merge_curation_report.json`, `dataset/final/`, `data.yaml`) still reflects the *previous* cap decision and is now stale relative to this one. Re-running `merge.py` (and, depending on scope, `dedup.py`/`split.py`/`generate_yaml.py`) is the explicit next step, deliberately held pending the student's go-ahead rather than run immediately — same "don't cascade a judgment call without confirmation" posture as every other student-decided threshold this session.

Also unresolved, deliberately not touched by this decision: Trash Bins is still below its 1,500 floor (1,104 images) and the ratio invariant still fails (4.08 vs 3:1) — both remain `docs/OPEN_QUESTIONS.md` #1, unrelated to today's fix.

---

## DEC-068: `cap_per_class.py` — Configurable `--hard-cap` Preset (1500 / 4500 / 9000), Floor Derived as `hard_cap // 3`

- **Date:** 2026-08-17
- **Status:** Accepted
- **Related:** DEC-042 (floor/hard_cap/instance_target/ratio-invariant policy this generalizes), DEC-067 (priority-source floors this interacts with), `docs/OPEN_QUESTIONS.md` (the still-open INSTANCE_TARGET-scaling and priority-source-starvation questions this decision does NOT resolve)

### Context

DEC-042 hardcoded a single hard_cap (4500, matched by every class's `cap` field in `config/classes.yaml`). Earlier this session, the student pushed back on the claim that "raising the cap to 9000 does nothing" — a claim that turned out to be true only *given* the crowdhuman instance sub-budget DEC-067 introduced, not true in general. The student wants to actually run the pipeline at different hard_cap values to see this tradeoff directly (default 4500, the real plan; 1500 and 9000 as comparison points) rather than reason about it hypothetically. This also matters for the still-open question of whether 4500's implied ratio (max/min = 4.08, see DEC-067) constitutes a meaningful class-imbalance problem — being able to run smaller/larger presets makes that an empirical question, not just a discussion.

### Decision

Added `--hard-cap {1500,4500,9000}` to `cap_per_class.py` (default 4500, `choices=`-restricted to the three presets rather than an arbitrary int — DEC-042 never established what an out-of-preset value would even mean). `FLOOR` is no longer a fixed module constant; it's derived every run as `hard_cap // 3`, directly applying DEC-042's own stated ratio invariant (hard_cap = 3× floor) rather than leaving floor pinned at 1500 regardless of hard_cap. At the 4500 default this derives floor=1500 — identical to before, a verified no-op. `INSTANCE_TARGET` (10000) is deliberately left unscaled — DEC-042 already establishes it as independently chosen, not formulaically tied to floor/cap, and no scaling rule was ever specified; inventing one now would be exactly the "guessing an undecided parameter" the script's own header already flags as out of scope. One consequence made explicit, not hidden: at `--hard-cap 9000`, `INSTANCE_TARGET` binds far more often (more classes hit the instance ceiling before the image ceiling), while at `--hard-cap 1500` it almost never binds (image ceiling arrives first for nearly every class) — a real behavioral shift, left as-is rather than "fixed."

Output path handling, to avoid a repeat of DEC-066's `dedup_report.json` overwrite mistake: the 4500 default writes to the canonical `dataset/reports/cap_report.json` path `merge.py` reads unconditionally. The 1500/9000 presets write to separate `cap_report_hardcap<N>.json` files — nothing downstream reads these yet, they exist purely for comparison, and must never silently clobber the canonical report.

**A real gap surfaced by testing this, not resolved by it**: at `--hard-cap 1500` (floor derived to 500), Person's ExDark priority source alone consumes the *entire* 1,500-image budget (2,658 ExDark candidates >> 1,500), leaving 0 images for crowdhuman — the exact "1500-preset starves second/third priority sources" risk flagged as an open question earlier this session, now concretely reproducible. Added detection (not a fix): each priority source's breakdown now carries `starved_next_priority_source: bool`, and `run()` prints an explicit `WARNING` when it fires. Verified via `--dry-run --hard-cap 1500`: fires correctly for Person (`exdark=1500, crowdhuman=0`), does not fire for Vehicle (`exdark=1312` doesn't exhaust the 1,500 budget, `roboflow_me5_u6rvg` still gets its remaining 188).

### Rationale

Deriving floor from hard_cap (rather than adding a second independent `--floor` flag) directly encodes DEC-042's own stated invariant instead of allowing a preset combination that violates it. Restricting to three named presets (not an arbitrary `type=int`) matches what was actually asked for and avoids inventing behavior for hard_cap/floor combinations nobody has reasoned about. Detecting-and-warning on priority-source starvation (rather than silently designing around it, e.g. giving every priority source its own dedicated sub-floor) was chosen because there is no established rule yet for how to fairly split a shrunken hard_cap across multiple priority sources — that is a real open question for the student to decide, same posture as every other undecided threshold this session, not something to resolve unilaterally.

### Consequences (verified via `--dry-run` at all three presets, then a real run at the default)

- `--hard-cap 4500` (default): real run executed, `dataset/reports/cap_report.json` regenerated — numbers reproduce DEC-067's run exactly (Person 3,404 img/10,007 inst, Vehicle 4,500 img/6,756 inst, ratio 4.08). Confirms the new code path is a true no-op at the default.
- `--hard-cap 1500`: NOT run for real (student doesn't need this yet). `--dry-run` confirms floor derives to 500, every class collapses toward the 1,500 ceiling, and the starvation warning fires exactly for Person (the one class where a single priority source's candidate pool exceeds the shrunken cap).
- `--hard-cap 9000`: NOT run for real. `--dry-run` confirms Vehicle grows to 7,373 images (up from 4,500) before `INSTANCE_TARGET` binds — directly reproducing the plateau-vs-growth tradeoff discussed earlier this session, now runnable rather than simulated.
- Still unresolved, deliberately not touched here: whether `INSTANCE_TARGET` *should* scale with hard_cap (no formula exists to apply), and how to fairly allocate a shrunken hard_cap across multiple priority sources when the first one alone can exhaust it. Both remain open, tracked in `docs/OPEN_QUESTIONS.md`.
- Cascade to `merge.py` and beyond is still explicitly gated on the student's go-ahead (DEC-067's posture, unchanged) — this decision only touches `cap_per_class.py` and its own report output.

---

## DEC-069: Batch Resolution of `docs/OPEN_QUESTIONS.md` — New Trash Bins Source, traffico_y1 Closed, Starvation Fix, Split Ratio, Ultralytics Compatibility Rule

- **Date:** 2026-08-18
- **Status:** Accepted
- **Related:** DEC-039/051/058 (Trash Bins), DEC-032/037 (traffico_y1), DEC-067/068 (cap_per_class.py priority sources), DEC-063 (split.py)

### Context

Student answered all 8 `docs/OPEN_QUESTIONS.md` items in one message. Several required real work, not just recording an answer — each checked against real data/APIs before acting, not assumed.

### Decision

1. **Trash Bins secondary (OPEN_QUESTIONS #1)**: checked 2 candidate Roboflow projects via SDK before adding either.
   - `eyecue/trashcan-detection-pihfn` — clean single class `Trashbin`, added to `config/datasets.yaml`, pulled (559 images), converted (0 dropped/clipped/invalid). Real result after a full `cap_per_class.py`+`merge.py` rerun: Trash Bins grew from 1,104 → **1,663 images**.
   - `ronits-workspace-e52mh/nyb` — real class list (`bin_elevated`/`bin_caged`/`bin_ground` alongside `z_action`/`z_no_action`/`trap_object`, an apparent unrelated pest/wildlife camera-trap context) and non-monotonic version sizes made this a real judgment call, not a clean add — deliberately NOT included, documented in `datasets.yaml` for the student to decide after looking at real images themselves.
   - License could not be verified for either (no SDK field, `WebFetch` 403 on both Universe pages, consistent with earlier blocks this project has hit) — flagged for the student to confirm directly.
   - **New finding, not previously visible**: with Trash Bins healthier, **Doors (1,337 images) is now the actual ratio-invariant minimum**, not Trash Bins. Ratio improved from 4.08 to 3.37 but is still above the 3:1 target — the DEC-042 recompute question in OPEN_QUESTIONS #1 is now about Doors, not Trash Bins.
2. **traffico_y1 (OPEN_QUESTIONS #2)**: formally closed. `audit_status: pending` → `benched` in `config/datasets.yaml`, same pattern as `jeep_hozhs`/DEC-037 — me5_u6rvg already covers what it would have added (DEC-053).
3. **cap_per_class.py priority-source starvation (OPEN_QUESTIONS #6)**: root-caused and fixed, not just scaled. The earlier open question ("scale the subfloor to hard_cap") undersold the actual bug — crowdhuman's *instance* subbudget wasn't what caused Person's starvation at `--hard-cap 1500`; ExDark's own *image* floor, unbounded, was. Fixed with a general per-priority-source minimum image reservation (`per_source_floor = floor // len(priority_sources)`), so no earlier priority source can fully exhaust a later one's share. `PRIORITY_SOURCE_INSTANCE_SUBBUDGET` also renamed to `..._BASE` and now scales proportionally via `scaled_instance_subbudget()`, per the student's literal request. Verified as an exact no-op at the default `--hard-cap 4500` (Person/Vehicle's real priority-source selections unchanged) before trusting it; verified it actually fixes the `--hard-cap 1500` case (Person: exdark 1250/crowdhuman 250, was 1500/0; Vehicle: exdark 1250/me5_u6rvg 250, was 1312/188).
4. **Split ratio (OPEN_QUESTIONS #8)**: changed from 75/12.5/12.5 to **70/15/15**, matching Ultralytics Academy's own documented baseline. Verified first (student explicitly asked) whether this even matters given Ultralytics might auto-split: confirmed via WebSearch that `ultralytics.data.utils.autosplit()` is optional/manual-invoke only — standard training requires pre-split directories, so `split.py` is necessary, not redundant.
5. **Standing rule recorded** (student's explicit request, not itself a dataset decision): added a subsection to `AGENTS.md`'s existing "Deployment Pipeline Awareness" section requiring every pipeline claim about Ultralytics' training-loop behavior to be verified against real docs/source, not assumed — using the autosplit finding above as the concrete example.
6. **`cap_per_class.py` + `merge.py` rerun** with all of the above combined (new eyecue source + starvation fix): `dataset/reports/cap_report.json` and `dataset/merged/` regenerated. **52,756 unique images merged** (up from 51,529). Per-class real counts: Person 4,152, Vehicle 4,645, Motorcycle 3,861, Pole 4,500, Animals 4,547, Stairs 4,500, Escalator 4,255, Doors 1,337, Chairs 3,442, Tables 4,662, Tricycle 3,495, Potholes 2,867, Trash Bins 1,663, Elevator 4,500, Pedestrian Lane 2,158, Bicycle 3,638.

### Rationale

Every item was checked against real data/APIs rather than answered from the question text alone — matching this project's dominant pattern all session. The starvation fix in particular needed root-causing rather than literal-instruction-following: scaling only the existing instance subbudget (as literally requested) would not have fixed the actual "crowdhuman=0" case, since that was an image-budget problem, not an instance-budget one.

### Consequences

- `docs/OPEN_QUESTIONS.md` updated: #1, #2, #6, #8 substantially resolved (details/verification above); #1 now centers on Doors, not Trash Bins.
- `dataset/reports/merge_report.json`, `dataset/reports/cap_report.json` reflect this run — anything downstream (`dedup_report.json`, `dataset/final/`) built before this is now stale until re-run (see DEC-070).

---

## DEC-070: `fiftyone_review_processed.ipynb` Extended (merged/final + flagged-only view); Full-Scale Dedup Attempted, Found Impractical Locally

- **Date:** 2026-08-18
- **Status:** Accepted
- **Related:** DEC-054 (notebook originally built), DEC-057 (box_audit.py flagged reports), DEC-062 (dedup.py, sampled near-duplicate check)

### Context

Two more `docs/OPEN_QUESTIONS.md` items needed real work: #3/#4 (the student didn't understand what Stage 5.3's flagged-box review actually required, and had no way to visually inspect it) and #7 (explicit request to run the near-duplicate dedup check at full scale locally, accepting "over an hour").

### Decision

- **Notebook extended** (`notebooks/fiftyone_review_processed.ipynb`): `source_key` now also accepts `"merged"` and `"final/<split>"` (via `merged_dir()`/`final_dir()`, both already in `file_utils.py`) alongside the original processed-source case. New `flagged_report_path` option loads a `box_audit.py`-style flagged-boxes report (list of `{label_path, class, cx, cy, w, h, reasons}`), restricts the loaded set to only images with ≥1 flagged box, and marks the *specific* flagged detection(s) with a `flagged` attribute (matched by rounded coordinates, not raw float equality) so they're distinguishable from an image's other, unflagged boxes — this is what actually answers "how do I review this," not just "how do I browse this."
  - Verified for real, not just written: `elevator_status_s4lrk_flagged.json` (844 flagged entries across 750 distinct images) loads to exactly 750 images and exactly 844 `flagged == True` detections (checked via `count_values`, not a misused `count(field, expr)` call that initially and incorrectly suggested only 750 — caught before trusting it). Also verified plain-source, `merged`, and `final/train` modes load correctly.
- **Full-scale dedup (`dedup.py --full-scale`, new flag)**: added and launched as a background run per the student's explicit instruction. **Result: projecting ~5 hours, not the "over an hour" the student anticipated** — running at 2-4 img/s versus the ~28 img/s clean benchmark and even the previously-documented partial-degradation case (~9-13 img/s) that motivated the original 6,000-image sample bound in the first place. Left running rather than killed (killing loses the fast, already-complete exact-duplicate results too — `dedup.py` only writes its report once, at the end, no incremental checkpointing) — **but this is flagged prominently to the student as a real finding, not left to silently run 5 hours unattended.** Matches the "worse comes to worse, RunPod" fallback the student themselves pre-authorized.

### Rationale

The flagged-report matching needed per-box, not per-image, granularity — a naive "does this image have a flag" check would have lost the distinction between an image's flagged and unflagged boxes, which is the entire point of the review. The dedup full-scale attempt was worth trying locally first exactly as asked, but the result itself is the useful information — better to surface a real 5-hour number than let a background job silently run far past what was actually agreed to.

### Consequences

- `notebooks/README.md` and the notebook's own top cell describe the new modes — not yet updated in this pass, worth a follow-up touch.
- `scripts/build/split.py` and `scripts/build/generate_yaml.py` reruns (needed regardless, since DEC-069's fresh `merge.py` output makes their existing results stale) are **blocked on the dedup question being resolved** — `split.py` reads `dedup_report.json` for duplicate-group leakage prevention. Waiting on the student's call: let the 5-hour run finish, kill it and accept the original 6,000-sample result, or move to RunPod.

---

## DEC-071: Second Doors Source Added (`door_detection_zqt59`) — Bottleneck Likely Resolved; Dedup Confirmed Stalled Post-Sleep, Not Corrupted-Display

- **Date:** 2026-08-18
- **Status:** Accepted
- **Related:** DEC-069 (Doors identified as the new post-eyecue bottleneck), DEC-070 (full-scale dedup launched)

### Context

The student found a second Doors candidate (`nathaly-espinoza/door-detection-zqt59`) directly addressing DEC-069's finding that Doors (1,337 images) is now the ratio-invariant bottleneck. Separately, the student's laptop slept mid-run during the background full-scale dedup from DEC-070; needed to confirm whether the process survived and re-assess the real ETA rather than trust the pre-sleep number.

### Decision

- **`door_detection_zqt59` added and pulled**: checked via Roboflow SDK before adding (same discipline as `trashcan_detection_pihfn`) — real class list is `door` (1908 instances), `knob` (376), `hinged` (1739), `lever` (1239). Only `door` is an unambiguous canonical match; `knob`/`lever` read as door-hardware sub-parts (same pattern as `cv_project_hovyc`'s Exit-signage classes, dropped). `hinged` is genuinely ambiguous from the class name alone — could be additional real door instances or another hardware sub-part — filtered out for this pull, flagged in `datasets.yaml` for the student's own visual check given its size (1,739 instances) is large enough to be worth revisiting.
  - Pulled version 3 (5,173 images, latest). Converted with `native_class_filter: "door"`: **4,493 images, 4,888 boxes kept, 8,643 non-canonical (knob/hinged/lever) boxes correctly dropped, 0 clipped, 0 invalid.**
  - Sits in `dataset/processed/door_detection_zqt59/` — **not yet folded into `dataset/merged/`**; requires a `cap_per_class.py` → `merge.py` rerun, deliberately not done yet (see Consequences).
- **Dedup background process (PID 48994) confirmed alive, not dead**: `ps` shows it still running; progress advanced from 6% (pre-sleep) to 36% (18,999/52,756) by the time of this check. CPU time (~4h) trailing wall-clock elapsed (~5.2h) by a gap consistent with a genuine sleep-induced stall, not a display glitch. **Post-resume throughput has not recovered to the original benchmark**: sustained ~1.1 samples/s over multiple checks (vs. the original 3.1 samples/s), putting the live ETA at ~9h remaining — worse than DEC-070's already-bad ~5h estimate, not an artifact that resolves itself. No `caffeinate` or equivalent is currently protecting the process from a repeat stall.

### Rationale

Same SDK-verification discipline applied to every prior source candidate (eyecue, `nyb`, `cv_project_hovyc`) — check the real class list before trusting a URL. Re-checking the dedup process rather than assuming the student's "I think it stopped" impression was correct, or assuming the opposite (that it was fine) — `ps`/timestamps are ground truth, guesses aren't.

### Consequences

- **`cap_per_class.py`/`merge.py` must NOT be re-run while `dedup.py` is still reading `dataset/merged/`**: `merge.py` does `shutil.rmtree()` on `dataset/merged/images|labels/` before rewriting them (DEC-059's stale-output fix) — running it now would delete the directory out from under the in-flight dedup read, likely crashing it or silently corrupting its result. This is a hard ordering constraint, not just a nice-to-have.
- Whatever happens to the current dedup run, **a follow-up merge+dedup cycle is needed regardless** once `door_detection_zqt59` (and any `elevator_status_s4lrk` label cleanup) are ready to fold in — the current run's ~52,756-image pool doesn't include the new Doors images at all. This weakens the case for treating "unblock split.py fast" as a reason to kill the current run: split.py can't be finalized until curation work (new Doors source + elevator review) is done and re-merged anyway, which was already going to require another pass.
- Student decision on the dedup run's fate (let finish / kill for RunPod / kill and accept the stale Aug-14 sample) still pending — re-asked directly.

---

## DEC-072: Full-Scale Dedup Run Killed (Student Decision); Write-Back Script Built for `fiftyone_review_processed.ipynb`

- **Date:** 2026-08-18
- **Status:** Accepted
- **Related:** DEC-070 (run launched), DEC-071 (found still stalled post-sleep, ~9h ETA)

### Context

Two things converged: (1) DEC-071 already established a follow-up merge+dedup cycle is needed regardless (new Doors source not yet merged), so continuing to burn hours validating an already-stale 52,756-image pool had diminishing return; (2) the student needs to restart their computer for resource reallocation, which would have killed the process anyway. Separately, the notebook's flagged-box review (Stage 5.3) had a real gap: FiftyOne's in-App annotation editor auto-saves to the session's live dataset, but nothing wrote those edits back to the actual `.txt` label files — a review session would produce nothing durable.

### Decision

- **Killed the dedup process** (`kill 48994`, confirmed dead, exit 143/SIGTERM as expected). Student's explicit call, made with full knowledge of the ~9h remaining estimate and the "another run is needed anyway" reasoning. Computer is safe to restart.
- **Write-back mechanism built** in `notebooks/fiftyone_review_processed.ipynb` (new markdown + code cell after the App-launch cell): reads the live, possibly-App-edited `dataset` object, converts each sample's `ground_truth.detections` back to YOLO format (inverse of the load transform), and writes to a new `dataset/processed/<source>/labels_reviewed/` staging folder — **not** the original `labels/` — printing an added/removed/modified/unchanged summary per run. Promoting reviewed files over the originals is a deliberate, separate manual step, not automated, since this path hasn't been used for a real correction pass yet.
  - The load cell (`p0review3build`) also now stashes `source_label_filename` on every sample (needed so write-back knows which file a sample's edits belong to) and attaches `flag_reasons` (the actual `box_audit.py` reason strings, e.g. `"large_area_outlier (>0.31)"`) as a visible detection attribute, not just the boolean `flagged`.
- Verified for real, not just written: simulated App edits (deleted one detection, moved another's bounding box, added a new one) against the live 750-image flagged dataset, ran the write-back logic, confirmed `added=1 removed=1 modified=1 unchanged=747` exactly matching the simulated edits, and confirmed byte-for-byte that the original `labels/` files were never touched (re-read from disk post-write-back, line counts matched pre-edit state).

### Rationale

Killing now rather than "let it finish for the data point" was the student's own reasoning, not just accepted at face value: the current run's coverage is already known-incomplete (missing the new Doors images), so its exact/near-duplicate results would need re-validation in the next pass regardless of whether this one finished. A staging-folder write-back (vs. overwriting `labels/` directly) matches this project's general caution around first-use, unverified-in-production write paths touching the only copy of curated annotations.

### Consequences

- No dedup report currently reflects the post-eyecue, post-`door_detection_zqt59` merged pool. The Aug 14 sampled report is the only one on disk; it predates both additions.
- `cap_per_class.py` → `merge.py` → `dedup.py` → `split.py` → `generate_yaml.py` all still need a final rerun, now explicitly deferred until curation (elevator review promotion, any further Doors work) is finished — not mid-flight, to avoid another wasted multi-hour pass.
- RunPod-vs-local decision for that final dedup pass is still open — student asked for a grounded speed estimate, answered inline in conversation (not a doc-worthy number, no benchmark run yet to cite).
- `labels_reviewed/` needs no new `.gitignore` entry — it's a subfolder of `dataset/processed/<source>/`, already wholesale-ignored (`.gitignore:36`).

---

## DEC-073: `door_detection_zqt59`'s `hinged` Class Confirmed Hardware (No Action Needed); `ronits-workspace-e52mh/nyb` Dropped; License Confirmed CC BY 4.0 for Both New Roboflow Sources

- **Date:** 2026-08-18
- **Status:** Accepted
- **Related:** DEC-069 (`trashcan_detection_pihfn` added, `ronits-workspace-e52mh/nyb` flagged but not added), DEC-071 (`door_detection_zqt59` added, `hinged` flagged as ambiguous), `docs/OPEN_QUESTIONS.md` #1 (all three sub-items closed by this entry)

### Context

Three loose ends from DEC-069/071 needed the student's own judgment, not something resolvable from API metadata alone: whether `ronits-workspace-e52mh/nyb` (the mixed pest/wildlife-camera-trap Trash Bins candidate) was worth a closer look; whether `door_detection_zqt59`'s `hinged` class (1,739 instances, filtered out pending review) was additional real door instances or hardware; and the license for both `trashcan_detection_pihfn` and `door_detection_zqt59`, which neither the Roboflow SDK nor WebFetch could surface.

### Decision

- **`ronits-workspace-e52mh/nyb` dropped.** Student's call: not worth pursuing — Trash Bins is no longer the bottleneck now that `trashcan_detection_pihfn` is in (DEC-069), and the mixed-context class list was reason enough on its own. Comment block in `config/datasets.yaml` updated to record the closure (not deleted, same as every other considered-but-declined source in this file).
- **`hinged` confirmed to be door-hinge hardware, not an additional door-type label.** Student's visual read: every image containing a `hinged` box also contains the door it's attached to, so `native_class_filter: "door"` (already in place since DEC-071) was already doing the right thing — dropping the hinge box while keeping the image, since the image's `door` box survives the filter regardless. No image is lost unless it contained *only* `hinged` boxes with no `door` box at all, which the student's review didn't find to be the case in this source. **No code change or re-pull needed** — this closes the open question as "already correct," not as a fix.
- **License confirmed CC BY 4.0 for both `trashcan_detection_pihfn` and `door_detection_zqt59`**, checked by the student directly on each Roboflow project page (neither the SDK nor `WebFetch` could surface it — SDK exposes no license field, WebFetch was blocked/403'd on both pages in earlier sessions). Recorded as a new `license:` field on both `config/datasets.yaml` entries, matching the field already used by `openimages`/`crowdhuman`/`exdark`.

### Rationale

Same pattern as every other source-inclusion judgment call in this project (DEC-037's jeepney swap, DEC-039's Trash Bins bench, DEC-069's eyecue-vs-nyb choice): when the deciding factor is real image content or licensing terms not exposed via API, it goes to the student rather than being guessed at. All three were flagged rather than silently resolved when originally found (DEC-069, DEC-071) — this entry closes them now that the student has looked.

### Consequences

- `docs/OPEN_QUESTIONS.md` #1 fully closed — all three sub-items resolved, nothing outstanding on the Doors/Trash Bins data-scope question.
- No re-pull, re-convert, or code change triggered by this entry. `door_detection_zqt59`'s existing `dataset/processed/roboflow_door_detection_zqt59/` output (4,493 images/4,888 boxes, DEC-071) remains correct as-is.
- Both sources are now confirmed license-clean (CC BY 4.0) for any external use, removing the last "unverified for external use" caveat DEC-069/071 had left open.

---

## DEC-074: Mistakenness Top-N Review Tooling Built for `fiftyone_review_processed.ipynb` — Bounded to 1,000 Images, Deferred-Full-Review Recommendation Superseded for This Slice

- **Date:** 2026-08-18
- **Status:** Accepted
- **Related:** DEC-060 (`run_mistakenness.py` built, 22,846 images scored), `docs/PLAN.md` Stage 5.5 ("Review and correct flagged samples... Todo")

### Context

Discussing whether to review Stage 5.5's mistakenness-ranked images before or after a first training baseline, the student asked directly why deferring was recommended, then — after the tradeoffs were laid out (partial 7/16-class coverage, a proxy model's disagreement being a heuristic not a ground truth, opportunity cost of reviewing 22,846 images against a baseline-informed alternative) — proposed a middle ground: review a bounded top slice (1,000 images, ~1 hour of their own time) rather than either extreme. No tooling existed for this; `fiftyone_review_processed.ipynb`'s existing modes are all single-source (`source_key`), but mistakenness ranking spans every source contributing to the 7 COCO-eligible classes at once.

### Decision

Added a new, clearly-separated section to `fiftyone_review_processed.ipynb` (4 new code cells + 2 markdown, after the existing box-audit write-back):

- **Build**: takes `mistakenness_report.json`'s top `MISTAKENNESS_TOP_N` (1,000) ranked entries, resolves each to a real image path via `build_stem_index()`, and — since the report only stored `gt_box_count`/`pred_box_count`, not actual box coordinates — **re-runs yolov8n inference fresh** for just this slice, importing `COCO_CROSSWALK`/`CANONICAL_KEY_TO_NAME`/`ELIGIBLE_CANONICAL_IDS`/`load_eligible_ground_truth` directly from `run_mistakenness.py` rather than reimplementing them, so this can't silently drift from what actually produced the ranking. Builds a FiftyOne dataset with both `ground_truth` (eligible-classes-only) and `predictions` (proxy model output, with confidence) as separate fields, plus the real `mistakenness` score as a sortable/filterable field — letting the student see *why* each image was flagged (the actual disagreement), not just that it was.
- **Write-back**: a real design difference from the box-audit write-back, not a copy-paste — `ground_truth` here only ever held the 7 eligible classes' boxes, so blindly writing it back out would silently delete any other canonical-class box an image happens to have. Instead, reads each original label file fresh, splits it into eligible/non-eligible lines by class id, keeps non-eligible lines exactly as they were, and only replaces eligible lines with the current (possibly edited) `ground_truth` state. Also deliberately does **not** clear the whole `labels_reviewed/` folder before writing (unlike the box-audit write-back) — a 1,000-image, multi-source review is more likely to span several sittings than the single-source elevator/stairs case, and shouldn't wipe out another already-promoted source's files on every re-run.
- Verified for real before handing over, not just written: ran the actual build path end-to-end on a real 20-image slice (real inference, real gt/pred box counts, e.g. one sample scored 0.9714 with gt=2/pred=3 — a real, visible disagreement). Separately verified the write-back merge logic against a real mixed-class label file (`open_images/3b6b7e47e005e7cb.txt`, Person + 2 Trash Bins boxes): simulated deleting the eligible Person box, confirmed both non-eligible Trash Bins lines survived byte-for-byte in the merged output, and confirmed the original file on disk was untouched.

### Rationale

The earlier "defer the whole mistakenness review past a first baseline" recommendation was reasonable as a default but overstated for a *cheap, bounded* slice — the real argument was against spending many hours reviewing a partial-coverage heuristic before ever training anything, not against any review at all. A ~1,000-image, ~1-hour pass costs little against that opportunity-cost argument while still catching anything glaringly wrong early. Reusing `run_mistakenness.py`'s own crosswalk/eligibility/box-conversion logic directly (matching `final_merge_curation.py`'s already-established pattern of doing the same) was chosen over reimplementing similar logic, so the review tool can never disagree with what actually produced the ranking it's reviewing.

### Consequences

- `dataset/processed/<source>/labels_reviewed/` is now a write target for **two** independent review flows (box-audit and mistakenness) that can touch different or overlapping sources — both write per-file, neither wholesale-clears a source's folder except the box-audit flow (single-source, single-sitting, safe to clear each run).
- Promotion (copying `labels_reviewed/*.txt` over the real `labels/`) remains manual for both flows — no auto-promote script exists yet.
- Doesn't change anything about Stage 5.7's `final_merge_curation_report.json` (the post-merge mistakenness pass) — that one's still stale and still deferred, same reasoning as before, just not addressed by this entry.

---

## DEC-075: `box_audit.py` Extended with `--pool merged` — Box Review Reordered to Run Post-Cap/Merge, Not Pre-Cap

- **Date:** 2026-08-18
- **Status:** Accepted
- **Related:** DEC-057 (original box_audit.py, pre-cap only), DEC-058 (`cap_per_class.py`, confirmed to not consume box-quality signal)

### Context

Auditing the full flagged-box picture (not just elevator/stairs) surfaced that `box_audit.py`'s Tukey-fence heuristic flags boxes across **all 16 sources**, not just the two with a dedicated reviewable list — 84,950 boxes total (14.6% of the 581,872-box pre-cap pool), of which elevator_status_s4lrk + stairs_i2yia's reviewable lists covered only 1,077 (1.3%). The other 14 sources' flags existed only as a truncated 300-entry sample inside `box_audit_report.json`, unreviewable at full detail. crowdhuman alone accounted for 68,430 (80% of all flags).

Separately, checking whether `cap_per_class.py`'s selection logic uses box-quality/flag data at all (it doesn't — confirmed by inspection, selection is purely priority-source + seeded-random against instance/image targets) exposed that auditing pre-cap wastes review effort proportional to each source's cap discard rate. Real numbers from the current (pre-Doors) merge: crowdhuman keeps only 729 of 19,370 candidate images (96.2% discarded) — meaning up to 96% of any pre-cap crowdhuman review would target images that never reach `dataset/merged/` at all. Other sources lose less (open_images 47.9%, stairs_i2yia 23.3%, elevator_status_s4lrk 24.1%), but the effect is universal and free to eliminate.

### Decision

- **`box_audit.py` gets a new `--pool {processed,merged}` flag** (default `processed`, unchanged existing behavior). `--pool merged` audits `dataset/merged/` instead: recovers each box's origin source + original filename from `merge.py`'s own `<source>__<original>` prefix (`file_utils.prefixed_filename`'s format) via a new `_split_prefixed_stem()` helper, then runs the identical per-(source, class) Tukey-fence logic (refactored into a shared `audit_boxes()` core, used by both pools) against that recovered population.
- In `merged` mode, writes a **full** flagged-boxes JSON per source (not the 300-entry sample) — cheap now that cap has already cut most sources down substantially. Filenames follow the existing `<source-without-roboflow-prefix>_flagged.json` convention (`elevator_status_s4lrk_flagged.json`, `stairs_i2yia_flagged.json`) so every source's flagged list is directly loadable by `notebooks/fiftyone_review_processed.ipynb`'s existing `flagged_report_path` mechanism with **zero notebook changes** — label paths are written as the original (unprefixed) filename, matching what that source's own `dataset/processed/<source>/labels/` expects.
- **Pipeline reordering**: box-shape review now happens *after* `cap_per_class.py` → `merge.py`, not before. Verified via a real dry-run against the current on-disk (pre-Doors) merged pool: crowdhuman's flagged count drops from 68,430 → 133 (99.8%), open_images 8,851 → 5,211, elevator_status_s4lrk 844 → 645, stairs_i2yia 233 → 178; exdark stays flat at 2,154 (100% cap survival via its floor reservation). Global class balance from the merged-pool parse matched `merge_report.json`'s own `real_post_merge_class_counts` exactly, confirming the source/filename recovery is correct.
- Real (non-dry-run) generation of the merged-pool report/flagged lists is **deliberately deferred** — the current `dataset/merged/` predates this session's Doors merge, so running it for real now would produce lists for a pool about to be replaced. Run `python3 scripts/preprocess/box_audit.py --pool merged` for real right after the next `cap_per_class.py` → `merge.py` pass (which folds in Doors).

### Rationale

`cap_per_class.py` not using box-quality signal at all means pre-cap auditing buys nothing for review purposes — it only costs wasted human review time on images that never reach training data. Reusing the existing per-(source, class) Tukey-fence/flagged-list machinery (via a shared `audit_boxes()` core) rather than writing a parallel merged-only script keeps both pools' flagging logic guaranteed identical, and preserving the existing flagged-JSON naming/label-path convention means the notebook needs no changes to review any of the 16 sources this way, not just elevator/stairs.

### Consequences

- The pre-cap `processed` pool mode (default, unchanged) remains useful as a cheap diagnostic — spotting a systemic per-source labeling problem early, before deciding cap priority — but is no longer the mode used to generate lists intended for box-by-box review.
- `elevator_status_s4lrk_flagged.json`/`stairs_i2yia_flagged.json` will be **overwritten** by the next real `--pool merged` run with smaller, more relevant (post-cap) lists — intentional, supersedes the pre-cap versions from DEC-057/070.
- Every other source (crowdhuman, open_images, exdark, etc.) will get its own reviewable flagged list for the first time once `--pool merged` runs for real — Stage 5.3's review scope question (raised this session) is now answerable at low cost, not blocked on hand-building per-source tooling.
- No `dataset/processed/` or `dataset/merged/` files are touched by this change — read-only, writes only to `dataset/reports/`, same posture as the original script.

---

## DEC-076: DEC-018r's "Near-Exhaustive" Roboflow Review — Scoped to Risk-Ranked Full Pass, Not a Thorough/Skip Split

- **Date:** 2026-08-18
- **Status:** Accepted
- **Related:** DEC-018r (original "may be reviewed near-exhaustively" commitment for smaller Roboflow pools), DEC-019 (large-scale sources explicitly exempted from manual review), DEC-075 (`box_audit.py --pool merged`, source of the flag-rate ranking used here), `docs/OPEN_QUESTIONS.md` #4

### Context

DEC-018r (2026-08-06) committed to "near-exhaustive" review of Roboflow-native pools, on the assumption this was practical at their scale. Checking that assumption for real this session: the 12 currently-merged small Roboflow pools total 30,072 images / 41,793 instances post-cap (cap only cut this group ~23%, since most of these sources already sit at or under their per-class caps — unlike crowdhuman's 96% cut). `docs/OPEN_QUESTIONS.md` #4 had been marked "Resolved," but that only confirmed `fiftyone_review_processed.ipynb` is *capable* of browsing a full pool — nobody had actually decided how the review itself would be scoped, and `docs/PLAN.md`'s own status table still listed it "Todo."

Discussed four options (exhaustive-as-decided, bounded top-N per source, risk-prioritized-skip-the-rest, defer-decide-later). Student rejected all four as posed and proposed a fifth: rank every small pool by risk, but not as a binary include/exclude split — go through **every image in every pool regardless**, varying *pace* (slow/thorough vs. fast skim) by that pool's rank.

### Decision

- **No source gets skipped.** Every one of the 12 small Roboflow pools (13 once `door_detection_zqt59` merges) gets a full pass through `notebooks/fiftyone_review_processed.ipynb` (`source_key` set to that source, `flagged_report_path=None` so every image is visible, not just flagged ones) — this is what "near-exhaustive" ends up meaning in practice, honoring DEC-018r's original commitment rather than quietly narrowing it.
- **Pace varies by a risk ranking**, computed from `box_audit.py --pool merged`'s per-source flagged-box rate (Tukey-fence outliers ÷ total boxes), cross-checked against DEC-031's already-documented Stairs/Elevator box-shape defect. Real ranking (post-cap, pre-Doors): `roboflow_cv_project_hovyc` (14.38%), `roboflow_elevator_status_s4lrk` (12.47%, DEC-031), `roboflow_stairs_i2yia` (12.32%, DEC-031), `roboflow_pothole_vhmow` (12.11%), `roboflow_escalator_stairs` (10.79%, DEC-031), `roboflow_pedestrian_and_animal_crossing` (10.23%), `roboflow_elevator_awvus` (9.31%, DEC-031), `roboflow_utility_poles_44tzx` (8.44%), `roboflow_pole_detection_z76mb` (7.97%), `roboflow_augmented_tricycle` (6.16%), `roboflow_me5_u6rvg` (4.49%), `roboflow_trashcan_detection_pihfn` (3.54%).
- **No hard cutoff between "thorough" and "skim" tiers** — student's explicit call: the rank is a pacing guide for their own attention, not a scope gate. `roboflow_escalator_stairs` (7,560 images, the largest pool) stays at its ranked position despite its size — its size is a structural consequence of feeding two canonical classes (Stairs *and* Escalator, each capped independently at 4,500; `merge.py` unions both classes' selections without dedup against a source's other class-quota usage), not evidence it should be treated specially.
- **Update 2026-08-18, same day**: `door_detection_zqt59` merged for real (DEC-077), ranking recomputed with it included — it takes **#1** at 15.88% (3,460 images), ahead of `cv_project_hovyc` (now 13.63%, its image share having dropped to 1,040 once Doors' cap selection split between both sources). Full 13-source ranking (post-Doors): door_detection_zqt59 15.88%, cv_project_hovyc 13.63%, stairs_i2yia 12.32%, elevator_status_s4lrk 12.20%, pothole_vhmow 12.11%, escalator_stairs 10.79%, pedestrian_and_animal_crossing 10.23%, elevator_awvus 9.31%, utility_poles_44tzx 8.44%, pole_detection_z76mb 7.97%, augmented_tricycle 6.16%, me5_u6rvg 4.49%, trashcan_detection_pihfn 3.54%.
- **`elevator_awvus`'s DEC-031 status does not override its measured 9.31% rank** — asked directly, student's explicit answer: the rank stays purely the hard number `box_audit.py` produces; DEC-031-known-issue status isn't a manual boost. The human-judgment layer belongs to the student during the actual review pass ("it will be up to me... to be more careful based on what i actually see"), not baked into the ranking itself.

### Rationale

A binary thorough/skip split risked either under-covering pools that didn't make a cutoff (leaving DEC-018r's commitment quietly unmet, the exact gap this session found) or over-committing time uniformly regardless of actual risk signal. Variable-pace-across-the-full-pool keeps the commitment intact while still using the real, data-driven flag-rate signal to direct where careful attention actually pays off, rather than a rate number silently deciding what gets looked at at all.

### Consequences

- `docs/OPEN_QUESTIONS.md` #4 should be updated to reflect this as the actual resolution (a scoped review *approach*, not just tooling availability) — pending next docs pass.
- `docs/PLAN.md`'s Stage 5.3 "Near-exhaustive review for smaller Roboflow pools" row stays "Todo" until the student has actually gone through all 12/13 pools — this decision sets the method, not completion.
- No new tooling needed — `flagged_report_path=None` (full-pool browsing) already existed in the notebook; this decision is about how the student uses it, not a code change.

---

## DEC-077: `cap_per_class.py` + `merge.py` Rerun — Doors Folded In, Ratio Invariant Now Met

- **Date:** 2026-08-18
- **Status:** Accepted
- **Related:** DEC-069 (ratio invariant last measured 4.08, Doors identified as the bottleneck), DEC-071 (`door_detection_zqt59` acquired, deliberately held out of the merge until now), `docs/OPEN_QUESTIONS.md` #1

### Context

`door_detection_zqt59` (4,493 images) had been sitting processed-but-unmerged since DEC-071 (2026-08-18, earlier same day) — deliberately held back while the box-audit review notebook and the (later-killed, DEC-072) dedup run were in flight. With those blockers cleared and the review-scope questions (DEC-075, DEC-076) settled, this was the first point Doors' new source could actually be folded into the trainable pool.

### Decision

Ran `cap_per_class.py` then `merge.py` for real. Real results:

- **Doors: 1,337 → 5,830 candidates → 4,500 selected images / 5,029 instances** (hit the hard image cap, `stop=image_hard_cap` — the first time Doors has had *more* candidates than the cap allows, rather than being candidate-starved).
- **Ratio invariant now met**: `max=Vehicle(4,500) / min=Trash Bins(1,663) = 2.71` (≤ 3.0) — resolves `docs/OPEN_QUESTIONS.md` #1's last open sub-item ("if Doors is still short after the merge... fall back to (a) accept ratio or (b) recompute cap") without needing either fallback.
- **Merged pool: 52,756 → 55,917 images** (up 3,161 — mostly Doors' net gain, since `cv_project_hovyc`'s prior 1,337 was already counted). Verified on disk: `dataset/merged/images/` and `labels/` both contain exactly 55,917 files, `door_detection_zqt59` contributes 3,460 of Doors' new 4,500-image selection directly.
- Every other class's real post-merge count moved only marginally (e.g. Person instances 12,745 → 12,814, Tables images 4,662 → 4,675) — expected noise from the seeded-random general pool, not a Doors side-effect.

### Rationale

Mechanical rerun of already-built, already-decided logic (DEC-058/059) — no new policy decisions, just executing the pipeline now that its inputs (Doors) and blockers (review-scope questions) are resolved.

### Consequences

- `docs/OPEN_QUESTIONS.md` #1 can be closed entirely now — its last open sub-item is resolved.
- The `dataset/merged/` pool driving `box_audit.py --pool merged` (DEC-075) is no longer stale — real (non-dry-run) flagged-list generation, deferred in DEC-075 for exactly this reason, can now proceed.
- `elevator_status_s4lrk_flagged.json`/`stairs_i2yia_flagged.json` will be regenerated (and every other source will get one for the first time) against this pool, not the pre-Doors one DEC-075's dry-run smoke-tested against — numbers will shift slightly from what was shown then.
- Downstream (dedup, split, generate_yaml) all still pending a rerun against this pool, per the established plan.

---

## DEC-078: Whole-Image Removal via App Tag (`exclude`), Not Sample Deletion — `merge.py` Reads a Per-Source Exclusion File

- **Date:** 2026-08-18
- **Status:** Accepted
- **Related:** DEC-072 (box-audit write-back cell, the mechanism this extends), DEC-076 (the risk-ranked full-pool review this unblocks)

### Context

Starting the DEC-076 full-pool review surfaced a real gap: the review notebook could edit an image's boxes, but had no way to remove an image from the dataset entirely (wrong content, duplicate, doesn't belong). The obvious approach — delete the sample in the FiftyOne App, detect its absence in the write-back cell — was checked against the actual write-back code (`for sample in dataset: ...`) and confirmed to silently do nothing: a deleted sample just isn't iterated, so no file is written for it either way, and the original image/label in `dataset/processed/<source>/` is never touched. Asked whether a simpler or more robust alternative existed before building the deletion-diffing version.

### Decision

- **Tag, don't delete.** Verified via `docs.voxel51.com`: FiftyOne's App supports tagging samples natively (a tag icon above the sample grid, works on single or multi-selection) — no plugin needed, same "native App feature" category as the box editing already in use. The write-back cell (`notebooks/fiftyone_review_processed.ipynb`, cell `2343228f`) now checks `"exclude" in sample.tags` per sample.
- Newly-excluded filenames are merged with any existing exclusions for that source (read from `dataset/reports/<source>_excluded.json` if present, unioned, rewritten) — same "don't clobber an earlier sitting's work" posture as DEC-074's mistakenness write-back, since a full-pool review spans multiple sessions.
- An excluded sample still gets its `labels_reviewed/` file written normally — exclusion is tracked as a separate, parallel signal, not a special case in the box-writing logic. Untagging before a later write-back run removes it from the exclusion list on that run.
- **`merge.py` is the enforcement point**, not `cap_per_class.py`. New `load_excluded_pairs()` reads every `dataset/reports/*_excluded.json` (globbed, each file's own `"source"` field is authoritative — not parsed from the filename) and subtracts that set from `cap_report.json`'s selected union before copying. `cap_per_class.py`'s selection is **not** re-run — an excluded image's quota slot is simply lost, not backfilled with a replacement candidate, to avoid cascading the seeded-random selection elsewhere just because one image got excluded.
- Verified for real, not just written: smoke-tested against 2 real `roboflow_trashcan_detection_pihfn` filenames actually present in `cap_report.json`'s selection — `merge.py --dry-run` correctly dropped the union from 55,917 to 55,915, then the smoke-test exclusion file was deleted, leaving no trace on real state.

### Rationale

Tag-based exclusion is both simpler and more robust than the deletion-diffing alternative it replaced before being built: no need to snapshot the original file list at load time and diff it against the live dataset at write-back time, and — more importantly — it makes the student's intent explicit and visible (a tag persists and is inspectable in the App) rather than inferred from a side effect (a sample's absence, which could have other causes). It's also reversible mid-session: untag to change your mind, no full source reload required.

### Consequences

- `dataset/reports/<source>_excluded.json` is a new, small, per-source file — one for any source where at least one image gets excluded during review.
- `merge_report.json` gains an `excluded_count` field.
- A class that loses an excluded image ends up marginally under its `cap_report.json` figure — expected, not a bug; re-running `cap_per_class.py` to backfill was deliberately not automated (see Decision above).
- This mechanism is currently only wired into the box-audit-style write-back cell (the one used for DEC-076's full-pool review), not the separate mistakenness write-back cell — not needed there yet, could be extended the same way if the student wants it.

---

## DEC-079: Predictions Overlay Generalized to Full-Pool Review (`show_predictions`); Per-Label `accept` Tag Promotes a Prediction into Ground Truth

- **Date:** 2026-08-18
- **Status:** Accepted
- **Related:** DEC-074 (mistakenness top-N section, source of the reused inference logic), DEC-078 (`exclude` sample-tag pattern this mirrors at the label level), `docs/preprocessing.md` (pre-existing, narrower "Model-Assisted Pre-Labeling" procedure this generalizes)

### Context

Student flagged missing-label detection as a real problem across the DEC-076 full-pool review (~33k images) and asked to automate it as much as possible. Checked the actual ceiling first: any pretrained-model assist is bounded by COCO's vocabulary, which only overlaps 7 of 16 canonical classes (Person, Vehicle, Motorcycle, Bicycle, Animals, Chairs, Tables — `run_mistakenness.py`'s `COCO_CROSSWALK`). Student confirmed this ceiling is acceptable — the classes they're most worried about missing labels for are exactly the COCO-eligible ones (incidental background objects, e.g. a person walking through a Door-labeled scene), not each source's own deliberately-labeled primary class. Matches `docs/preprocessing.md`'s existing (but only ever applied once, to `augmented_tricycle`) "Model-Assisted Pre-Labeling for Missing Classes" procedure.

Also asked why the mistakenness section caps at `MISTAKENNESS_TOP_N = 1000`. Checked rather than assumed: benchmarked real `yolov8n` inference on this machine against real project images — **32.4 img/s**, meaning even the largest single small-pool source (`escalator_stairs`, 7,560 images) finishes in under 4 minutes, and all 13 small pools combined would take ~17 minutes if run eagerly. The 1,000 cap is a *student review-time* budget (DEC-074: "~1hr"), not an inference-cost constraint — doesn't apply to a per-source overlay used alongside a full-pool pass the student is already committed to.

Finally asked for the acceptance mechanism, correctly anticipating that naively merging `predictions` into `ground_truth` would create redundant duplicate boxes (one from each field, over the same real object). Checked FiftyOne's real capabilities via `docs.voxel51.com` before designing: individual **label-level** tagging (not just sample-level) is a native App feature — "any label or collection of labels can be tagged at any time in the sample grid or expanded sample view" — and there's a Patches view (`Patches > Labels > predictions`) that shows each individual detection as its own thumbnail for faster triage. No built-in "copy field A's label into field B" App action exists; that part needed code.

### Decision

- **`show_predictions` (new toggle, `notebooks/fiftyone_review_processed.ipynb`, cell `p0review2source`, default `False`)**: when set, the main load cell (`p0review3build`) runs `yolov8n` inference over *every* image in whatever `source_key` is currently loaded (no top-N cap) and attaches a `predictions` field alongside `ground_truth`, reusing `run_mistakenness.py`'s `COCO_CROSSWALK`/`CANONICAL_KEY_TO_NAME` directly (not reimplemented, so it can't drift from what that script does).
- **Acceptance mechanism**: tag a specific predicted box `accept` in the App (click it, tag from the Labels list or Patches view) — a per-*label* tag, distinct from DEC-078's per-*sample* `exclude` tag. The write-back cell (`2343228f`) now promotes any `accept`-tagged prediction into `ground_truth` (appending, not replacing) *before* its existing write logic runs, so the rest of the pipeline treats it like any other ground-truth box. Predictions themselves are never written to `labels_reviewed/` or anywhere else — only `ground_truth`'s post-promotion state does — so the redundant-duplicate-box concern doesn't arise: nothing merges both fields wholesale, only the specific boxes the student explicitly accepted.
- Verified for real: benchmarked inference (32.4 img/s, real images), then a full mechanism smoke test against 10 real `roboflow_trashcan_detection_pihfn` images — found 4 real predictions (confirming the incidental-Person pattern), simulated tagging one `accept`, confirmed `ground_truth` grew 3→4 boxes exactly, simulated tagging the same sample `exclude` simultaneously and confirmed both mechanisms coexist cleanly on one sample.

### Rationale

Generalizing the existing mistakenness-section inference logic (rather than writing a second, separate implementation) keeps the crosswalk/model/device-selection logic in one place. Label-level tagging for acceptance (rather than, say, a confidence-threshold auto-accept) keeps a human decision in the loop for every promoted box, consistent with this project's standing posture that scripts flag/assist but don't silently decide — same principle as `box_audit.py`'s flag-don't-fix stance and DEC-078's tag-don't-infer choice for exclusion.

### Consequences

- `show_predictions = True` adds real time to the load cell (a few minutes for large sources) — off by default so the existing fast path is unaffected when not wanted.
- The write-back cell now has three stages in order: promote accepted predictions → track exclusions → write ground_truth to `labels_reviewed/`. Order matters: promotion must land in `ground_truth` before the write logic reads it.
- No change to the mistakenness section itself (`MISTAKENNESS_TOP_N` stays 1,000) — that cap remains a deliberate review-time bound, not something this entry argues should change.
- Doesn't help with the 9 non-COCO classes at all — explicitly out of scope, not a gap to revisit without a trained model of this project's own first.

---

## DEC-080: Full-Pool Review Restricted to the Merged (Post-Cap) Pool by Default (`restrict_to_merged`)

- **Date:** 2026-08-19
- **Status:** Accepted
- **Related:** DEC-075 (same fix, already applied to `box_audit.py`), DEC-076 (the full-pool review this corrects), DEC-078 (why an unselected image can't be backfilled)

### Context

Student caught a real discrepancy while actually using the notebook: `door_detection_zqt59` was described as contributing 3,460 images to the trained pool, but the full-pool review cell (`flagged_report_path=None`) reported loading 4,493 — the entire raw `dataset/processed/roboflow_door_detection_zqt59/` pool, not the 3,460 `cap_per_class.py` actually selected into `dataset/merged/`. Root cause: `source_key` browsing for a plain processed source always resolved to `processed_dir(source_key)/images` with no awareness of `cap_report.json`'s selection at all.

This is the exact problem DEC-075 already fixed for `box_audit.py` (auditing the full pre-cap pool wastes effort on images that get discarded) — it just never got applied to the notebook's own full-pool browsing mode. The flagged-mode path (`flagged_report_path` set) was accidentally already correct, since `box_audit.py --pool merged` only ever scans `dataset/merged/` in the first place — the gap was specific to `flagged_report_path=None`, DEC-076's actual review mode.

### Decision

New `restrict_to_merged` toggle (default `True`) in the config cell. When on (the default) and browsing a plain processed source with `flagged_report_path=None`, the build cell filters to only images present in `dataset/merged/images/` under that source's prefix — i.e., exactly what `cap_per_class.py` selected, recovered via the same `<source>__<filename>` convention `box_audit.py --pool merged` (DEC-075) already uses. Prints a count of how many were skipped. Set to `False` to see the full raw pool anyway (e.g. deciding whether a source needs a manual cap bump).

Verified for real: `door_detection_zqt59` — 4,493 in `dataset/processed/`, 1,033 correctly skipped, 3,460 reviewed. Exact match to the real `dataset/merged/` count.

### Rationale

An unselected candidate has no path to reaching the trained model under the current design — DEC-078 established that an excluded image's cap slot is not backfilled by re-running `cap_per_class.py`, so there was never a mechanism that would pull a currently-unselected image into the pool based on review outcome. Reviewing it costs real time for zero possible effect, identical in kind to the crowdhuman case DEC-075 already solved — this was a gap in applying that same principle consistently, not a new tradeoff.

### Consequences

- Every small-pool source's real full-pool review count is now smaller than `dataset/processed/<source>/`'s raw count, sometimes substantially (Doors: 4,493 → 3,460, a 23% cut). The DEC-076 risk ranking's flag-rate percentages don't change (already computed against the merged pool via DEC-075), but the number of images a full-pool pass actually has to look at drops for every source with unselected candidates.
- If `cap_per_class.py`/`merge.py` are re-run later (e.g. after further source changes), `dataset/merged/` membership shifts and a source's "restricted" pool for review should be treated as current-as-of-last-merge, same caveat `box_audit.py --pool merged` already carries.

---

## DEC-081: `dataset.classes` Set on Every Review Dataset — Fixes the App's "Import Your Dataset Schema" Block on Drawing New Boxes

- **Date:** 2026-08-19
- **Status:** Accepted
- **Related:** DEC-072/078/079 (the review notebook's editing mechanisms this unblocks)

### Context

Student hit a real wall moving to their second source (`roboflow_cv_project_hovyc`): existing box edits, `accept`/`exclude` tagging all worked fine, but clicking the App's "Annotate" tab to draw a brand-new box showed "Annotate faster than ever — Import your dataset schema..." instead of letting them draw. Student shared the dataset's field-schema JSON, which showed the real gap directly: every field's `default_label_schema` listed attributes (`confidence`, `id`, `index`, `mask_path`, `tags`) but **no `classes` list anywhere** for `ground_truth`/`predictions`.

Checked via `docs.voxel51.com` before proposing a fix: FiftyOne's Annotate tab has required an explicit "Annotation Schema" per dataset since v1.16–1.18 — by default no fields are auto-included, and a field needs a defined class list to offer a class-selection dropdown for a *new* detection (existing-box geometry edits and sample/label tagging don't need one, which is exactly why those already worked). `dataset.classes` — a dict of `{field_name: [class, ...]}` — is confirmed as a real, long-standing, non-Enterprise SDK property (documented under core "FiftyOne Concepts," not the newer Ontology system), and was empty (`{}`) on the live dataset, confirming it as the actual gap.

### Decision

- Applied directly to the two datasets already live in the student's session (`review_roboflow_cv_project_hovyc`, `review_roboflow_door_detection_zqt59`) without requiring a rebuild — `dataset.classes = {"ground_truth": CANONICAL_NAMES, "predictions": CANONICAL_NAMES}` then `dataset.save()` (required for in-place property changes to persist, per FiftyOne's own docs). Verified for real: reloaded each dataset fresh from the backing DB in a separate process afterward and confirmed the 16-name list actually persisted, not just held in the mutated Python object.
- Added the same two lines to the build cell (`p0review3build`) right after `dataset.add_samples(samples)`, so every source from here on gets this automatically — no more per-source manual patching needed.

### Rationale

Standard SDK mechanism over the newer Annotation Ontology system (`fo.AnnotationOntology`/`fo.apply_ontology()`), which surfaced during research but reads as built for reusable/versioned schemas shared across datasets — heavier than needed here, and not confirmed to even be available outside FiftyOne Enterprise. `dataset.classes` is the documented, minimal mechanism for exactly this need (declaring valid class strings per label field) and was directly confirmed as the missing piece by the student's own shared schema JSON.

### Consequences

- Every remaining source's review dataset will have its Annotate tab class dropdown pre-populated from `config/classes.yaml`'s 16 names — the "Import your dataset schema" prompt should no longer need any manual class configuration; whatever remains of that prompt should just reflect the classes that are now already declared.
- Not fully verified end-to-end against the App's own UI (couldn't click through it directly) — confirmed the SDK-level fix (classes now populated, persisted) but the student still needs to confirm the Annotate tab itself now behaves correctly.

---

## DEC-084: `pothole_vhmow`'s 1302→871 Image Gap Investigated — Correct Version Downloaded, Gap Is Zero-Annotation Images Dropped by Design

> Renumbered from DEC-082 to DEC-084 (2026-08-20) — a numbering collision from two concurrent sessions editing this file (this repo has a deliberate split-agent setup, see the architectural handoff doc). This entry had zero external cross-references at the time of the collision, unlike the other DEC-082 (`stairs_i2yia`/`escalator_stairs`/`elevator_status_s4lrk`), which was already referenced in 17+ places — renumbering this one instead was the lower-cost fix. Content below is unchanged.

- **Date:** 2026-08-20
- **Status:** Accepted
- **Related:** DEC-034 (`pothole_vhmow` added as co-primary Potholes source); `datasets.yaml:492`'s `pinned_version: 18` comment

### Context

Student, reviewing `pothole_vhmow`'s box quality in FiftyOne, separately noticed a large fraction of exact/near-duplicate images in the same source, and asked whether the wrong Roboflow version might have been downloaded — recalling the project should have 1302 images, while only 871 are in `dataset/processed/roboflow_pothole_vhmow/`.

Verified directly against Roboflow's own bundled export metadata rather than re-deriving from `datasets.yaml` alone: `dataset/raw/roboflow_projects/pothole_vhmow/README.roboflow.txt` states "pothole - v18 2024-06-11 7:34pm" and "The dataset includes 1302 images"; `data.yaml` confirms `version: 18` and `url: .../pothole-vhmow/dataset/18` — both an exact match to `datasets.yaml:492`'s `pinned_version: 18` and to the student's own memory of 1302. All 1302 images are present on disk under `train/images` (this version's project owner put 100% of images in the train split, 0 in valid/test — a project configuration choice on Roboflow's side, not a partial/broken download).

Checked the 1302 raw label files directly: 431 are completely empty (0 annotated boxes), 871 contain at least one box (2,189 box lines total, 0 degenerate). `yolo_to_intermediate.py` (used by every Roboflow source, not just this one) drops any image whose label file yields zero valid boxes after conversion (`images_dropped_empty` stat) — exactly accounting for the 1302 → 871 gap, and matching the existing conversion log exactly ("`pothole_vhmow` 871/2,189" in the Stage 5.2 conversion run entry).

### Decision

No re-download needed — the correct version (v18, 1302 images) was pulled correctly. The 871-image processed count is correct, expected behavior given the existing (pipeline-wide, not source-specific) policy of dropping zero-annotation images during conversion. Not treated as a bug.

### Rationale

Cross-checked against Roboflow's own export metadata (`README.roboflow.txt`, `data.yaml`) rather than trusting `datasets.yaml`'s comment alone, since that comment could itself have been stale or wrong — it wasn't. The empty-label-file count (431) exactly closes the gap to 1302 with no unexplained remainder, and the drop behavior is a general converter policy already applied uniformly to every other Roboflow source, not something unique to `pothole_vhmow` that would suggest a source-specific error.

### Alternatives Considered

- **Re-download `pothole_vhmow` to rule out a corrupted/partial pull**: Rejected — the raw files' own embedded metadata (README/`data.yaml`, generated by Roboflow at export time, not by this project's tooling) already independently confirms version and count; a re-download would reproduce the identical 1302/431/871 split, since the 431 zero-box images are how the source project itself was annotated, not an artifact of this pipeline's download step.

### Consequences

- The duplicate-data finding from the same review session (22 exact-duplicate groups/44 files, plus a partial near-duplicate signal from `dedup_report.json`'s sampled check) is a separate, still-open, and still-real issue — unaffected by this investigation. Confirmed not to be an artifact of a wrong-version download either (same raw v18 pull verified); genuinely duplicate content within the correct version.
- Open, not yet decided: whether zero-annotation images should be kept project-wide as explicit background/negative examples instead of always being dropped by `yolo_to_intermediate.py`. This is a pipeline-wide converter policy question (affects every Roboflow source with any zero-box images, not just `pothole_vhmow`) — flagged here, not decided, since it wasn't what the student asked about.

---

## DEC-082: `stairs_i2yia`/`escalator_stairs`/`elevator_status_s4lrk` Failed on Direct Visual Review; Two Replacement Sources Added and Pulled; Escalator's Schema Slot Reserved, Not Renumbered

- **Date:** 2026-08-20
- **Status:** Accepted
- **Related:** DEC-031 (original box-shape finding this supersedes with a stronger read), DEC-057/DEC-075 (box_audit.py's flagged lists for these sources, now understood to undercount the real defect), DEC-042 (floor/ratio-invariant policy affected by losing two sources), `docs/OPEN_QUESTIONS.md` #4 (the DEC-076 full-pool review that surfaced this)

### Context

Partway through the DEC-076 risk-ranked full-pool review, the student's direct visual read on 3 sources diverged sharply from what the automated tooling had shown: `stairs_i2yia` and `escalator_stairs` had duplicate/near-duplicate images and boxes that don't land on the annotated object at all (not just imperfect shape), and `elevator_status_s4lrk` had large corrupted black regions baked into the source images themselves. The student's framing: "it would be like i annotated the images myself."

Before accepting this over the existing `box_audit_report.json` flagged-rate numbers (which showed these sources in the same range as several others, not obvious outliers), independently sampled and rendered real image+box pairs — both flagged and random/unflagged — from all three sources plus `elevator_awvus` (the other Elevator source) for comparison. Also diffed one box's raw-export coordinates against its converted intermediate-schema coordinates to rule out a conversion-pipeline bug before concluding the defect was in the source data itself.

### Decision

- **All three marked `audit_status: failed`** in `config/datasets.yaml`, with real evidence recorded in each entry's `audit_note`:
  - `stairs_i2yia`: 3/3 independently-sampled images showed boxes not on the staircase — one on a floor rug below the actual stairs, one a vertical sliver on the image's far-left edge nowhere near the staircase visible in the center of frame.
  - `escalator_stairs`: checked its Stairs-labeled slice specifically (not just Escalator), since this source is dual-canonical (`canonical_classes: [stairs, escalator]`) and losing it affects Stairs' volume too. 2/3 Stairs-only samples were low-resolution (227×227), heavily grainy, monochrome; a third closely matched an already-checked full-size image (same stairwell, same rag-on-step, same shadow pattern) closely enough to read as a near-duplicate. Both native classes excluded, not just escalator content.
  - `elevator_status_s4lrk`: 3/3 independently-sampled images (1 flagged, 2 random) showed a large corrupted black polygon void baked into the source image itself (not a rendering artifact), with the box sitting mostly on that void instead of the real door/panel. One sample visibly carries an "alamy" stock-photo watermark. This overturns DEC-031's original "genuine usable subset" framing for this source.
  - Conversion-pipeline bug ruled out first: raw-export vs. converted-intermediate coordinates matched exactly (to float precision) on a real `escalator_stairs` box — whatever's wrong originates in the source's own export, not `yolo_to_intermediate.py`.
  - `elevator_awvus` checked as a comparison point: 3/3 independently-sampled images showed correctly-placed, tight boxes on real elevator doors — confirmed clean, stays the sole active Elevator source pending the new addition below. (Also visibly stock-photo-watermarked in-frame, same as the failed source — a provenance note, not a quality one.)
- **Two replacement sources found by the student, SDK-checked, pulled, and converted for real:**
  - **`stair_gaptw`** (`school-4awgw/stair-gaptw`): single native class `stair`, 1,766 instances. Pulled 1,564 images (v1, only version). Converted: **1,564 images, 1,766 boxes kept, 0 dropped/clipped/invalid** — clean. License recovered from the real downloaded `data.yaml`'s `roboflow.license` field (not exposed via the SDK's `Project.license` attribute, which returned `None` for every source checked this way so far): **Public Domain**.
  - **`elevator_status_0iq4p`** (`elevator-0iq4p/elevator-status` — a distinct Roboflow project from the failed `elevator_status_s4lrk`, different workspace, verified not a duplicate/mirror): 4 native classes, all elevator-*state* labels (`Open Elevator`, `Closed Elevator`, a label with a stray leading combining character that the SDK reported but the real downloaded `data.yaml` shows as a plain ASCII hyphen — `-Nobody was in the elevator` — an SDK-vs-real-export mismatch caught before trusting the convert run, same DEC-053 discipline; and `People are in the elevator.`). Explicit student decision: `Open Elevator`/`Closed Elevator`/`-Nobody was in the elevator` map to canonical **Elevator**, but `People are in the elevator.` maps to canonical **Person** instead — geometrically verified after conversion, not assumed: sampled 2 real "People are in the elevator."-labeled images post-convert and confirmed the boxes are tight, individual, per-person boxes (one image has 5 people, each with its own reasonably-tight box), not a single elevator-shaped box mislabeled as Person. Pulled 1,464 images (v2, latest — v1 identical size, no anomaly). Converted: **1,461 images, 2,868 boxes kept** (3 images dropped empty, 0 dropped non-canonical — every native class had a mapping), **1,687 clipped, 17 invalid**. The high clip rate was checked, not just accepted: visually confirmed on 4 real converted samples that boxes cluster at/past the frame edge because this source shoots tight, close-up elevator-door photos, not because of corrupt coordinates — `clip_bbox()` (DEC-057) is doing its documented job. License recovered the same way as `stair_gaptw`: **CC BY 4.0**, from the real `data.yaml`.
  - Both sources' `license:` field in `config/datasets.yaml` set directly from the verified value — no "could not verify via SDK" caveat needed this time, since the real downloaded `data.yaml` carries a `roboflow.license` field the `Project`/`Version` SDK objects apparently don't expose. **Worth carrying forward as a process note**: check the downloaded `data.yaml` before writing "license unverified" for any future Roboflow source — the SDK-only check used for `trashcan_detection_pihfn`/`door_detection_zqt59` (DEC-069/071/073) may have given up too early.
- **Escalator's schema slot (id 6) is reserved, not renumbered, for now.** `escalator_stairs' was Escalator's only source; no replacement has been found yet. Checked `config_loader.py` before deciding how to represent this: `EXPECTED_NC` (16) and `CANONICAL_NAMES` are hardcoded module constants, and `load_classes()` validates `classes.yaml` against them exactly — there is no existing status-driven exclusion mechanism (`Tricycle`'s `status: possible` is confirmed to be a documentation-only annotation; `config_loader.py` doesn't branch on `status` anywhere, Tricycle is fully counted as one of the 16 regardless). Two real mechanisms exist to actually drop a class: renumber (nc→15, nc 7-15 shift down, every already-converted label file referencing those ids needs remapping — this project already did an analogous migration once, DEC-038, though at a much earlier and cheaper point before most conversion had happened) or reserve the slot (nc stays 16, Escalator keeps id 6 with near-zero real data, zero file-touching cost). Student's explicit call: **reserve it for now** and brainstorm a possible replacement class first; **renumber only if no replacement with adequate data can be found.** Nothing in `classes.yaml`/`config_loader.py` changed by this entry — this is a documented holding pattern, not a code change.

### Rationale

The automated `box_audit.py` flagged-rate ranking (DEC-076) undercounts a defect that's uniform across most of a source's own images, by construction — Tukey's fences flag statistical outliers *relative to that source's own distribution*, so a source that's consistently bad in a similar way across the majority of its images produces few outliers, without that meaning the un-flagged majority is fine. This is why the student's direct visual read caught something the ranking didn't surface as urgent. Independently re-verifying with fresh samples (rather than taking either the automated numbers or the verbal description alone) follows this project's standing discipline of checking real data before trusting or repeating a claim — applied here to the student's own claim, not just to a script's output, which is exactly the same bar DEC-053/DEC-066 already held code changes to.

Reserving Escalator's slot rather than immediately renumbering keeps today's change bounded (config + two source pulls, no mass label-file rewrite) while leaving the more consequential, harder-to-reverse schema surgery for a moment when it's actually known to be necessary — consistent with this project's repeated pattern of not resolving a consequential, discussion-worthy change unilaterally (DEC-042's cap-recompute trigger, DEC-078's tag-don't-delete choice) when a cheaper path might make it unnecessary.

### Consequences

- `dataset/processed/roboflow_stair_gaptw/` and `dataset/processed/roboflow_elevator_status_0iq4p/` now exist and are ready for the next `cap_per_class.py` → `merge.py` pass. `roboflow_stairs_i2yia/`, `roboflow_escalator_stairs/`, `roboflow_elevator_status_s4lrk/` remain on disk (not deleted) but are excluded from that pass by their `failed` status, same posture as `stairs_lusiz`/`stairs_hsatv`/`jeep_hozhs`.
- **Stairs' realistic near-term pool shrinks substantially**: with both `stairs_i2yia` and `escalator_stairs` out, Stairs' active candidate pool is essentially `stair_gaptw` alone (1,564 images) plus whatever `stairs_lusiz`/`stairs_hsatv` could contribute if ever relabeled (currently inactive). That's barely above DEC-042's 1,500 floor, with no cushion for further box-audit/dedup attrition — worth tracking the real post-cap number closely, not assuming "solved."
- **Escalator has zero active source** until either a replacement is found or the slot is formally renumbered away. `classes.yaml`'s `cap: 4500` for Escalator is now aspirational, not achievable, until one of those happens.
- `cap_per_class.py` → `merge.py` → `dedup.py` → `split.py` → `generate_yaml.py` all need a rerun regardless (already true before this entry, per DEC-072/077) — this entry adds "reflects the failed/added source changes" to what that rerun needs to pick up.
- `docs/OPEN_QUESTIONS.md` #4 (the DEC-076 full-pool review) should note this as a real finding from that review, not a side discussion.
- Escalator replacement-class brainstorm is a separate, open thread — not resolved by this entry.

---

## DEC-083: Pole → Open Images "Street Light"; Escalator's Reserved Slot Filled with "Shelf"; Five New Roboflow Sources Added (Potholes, Pedestrian Lane, Vehicle/Motorcycle/Tricycle); Four Superseded Roboflow Sources Benched

- **Date:** 2026-08-20
- **Status:** Accepted
- **Related:** DEC-082 (Escalator's slot reserved pending a replacement class or renumber — this entry fills it), DEC-053/DEC-069/DEC-071/DEC-073 (the SDK-verify-before-trusting discipline applied throughout), `docs/OPEN_QUESTIONS.md` #9/#10 (both closed by this entry)

### Context

Continuing from the Escalator-replacement brainstorm (DEC-082): the student browsed Street light/Cart/Shelf/Countertop candidates in `fiftyone_preview.ipynb`'s new ad-hoc section and picked Shelf, rejecting Cart (mostly horse-drawn carts on inspection, no clean shopping-cart filter) and Countertop (mostly residential kitchen counters). Separately, reviewing `fiftyone_review_processed.ipynb`'s per-source checklist (Notebook version 9 — a real, substantial review effort that happened directly in the App/notebook, not narrated in this conversation), the student decided to swap out several more unreviewed/lower-confidence Roboflow sources for ones they'd sourced themselves, and asked for the current per-class distribution before finalizing anything (delivered inline, not as a separate DEC entry — informational only).

### Decision

**Pole: switched from Roboflow to Open Images.** `pole_detection_z76mb` and `utility_poles_44tzx` (both unreviewed, neither confirmed defective) benched in favor of Open Images `"Street light"` — 44,697 boxes / 11,226 images in the cached train split alone, gold-standard annotation, zero Roboflow-quality risk. `classes.yaml`'s `pole:` block restructured from `primary_providers` (2 Roboflow entries) to a single `primary: {source: open_images, native_class: "Street light"}`, matching the Chairs/Tables/Animals pattern — picked up automatically by `acquire_openimages.py`'s `get_openimages_targets()` (confirmed by reading it: it iterates every `classes.yaml` entry whose `primary.source == "open_images"`, no hardcoded class list, no script change needed). Pulled and converted for real: **2,154 images / 8,311 instances** (2,159 raw pulled, 5 dropped empty). Spot-checked: a real street light correctly boxed on a real street scene.

**Escalator's reserved slot (id 6) filled with Shelf**, not renumbered. `classes.yaml`'s `names`/`hailo_runtime_names` (id 6: Escalator → Shelf) and `scripts/utils/config_loader.py`'s hardcoded `CANONICAL_NAMES[6]` updated together, then re-verified via `python3 scripts/utils/config_loader.py` (its own validation checks `names` against `CANONICAL_NAMES` exactly — confirms both stayed in sync). Provider: Open Images `"Shelf"`. A suitable replacement was found, so the renumber-to-15-classes fallback (DEC-082's stated condition for when to actually renumber) was never triggered. Pulled and converted for real: **2,386 images / 9,471 instances** (2,756 raw pulled, 370 dropped empty). Spot-checked on 2 samples: one legitimate full-frame shot of a CD/media shelf (the shelf genuinely fills the frame, not a lazy annotation — same "real full-frame content, not a defect" case seen elsewhere this session), one a real storefront display with 3 individual shelf units correctly, tightly boxed — the more representative sample of the two.

**Five new Roboflow sources added, SDK-checked, pulled, and converted for real:**

| Source | Real native classes kept → canonical | Pulled | Converted |
|---|---|---|---|
| `pothole_voxrl` (replaces `pothole_vhmow`) | `pothole` → potholes | 665 img | 665 img / 1,739 boxes, 1 invalid |
| `wtf_dwvgm` (replaces `pedestrian_and_animal_crossing`) | `crosswalk`, `Crosswalks` → pedestrian_lane (`object` dropped) | 1,332 img | 1,326 img / 1,592 boxes |
| `revised_pedestrian_obstacle` ("General Filipino Outside Dataset") | `vehicle`→vehicle, `stairs`→stairs, `crosswalk`→pedestrian_lane, `person`→person (`animal`/`bike`/`hazard-sign` dropped) | 6,342 img | 4,323 img / 12,193 boxes |
| `dlsu_d_vehicle_type_detection` ("Filipino Related Road Vehicles") | `Tricycle`→tricycle, `Motorcycle`→motorcycle, 6 named types→vehicle (`Sedan`/`Sports Utility Vehicle`/`Hatchback`/`Pickup Truck`/`Bicycle`/`Electric Bike` dropped) | 37,556 img | 21,142 img / 34,697 boxes |
| `roitrikee` (Tricycle supplement) | `Tricycle` → tricycle | 665 img | 665 img / 860 boxes |

Real bug caught and fixed during this pass: `wtf_dwvgm` was first configured with `native_class_filter` as a bare list (`["crosswalk", "Crosswalks"]`) — not one of `yolo_to_intermediate.py`'s 4 documented filter forms (dict, string, no-filter+singular, no-filter+plural). It silently fell through to the "no filter, blanket-map every native class" case, pulling in the junk `object` class too. Caught from the conversion report's own `"0 dropped non-canonical"` (should have been >0 with `object` present) before trusting the output — same "verify the report, don't just trust it ran" discipline as DEC-066's smoke-test-overwrite catch. Fixed to the documented dict form (`{pedestrian_lane: ["crosswalk", "Crosswalks"]}`) and reconverted; corrected result above.

Two deliberate, flagged judgment calls, not script defaults:
- `revised_pedestrian_obstacle`'s `vehicle` class has motorcycles mixed in under one label with no sub-filter to split them — **explicitly asked the student rather than guessing**; answer was to include it as-is and catch individual bad boxes during the same visual-review pass every other source goes through (DEC-078's exclude-tag workflow), not to exclude the class outright.
- `revised_pedestrian_obstacle`'s `person` class (3,933 instances) wasn't on the student's explicit keep or drop list — defaulted to keep, flagged clearly rather than silently decided either way.
- `dlsu_d_vehicle_type_detection`'s `Hatchback`/`Pickup Truck`/`Bicycle` (not explicitly mentioned by the student, unlike the explicitly-dropped `Sedan`/`Sports Utility Vehicle`) were dropped on the assumption they follow the same "already covered by Open Images" logic — flagged as an extrapolation, not a confirmed instruction.

Spot-checked post-conversion, not just trusted from the numbers: `revised_pedestrian_obstacle`'s `vehicle` boxes (2 real samples) landed correctly on vans/cars with no visible motorcycle contamination in the sampled images (doesn't rule it out elsewhere in the pool — still flagged for the planned review pass); `dlsu_d_vehicle_type_detection`'s `vehicle` and `tricycle` boxes (1 sample each) landed correctly on a goods truck and a real Philippine sidecar tricycle respectively.

**Four superseded Roboflow sources marked `audit_status: benched`** (not `failed` — none independently confirmed defective this session, same DEC-037 benched-vs-failed distinction): `pole_detection_z76mb`, `utility_poles_44tzx` (Pole → Street light), `pothole_vhmow` (→ `pothole_voxrl`), `pedestrian_and_animal_crossing` (→ `wtf_dwvgm` + `revised_pedestrian_obstacle`). Note: `utility_poles_44tzx` is dual-purpose (`canonical_classes: [pole, potholes]`) but its real Potholes contribution was already zero in the converted output before this change — benching it has no Potholes-side effect.

**Reinstate-if-insufficient conditional resolved for all three affected classes — none needed reinstating.** Real post-add active totals, computed directly from `dataset/processed/`:
- Potholes: 2,661 images (down slightly from 2,867, still comfortably above the 1,500 floor)
- Pedestrian Lane: 2,745 images (up from 2,158)
- Stairs: 2,294 images (up from DEC-082's thin 1,564 — `revised_pedestrian_obstacle`'s `stairs` class turned out to be a real, unplanned bonus here)

Also real, incidental gains from the new sources' multi-class content: Vehicle 12,953 → 29,778 images, Motorcycle 3,861 → 8,990, Tricycle 3,495 → 7,293, Person 27,006 → 29,156.

### Rationale

Same SDK-verify-before-trusting discipline this project has held all session (DEC-053/069/071/073) — every native class name was checked against the real downloaded `data.yaml`, not assumed from the SDK's `project.classes` dict or the student's own memory of a project's labels, and it paid off immediately (the `wtf_dwvgm` list-vs-dict bug would have silently shipped 4 junk boxes as Pedestrian Lane otherwise). Asking the student directly on the one genuinely ambiguous, previously-unresolved call (`vehicle`'s motorcycle contamination) rather than guessing matches this project's standing pattern of surfacing real judgment calls instead of silently resolving them (DEC-042's cap-recompute trigger, DEC-078's tag-don't-infer choice).

### Consequences

- `cap_per_class.py` → `merge.py` → `dedup.py` → `split.py` → `generate_yaml.py` still need a rerun (unchanged from DEC-082) — this entry adds more source changes to what that rerun picks up, doesn't change the fact that it's still pending.
- `docs/OPEN_QUESTIONS.md` #9 (Escalator replacement) and #10 (Stairs volume) both close — #9 resolved (Shelf), #10 resolved (2,294 active images, comfortably clear of the floor).
- A real numbering collision was found and fixed while writing this entry: another concurrent session (this repo's deliberate split-agent setup) had independently added its own DEC-082 (a `pothole_vhmow` image-count investigation, unrelated to this one) — renumbered to DEC-084 since it had zero external cross-references, versus this conversation's DEC-082 already being referenced in 17+ places. Worth the student's awareness that two sessions have been editing this repo's docs concurrently — re-read `docs/OPEN_QUESTIONS.md` and `TASKS.md` fresh next session rather than assuming either agent's view is complete.
- `dataset/processed/roboflow_pole_detection_z76mb/`, `roboflow_utility_poles_44tzx/`, `roboflow_pothole_vhmow/`, `roboflow_pedestrian_and_animal_crossing/` remain on disk (not deleted) but excluded from the next merge, same posture as every other benched/failed source.
- Not yet done: real per-class re-audit of box quality for the 5 new sources via `box_audit.py` (will happen naturally on the next `--pool merged` run) — SDK/conversion-level verification was done here, not a full Stage 5.3 pass.

---

## DEC-085: Post-DEC-083 Verification Pass — Real `wtf_dwvgm` Stale-File Bug Found and Fixed, `classes.yaml` Documentation Resynced Across 7 Classes, One Pre-Existing Floor-Compliance Risk Flagged

- **Date:** 2026-08-21
- **Status:** Accepted
- **Related:** DEC-082, DEC-083, DEC-084 (the entries being verified), DEC-042 (floor policy relevant to the Potholes finding), DEC-057 (`yolo_to_intermediate.py`'s conversion mechanics)

### Context

Student asked for a direct verification of the DEC-082/083/084 work after the prior session ended on a context-limit compaction, rather than taking the prior session's own summary on trust. Re-derived every headline claim from real on-disk data instead of re-reading the prior session's written account: re-ran `config_loader.py`, recounted every new/changed source's converted images/boxes directly from `dataset/processed/`, and cross-checked `classes.yaml`'s per-class documentation against `datasets.yaml`'s actual `audit_status` values.

### Decision

**Verified correct, no changes needed:** `classes.yaml`/`config_loader.py`'s Escalator→Shelf rename (`nc: 16`, `CANONICAL_NAMES[6]`, `hailo_runtime_names` all in sync, validation passes). Real converted counts for Pole (2,154 img/8,311 inst), Shelf (2,386 img/9,471 inst), `stair_gaptw` (1,564/1,766), `elevator_status_0iq4p` (1,461/2,868), `pothole_voxrl` (665/1,739), `revised_pedestrian_obstacle` (4,323/12,193), `dlsu_d_vehicle_type_detection` (21,142/34,697), and `roitrikee` (665/860) all match DEC-082/083's documented numbers exactly. No duplicate/colliding `DEC-08x` headers remain in this file. Final active-source totals for Stairs (2,294), Vehicle (29,778), Motorcycle (8,990), Tricycle (7,293), and Person (29,156) all reproduce exactly from a fresh, independent recount.

**Real bug found and fixed: `wtf_dwvgm` had 4 stale, contaminated files on disk.** `convert_project()` in `yolo_to_intermediate.py` writes output files for every image that currently qualifies under the mapping, but never deletes previously-written output files that no longer qualify after the mapping changes. DEC-083's own bare-list-filter bug (`native_class_filter` as a list instead of the required dict form) caused a first, buggy conversion run that blanket-mapped every native class — including the junk `object` class — into Pedestrian Lane; 4 images whose only raw box was `object` were written to disk under that run. Fixing the filter and reconverting produced the correct stats (1,326 images/1,592 boxes, recorded accurately in DEC-083's text and in `yolo_to_intermediate_report.json`), but the second run's file-writing loop only writes images that currently qualify — it never swept the 4 stale files left behind by the first run. Confirmed via raw-label cross-reference (exactly the 4 images whose raw label file contains only native index 2 / `object`, content: single thin slivers at frame edges, not real crosswalk boxes) and mtime (the 4 stale files timestamped a minute before the other 1,326 — consistent with a first, since-superseded run). Deleted the 4 stale image+label pairs; `dataset/processed/roboflow_wtf_dwvgm/` now holds exactly 1,326 images/1,592 boxes, matching the report. Confirmed this hadn't propagated to `dataset/merged/` or `dataset/final/` (the cap/merge/dedup/split cascade rerun is still pending, per DEC-083's own Consequences) — contamination was fully contained to `dataset/processed/`. Checked the other two sources with a similar "wrong value caught, then corrected" history this session (`elevator_status_0iq4p`'s SDK-vs-real garbled string, `stair_gaptw`) — neither has a stale-file mismatch, since in both cases the correct value was used before the converter was ever run, not after.

**`classes.yaml`'s per-class `primary_providers`/`secondary_providers` documentation was stale for 7 of the 9 classes DEC-082/083 touched.** Only the `pole:`/`shelf:` blocks (clean single-source swaps) were updated during the original work; `stairs:`, `potholes:`, `vehicle:`, `motorcycle:`, `tricycle:`, `pedestrian_lane:`, and `elevator:` still listed only pre-DEC-082/083 sources — newly-added sources (`stair_gaptw`, `elevator_status_0iq4p`, `pothole_voxrl`, `wtf_dwvgm`, `roitrikee`, plus `dlsu_d_vehicle_type_detection`'s and `revised_pedestrian_obstacle`'s multi-class contributions) were absent, and `elevator:`'s entry for `elevator_status_s4lrk` still carried DEC-031's original "partially usable" verdict with no note that DEC-082 overturned it. Checked `get_eligible_projects()` (`scripts/acquire/acquire_roboflow.py:53`) before treating this as more than cosmetic: confirmed both acquisition and conversion are driven entirely by `datasets.yaml`'s `audit_status`/`pinned_version` fields, so the staleness never affected pipeline behavior — but `classes.yaml`'s own header claims it is "the AUTHORITATIVE class list and source mapping," and it gave a wrong picture of current sourcing for 7 classes. Resynced all 7 blocks in place, following the file's existing convention of keeping benched/failed entries listed with an inline `note:` rather than deleting them (matching the pattern already used for `jeep_hozhs`). Also resynced `doors:` for `door_detection_zqt59` — a gap from DEC-071/073 (an earlier session, predating DEC-082/083), found incidentally while auditing the same class of staleness.

**Pedestrian Lane's active total corrected from DEC-083's stated 2,745 to 2,741** — a direct consequence of the `wtf_dwvgm` fix above (1,330 stale → 1,326 correct, a delta of 4 images matching exactly).

**Flagged, not resolved: Potholes' active total (2,661, matching DEC-083 exactly) depends on `dataset_ninja_road_damage_detector` counting as an active source, and that's genuinely ambiguous.** That source has no `audit_status` field at all in `datasets.yaml` (unlike its sibling `dataset_ninja_pothole_detection`, which explicitly has `audit_status: approved`), and its own config `notes` field describes it as DEC-034's deprioritized fallback — "kept as a fallback if the combined volume proves insufficient... if ever activated" — language that reads as *not currently active*. Its processed output (1,331 images) has existed on disk since 2026-08-13, predating DEC-082/083 entirely, so this wasn't introduced by this session's changes, and DEC-083's own arithmetic (2,867 before → 2,661 after) is internally consistent with treating it as active, same as it was apparently already being treated before DEC-083. The consequence is real either way: **without** `road_damage_detector`, Potholes' active total is 665 (`pothole_voxrl`) + 665 (`dataset_ninja_pothole_detection`) = **1,330 — below DEC-042's 1,500 floor.** Not resolved here since it's a pre-existing ambiguity, not something DEC-082/083 changed — flagged in `docs/OPEN_QUESTIONS.md` (item #11) for the student to decide: either make the "active" status explicit (`audit_status: approved`, matching its sibling) since it's apparently load-bearing, or take "fallback only" literally and find Potholes another real source.

### Rationale

Re-deriving every number from raw files rather than re-trusting the prior session's own JSON reports or written summary follows this project's own standing discipline (DEC-053/066/069/071/073/083) — applied here to the prior session's own output, the same bar already held for source data and script output. The `wtf_dwvgm` finding shows exactly why that bar matters: the *stats* were computed correctly after the filter fix, and would have looked clean to a report-only check — only a direct disk recount caught that the *files* didn't match the *report*.

### Alternatives Considered

- **Trust DEC-082/083's own numbers without re-deriving them**: Rejected — that's exactly the "trust a report without checking the underlying files" gap that produced the `wtf_dwvgm` bug in the first place; re-verifying a prior session's claims deserves the same scrutiny as any other unverified claim.
- **Silently resolve the `dataset_ninja_road_damage_detector` ambiguity** (either bench it or mark it approved): Rejected — it's a genuine judgment call (literal reading of its own "fallback only" note vs. its apparent load-bearing role in clearing the Potholes floor) that the student should make with the tradeoff visible, not have decided for them.

### Consequences

- `dataset/reports/yolo_to_intermediate_report.json`'s `wtf_dwvgm` entry (1,326/1,592) now matches disk again.
- `docs/OPEN_QUESTIONS.md` gets a new item (#11) for the `dataset_ninja_road_damage_detector` active-status ambiguity and its floor-compliance implication for Potholes.
- `cap_per_class.py` → `merge.py` → `dedup.py` → `split.py` → `generate_yaml.py` rerun is still pending (unchanged from DEC-082/083) — this entry's file deletion means that future rerun picks up 4 fewer (correct, not stale) `wtf_dwvgm` images than it would have yesterday.
- No prior decision or number the student already approved needed to change — this entry is a verification/correction pass, not a new sourcing decision.

---

## DEC-086: `dataset_ninja_road_damage_detector` Activated for Potholes; Real Stage-5.4 Gating Bug Found and Fixed — `cap_per_class.py` Had No `audit_status` Awareness At All

- **Date:** 2026-08-21
- **Status:** Accepted
- **Related:** DEC-085 (flagged the `road_damage_detector` ambiguity), DEC-034 (original deprioritization), DEC-042 (floor policy), DEC-082/083 (the 7 sources this bug would have silently reincluded), DEC-077 (last real cap/merge run, confirmed unaffected)

### Context

Student's direct answer to DEC-085's flagged open question (`docs/OPEN_QUESTIONS.md` #11): "lets use the dataset_ninja_road_damage_detector, it is indeed the case that we need that fallback now." Implementing this — setting an explicit `audit_status` and verifying it would actually change Potholes' candidate pool — required tracing exactly how `cap_per_class.py` decides which processed sources feed each class's candidate pool. That trace surfaced a bug substantially bigger than the setting being flipped.

### Decision

**`dataset_ninja_road_damage_detector` activated.** `config/datasets.yaml` gets an explicit `audit_status: approved` (previously had no `audit_status` field at all), matching its sibling `dataset_ninja_pothole_detection`. Corrected a second, separate stale claim found in the same entry while doing this: its `notes` field said conversion was "unverified... not attempted yet" — false; `scripts/acquire/acquire_datasetninja.py`'s `SOURCES` dict has processed this project unconditionally since DEC-041/049, and its real converted output (1,331 images/1,331 labels, `dataset/processed/dataset_ninja_road_damage_detector/`) was directly re-verified: every box is canonical `pothole`, alligator/lateral/longitudinal crack correctly dropped. `classes.yaml`'s `potholes:` block's matching entry updated with the same activation note.

**Real bug found and fixed: `cap_per_class.py` (Stage 5.4) had no `audit_status` awareness anywhere.** Traced the full path a source takes from `dataset/processed/` into the training pool: `run()` calls `discover_processed_sources()` (globs every `dataset/processed/<name>/` with an `images/`+`labels/` pair, no status check) and passes the result straight into `build_class_index()` (indexes every label file's class ids, again no status check) and then `cap_class()` (pools every non-priority candidate via pure random selection — read in full, confirmed no `audit_status`/`datasets.yaml`/`classes.yaml` reference anywhere in the function). **This means a source marked `failed`/`benched` *after* it was already pulled and converted would still have been silently eligible for the next real cap/merge run** — directly contradicting what DEC-082 and DEC-083 both explicitly claimed ("excluded from the next merge, same posture as every other benched/failed source"). Concretely, the 7 sources benched/failed today (`stairs_i2yia`, `escalator_stairs`, `elevator_status_s4lrk`, `pole_detection_z76mb`, `utility_poles_44tzx`, `pothole_vhmow`, `pedestrian_and_animal_crossing`) all still have real processed data on disk and would all have silently re-entered Stairs/Elevator/Pole/Potholes/Pedestrian Lane's candidate pools on the next run.

Added `get_inactive_processed_source_keys()` to `scripts/utils/config_loader.py` — returns the set of processed-dir-style source keys (`roboflow_<key>` for Roboflow projects, bare top-level key for everything else) whose `audit_status` is `failed`/`benched`/`blocked`; a source with no `audit_status` field at all (crowdhuman, exdark, open_images, and — until this entry — `dataset_ninja_road_damage_detector`) is treated as active, matching the "not in {failed,benched,blocked}" convention DEC-085's own recount script already used. Wired into `cap_per_class.py`'s `run()`, filtering `sources` right after `discover_processed_sources()`, before any candidate-index building. Deliberately scoped to `cap_per_class.py` only, not `discover_processed_sources()` itself — `box_audit.py`'s non-`--pool merged` mode also calls `discover_processed_sources()`, and legitimately wants visibility into failed sources too (that's literally how DEC-082's review found the defects that got `stairs_i2yia`/`escalator_stairs`/`elevator_status_s4lrk` failed in the first place — auditing them before they were excluded). Changing that shared function's meaning would have silently narrowed a diagnostic tool's scope as a side effect of fixing a training-pool-eligibility bug — two different questions ("what exists on disk" vs. "what should train the model") that deserve two different answers, not one.

Also fixed a second, smaller stale-doc issue found in the same file while in there: `cap_per_class.py`'s own module docstring claimed "`merge.py`... mechanically pools every `dataset/processed/<source>/`... it does not consult this report" — false, and contradicted by `merge.py`'s own docstring, which documents a scope correction where it was fixed to read `cap_report.json`'s selected union. The report-only design conclusion this paragraph was arguing for is still correct (kept as-is); only the stated reason was wrong.

**Verified via `--dry-run` that the fix produces exactly the numbers everyone has been assuming were already true:**

| Class | Candidates (fixed) | Sources correctly excluded |
|---|---|---|
| Potholes | 2,661 | `pothole_vhmow` |
| Stairs | 2,294 | `stairs_i2yia`, `escalator_stairs` |
| Pedestrian Lane | 2,741 | `pedestrian_and_animal_crossing` |
| Pole | 2,154 (unchanged — pure Open Images already) | `pole_detection_z76mb`, `utility_poles_44tzx` |

Every one of these matches DEC-083/085's already-documented totals exactly — the fix makes the code finally do what the documentation already (prematurely) claimed. Ratio invariant still holds post-fix: max=Vehicle(4,500)/min=Trash Bins(1,663) = 2.71 (≤3.0).

**Confirmed the last real cap/merge run (DEC-077, 2026-08-18) was NOT affected by this bug** — checked the existing `cap_report.json` on disk directly: it legitimately includes `pothole_vhmow` (871), `stairs_i2yia` (1,559), `escalator_stairs` (4,429), `pole_detection_z76mb` (3,100), `utility_poles_44tzx` (5,089) as candidates, all of which were genuinely still-active sources on 2026-08-18, before any of today's bench/fail decisions existed. `dataset/merged/`'s current physical contents (if any) derive from that same, still-valid-for-its-time report. This bug only matters prospectively, for the still-pending next real run — nothing already trained on or physically merged is contaminated.

### Rationale

The bug was found by not stopping at "does the requested field change do what's asked" — tracing the field's actual functional effect (not just setting it and assuming it works) is the same discipline this project has applied to source data all session (DEC-053/066/069/071/073/083/085), applied here to the pipeline's own code path instead. Scoping the fix to `cap_per_class.py` rather than the shared `discover_processed_sources()` utility follows DEC-082's own reasoning pattern (prefer the smaller, more targeted change when a narrower one satisfies the actual need) — `box_audit.py`'s diagnostic use case and `cap_per_class.py`'s training-pool-eligibility use case are genuinely different questions that happened to share a utility function by coincidence, not by design.

### Alternatives Considered

- **Change `discover_processed_sources()` itself to filter by `audit_status`**: Rejected — would silently narrow `box_audit.py`'s diagnostic visibility into failed sources as a side effect, and that visibility has real, demonstrated value (DEC-082's review process).
- **Leave the Stage 5.4 gating bug for a separate future entry, only do the requested activation**: Rejected — activating `dataset_ninja_road_damage_detector` while `cap_per_class.py` still had no `audit_status` awareness at all would have been a documentation-only change with no verified functional effect; the whole point of tracing the fix through was to confirm the activation actually does something, which required finding and fixing this bug in the same pass.

### Consequences

- `docs/OPEN_QUESTIONS.md` #11 closes.
- `cap_per_class.py` → `merge.py` → `dedup.py` → `split.py` → `generate_yaml.py` rerun is still pending (unchanged) — but the next real run will now correctly exclude all 7 benched/failed sources with stale processed data, something no prior run in this project's history needed to rely on (none of them had this specific "benched/failed after first pull" situation until DEC-082/083).
- Worth a forward note for future bench/fail decisions on already-pulled sources: this gate is now real, so marking something `failed`/`benched` in `datasets.yaml` going forward will actually take effect on the next `cap_per_class.py` run, not just in documentation.

---

## DEC-087: Person/Vehicle/Motorcycle Priority-Source Reordering; `me5_u6rvg`/`augmented_tricycle` Benched Per Review-Checklist Policy; `cv_project_hovyc`/`trashcan_detection_pihfn` Cleaned Labels Promoted — Revealing Real Multi-Class Enrichment From the Review Pass

- **Date:** 2026-08-21
- **Status:** Accepted
- **Related:** DEC-067/069 (original `CLASS_PRIORITY_SOURCES` design), DEC-076 (review checklist), DEC-078/079 (exclude/accept-tag write-back mechanisms), DEC-082/083 (sources this reorders around), DEC-086 (the `audit_status` gating fix this builds on directly)

### Context

Student gave a consolidated set of instructions covering: (1) revised per-class source priority for Person/Vehicle/Motorcycle, (2) a standing policy that any Roboflow source still marked `[x]` (not `[/]`, i.e. never completed a full `fiftyone_review_processed.ipynb` write-back review) should be deactivated rather than reviewed — explicitly naming `me5_u6rvg` and `augmented_tricycle` — with a note this had been said before, (3) confirmation that `revised_pedestrian_obstacle`'s known motorcycle-in-vehicle contamination (flagged, not acted on, in DEC-083) needs active cleaning, not just flagging, and (4) that `cv_project_hovyc` and `trashcan_detection_pihfn`'s already-reviewed, cleaned labels (`labels_reviewed/`, DEC-078/079's write-back staging area) should be what the next merge actually uses.

Re-read `fiftyone_review_processed.ipynb`'s checklist cell (`b748ccf6`) fresh rather than trust a summarized memory of it — its own legend text described `[x]` as "not yet reviewed," which reads differently from the student's "failed the audit check" framing from earlier in this project. Resolved by checking each `[x]` row's actual evidence individually rather than picking one global reading: most `[x]`-marked, still-processed-on-disk sources genuinely were later confirmed defective via independent visual review (DEC-082) or superseded by an explicit replacement decision (DEC-083) — but `elevator_awvus` is also `[x]`-marked and was *not* one of those; DEC-082 directly visually confirmed it clean and kept it active. Surfacing this tension explicitly rather than silently applying "x means deactivate" universally is what caught it.

### Decision

**Priority-source order revised for Person, Vehicle, and (newly) Motorcycle** in `scripts/preprocess/cap_per_class.py`'s `CLASS_PRIORITY_SOURCES`:
- Person: `["exdark", "roboflow_revised_pedestrian_obstacle", "crowdhuman", "open_images"]` (was `["exdark", "crowdhuman"]`)
- Vehicle: `["roboflow_dlsu_d_vehicle_type_detection", "exdark", "roboflow_revised_pedestrian_obstacle", "open_images"]` (was `["exdark", "roboflow_me5_u6rvg"]`)
- Motorcycle: `["roboflow_dlsu_d_vehicle_type_detection", "exdark", "open_images"]` (new — previously used the plain `["exdark"]` default)

**Real consequence caught before finalizing, not assumed:** a first pass left `open_images` off all three lists (matching the student's initial framing of it as a "control group"/leftover pool), and a `--dry-run` showed this squeezed Open Images to **0 selected images** for both Person and Vehicle — the higher-priority sources' per-source floors alone already filled the entire 4,500 cap. This directly contradicted the student's stated intent (Open Images as the real source of sedan/SUV imagery for Vehicle, general diversity for Person). Flagged via `AskUserQuestion` rather than silently picking a side; student chose to give Open Images its own explicit floor too, added as the last tier in all three lists. Re-verified via `--dry-run`: Open Images now gets a real, non-zero guaranteed share (375 for Person, 375 for Vehicle, 500 for Motorcycle) instead of 0/thin leftovers.

`elevator_status_0iq4p`'s small Person contribution (~300 images, the "People are in the elevator." remap, DEC-082) deliberately stays un-prioritized, unlike Open Images — student's explicit framing: "somewhat negligible... if it appears... so be it."

**`me5_u6rvg` and `augmented_tricycle` benched** (`config/datasets.yaml`, `config/classes.yaml`) — both still `[x]` on the review checklist (never completed a full write-back review), student's explicit call to deactivate rather than review. Neither has specific evidence of a defect (unlike `stairs_i2yia`/`escalator_stairs`/`elevator_status_s4lrk`'s DEC-082 findings) — marked `benched`, not `failed`, consistent with the project's established distinction. `get_inactive_processed_source_keys()` (DEC-086) picks both up automatically; `--dry-run` confirms them correctly excluded and Vehicle/Motorcycle/Tricycle's candidate pools now driven by `dlsu_d_vehicle_type_detection` + the sources above instead.

**`elevator_awvus` kept active despite its `[x]` mark** — its row in the review checklist still cited DEC-031's original box-shape concern, which DEC-082 already independently overturned for this specific source (3/3 sampled images confirmed correctly-boxed). Applying "x means deactivate" here would have dropped Elevator's only confirmed-clean source, leaving just `elevator_status_0iq4p` (1,414 images). Checklist row corrected to state the DEC-082 finding plainly instead of the stale DEC-031-only framing; `config/datasets.yaml`/`classes.yaml` unchanged (already correct).

**`revised_pedestrian_obstacle`'s Vehicle-class motorcycle contamination**: `classes.yaml`'s note strengthened from "flag for review, not exclude" (DEC-083's original framing) to explicitly requiring active cleaning during the `fiftyone_review_processed.ipynb` pass — no automated sub-filter exists to separate motorcycles from its native `vehicle` label, so this remains a visual-review task, not something resolved by this entry. Flagged, not fixed.

**`cv_project_hovyc` and `trashcan_detection_pihfn`'s reviewed labels promoted** — `labels_reviewed/` copied over the real `labels/` for both (1,040 of 1,337 files for `cv_project_hovyc`, matching its `restrict_to_merged`-scoped review; all 559 for `trashcan_detection_pihfn`). Backed up originals first to `dataset/backups/pre_promotion_<source>_labels_<timestamp>/` (`dataset/processed/` is fully gitignored, no git-level safety net) since this is a real, non-trivially-reversible file overwrite. Sanity-checked before promoting: 484/1,040 and 95/559 files actually differed from the originals; total box counts increased (1,269→2,296 and 762→1,322 respectively), consistent with DEC-079's accept-tag prediction-promotion plus manual additions, not a corruption or mass-deletion.

**Real, unplanned finding surfaced by the promotion, investigated fully before trusting it:** a `cap_per_class.py --dry-run` immediately after the promotion showed Pole/Stairs/Trash Bins/Pedestrian Lane candidate counts increased by amounts unexplained by anything else changed this session. Independently re-verified the raw files hadn't been touched by a concurrent session (checked mtimes on `open_images`/`stair_gaptw`/`revised_pedestrian_obstacle` — all untouched; searched for any file modified in the last 3 hours outside the two promoted sources — none found) before concluding the promotion itself was the cause. Confirmed directly: `cv_project_hovyc` and `trashcan_detection_pihfn` are **no longer single-class sources** — the review process added real, visually verified boxes for other canonical classes visible in the same photos (a door photo naturally also shows the entrance's steps, a nearby pole, the sidewalk's pedestrian lane, etc.). Full breakdown:
- `cv_project_hovyc` (was Doors-only): doors 1,326, person 103, stairs 77, chairs 38, vehicle 37, pole 32, trash_bins 20, bicycle 17, tables 11, animals 5, pedestrian_lane 4, motorcycle 4.
- `trashcan_detection_pihfn` (was Trash-Bins-only): trash_bins 559, person 73, vehicle 20, chairs 9.

Spot-checked 2 real `cv_project_hovyc` samples (a Pole+Pedestrian Lane image — a street-corner photo with two correctly-boxed poles and a correctly-boxed crosswalk; a Stairs image — a townhouse door with its entrance steps correctly boxed) — both legitimate, tightly-boxed, real content, not review mistakes. `config/datasets.yaml`'s entries for both sources updated with this finding; their `canonical_class` field is now understood to reflect only their original Stage 5.2 conversion mapping, not current full content — functionally harmless, since `cap_per_class.py` reads actual box content, not that field, but worth knowing when reading the config.

Notebook (`fiftyone_review_processed.ipynb`) checklist cell rewritten (version 9→10): corrected `elevator_awvus`'s stale note, added the operating-policy statement reconciling the `[x]`-means-what tension, annotated every already-actioned row with its real disposition and DEC citation, and listed the 7 sources added since DEC-076's original ranking that have no review status on this list yet.

### Rationale

Tracing the actual numeric consequence of a configuration change before finalizing it (the open_images-squeezed-to-0 finding) rather than trusting the request's literal wording to "just work" follows the same discipline DEC-086 just established for the `audit_status` gating bug — a config/priority change is only as good as its verified real effect, not its stated intent. Investigating the unexplained candidate-count shift before writing it off as "probably fine" (rather than either alarm-firing about a hypothetical concurrent-session collision or silently accepting stale numbers) matches this project's standing bar: re-derive from real files, spot-check with real images, only then trust a number enough to document it.

### Consequences

- `dataset/processed/roboflow_cv_project_hovyc/labels/` and `roboflow_trashcan_detection_pihfn/labels/` now hold the reviewed, cleaned versions — the next `cap_per_class.py`/`merge.py` run will use these, not the pre-review originals. Backups of the originals exist under `dataset/backups/` if this ever needs reverting.
- `cap_per_class.py` → `merge.py` → `dedup.py` → `split.py` → `generate_yaml.py` rerun is still pending (unchanged) — now also picks up the priority reordering, the 2 newly-benched sources, and the promoted labels' multi-class enrichment.
- `revised_pedestrian_obstacle`'s Vehicle motorcycle-contamination cleanup remains a real, not-yet-done visual-review task — worth tracking explicitly rather than assuming DEC-083's original "flag for later" note covers it.
- The other 7 sources added in DEC-082/083 (`stair_gaptw`, `elevator_status_0iq4p`, `pothole_voxrl`, `wtf_dwvgm`, `revised_pedestrian_obstacle`, `dlsu_d_vehicle_type_detection`, `roitrikee`) still have no `fiftyone_review_processed.ipynb` review status at all — flagged on the notebook checklist, not resolved here.
- Final verified `--dry-run` state (all floors met, ratio invariant 2.67 ≤ 3.0): Person 4,500, Vehicle 4,500, Motorcycle 4,500, Pole 2,186, Animals 4,500, Stairs 2,371, Shelf 2,386, Doors 4,500, Chairs 2,984, Tables 4,500, Tricycle 3,798, Potholes 2,661, Trash Bins 1,683, Elevator 3,191, Pedestrian Lane 2,745, Bicycle 3,655.

---

## DEC-088: `hide_duplicates` Added to `fiftyone_review_processed.ipynb` for the Post-Dedup Review Pass; Confirmed `dedup.py` Checks Across Sources, Not Just Within One

- **Date:** 2026-08-21
- **Status:** Accepted
- **Related:** DEC-062 (dedup.py's original design), DEC-080 (`restrict_to_merged`, the pattern this follows), DEC-078 (exclude-tag mechanism, a different "remove from consideration" tool this complements)

### Context

Student described their actual working sequence, which deliberately reorders the documented pipeline: visual inspection of easily-checked sources first (done: `cv_project_hovyc`, `trashcan_detection_pihfn`) → merge → dedup → a **second** visual inspection pass, post-dedup → final merge curation → split. Rationale, student's own: doing dedup before the second inspection avoids spending review time re-drawing boxes on images that turn out to be duplicates of each other. Two follow-up requirements: (1) duplicate images must be **hidden entirely** from that second review pass, not just tagged/flagged for the student to notice and skip themselves; (2) confirm whether `dedup.py` actually checks across different source datasets, not just within one, since two different Roboflow projects could plausibly contain the same underlying photo.

### Decision

**Confirmed empirically: `dedup.py` checks across all sources simultaneously, and cross-source duplicates are real and common, not a hypothetical.** `dedup.py` builds one FiftyOne dataset from the *entire* `dataset/merged/images/` pool (all sources pooled together, not iterated per-source) before running both duplicate checks — cross-source detection was never a special case, it falls out of the pool being unified. Verified against the real (if stale) `dataset/reports/dedup_report.json`: **474 of 1,893 exact-duplicate groups (25%) and 78 of 520 near-duplicate groups (15%) span more than one source.** Most striking example: `roboflow_augmented_tricycle` and `roboflow_me5_u6rvg` (both benched this session, DEC-087, for unrelated reasons — never having gone through a full review) share dozens of byte-identical images under matching base filenames (e.g. `11_JPG_jpg...` appears in both) — strong evidence the two Roboflow projects were sourced from the same original dataset. Other real cross-source pairs found: `exdark`↔`open_images`, `open_images`↔`roboflow_escalator_stairs`, `roboflow_elevator_status_s4lrk`↔`exdark`.

**`hide_duplicates` added to `fiftyone_review_processed.ipynb`** (version 9→11, this entry plus DEC-087's checklist rewrite): a new per-source review toggle, following the exact pattern `restrict_to_merged` (DEC-080) already established. When `True`, for the currently-reviewed `source_key`: every image that appears as a `"duplicates"` member of any `dataset/reports/dedup_report.json` group (exact *or* near) is skipped entirely during the FiftyOne dataset build — never added as a sample, not shown with a badge, genuinely absent — per the student's explicit "hidden, not tagged" requirement. Each group's `"kept"` representative is deliberately *not* hidden, so exactly one reviewable copy survives per duplicate cluster. Refuses to run (raises, not a silent under-hide) if `dedup_report.json`'s `images_checked` doesn't match `dataset/merged/`'s current real image count — the same staleness guard `split.py`'s own `load_duplicate_groups()` already uses, applied here for the same reason: hiding based on a stale report (e.g. one computed before this session's new sources merged in) could hide the wrong images or miss real ones.

**Deliberately scoped to hiding from review, not physically removing anything.** No change to `merge.py`, `dedup.py`, or `split.py` — duplicates still physically exist in `dataset/processed/`/`dataset/merged/`/`dataset/final/` exactly as before; `split.py`'s existing union-find duplicate-grouping (keeping a whole cluster in one split) is unaffected and still needed regardless. This entry only changes what one review notebook *displays*, matching the student's own framing ("in fiftyone").

### Rationale

Verifying the cross-source claim against real report data rather than reasoning abstractly about whether it's *possible* follows this project's standing discipline — and it mattered here, since the answer (25% of exact-duplicate groups cross sources) is large enough to materially change how seriously the student should treat per-source-only review as insufficient. Scoping the fix to the review notebook's display logic (not the physical pipeline) keeps this change small and reversible, consistent with not pre-empting the larger "should duplicates be physically pruned somewhere" question the student hasn't decided yet — flagged as open, not resolved here.

### Consequences

- `hide_duplicates` has no usable effect yet — the only `dedup_report.json` on disk is the stale 2026-08-14 6,000-sample result, predating this week's changes entirely. It will raise (staleness guard) if turned on today. Real use requires a fresh `dedup.py` run against the current merged pool first (see DEC-089 for RunPod planning toward that).
- Whether duplicates should ever be *physically* removed from the pipeline (not just hidden during manual review) remains an open, undecided question — `split.py`'s current design keeps every image, just groups duplicates to the same split. Not changed by this entry.
- Syntax-verified (`ast.parse` on every code cell) since it couldn't be run for real against live data this session — no `dedup_report.json` exists yet that matches the current pool.

---

## DEC-089: RunPod Full-Scale Dedup Executed for Real — 66,907-Image Pool, Real Results, and the Operational Lessons From Actually Running It

- **Date:** 2026-08-22
- **Status:** Accepted
- **Related:** DEC-062 (dedup.py's original design), DEC-068/069 (`--hard-cap` presets), DEC-072 (killed local full-scale attempt, the reason this moved to RunPod), DEC-088 (`hide_duplicates`, which this unblocks), `docs/RUNPOD_DEDUP_PLAN.md` ("RunPod execution specifics" section — reference for upload/smoke-test approach; its Phase 1-3 union-of-4500/9000 sequence was superseded before this run by a single flat `hard_cap=10000` selection, already reflected in that doc)

### Context

The 66,907-image `dataset/merged/` pool (built from `cap_report_hardcap10000.json`, hard_cap=10000 = 9000 real cap + 1000 slack, per the student's "one dedup investment, ever" design) needed a real GPU dedup run — the only report on disk was a stale 2026-08-14, 6,000-image local sample predating this week's fixes entirely. This entry records the run actually happening, plus the real operational friction hitting nearly every step, since several of these are non-obvious and worth not rediscovering next time a pod gets provisioned for this project.

### Decision

**Ran to completion.** `dedup.py --full-scale --num-workers 16` on a RunPod pod, executed via a purpose-built local wizard script (`runpod_dedup_wizard.sh`, gitignored, not committed — ephemeral by design per the mattpocock wizard skill). Real results, verified against the live pool before trusting them:

- `images_checked: 66907`, `near_duplicate_sample_size: 66907` — true full-scale, zero sampling on either check, confirmed equal to the real merged-pool count before download.
- `exact_duplicate_groups: 22`, `exact_duplicate_files: 23` — small, as expected for filehash-exact matches across independently-sourced data.
- `near_duplicate_groups: 5868`, `near_duplicate_files: 12692` (~19% of the pool) — high but plausible given several sources are video-frame-derived or augmented; full coverage was confirmed, so this reflects the data's actual near-duplicate density, not an under-thorough run.
- `near_duplicate_threshold: 0.2` — unchanged from FiftyOne Brain's own default. **Left explicitly uncalibrated for this run**: the report's own `near_duplicate_threshold_source` field notes the `[0.1, 0.25]` guidance may be tuned for FiftyOne's default embedding model, not `mobilenet-v2-imagenet-torch` (used here for ~9x measured speed). Not resolved by this run — a local, GPU-free visual spot-check of a few groups is the recommended next step before treating 0.2 as validated.
- GPU: **RTX A6000** substituted for the originally-planned RTX 3090 — 3090 stock disappeared from the marketplace between planning and deployment (normal churn, not scarcity specific to this project). A6000 met the same sizing bar (8 vCPU / 50GB RAM / 48GB VRAM) at a comparable $0.53/hr.
- `--num-workers`: swept empirically (8/16/32/48/64 at `--limit 2000`) rather than assumed from vCPU count (pod had 96 vCPUs, misleadingly suggesting "higher is better"). **16 won clearly** (22.7 samples/s on the embeddings pass, the peak of the sweep) — throughput *degraded* monotonically past that point (32→16.8, 48→12.5, 64→8.5 samples/s). This reproduces, on CUDA, the same DataLoader-worker-oversubscription regression `dedup.py`'s own docstring already documented for MPS — confirms it's structural (single-GPU-bound embedding pass, workers only prefetch) rather than an MPS-specific quirk.
- **Region**: the plan's original EU-only constraint (driven by 3090 stock at the time) was explicitly dropped mid-session in favor of "whatever datacenter has both storage capacity and a suitable GPU right now" — repeated capacity/storage-cluster errors made strict region-pinning counterproductive given no correctness dependency on region, only upload latency.
- **Network Volume: considered and rejected for this run.** ~$3.50 upfront for 50GB, and volume-compatible datacenters had materially worse/pricier GPU selection (mostly no 3090, cheaper options limited). Dedup is explicitly one-time (student's standing "no intentions to rerun" position); a volume's value is surviving Stop across *many* sessions, which doesn't apply here. Deferred to the training phase instead, which genuinely will span multiple sessions.

**Real incidents hit during execution, and the fixes now baked into the wizard** (useful precedent for the next RunPod session, training or otherwise):
- SSH key auth failed (`password:` prompt) because the account's SSH public key was added to RunPod Settings *after* the first pod was already deployed — RunPod bakes account keys in at pod boot, not retroactively. Fix: register the key before deploying, or redeploy after registering.
- `tar`-over-SSH from macOS to the pod produced per-file `Cannot change ownership to uid 501, gid 50: Operation not permitted` plus AppleDouble (`._*`) sidecar-file warnings — cosmetic but overwhelming at 66,907 files, and the `._*` files are real files on the pod that `list_images()`'s `Path.glob()` does **not** skip (verified directly: unlike shell globbing, `pathlib.Path.glob("*")` matches dot-prefixed names), so they'd have polluted `dedup.py`'s pool count if left in place.
- **Upload mechanism switched from `tar` to `rsync`** (`-a --no-owner --no-group --partial`) mid-session specifically for resumability, after a Stop-triggered GPU reclaim (below) made "redo the whole multi-GB upload" a real, repeated cost rather than a hypothetical. `rsync` also sidesteps the AppleDouble problem entirely (doesn't create sidecar files). One sharp edge: `rsync`'s receiver does **not** create multiple missing parent directory levels the way `tar` extraction implicitly does — the wizard now runs `mkdir -p` on the full destination path before invoking `rsync`, after hitting exactly this failure once.
- Pod's Python was PEP 668 "externally-managed-environment" (Debian/Ubuntu 3.12) — bare `pip install fiftyone` refused; fixed with `--break-system-packages`, judged safe/appropriate for a disposable single-purpose container rather than setting up a venv (which would've meant threading activation through every later remote command).
- `time python3 ...` failed remotely (`bash: line 1: time: command not found`) because a preceding `VAR=value` prefix on the same command strips `time`'s bash-reserved-word status, making bash search for a literal `/usr/bin/time` binary that doesn't exist on a minimal pod image. Fixed by wrapping `time` around the *local* `ssh` invocation instead of embedding it in the remote command string — measures the same wall-clock duration without depending on anything installed on the pod.
- **A Stopped pod's GPU got reclaimed overnight** ("Your Pod's GPUs are no longer available" — Community Cloud doesn't guarantee a Stopped pod's physical GPU stays reserved). RunPod's "Automatically migrate" option reported no instances available even though the Deploy tab showed 3090 stock elsewhere — migrate is scoped to the pod's existing storage cluster/datacenter, not the general marketplace. No data was actually lost (the pod was still mid-upload; the source images live locally regardless), but recovering cost close to a full calendar day. **Operating rule adopted for the rest of this run**: keep the pod *Running* continuously through upload → smoke tests → real run → download rather than Stopping between steps — Stop is what exposes the reclaim risk, a Running (even idle) pod keeps its GPU allocated.
- Post-run, the pod was deliberately **kept Running (not Stopped, not Terminated)** as time-boxed insurance against needing a fast re-run (e.g. if the threshold spot-check above finds 0.2 is miscalibrated) without repaying the multi-hour upload cost — explicit tradeoff given the real ~8-hour cost of this run (dominated by upload, not GPU compute) against the ~$0.53/hr idle cost of leaving it up. To be terminated once the threshold spot-check is done and the report is judged satisfactory, not left open-ended.

### Rationale

Recording the operational incidents alongside the results, not just the final numbers, matches this project's standing discipline of writing down *why*, not just *what* — several of these (the `Path.glob()` dotfile behavior, `rsync`'s parent-directory requirement, `time`'s reserved-word interaction with `VAR=value` prefixes, Stop-vs-Terminate's real reclaim risk) are non-obvious enough that re-deriving them from scratch next time would cost real hours again. The GPU/region substitutions were driven by live marketplace conditions, not a change in the underlying sizing logic (`docs/RUNPOD_DEDUP_PLAN.md`'s vCPU/RAM-over-GPU-tier reasoning held throughout and picked correct replacements both times).

### Consequences

- `dataset/reports/dedup_report.json` is now real and current — `hide_duplicates` in `fiftyone_review_processed.ipynb` (DEC-088) is unblocked; its staleness guard will pass.
- The near-duplicate threshold (0.2) remains an open, unverified calibration question — flagged, not resolved. A local visual spot-check (no GPU needed) is the recommended next action before the next session treats this report as fully validated.
- `runpod_dedup_wizard.sh` is gitignored and not part of the repo's tracked history by design (ephemeral per its originating skill) — the operational lessons above are captured here specifically so they aren't lost along with the script.
- Per the original session scope, this entry closes out "get a verified real dedup report" — capping back to 5,500, the post-dedup review pass, and designing "exclude then trim" remain explicitly next-session work.

---

## DEC-090: Near-Duplicate Threshold Reviewed and Accepted; Pedestrian Lane's Post-Dedup Floor Shortfall Closed With a New Source, Added Without Re-Running GPU Dedup

- **Date:** 2026-08-24
- **Status:** Accepted
- **Related:** DEC-089 (the real dedup run this reviews/extends), DEC-062 (dedup.py's original exact/near coverage asymmetry design, which is what made this possible without new guard code), DEC-082 (the exclude-tag precedent for defective-box sources, applied again here)

### Context

DEC-089 flagged the 0.2 near-duplicate threshold as unverified for the `mobilenet-v2-imagenet-torch` embedding space, and left Pedestrian Lane's real post-dedup count at risk relative to the student's floor policy. Both needed resolving before the next session's cap-to-5,500 work could proceed on solid ground.

### Decision

**Near-duplicate threshold: reviewed and accepted, not retuned.** Built `notebooks/fiftyone_near_dup_inspection.ipynb` (split out from `fiftyone_review_processed.ipynb` to keep concerns separate) — loads every near-duplicate group, supports individual and bulk-by-source `false_positive` tagging, exports findings to `dataset/reports/near_duplicate_false_positives.json`. Confirmed a real FiftyOne Brain caveat concretely (not just from documentation): exported `distance` values can be up to 8.10, far past the 0.2 cutoff, because `neighbors_map` reports distance to a different surviving neighbor than the one that triggered the original flag — group membership stays trustworthy, the distance number doesn't. Student tagged 2,852 false positives (individual + bulk-by-source, notably all of `exdark`/`open_images`'s flagged duplicates). **Student's own empirical finding, worth preserving**: false-positive-ness consolidates cleanly by source — a source's flagged near-duplicates are either mostly genuine or mostly false positives, rarely a real mix — which is what made a 2,852-image review tractable by hand. Student is satisfied with this as the final word on the threshold for this dedup pool; not revisiting further.

**Potholes' 68.9% dedup impact (DEC-089) resolved as expected via this review**: concentrated almost entirely in `roboflow_pothole_voxrl` (94.7% flagged, likely embedding-collision from narrow subject matter — pothole photos are visually similar as a category, not necessarily near-duplicates of each other). Post-correction: Potholes recovered to 1,684 images (safely above the 1,500 floor). Doors/Elevator/Stairs remained heavily impacted after the same review effort — treated as more likely genuine duplication (video-frame-derived sources), not pursued further since all three sit above 1,500 regardless (student's explicit policy: "as long as its above 1500 images it doesn't matter if even 90% of the images are cut").

**Pedestrian Lane did not clear the floor from review alone** (1,362 post-correction) — closed by adding a new source, `roboflow_crosswalk_detector_lz3hc` (202 images, single `crosswalk`→Pedestrian Lane class, CC BY 4.0, student-forked from `insu-park/crosswalk-detector` and regenerated with augmentation explicitly disabled — the original project's only two versions were 2.4x/5.35x augmented, would have reintroduced exactly the duplication problem being solved). Added via `config/datasets.yaml`, acquired, converted (269 boxes, 0 dropped).

**Added without re-running GPU dedup**, by design, not by skipping the check: built `scripts/preprocess/dedup_extend_exact.py`, which extends only exact-duplicate coverage (cheap, filehash-based, CPU-only, idempotent) for a newly-added source against the live merged pool — run for real, 0 matches found. `images_checked` grew honestly to 67,109; `near_duplicate_sample_size` and the new `near_duplicate_covered_source_keys` field (the 16 sources with real GPU near-dup coverage, captured from the live pool before this addition, while still knowable) were deliberately left untouched, so the new source's lack of near-duplicate coverage stays an honest, visible gap rather than an implicit assumption. **Verified directly, not just reasoned about**: `split.py`'s and `hide_duplicates`' existing staleness guards (DEC-062/088) both pass cleanly against the new 67,109-image state with zero code changes — both guards only ever gated on `images_checked` vs. live pool size, an asymmetry the pipeline was already designed to tolerate for the unrelated original reason of local-run near-dup sampling. `cap_per_class.py --hard-cap 10000` and `merge.py` re-run for real; `dataset/merged/` now 67,109 images, verified on disk.

**Real defect found during the new source's manual review, not yet fully resolved**: some of the 202 images are portrait content stored as a landscape pixel grid with no EXIF orientation data, so their boxes (correct for the intended orientation) land in the wrong place relative to the raw file as exported. Confirmed concretely for one sample (visual inspection + box-position cross-check). An automated detection heuristic (boxes anchored at the left edge) was tried and rejected — too many false positives from legitimate edge-touching compositions in web-sourced photos; this needs human eyes, same as the false-positive review did. Resolution: handle via the same `exclude` tag workflow already established for defective-box sources (DEC-082), not a geometric fix. **Review deliberately left unfinished** — student is doing it by hand, and is intentionally deferring full completion until after the next session's cap-to-5,500 step narrows down what actually needs reviewing (same review-efficiency logic as DEC-088).

**Reversibility**: full backup of everything this changed sits at `dataset/backups/pre_crosswalk_add_20260824_190458/` (`cap_report_hardcap10000.json`, `dedup_report.json`, `config/datasets.yaml`, `config/classes.yaml`). `dataset/merged/` itself was not separately backed up — `merge.py`'s destructive-rebuild design means it's already fully reproducible from the backed-up cap report.

### Rationale

Both problems (threshold calibration, floor shortfall) were solved without touching the RunPod GPU investment DEC-089 was built to be one-time — the threshold via human review with tooling, the floor via a source addition whose duplicate-coverage gap is tracked honestly rather than assumed away. Verifying the staleness guards actually passed (not just arguing they should) matches this project's standing discipline of confirming against real execution before trusting a design.

### Consequences

- `roboflow_crosswalk_detector_lz3hc`'s box-orientation cleanup is real, open, unfinished work — not to be assumed done by a future session reading only this entry.
- `near_duplicate_covered_source_keys` in `dataset/reports/dedup_report.json` is now the authoritative record of which sources have real GPU near-dup coverage; any future source added the same way (`dedup_extend_exact.py`) should leave it untouched, same pattern.
- Next session's cap-to-5,500 work picks up the new 67,109-image pool automatically — no special-casing needed, `crosswalk_detector_lz3hc` is just another eligible candidate in `cap_per_class.py`'s pool now.

---

## DEC-091: Fresh 5,500-Cap Selection Run and Verified as a True Subset of the 10,000 Selection

- **Date:** 2026-08-24
- **Status:** Accepted
- **Related:** DEC-090 (the 67,109-image pool this reads from), DEC-042 (hard_cap/floor/ratio-invariant policy this run reports against)

### Context

DEC-090 left the fresh `cap_per_class.py --hard-cap 5500` run and its subset-verification as explicitly next-session work — the review slice can't safely be treated as a prefix of the 10,000 selection without re-running the actual selection logic, since a raw slice could disregard `CLASS_PRIORITY_SOURCES` and starve later-priority sources.

### Decision

Ran `python3 scripts/preprocess/cap_per_class.py --hard-cap 5500` against the current 67,109-image `dataset/processed/` pool (9 benched/failed sources correctly excluded via `get_inactive_processed_source_keys()`, DEC-086). Wrote `dataset/reports/cap_report_hardcap5500.json`. **Verified directly** (not assumed): for every one of the 16 classes, every `(source, filename)` pair in the 5,500 selection is also present in `dataset/reports/cap_report_hardcap10000.json`'s selection — the true-subset property holds under the current config, same as the earlier (now-stale) verification found for an older config.

All 202 `roboflow_crosswalk_detector_lz3hc` images made it into Pedestrian Lane's 5,500 selection (source pool small enough to be fully claimed) — the upcoming review pass will see the complete crosswalk source, not a partial slice, so finishing its exclude-tagging (deferred from DEC-090) covers everything relevant.

Two things this run surfaces, not new problems but worth recording as visible at this cap level:
- **Trash Bins** (1,683 images) sits just under the 5,500-cap's derived floor (1,833 = 5500 // 3) — `floor_met=False`. It was already known to be a small class; this is the same shortfall pattern as Pedestrian Lane's pre-DEC-090 state, just not (yet) large enough to warrant its own new-source fix.
- **Ratio invariant misses 3:1** (max=Person 5,500 / min=Trash Bins 1,683 = 3.27) — `recompute_hard_cap_trigger` fired (would suggest recomputing hard_cap as 3×1683=5049), **not applied**, same standing student-owned call as DEC-042's ratio invariant and `docs/OPEN_QUESTIONS.md` #1.

### Rationale

Re-verifying the subset property rather than reusing the earlier (pre-crosswalk-source, pre-DEC-090) confirmation matches this project's standing discipline of not trusting a check performed against a since-changed config — `CLASS_PRIORITY_SOURCES`, the source pool, and the benched-source list have all changed since the original verification.

### Consequences

- `dataset/reports/cap_report_hardcap5500.json` is now the real input for the next step: reviewing this slice in `fiftyone_review_processed.ipynb`.
- Trash Bins' floor shortfall and the ratio-invariant miss are flagged for visibility, not resolved here — they existed at the 10,000-cap level too (in less visible form) and are the student's call per `docs/OPEN_QUESTIONS.md` #1, same as before.
- "Exclude then trim to 4,500" (the next real step) still has no implementation — `merge.py` already has an exclude-subtraction mechanism (`load_excluded_pairs()`), but nothing trims the post-exclusion counts back down to a true 4,500 per class; confirmed by direct code reading, not assumption.

**Same day, follow-up:** re-ran `box_audit.py --pool merged` fresh against the current `dataset/merged/` (still the 10,000-cap pool at time of running — the merge hasn't been rebuilt at 5,500 yet) to get current per-source flag rates, since the existing `merged_box_audit_report.json` predated dedup/DEC-090's crosswalk-source addition/9 benched sources and was no longer trustworthy for ranking. Built a "review #2" checklist (`notebooks/fiftyone_review_processed.ipynb`, new bottom cell, notebook version bumped to 20) covering all 17 active sources for the first time — the original DEC-076 checklist only ever covered Roboflow sources. Roboflow sources ranked first (by flag rate, high to low), non-Roboflow sources (crowdhuman, exdark, open_images, both Dataset Ninja sources) ranked the same way but placed as a lower-priority block at the bottom, per the student's explicit instruction. Confirmed via `docs/DECISIONS.md` history that none of the 5 non-Roboflow sources have ever gone through a full write-back review in this notebook — all start unchecked. `cv_project_hovyc`/`trashcan_detection_pihfn` carried forward as already-reviewed (DEC-087 promotion) rather than reset to unreviewed.

**Same day, second follow-up — two student questions, both checked directly, not assumed:**

1. *"Are the datasets only scoped to what's in dedup — mark unused Roboflow sources as failed."* Checked `config/datasets.yaml` in full (parsed via PyYAML, not grepped by eye): all 25 Roboflow project entries already carry an explicit `audit_status` — 12 `pending` (the current active set, exactly matching the 12 active Roboflow sources counted above) and 13 already `failed`/`benched`, every one with a documented reason already in its `audit_note`. Four of the 13 (`stairs_lusiz`, `stairs_hsatv`, `traffico_y1`, `jeep_hozhs`) were never even pulled (no `dataset/processed/` directory), so they were never candidates for dedup or anything downstream in the first place. No config change was needed — the ask was already satisfied by existing bookkeeping, confirmed rather than assumed.

2. *"Does the review notebook show only non-duplicate images?"* Yes — `hide_duplicates` (DEC-088) already exists in the build cell (`p0review3build`), default `False`. **Found and fixed a real bug while checking it**: the mechanism read `dataset/reports/dedup_report.json`'s duplicate groups directly, with no awareness of `dataset/reports/near_duplicate_false_positives.json` (DEC-090's 1,568-pair — 2,852-image — threshold review) at all. Turning `hide_duplicates = True` as-is would have silently re-hidden every image the student already spent real review time confirming was NOT a duplicate. Fixed: the build cell now loads `near_duplicate_false_positives.json` (when present) and excludes any `"duplicate"` filename it lists from the near-duplicate hide-set before applying it — exact-duplicates are left untouched, since that review never covered them (byte-identical, no ambiguity to false-positive-check). Notebook version bumped to 21.

**2026-08-25: `roboflow_pothole_voxrl` traced to 35/665 visible under `hide_duplicates`, and a second real tag-loss incident found and fixed.** Diagnosing why only 35 of 665 images stayed visible led to computing the source's true near-duplicate flag rate directly: 630/665 = 94.7%, exact match to DEC-090's own already-documented number for this source. But `pothole_voxrl` had **zero** entries in `near_duplicate_false_positives.json`, contradicting the student's memory of having bulk-tagged it (along with `roboflow_cv_project_hovyc`, `crowdhuman`, `dataset_ninja_pothole_detection`, `dataset_ninja_road_damage_detector`, `roboflow_trashcan_detection_pihfn`) false-positive in `fiftyone_near_dup_inspection.ipynb`, with a remembered console output of "Tagged 2852 duplicate-role samples." Root cause, confirmed directly with the student: the bulk-tag cell ran (2,852 real tags applied live), but the write-back/export cell was never run afterward, and the kernel has since been restarted — the 2,852 tags are gone, unrecoverable, need to be redone.

**Underlying gap, now fixed**: unlike `fiftyone_review_processed.ipynb`'s build cell, `fiftyone_near_dup_inspection.ipynb`'s build cell (`aaf4b3fc`) had no rebuild guard and never set `persistent = True` — any re-run (including an innocuous kernel-restart + rerun-from-top) would `fo.delete_dataset()` and silently wipe every tag with no warning. This is the exact same class of bug DEC-087's session already found and guarded against in the *other* notebook (the `cv_project_hovyc`/`door_detection_zqt59` tag-loss incidents) — just never ported over here. Fixed three ways: (1) added the same guard pattern (refuses to rebuild a persistent dataset unless `FORCE_REBUILD = True`), (2) added `spotcheck_dataset.persistent = True` after the build, (3) inserted a new "resume without rebuilding" cell (`resume_no_rebuild`) mirroring the review notebook's own, so a restart no longer requires touching the guarded build cell at all. Also fixed a real, separate bug caught in the same pass: `BULK_FALSE_POSITIVE_SOURCES` had `"dataset_ninja_road_damage"` (no such source key) instead of the real `"dataset_ninja_road_damage_detector"` — that entry would have silently matched 0 samples even on a successful run. Notebook version bumped to 2.

**Not yet done**: the 2,852-tag bulk-tag pass itself needs to be redone from scratch (now with the typo fixed and the guard in place) and, critically, followed immediately by the write-back cell in the same sitting — the guard/persistence fixes prevent a *silent* loss on rebuild, they don't replace actually running write-back.

---

## DEC-092: `fiftyone_review_processed.ipynb`'s Live Session Confirmed to Silently Drop Tags Between Checkpoints — Root Cause Not Found, Recovery Method and New Operating Rule Established Instead

> **RESOLVED 2026-08-26 by DEC-097.** The mechanism was FiftyOne's in-process `Sample` cache, not data loss: the kernel was serving cached objects from before the tags were applied, while MongoDB held them the whole time. That is why every hypothesis below came back clean — they were all searching for a deletion that never occurred. Fix: `dataset.reload()` before reading the live dataset. The backup-before-write-back rule this entry established still stands.

- **Date:** 2026-08-26
- **Status:** Accepted
- **Related:** DEC-091 (the 5,500-cap review pass this bug was hit during, reviewing `roboflow_pothole_voxrl`), the mentor-mode memory's existing multi-escalation record of tag/edit loss in this same notebook (`cv_project_hovyc`, `door_detection_zqt59`, the "App-reverts-to-hovyc" saga) — this is a new incident in that same family, not a repeat of a previously-solved one.

### Context

While reviewing `roboflow_pothole_voxrl` (665 images), a `dataset.export(..., dataset_type=fo.types.FiftyOneDataset, export_media=False)` backup taken at 02:46 showed 449 predictions tagged `accept` (246 of them manually tagged below the 0.7 auto-accept confidence threshold) and 1,995 `ground_truth` boxes (including boxes manually copied from predictions into `ground_truth` directly, to reassign them to a canonical class outside the COCO-crosswalk predictions schema). Running write-back at 03:01 — 15 minutes later, no kernel restart, no rebuild — printed "Promoted 208", not 449. A live pre-check run immediately before write-back (`sum(... "accept" in det.tags ...)`) independently confirmed 208 live, at that moment, before write-back touched anything — proving write-back itself was reporting real, correctly-read live state, not a code bug in the promotion logic.

### Decision

**Root cause not found, and stopped actively chasing it** — several concrete hypotheses were raised and each was directly disproven with real evidence, not assumption:
- *Wrong-field mis-tagging* (found 3 `ground_truth` detections tagged `accept` in an earlier backup) — real, but the student confirmed some of the missing boxes were legitimate class-reassignment copies into `ground_truth`, not just misclicks, and the 02:46 backup showed 0 such mis-tags, so this doesn't explain the 449→208 drop.
- *`FORCE_REBUILD` used to fix an earlier 35-image bug* (real, confirmed by the student) — ruled out by hard evidence: the 21:42 backup, taken *after* that rebuild, already showed the correct 665 images, so the rebuild predates the 449→208 window by hours, not the cause.
- *A duplicate/zombie Jupyter kernel silently holding the same dataset* (the exact confirmed root cause of a prior incident in this same notebook family) — **directly checked this time, not assumed**: `lsof -nP -iTCP:5151`/`5152` showed exactly one kernel, one browser tab, one App server for each port; connecting directly to the two other live kernels on the machine (via `jupyter_client.BlockingKernelClient` against their connection files) confirmed neither had a `dataset`, `session`, or `source_key` variable at all. Cleanly ruled out.
- *Fresh predictions regenerated mid-session* (would explain both the loss of manual low-confidence accepts and the near-but-not-exact match between 208 and the backup's 203 auto-only count, via MPS inference non-determinism) — the student directly confirmed they ran nothing that would regenerate predictions.

With every concrete lead disproven and the student explicitly redirecting effort away from further diagnosis, the session pivoted to recovery instead: **all three checked hypotheses being cleanly ruled out, rather than one landing, is itself the honest state of this investigation — the actual mechanism remains unknown.**

**Recovery method used, real and verified**: rather than trusting the live session's write-back again, a standalone script (`writeback_from_backup.py`, scratch — not committed to the repo, one-off) read the 02:46 backup's `samples.json` directly and reproduced the write-back cell's exact logic (promotion, YOLO coordinate conversion, per-file diff classification, `*_excluded.json` update) against that static export, entirely bypassing the live kernel/MongoDB session. Result verified independently by recounting boxes straight from the written files: 2,444 total, matching 1,995 (backup's `ground_truth`) + 449 (promoted) exactly. The previous, incomplete 208-based `labels_reviewed/` output was moved aside (not deleted) to `labels_reviewed_208promoted_bak_20260826_031001/` before being overwritten.

**New standing operating rule for this notebook, going forward**: take a backup (the existing cell 9 mechanism) *immediately* before considering a review session done, then either (a) run write-back in the very same sitting, no gap, or (b) if anything about the live session seems off, do **not** trust write-back's live output — recover from the backup file directly the same way this incident was resolved, rather than assuming the live dataset still matches what was last confirmed. The live session has now demonstrated, twice in this project (this incident and the earlier `cv_project_hovyc`/`door_detection_zqt59` ones), that it can silently diverge from what a human just confirmed was there, without an explanation either investigation was able to pin down.

### Rationale

Continuing to chase an increasingly-expensive, evidence-resistant root cause stopped being the right use of time once every concrete, checkable hypothesis had been individually disproven — the student's redirect to "make what we have durable" over "find out exactly why" was the correct call given the actual goal (a usable, correct `labels_reviewed/`) didn't require the explanation. Recording the ruled-out hypotheses in full, not just the outcome, matters because a future recurrence shouldn't re-spend effort re-disproving the same three theories.

### Consequences

- `roboflow_pothole_voxrl`'s `labels_reviewed/` is now correct and verified (2,444 boxes) — still not promoted over the real `labels/`, that step remains manual and separate, per the notebook's existing design.
- **Every future review session in this notebook should follow the new operating rule above** — this is now a standing practice, not a one-off workaround, until (if ever) the actual mechanism is found.
- The mentor-mode memory has been updated with this same rule so it persists across sessions without needing to be rediscovered from this entry.
- If this recurs, the three ruled-out hypotheses above don't need re-checking from scratch — start from "what else could silently mutate a persistent FiftyOne dataset's tags with no rebuild, no restart, no competing kernel, and no predictions regeneration" as new territory, not from these three again.

---

## DEC-093: Write-Back's Prediction-Promotion Was Not Idempotent — Every Re-Run Duplicated Every Previously-Promoted Box

- **Date:** 2026-08-26
- **Status:** Accepted
- **Related:** DEC-092 (the same review session this was found in, immediately after recovering from that incident)

### Context

After recovering `roboflow_pothole_voxrl`'s review work per DEC-092, the student restarted the kernel, resumed the main (persistent) dataset via the existing "resume without rebuild" cell, added more `ground_truth` boxes by hand, and inspected the result via cell 13's read-only preview (port 5152) before running write-back again. That inspection found duplicated `ground_truth` boxes.

### Decision

**Confirmed a real, unambiguous code bug, not a repeat of DEC-092's unexplained mechanism.** The write-back cell's (`p0review... write-back`, cell 12) promotion loop:

```python
accepted = [det for det in sample["predictions"].detections if "accept" in (det.tags or [])]
...
sample["ground_truth"].detections.extend(fo.Detection(label=det.label, bounding_box=det.bounding_box) for det in accepted)
```

filters purely on the `accept` tag and never clears it, and never records anywhere that a given prediction has already been promoted. Every re-run of write-back re-selects every prediction still tagged `accept` — including ones promoted in an earlier run — and appends a fresh duplicate `Detection` for each. A dataset that's had write-back run N times has every once-promoted box duplicated N times. This dataset had write-back run at least twice live (the 208-promotion run, then again after the DEC-092 recovery + kernel restart), so duplication was real and expected, not a fluke.

**Fixed**: the promotion filter now also excludes anything already tagged `promoted`, and after promoting, each promoted prediction gets `accept` swapped out for `promoted` on its own tags (`accept` stays as a historical record of intent; `promoted` is the new idempotency guard the filter checks). A prediction can now only ever be promoted once, regardless of how many times write-back runs afterward.

**Live duplicates from before the fix were removed** via a one-off deduplication pass against the live dataset — for each sample, `ground_truth.detections` deduplicated by exact `(label, bounding_box)` match, keeping one copy of each. Safe by construction: two independently-drawn boxes sharing floating-point-identical coordinates is not a realistic collision; every actual duplicate here was a byte-for-byte copy created by the bug itself.

### Rationale

This is a straightforward correctness bug once identified — a missing idempotency guard — not a design tradeoff with alternatives worth weighing. Fixing it at the tag level (rather than, say, deduplicating at write-time) keeps the fix visible and inspectable in the App (a `promoted`-tagged prediction is still there to look at) rather than silently masking the symptom downstream.

### Consequences

- **Checked the two other sources ever promoted through this notebook — both were actually affected, not hypothetically.** `roboflow_cv_project_hovyc`: 165 files, 554 duplicate `(label, bounding_box)` lines. `roboflow_trashcan_detection_pihfn`: 95 files, 420 duplicate lines. Both were already promoted over the real `dataset/processed/<source>/labels/` (DEC-087), meaning this duplication had been silently sitting in the **live, active candidate pool** — not just a staging folder — since 2026-08-21, affecting every `cap_per_class.py`/`merge.py`/`box_audit.py` run since. Fixed directly: current (duplicated) `labels/` backed up to `labels_pre_dedup_bak_20260826_035316/` for each source (reversible, not deleted), then deduplicated in place by exact `(label, bounding_box)` match. Verified 0 duplicate files remain in either source afterward.
- **Downstream reports are now stale relative to the corrected box counts**, though the *image* selection itself is unaffected (deduplication only removes redundant boxes within an already-selected image's label file, never removes an image) — `box_audit_report.json`/`merged_box_audit_report.json`'s per-source instance totals for these two sources overcounted by the duplicate amount, and any per-class instance totals that included them (Doors, Person, Stairs, Chairs, Vehicle, Pole, Trash Bins, Bicycle, Tables, Animals, Pedestrian Lane, Motorcycle for `cv_project_hovyc`; Trash Bins, Person, Vehicle, Chairs for `trashcan_detection_pihfn`, per DEC-087's contribution breakdown) are very slightly inflated in any report generated before this fix. Not re-run here — flagged for whoever next runs `box_audit.py`/`cap_per_class.py` for real to pick up the corrected counts naturally.
- `roboflow_pothole_voxrl`'s live `ground_truth` is deduplicated as of this fix; its `labels_reviewed/` needs a fresh write-back run (now safe, idempotent) to reflect the corrected state.

**Same day, follow-up — the DEC-092 recovery script inherited this exact bug.** Checking whether *other* reviewed sources were affected (a direct, full scan of every processed source's `labels/` and any `labels_reviewed/` for internal duplicate `(label, bounding_box)` lines, not assumed clean) found `roboflow_pothole_voxrl`'s `labels_reviewed/` itself had 150/665 files, 450 duplicate lines — because the standalone recovery script written for DEC-092 faithfully reproduced the *original, buggy* promotion logic (it predates this fix), and the 02:46 backup it read from already contained the first write-back's 208 promoted boxes inside `ground_truth`, with those same predictions still tagged `accept` — so the recovery script re-promoted them a second time, same failure mode, different code path. Deduplicated in place, same method as above. `cv_project_hovyc`/`trashcan_detection_pihfn`'s stale (unused, pre-promotion) `labels_reviewed/` folders were also deduplicated for consistency, though only their already-fixed real `labels/` actually matters downstream.

**Two small, separate, pre-existing findings from the same full scan — explicitly NOT the same bug, not fixed here**: `crowdhuman` (6 duplicate lines across 19,370 files) and `open_images` (3 duplicate lines across 31,011 files) have a handful of internal duplicate boxes in their real `labels/`. Neither source has ever been touched by this notebook's write-back — no `labels_reviewed/` exists for either — so this has a different, unidentified origin, most plausibly the Open Images cross-folder merge (DEC-052) for `open_images`, unknown for `crowdhuman`. Flagged, not investigated or fixed — negligible in scale (well under 0.1% of files each) and a real, separate root-cause question if ever worth pursuing.

### Resolution — and a methodology lesson worth more than the fix

After all three fixes above landed and the student re-ran dedup → write-back, they still reported seeing duplicates in the read-only verify App (port 5152). Three further hypotheses were proposed and **all three were wrong**: a stale notebook tab holding pre-fix write-back code (disproven — the student pasted the cell's real content, the fix was present), the `predictions` overlay rendering on top of `ground_truth` (disproven — the student pointed out predictions are already promoted into `ground_truth` in the verify dataset, so there is no separate predictions field to toggle off there), and a retroactive-marking gap for predictions promoted before the `promoted` tag existed (real in principle, but returned 0 — nothing to fix).

**What actually resolved it was abandoning mechanism-guessing and reading the artifact directly.** A scan of `labels_reviewed/` on disk — the exact files the verify App renders — proved it already clean: 2,262 boxes across 665 files, **0 exact duplicate lines**, and only 3 high-IoU (>0.8) same-class pairs in the entire source, all three being two genuinely different accepted predictions of the same object (the model predicting one car twice at slightly different sizes), not promotion artifacts. An earlier "near duplicate" heuristic (same class, centers within 0.05, sizes within 0.05) had produced 79 false hits and was discarded as unfit — it flagged genuinely distinct small potholes clustered together in one image; IoU is the correct measure for "visually the same box," coordinate proximity is not.

With "disk is clean" and "App shows duplicates" both established as facts, only one explanation remained: **the App was not reading current disk state.** The verify cell builds `verify_dataset` as a `persistent=False` snapshot at the moment the cell runs, then binds the App to it — refreshing the browser does nothing, only re-running the cell rebuilds. The snapshot under inspection had been built while `labels_reviewed/` still contained the recovery script's un-deduplicated output. Re-running the cell resolved it; no further code or data fix was needed, because none was outstanding.

**Lesson, recorded because it cost several wrong turns across this and DEC-092**: when a symptom persists after a fix that provably landed, check the concrete artifact (the file on disk, the actual bytes) before theorizing about mechanisms. Three consecutive plausible-sounding theories were disproven at real cost; one direct read of the files produced a provable answer immediately. This is the same discipline the rest of this project already applies to pipeline results ("verified directly, not just reasoned about") — it applies to debugging too.

### Queued fix, agreed but NOT yet applied — the rebuild guard has a hole

Found 2026-08-26 while rebuilding `roboflow_revised_pedestrian_obstacle` with `hide_duplicates = True`: the build cell's guard only raises on `existing.persistent and not FORCE_REBUILD`. But the build cell *always* creates the dataset as `fo.Dataset(dataset_name, persistent=False)`, and persistence is only turned on by a **separate, manually-run cell** (`dataset.persistent = True`). So **a freshly-built dataset has no protection at all** until the student remembers to run that separate cell — re-running the build cell in that window silently deletes and rebuilds, discarding any tagging done in the meantime. Confirmed live: with `FORCE_REBUILD` correctly set back to `False`, re-running the build cell still rebuilt (and re-ran YOLO inference), because the dataset wasn't persistent yet. No work was lost this time (nothing had been tagged), but this is exactly the shape of the `cv_project_hovyc`/`door_detection_zqt59` tag-loss incidents.

Contributing factor worth noting: the instruction given was "set `FORCE_REBUILD = False` again" without stating that the value applies to the *next* run and does not require re-running the cell — the student re-ran it to "apply" the change, which is what triggered the rebuild. Guidance for this flag should say so explicitly.

**Agreed fix, deferred to a stopping point at the student's request** (notebook was mid-session and open; per the standing rule, substantial edits wait for a clean break): create the dataset as `persistent=True` directly in the build cell, so the guard is armed from the moment the dataset exists rather than depending on a remembered follow-up step. The existing `dataset.persistent = True` cell then becomes redundant but harmless — keep it rather than removing it, since kernel history and muscle memory both reference it.

### Final state of `roboflow_pothole_voxrl`

- `labels_reviewed/`: 665 files, **2,262 boxes**, verified free of exact duplicates. Not yet promoted over the real `labels/` — that step remains manual and deliberate, unchanged.
- 3 genuine double-prediction pairs (`img-286`, `img-521`, `img-596`, all class 1/Vehicle, IoU 0.81–0.95) remain as real content for the student to delete by hand if desired — not duplicates in the bug sense, two distinct accepted predictions of one object.
- The notebook gained a permanent, reusable "check/remove duplicate `ground_truth` boxes" cell (markdown + code, `dedup_gt_header`/`dedup_gt_code`) sitting between the write-back warning and the write-back cell itself, so this no longer depends on pasting a snippet from a chat log. Notebook version 24.

---

## DEC-094: Roboflow Sources Found Pre-Augmented; De-Augmented at the Processed Layer and Guarded at Split, Exposing Four Classes as Genuinely Below the Floor

- **Date:** 2026-08-26
- **Status:** Accepted
- **Related:** `docs/OPEN_QUESTIONS.md` #17 (the finding, options, and remaining decisions), DEC-089/DEC-090 (the GPU dedup run this complements rather than replaces), DEC-042 (the floor/cap/ratio policy this re-measures), DEC-091 (the 5,500-cap run this supersedes)

### Context

Mid-review of `roboflow_revised_pedestrian_obstacle`, the student reported still seeing many duplicates even with `hide_duplicates = True`. Investigating the actual files — rather than theorizing, per DEC-093's methodology lesson — showed the cause was not a dedup failure: **most Roboflow sources in this project ship pre-augmented**, and the existing embedding-based near-duplicate check structurally cannot catch that.

### Decision

**The finding, proven not inferred.** Roboflow exports as `<original>_<origext>.rf.<hash>.<ext>`, so augmented copies of one photo share a base name. Base image `100` in `revised_pedestrian_obstacle` exists as 7 files; two carry boxes `14 0.462500 …` and `14 0.537500 …` — identical width/height with the x-center mirrored exactly around 0.5, i.e. a **horizontal flip**. Others share identical box geometry with differing md5 (brightness/noise variants). Active pool: **97,933 files from 75,679 distinct photos**. Worst offender `door_detection_zqt59` at 6.2x (4,493 files, 724 photos). Clean at 1.00x: `pothole_voxrl`, `cv_project_hovyc`, `crosswalk_detector_lz3hc`, and every non-Roboflow source.

**Why not simply re-run dedup at a higher threshold** (the student's own question, answered with measurement rather than argument): measured recall of dedup@0.2 on real augmented siblings is **46.6%** — of 3,539 base-sibling pairs in `revised_pedestrian_obstacle`, it grouped 1,648. `mobilenet-v2-imagenet-torch` features are not flip-invariant, so a flipped copy is genuinely distant in that space; raising the threshold far enough to catch flips would flag masses of unrelated images (the student already hand-reviewed 3,829 false positives at 0.2). Filename base-name matching gives 100% precision and recall instantly. **Explicitly decided: do not re-run GPU dedup for this** — ~8 hours, mostly upload, for a strictly worse signal than a regex. The RunPod pod is kept for the training phase instead.

**Applied — option B, de-augment at the processed layer.** New `scripts/preprocess/deaugment_sources.py` keeps the alphabetically-first file per base group and **moves** (never deletes) the rest into per-source `images_augmented_aside/` + `labels_augmented_aside/`; `--revert` restores completely. Real run: **22,254 files moved aside** across 9 active sources. Verified after: image/label pairing exact in every affected source, zero sibling groups remaining. Labels travel with their image so no orphan labels are left for `box_audit.py` to count.

**Applied — option C, guard at split.** New `augmented_sibling_groups()` in `scripts/build/split.py`, unioned into the existing duplicate-group list. `assign_splits()` already ran real union-find (added earlier for overlapping exact/near-dup groups), so sibling groups merge into the same connected components rather than one grouping source silently overriding the other. Grouping is keyed on `(source, base)`, so two sources that both uploaded a `100.jpg` are never merged — unit-tested directly against a synthetic pool. Post-B this correctly reports **0 groups**: it is now a standing safety net for a future augmented source or a `--revert`, not an active fix. That is the intended end state.

**Shared helper, deliberately not duplicated.** `roboflow_base_name()` lives in `scripts/utils/file_utils.py` and is imported by both scripts. This was a direct response to a real failure earlier in the same session: two ad-hoc copies of the regex (one matching only `_jpg`, one matching `_jpg|_jpeg|_png`) produced **different answers for the same source** (`dlsu_d_vehicle_type_detection` 10,574 vs 8,731 bases), which is exactly the drift the project's "import rather than reimplement" convention exists to prevent.

**Pipeline re-run for real** against the de-augmented pool: `cap_per_class.py --hard-cap 5500` → `merge.py --cap-report cap_report_hardcap5500.json`. `dataset/merged/` is now **44,606 images** (was 67,109). `split.py --dry-run` passes cleanly (31,243 / 6,729 / 6,634), correctly reporting the dedup report as a tolerated superset (67,109 > 44,606).

### Rationale

De-augmenting at the processed layer rather than re-exporting from Roboflow (option A) preserves completed review work — a re-export changes every filename, which would have invalidated `trashcan_detection_pihfn`'s promoted review and `revised_pedestrian_obstacle`'s in-progress one. The tradeoff accepted knowingly: the surviving file may itself be a transformed copy rather than the pristine original, which is harmless for training (a correctly-transformed image with correctly-transformed boxes is valid ground truth) and unidentifiable from Roboflow's naming anyway. Option A remains available and is the right tool if a specific class needs genuinely new imagery rather than honest counting of what exists.

### Consequences

- **Four classes are now below DEC-042's 1,500 floor**, on real post-merge counts: **Stairs 1,417 · Trash Bins 1,339 · Elevator 1,338 · Pedestrian Lane 1,193**. Doors (−66%, to 1,980) and Tricycle (−53%, to 1,784) cleared but thinned sharply. Ratio invariant degraded to 4.61 (was 3.27). **Framing that matters: de-augmentation did not cause this shortfall, it revealed one that already existed** — if the floor exists to guarantee enough genuinely distinct data, those four classes were never actually clearing it; augmented copies were padding the count. Student's call on what to drop, bench, or backfill; explicitly not decided here. Provisionally accepted so far: Stairs and Trash Bins staying low. Pedestrian Lane is under consideration for benching. **Elevator is newly surfaced and unruled.**
- **`roboflow_trashcan_detection_pihfn` shrank from 559 to 215 files after promotion** — its reviewed labels survive for the kept variants; `labels_reviewed/` entries for moved-aside files are now orphaned but harmless.
- Any review dataset built before this run points at moved files and must be rebuilt. `dataset/reports/deaugment_report.json` records exactly what moved.
- Recorded for the drop decision: 11 of 16 classes survive dropping Roboflow entirely (Open Images / ExDark / CrowdHuman / Dataset Ninja cover them). The five with **no non-Roboflow fallback** are Doors, Stairs, Elevator, Tricycle, Pedestrian Lane. The only Roboflow sources droppable at zero cost to another class are `wtf_dwvgm` (475 distinct) and `crosswalk_detector_lz3hc` (202), both Pedestrian-Lane-exclusive.

---

## DEC-095: Per-Source Naming Audit Explains Where De-Augmentation Worked and Where It Could Not — Cross-Source Duplicates Found and Guarded at Split

- **Date:** 2026-08-26
- **Status:** Accepted
- **Related:** DEC-094 (the de-augmentation run this audits and extends; supersedes its "KNOWN LIMITATION" paragraph with measured numbers), `docs/OPEN_QUESTIONS.md` #17, DEC-063 (duplicate-aware splitting), DEC-042 (floor/cap policy)

### Context

After DEC-094, the student asked whether the de-augmentation had actually been *tailored per source* — whether each dataset's naming scheme had been analysed and the collapse fitted to it — observing that "in some datasets it did work effectively, but in some it didn't."

It had not. DEC-094 applied one generic Roboflow regex uniformly to all 26 sources. That was acknowledged rather than defended, and a per-source audit was run to find out whether the uniform rule was actually wrong or merely looked untailored.

### Decision

**The uniform rule was correct, and the reason is now a stated classification.** Two measurements — `rf%` (share of files carrying the `.rf.<hash>` export suffix) and `ratio` (files ÷ distinct base names) — sort every source into three tiers with no overlap:

| tier | rf% | ratio | sources | de-augmentation outcome |
|---|---|---|---|---|
| not Roboflow | 0% | 1.00 | `crowdhuman`, `exdark`, `open_images`, both `dataset_ninja_*` | no-op **by construction** |
| Roboflow, augmentation OFF | 100% | 1.00 | `crosswalk_detector_lz3hc`, `cv_project_hovyc`, `pothole_voxrl` | no-op **correctly** — nothing to collapse |
| Roboflow, augmentation ON | 100% | 1.38–6.21 | the 9 sources DEC-094 processed | worked as intended |

The uniform regex never *failed* on tiers 1–2; there was nothing there to catch. A per-source rule would have produced identical output.

**Roboflow augments only the train split — proven, not assumed.** Cross-referencing processed group sizes against `dataset/raw/roboflow_projects/*/{train,valid,test}`:

```
roboflow_trashcan_detection_pihfn
  train    172 bases   sizes: 3x172     <- every train image -> exactly 3
  valid     22 bases   sizes: 1x22      <- untouched
  test      21 bases   sizes: 1x21      <- untouched
```

`elevator_status_0iq4p` matches (419/424 train at 3x; 121/122 valid at 1x). Roboflow's default 3× setting is therefore recoverable from filenames alone, and **size-1 groups are not misses — they are val/test images**. This retires the concern that de-augmentation had silently skipped a large fraction of each source.

**Three failure modes identified, only one of which is fixable by code.**

1. **`door_detection_zqt59` was exported, re-uploaded, and re-exported.** Its *valid* and *test* splits carry group sizes 5 and 7, which train-only augmentation cannot produce; 72.4% of its base names end in `_JPG`, the fossil of a prior export. Pass 2 treated pass 1's augmented copies as fresh originals — hence 4,493 files from 724 photos and compounding odd group sizes (3/5/7/9). Not recoverable from filenames.
2. **Pre-upload augmentation is structurally invisible to base-name matching.** Bases such as `IMG-20210825-WA0006flip_output_output` carry the transform *inside* the name, so each reads as its own photo. Measured: `escalator_stairs` 158/2,663 bases, `revised_pedestrian_obstacle` 56/2,100, `stairs_i2yia` 29/1,372. Not recoverable from filenames.
3. **Cross-source duplicates — the one with training consequences, and now fixed.** **173 base names / 346 files in `dataset/merged/` are the same photo under two different source keys.** `split.py` grouped per `(source, base)` by design, so these were free to straddle splits, and `dedup.py`'s embedding check links only **29 of the 346**.

**Applied — `cross_source_duplicate_groups()` in `scripts/build/split.py`**, unioned into the duplicate-group list alongside the existing dedup and augmented-sibling groups (`assign_splits()`'s union-find merges overlaps). Guarded by `_is_distinctive_base()`: a base must be ≥10 characters **and** contain at least one letter, so coincidental collisions (`5.jpg`, `100.jpg`, bare numeric stems) are never merged while genuine hits are kept. Live pool: **173 groups / 346 files**; `split.py --dry-run` passes with zero leakage at 31,225 / 6,686 / 6,695.

### Rationale

Kept as a **separate function** from `augmented_sibling_groups()` rather than folded into it, because the two describe different events: one source exporting a photo many times, versus several sources each shipping the same photo once. They also have different life expectancies — `deaugment_sources.py` drives the sibling count to zero at the source, but it operates within one source directory and therefore **cannot** collapse cross-source duplicates. The sibling grouping is a dormant safety net; this one is a permanent, load-bearing fix.

The distinctiveness guard is deliberately conservative. Merging two unrelated images into one split-group costs only a small loss of shuffling freedom; merging by coincidence across a whole pool of short numeric stems would corrupt the split. Erring toward missing a few genuine matches is the cheaper error.

### Alternatives Considered

- **Key cross-source grouping on base name alone, unguarded.** Rejected: `dataset_ninja_pothole_detection`'s `potholes5` and a Roboflow `5.jpg` are unrelated, and the pool contains many such stems.
- **Perceptual/content hashing to catch all three failure modes at once.** Rejected for now — it would catch modes 1 and 2 as well, but it is a new pipeline stage with its own threshold to tune and review, and modes 1–2 affect count honesty rather than leakage. Recorded as available if a class later needs its true distinct-photo count established.
- **Re-exporting `door_detection_zqt59` from Roboflow with augmentation off** (DEC-094's option A). Still available and still the correct tool if Doors needs genuinely more distinct imagery; not triggered by this audit.

### Consequences

- **~317 files of previously-undetected train/val leakage are closed** (346 cross-source duplicate files, of which dedup already linked 29).
- **`roboflow_door_detection_zqt59` was built by scraping Open Images.** Verified visually: `open_images__01b15d5fcabae5b0.jpg` and its `door_detection_zqt59` counterpart are the same room with the same burned-in `3/29/2009 14:46` camera timestamp, differing only by Roboflow's 416×416 resize. 12 such pairs reach the merged pool. Relevant to the pending drop decision — this source contributes less independent data than its file count suggests.
- Largest overlaps recorded for that same decision: `revised_pedestrian_obstacle + wtf_dwvgm` (101 bases), `elevator_awvus + elevator_status_0iq4p` (53), `open_images + door_detection_zqt59` (12).
- `split.py`'s stats now report `augmented_sibling_groups` / `_files` and `cross_source_duplicate_groups` / `_files`, so both grouping sources are visible in `split_report.json` instead of only in stdout.
- **Failure modes 1 and 2 remain open and are not code-fixable.** Distinct-photo counts for `escalator_stairs`, `stairs_i2yia`, `revised_pedestrian_obstacle`, and `door_detection_zqt59` are still overstated by an unmeasured amount. This does not affect leakage (embedding dedup and the split guards still apply) but does mean DEC-042's floor is measured against a mild overcount for those sources.

---

## DEC-096: Two Roboflow Sources Re-Acquired Clean (Re-Pin and Fork), Four Silent Data-Corruption Bugs Fixed, De-Augmentation Strategy Settled Per Source

- **Date:** 2026-08-26
- **Status:** Accepted
- **Related:** DEC-094 (de-augmentation at the processed layer), DEC-095 (per-source naming audit that motivated this), DEC-089 (the RunPod dedup run whose verdict is carried across re-exports here), DEC-090 (the crosswalk fork precedent + `dedup_extend_exact.py`), DEC-093 (write-back idempotency), DEC-092 (backup-before-write-back rule), DEC-042 (floor/cap policy)

### Context

DEC-095 established that Roboflow augmentation is baked into a *version* at generation time and cannot be filtered at download. That reframed the problem: several sources were not "badly augmented", they were **pinned to the wrong version**. Acting on it exposed four separate bugs that were silently corrupting data, none of which had visible symptoms.

### Decision

**A version audit across all 22 Roboflow projects, comparing each version's image count to the project's source-image pool.** Nine were augmented. Seven had no clean version and would genuinely require a fork; **two were avoidable and had simply been pinned wrong**:

| project | was pinned | multiplier | clean version |
|---|---|---|---|
| `door_detection_zqt59` | v3, 5,173 images | **2.60x** | **v1, 1,993** |
| `trashcan_detection_pihfn` | v2, 559 images | **2.59x** | **v1, 216** |

The prior config comment on door read *"latest, 5173 images (up from v2's 4747, v1's 1978)"* — version image count was read as more data when it was the same photos multiplied. That framing is the trap worth remembering.

**`door_detection_zqt59` re-pinned v3 -> v1.** 1,978 raw -> 1,704 converted -> 1,601 after the dedup keep-list. This also resolved DEC-095's open puzzle about why this source behaved incoherently (multiplier 9, only 12.8% of val/test obeying the train-only rule): it was a 2.6x augmented export layered on top of a source pool that *already* contained ~2.4x pre-upload augmentation. Verified from v1's own splits — its `valid` has 251 base-groups of size 3 and `test` 147, and Roboflow never augments val/test, so those triples arrived as separate uploads. Confirmed visually (`Door0444` is one scene with a hue shift and crop). De-augmentation is therefore *correct* for this source, contradicting DEC-095's reading of it: all 474 multi-file bases are distinctive 16-hex or `Door####` names with zero camera-roll names, so same-base genuinely means same photo. Collapsed 1,601 -> **654**.

**`revised_pedestrian_obstacle` forked and regenerated.** All four upstream versions were 1.80x augmented with **no clean version to re-pin**, so a fork was genuinely required. The student's visual judgment led here and was right, while objective metrics initially pointed the wrong way: Laplacian variance read this as the *sharpest* source in the project (3,428 vs open_images' 1,135) because the augmentation recipe was heavy **noise**, and noise inflates that metric. Fork v1 generated through the SDK (`project.generate_version`) with `augmentation: {}` and `preprocessing: {auto-orient: true}` — **3,523 images, exactly 1.00x the source pool**. Median noise dropped 7.64 -> 3.55. **No resize**, deliberately: ultralytics letterboxes to `imgsz` at train time, so an export-time resize is pure irreversible loss, and every non-Roboflow source in this project is already native-resolution while the Roboflow ones had been squashed to between 224x224 and 720x720.

**Dedup verdicts carried across both re-exports rather than re-run.** New `build_dedup_keeplist.py` freezes a source's dedup verdict into a base-name keep-list before re-acquisition; `apply_dedup_keeplist.py` applies it after. The join key is the Roboflow base name, which survives a re-export while every `.rf.<hash>` filename changes. Coverage was **97.8%** for door and **99.8%** for revised_pedestrian_obstacle, so DEC-089's 8-hour GPU run was preserved for both. Scope stated honestly in the tooling: near-duplicate verdicts are claims about *content* and carry over soundly; exact-duplicate verdicts do not (different bytes) and were re-established with `dedup_extend_exact.py`, which found **0** for both — matching DEC-090's crosswalk result. A base is dropped only when every file carrying it was flagged AND none was a group's kept representative, and the tool aborts if under 50% of keep-list bases appear in the re-export.

**Four silent data-corruption bugs found and fixed.** Each produced plausible-looking output:

1. **`acquire_roboflow.py` never cleared its destination.** The SDK's `overwrite=True` only overwrites files it writes; it leaves a previous version's files behind. Re-pinning door produced `train/images/` with **6,137 files** (v3's 4,509 blended with v1's 1,628) while `data.yaml` correctly read `.../dataset/1`. Fixed with `rmtree` before download.
2. **`yolo_to_intermediate.py` had the same bug.** A 1,704-image conversion left `images/` holding **3,790**. Fixed, scoped to only `images/` and `labels/` so `labels_reviewed/` — which holds hand-review work — is never touched.
3. **`dedup_extend_exact.py`'s guard fired unconditionally.** Its `images_checked > merged_count` clause assumed a prior run of itself, but DEC-089's dedup ran against a 66,907-image pool, so `images_checked` is permanently 67,109 while `dataset/merged/` is whatever the current cap produces. It blocked both legitimate extensions. Narrowed to the `already_referenced` check, which is what actually encodes the intent.
4. **The same script silently shrank the coverage record.** `images_checked = merged_count + len(new_images)` was correct for DEC-090's crosswalk (extended *before* merging) but destructive here: it drove 67,109 down to **48,000**, below `near_duplicate_sample_size` (66,907) — an incoherent state claiming the near-duplicate check covered more images than the exact check ever saw. Made monotonic with `max()`; restored from backup and re-ran.

**De-augmentation strategy settled per source, not globally.** DEC-095 added a `provenance` strategy (collapse only groups provably inside train at the measured multiplier). The discriminator for whether the cheaper `filename` strategy is safe turns out to be **camera-roll naming**, not the `_is_distinctive_base()` test used for cross-source grouping — within one Roboflow project a bare sequential number like `102` is Roboflow-assigned and unique, while `IMG_7497` is user-supplied and collides. Measured share of multi-file bases with camera-roll names: `dlsu` **40%** (RISKY — this is where the verified `IMG_7497` collision lives, a motorcycle and a row of buses merged as one), `elevator_awvus` 11%, everything else 0–1%. Applied `filename` to `door`, `revised_pedestrian_obstacle`, `roitrikee`, `stair_gaptw`, `wtf_dwvgm`, `elevator_status_0iq4p`; left `dlsu` and `elevator_awvus` on `provenance`.

**`elevator_status_0iq4p` marked `audit_status: failed`.** Student caught on visual inspection that its boxes do not wrap elevators. The native class names read correctly ("Open Elevator", "Closed Elevator") so the mapping looked sound, but the annotator was classifying elevator *state* and anchored every box on the **floor-indicator display above the door**. Confirmed objectively by geometry: **31% of its boxes are wider-than-tall versus 1% in `elevator_awvus`**, which is what boxing a horizontal indicator strip instead of a tall door produces, while median box area is near-identical (0.125 vs 0.142) — so the defect is orientation, not scale. Unusable: it would teach the detector to find small rectangular displays. Not a relabel candidate at any sane cost.

**Review notebook v25 -> v28.** (a) The build cell now restores `exclude` sample tags from `<source>_excluded.json`, which both makes prior exclusions visible after a rebuild and puts `exclude` into the App's tag vocabulary so it is a checkbox rather than a retyped string. (b) When a source has no prior exclusions it seeds `exclude` on the first sample, printing the filename and instructing that it be un-tagged — the student's explicit call, since FiftyOne offers a tag only when some sample carries it and a zero-sample tag ceases to exist (verified against 1.20; `dataset.tags` is a dataset-level label, unrelated). (c) **The build cell now creates datasets with `persistent=True`.**

### Rationale

(c) above is the most important line in this entry. The rebuild guard only refuses when `existing.persistent` is True, but the cell created datasets as `persistent=False` — leaving every freshly built review dataset unprotected until the student manually ran the `dataset.persistent = True` cell. Any re-run of the build cell inside that window silently deleted the review. **This destroyed `door_detection_zqt59`'s review during this very session** (12 exclusions and 9 Stairs / 3 Bicycle / 2 Pole hand-drawn boxes, classes that existed nowhere else), recoverable only because a snapshot backup happened to exist — and it had already destroyed that source's tags once before, plus `cv_project_hovyc`'s 71 exclusions and 277 accepted predictions. Three destructions, one root cause, a fix that was identified earlier and deferred. The tradeoff — abandoned review datasets now survive restarts and need explicit deletion — is trivially worth it.

### Consequences

- **`dataset/merged/` is 45,132 images.** Class counts: Vehicle/Person at cap, Stairs **1,382**, Doors **1,910**, Elevator **1,350**, Pedestrian Lane **1,286**, Trash Bins 1,683, Tricycle 1,810. Ratio invariant degraded to 4.28. `split.py --dry-run` passes clean at 31,457 / 6,930 / 6,745 with zero leakage.
- **Stairs, Pedestrian Lane and Elevator are tracked as class-removal candidates** at the student's request — explicitly NOT decided, and not to be recorded as dropped. Deprioritised for cleaning because they are 100% single-candidate-class sources: `elevator_awvus` (1,350), `stair_gaptw` (967), `wtf_dwvgm` (475), `crosswalk_detector_lz3hc` (202) — 2,994 images of review deferrable at zero risk to any other class. `revised_pedestrian_obstacle` is explicitly NOT on that list despite its name: only 16% of its boxes are candidate classes, and it is the largest clean Person/Vehicle contributor.
- **`cap_per_class.py` indexes by the class ID present in each label file, not the source's configured `canonical_class`** — verified, and it matters now that hand-added boxes are introducing classes a source was never declared for (the door review added Stairs, Bicycle, Pole, Animals boxes to a Doors-only source). Those count normally.
- **`review_roboflow_cv_project_hovyc`'s live FiftyOne dataset is a stale pre-deduplication snapshot** and must never be written back: it holds 2,297 boxes against 1,742 on disk, the surplus being the 554 duplicates removed from `labels/` under DEC-093. Its disk state is correct and its review is complete; the dataset is simply obsolete. One `acce[t` typo'd prediction tag found in it was fixed surgically on disk rather than by re-running write-back, precisely to avoid reinstating those duplicates.
- `trashcan_detection_pihfn` remains on the augmented v2 by the student's explicit decision (keeping Trash Bins above the floor), reaffirmed after being shown the evidence. Its honest distinct-photo count from that source is **216**.
- Backups from this session: `dataset/backups/pre_door_repin_20260826/` (config, notebook v25 and v27, pre-extend dedup report), `deaugment_statusquo_20260826.json` (the 21,910-file filename-strategy state), `door_pre_restore_*`, `labels_reviewed_pre_restore_bak_*`.

---

## DEC-097: Root Cause Found for Every "Lost Review Edit" in This Project — FiftyOne's In-Process Sample Cache, Not Data Loss. DEC-092 Resolved.

- **Date:** 2026-08-26
- **Status:** Accepted
- **Resolves:** DEC-092 (recorded as "Root Cause Not Found"), `docs/OPEN_QUESTIONS.md` #15
- **Related:** DEC-093 (the duplication bug found while chasing this), DEC-096 (the review notebook this fixes)

### Context

The student reported a recurring pattern, stated precisely enough to be testable: *"when it's time for the write back... some stuff doesn't reflect, some hijinks happen, I end up just loading the backup and then do the actual write back which actually reflects the changes."*

That workaround is the diagnostic. A backup taken from the same `dataset` object moments earlier contained edits the write-back did not. Something was different between how the backup read the dataset and how write-back read it.

### Decision

**Root cause: FiftyOne caches `Sample` objects by id inside the Python process.** Once the kernel has materialised a sample, `for sample in dataset` returns that *same cached object* rather than re-reading the document. The App is a **separate process** — every box drawn, moved or deleted and every tag clicked goes to MongoDB, while the kernel keeps serving its stale copy.

**Measured directly, in a controlled probe** (kernel materialises a sample; a second connection writes to MongoDB exactly as the App would; then the kernel reads):

```
1. kernel materialises        -> 1 box,  tags=[]
2. DB actually contains       -> 2 boxes, tags=['exclude']
3. `for sample in dataset`    -> 1 box,  tags=[]            <-- STALE
4. after dataset.reload()     -> 2 boxes, tags=['exclude']  <-- correct
```

**And the second half, which explains the workaround exactly:**

```
iteration sees   -> 1 box,  tags=[]
dataset.export() -> 2 boxes, tags=['exclude']
```

`dataset.export()` queries MongoDB directly and is **always accurate**. Iteration is not. So the backup cell captured truth while the write-back cell — iterating the same object — wrote a snapshot from before the edits. Restoring that backup worked because `Dataset.from_dir` constructs fresh, uncached `Sample` objects.

**No cell in the notebook called `dataset.reload()`.** Fixed by adding it as the first executable statement of every cell that reads the live dataset: the write-back cell, the ground-truth dedup cell, and the retroactive-promotion cell. `reload()` re-reads documents from the DB and discards nothing unsaved, so it is cheap and unconditionally safe.

### Rationale

This retires a mystery that consumed a large share of two sessions and produced several confidently-wrong theories. **DEC-092 is now resolved**: its 449 → 208 tag drop was never data loss. The kernel was reading a cached snapshot from before those tags were applied, so the tags existed in MongoDB the whole time — which is exactly why recovery from a backup export worked and why every "disproven hypothesis" in that entry (mis-tagged field, stale FORCE_REBUILD, duplicate kernel, predictions regeneration) came back clean. They were all looking for a deletion that never happened.

It also explains, without any additional cause, the numbers in DEC-096's door incident: write-back wrote 730 boxes and 1 exclusion while the dataset genuinely held 733/12 at backup time and 789/12 later. Not three inconsistent states — one live state and two stale reads of it.

**Methodology note worth keeping.** The fix came from taking the student's workaround literally and asking what was mechanically different between the two code paths, rather than theorising about what might have deleted the data. DEC-093 already recorded "read the disk instead of reasoning about it" after a similar detour; the generalisation is: when a workaround reliably works, the difference between the working path and the broken one *is* the bug, and it is directly testable.

### Consequences

- Write-back now reflects the live App state on the first run. The backup-restore-writeback workaround is no longer needed — though the **backup itself remains mandatory before every write-back** (DEC-092's standing rule stands, and this finding makes the reason sharper: the export is the one read path guaranteed accurate).
- Review notebook is **v29**. Prior versions are backed up under `dataset/backups/pre_door_repin_20260826/` (v25, v27, v28).
- **Any FUTURE code that reads the live review dataset must call `dataset.reload()` first.** This is not specific to write-back; it applies to any cell or script that iterates a dataset the App may have touched. The three fixed cells carry the full explanation inline so it is not re-derived.
- The `verify_*` datasets are unaffected — they are built fresh from `labels_reviewed/` on disk, never from cached samples.
- Not changed: `dataset.export()`, `count_sample_tags()`, `count_label_tags()` and other aggregation-based calls were already reading the DB directly. That is why the diagnostics run against this project's datasets throughout this session gave correct numbers while the notebook's own write-back did not.

---

## DEC-098: Predictions Overlay Filtered by Same-Class IoU Against Ground Truth, Not by Containment or Class Exclusion

- **Date:** 2026-08-26
- **Status:** Accepted

### Context

Review has moved from specialty sources (Doors, Poles, Potholes — no COCO analog) onto sources whose own classes overlap the COCO-pretrained `yolov8n` used for the predictions overlay (DEC-074). On those, the overlay re-detects objects `ground_truth` already has a box for. Measured across all 15 live review datasets on 2026-08-26:

Measured with **both** contamination sources removed — already-promoted predictions, and predictions pixel-identical to a `ground_truth` box (promoted by an earlier write-back but never tagged, because the `promoted` tag postdates DEC-093). Each source's raw Roboflow export classes are shown, because they are what decides whether the rule can fire at all.

| source | dup_gt | self-match | raw export classes |
|---|---|---|---|
| `me5_u6rvg` | **33.1%** | 0 | `0,1,2,3,4,5,6,7` — genuinely multi-class |
| `roitrikee` | **25.1%** | 0 | `Tricycle` — entirely the alias rule |
| `augmented_tricycle` | **23.9%** | 0 | `Tricycle,tricycle` — entirely the alias rule |
| `pothole_voxrl` | 17.2% | 3 | `pothole` |
| `trashcan_detection_pihfn` | 4.2% | 140 | `Trashbin` |
| `cv_project_hovyc` | 2.9% | 277 | `Door,Exit up/down/left/right` |
| `door_detection_zqt59` | 2.7% | 0 | `door,hinged,knob,lever` |
| all 8 other sources | 0.0% | 0 | single-class specialty |

**Three sources carry the entire effect**, and all three are explained by their raw labels: `me5_u6rvg` labels eight classes of its own, and the two Tricycle sources are matched only through `GT_CLASS_ALIASES`. Every single-class specialty source lands at 0–4%.

An earlier cut of this table was wrong in both directions and is recorded here because the failure mode is easy to repeat. It reported `pothole_voxrl` at 83.6%, `cv_project_hovyc` at 69.0% and `door_detection_zqt59` at 62.7% — all artefacts of measuring against post-review `ground_truth` containing boxes promoted from the very predictions being tested. For `cv_project_hovyc`, 277 of its 289 matches were pixel-identical self-matches; its raw export contains no COCO class at all, so no real match was ever possible.

### Decision

A prediction is tagged **`dup_gt`** and has its `accept` withdrawn when it overlaps an existing `ground_truth` box of a **matching class** at **IoU >= 0.5**. It is never deleted. New notebook cell `dupgt_code` (v31), re-runnable against a live dataset.

"Matching class" is same-class, plus one documented cross-class alias set:

```python
GT_CLASS_ALIASES = {"Tricycle": {"Vehicle", "Motorcycle", "Bicycle"}}
```

Two classes of prediction are skipped **before** the IoU test and are not counted: those tagged `promoted`, and those matching a `ground_truth` box at **IoU >= 0.999** (`SELF_MATCH_IOU`).

**Cross-class alias pairs match on ANY overlap, not on IoU** — and only on a source whose `ground_truth` contains no box of the alias classes at all, decided from the dataset itself rather than a hand-kept list.

### Rationale

**IoU 0.5 is not tuned to taste.** The best-same-class-IoU distribution over 7,611 predictions is bimodal — 30.8% in `[0.0,0.1)` (same class, different object: a real find) and 64.9% at `>= 0.5` — with a 2.1% valley between. Suppression at 0.3 catches 5,049 and at 0.5 catches 4,940, a 2% spread across the whole valley.

**The alias set is evidence-driven, not guessed.** Tricycle accounts for **3,134 of the 3,406** cross-class matches at IoU >= 0.5 (93%); `gt=Tricycle / pred=Vehicle` alone is 2,322 at median IoU 0.848 — `yolov8n` boxing the identical object under a different name. The remaining 272 are a long tail across every other class pair, left visible deliberately: each alias added is a class-confusion the student can no longer see.

**The pixel-identical guard is geometric, not tag-based, and that is deliberate.** The `promoted` tag only exists for datasets reviewed after DEC-093. `cv_project_hovyc` was reviewed before it: its 277 promoted predictions carry `accept` with no `promoted` tag, and a tag-based guard alone reported it at 69.0% redundant against a true rate of 2.9%. `trashcan_detection_pihfn` has 140 in the same state. An IoU >= 0.999 test catches both without depending on review history — independent re-detection does not land on a hand-drawn or dataset-supplied box to six decimal places.

**Promoted predictions must be excluded before the test, not after.** Write-back copies a promoted prediction into `ground_truth`, so it thereafter matches its own copy at IoU 1.0. `door_detection_zqt59` scores 62.7% "redundant" post-write-back, of which **119 of 121 are that self-match** and mean nothing. Testing them asks whether a box duplicates itself.

### Alternatives Considered

- **Disable the overlay on overlapping sources.** Rejected: forfeits the unlabelled-instance detections that justify the overlay.
- **Filter by containment for SAME-class matches.** Rejected on measurement. Containment cannot separate "this box *is* the labelled object" from "this box is *inside* it." `gt=Tricycle / pred=Person` has **2,101** cases at containment >= 0.9 but median IoU **0.106** — a containment filter deletes 2,101 real person-in-tricycle detections; IoU >= 0.5 removes 31 and keeps 2,070.
- **Exclude whole classes per source.** Rejected: a source specialising in one class still misses instances of it, and a class-level filter hides exactly those. The IoU rule cannot — a missed instance has no ground_truth box to match, so its IoU is 0 and it always survives. This is why the specialty sources measure 0.0%: the rule is self-scoping and does not fire where ground_truth has nothing of that class.
- **Whitelist prediction classes per source** — on a Tricycle-only source show only `Person` predictions and drop `Vehicle`/`Motorcycle`/`Bicycle` outright. Simpler than containment (no thresholds at all) and kills 100% of the fragment noise rather than 763 of 1,207, and `roitrikee` is not a Vehicle or Motorcycle provider under DEC-087 in any case. **Rejected by the student on the grounds that a legitimately distant car in a tricycle photo would be suppressed, adding to the manual burden rather than reducing it.** The objection holds and is what the final rule preserves: **254** alias-class predictions on `roitrikee` touch no tricycle box at all and are kept, 201 of them occupying under 2% of the frame — the smallest at 0.070% — precisely the far-off vehicles a whitelist would have discarded and left to be drawn by hand.
- **Delete redundant predictions instead of tagging.** Rejected: where ground_truth is the wrong box, the prediction is the evidence. Tagging keeps it findable and makes the pass reversible and idempotent.

### Consequences

- Accepts withdrawn at build time: `me5_u6rvg` 4,445 -> 2,438; `augmented_tricycle` 1,570 -> 1,355; `roitrikee` 670 -> 522. Single-class specialty sources are effectively unaffected (`cv_project_hovyc` withdraws 0, `door_detection_zqt59` 0).
- **A separate finding from the same investigation, not caused by it:** `cv_project_hovyc`'s live FiftyOne dataset holds 2,297 ground_truth boxes against 1,743 on disk. The 554-box difference is entirely **exact-duplicate rows** stacked by the pre-DEC-093 non-idempotent promotion — de-duplicating the live dataset reproduces the disk total exactly, class for class. Disk is the correct copy and was already cleaned by the DEC-093 pass at 03:53:23 on 2026-08-26; the live dataset is the stale inflated one. No review work was lost and nothing was changed.
- `roitrikee` lands at **45.8%** (953 of 2,079), with zero alias predictions left overlapping a tricycle and all 867 `Person` predictions kept. `augmented_tricycle` at 23.9%, purely from the alias rule. Both are 0% without it.
- **`dlsu_d_vehicle_type_detection` is the safer of the two vehicle sources, not the riskier one.** It labels `Vehicle`/`Motorcycle`/`Tricycle`, so most of its duplication is caught by plain same-class matching — the robust rule with the clean valley — leaving the alias rule only the leftover. `roitrikee` labels only `Tricycle`, so every suppression there runs through the fuzzier alias path. Alias false-suppression risk on dlsu was measured against its own labels rather than assumed: just **2** real Vehicle/Motorcycle boxes out of 13,434 overlap a Tricycle box at IoU >= 0.5.
- **Deliberately not aliased: `Vehicle` <-> `Motorcycle`.** `dlsu_d_vehicle_type_detection` distinguishes them and is the priority source for both (DEC-087), so a disagreement there is worth seeing rather than hiding.
- **The alias rule's threshold is less robust than the same-class one, and this is a known limit.** Cross-class boxes align less tightly, so `roitrikee`'s alias-IoU distribution has no clean valley: 15.9% of alias-eligible predictions fall in `[0.2,0.5)` against 2.1% for the same-class case, and the dataset-wide rate moves 30.4% -> 25.1% between IoU 0.3 and 0.5. Same-class matching moves 2% across that range. 0.5 is kept for consistency, but a value in this band is a genuine choice for alias pairs rather than a read off a gap.
- **Why `roitrikee` is only 25.1% despite being near-entirely tricycle imagery:** 42% of its predictions (867 of 2,079) are `Person` — riders, which cannot and must not alias-match `Tricycle`. Of the 1,207 alias-eligible predictions, 36.3% have best-IoU in `[0.0,0.1)` against every Tricycle box in their image: background vehicles, or tricycles the dataset never labelled. Both groups survive correctly. A low percentage here is the rule working, not failing to fire.
- Hide `dup_gt` via the App sidebar's label-tag filter while reviewing.
- The pass is **idempotent** — verified by executing the cell twice against clones of `augmented_tricycle` and `door_detection_zqt59`: the second run reports 0 changes. It recomputes rather than accumulates, so raising `GT_MATCH_IOU` or deleting a ground_truth box un-tags predictions that are no longer redundant.
- Applies to datasets built from v31 on. Existing datasets get it by re-running `dupgt_code` against them — no rebuild needed.
- **Not addressed:** cross-class high-IoU pairs outside the alias set (272 cases) still appear as normal predictions. Promoting one stacks a wrong-class box on a correct one, so it remains a thing to watch for by eye.

---

## DEC-099: Pedestrian Lane Dropped from the Class Schema — Queued, Not Yet Executed

- **Date:** 2026-08-27
- **Status:** Accepted (execution queued)
- **Related:** DEC-042 (1,500 floor), DEC-087 (review-checklist policy). Resolves one of the three removal candidates the student had been tracking without committing; **Stairs and Elevator remain candidates and are explicitly NOT dropped.**

### Context

Pedestrian Lane entered `revised_pedestrian_obstacle`'s review already **below DEC-042's floor** — *"Per-Class Image Ceiling Set at 4,500 (Floor 1,500, 3:1 Ratio Invariant)"* — at 1,286 images against a 1,500 minimum.

Reviewing that source made it worse. 237 images were tagged `exclude`, of which **225 carry a Pedestrian Lane box**, cutting `revised_pedestrian_obstacle`'s contribution from 605 images to 380 and the class as a whole:

| | before | after |
|---|---|---|
| images | 1,286 | **1,061** (71% of floor) |
| boxes | 1,459 | 1,228 |

Remaining sources: `wtf_dwvgm` 475, `revised_pedestrian_obstacle` 380, `crosswalk_detector_lz3hc` 202, `cv_project_hovyc` 4. Closing the ~440-image gap to the floor would mean acquiring a new source, and the two largest existing contributors are both on the deprioritized list precisely because they feed only removal-candidate classes.

### Decision

**Drop `Pedestrian Lane` from the 16-class schema, taking it to 15.** Recorded now; execution deliberately deferred — nothing has been changed yet.

### Rationale

The exclusions were made on **quality** grounds, not to force this outcome. A class held above its floor by 225 images the student judged unusable was never really at 1,286; the count was measuring data that would have degraded the model. That argues for dropping the class rather than for keeping the images.

The student's stated reason for accepting the loss: high-quality Pedestrian Lane data is hard to source, and re-augmenting the existing set is out of scope for now.

### Alternatives Considered

- **Acquire another Pedestrian Lane source.** Rejected as out of scope; the student's judgement is that quality candidates are scarce, which the existing pool corroborates.
- **Keep the class below floor.** Rejected: at 1,061 it is 71% of the minimum, and DEC-042's floor is a training-viability threshold, not a target.
- **Keep the 225 excluded images.** Rejected: they were excluded on inspection, and propping a class up with known-bad data is the failure the floor exists to prevent.

### Consequences — Execution Plan (NOT yet done)

`Pedestrian Lane` is **id 14**, so only **`Bicycle` (15) re-indexes**, to 14. Every other class id is unchanged, which keeps the migration far smaller than a mid-schema removal would.

1. `config/classes.yaml` — remove the `pedestrian_lane` block; `config_loader.py`'s `CANONICAL_NAMES` drops to 15 (`nc: 16` -> `nc: 15`, regenerated by `generate_yaml.py`).
2. `config/datasets.yaml` — drop `pedestrian_lane` from `revised_pedestrian_obstacle`'s `canonical_classes` and its `native_class_filter` `crosswalk` entry.
3. **Two sources become empty and must be deactivated** — confirmed by scanning their converted labels, not assumed: `wtf_dwvgm` (540 boxes, all Pedestrian Lane) and `crosswalk_detector_lz3hc` (269 boxes, all Pedestrian Lane). `pedestrian_and_animal_crossing` also carries 2,158 but is already `benched`.
4. **Hand-review work must be migrated in place, never regenerated:** across all `labels_reviewed/`, **27 files contain Pedestrian Lane** and **34 contain Bicycle** and need re-indexing. Small, but irreplaceable — `labels/` is rebuilt from raw by `yolo_to_intermediate.py`, `labels_reviewed/` is not.
5. Derived layers need no migration, only a rebuild: `dataset/merged/` and `dataset/final/` are regenerated by the usual cascade (`cap_per_class.py` -> `merge.py` -> `dedup.py` -> `split.py` -> `generate_yaml.py`).

Scope measured across 212,261 label files: 6,915 contain Pedestrian Lane (8,168 boxes), 10,982 need re-indexing. All but the 27 + 34 reviewed files are regenerable.

**Not decided here:** whether `revised_pedestrian_obstacle`'s 237 exclusions should still be written back. They should — they were quality judgements about the images, and 12 of them carry Stairs boxes rather than Pedestrian Lane, so the exclusions retain meaning independently of this decision.

---

## DEC-100: Stairs, Elevator and Pedestrian Lane Dropped — Schema 16 -> 13, Executed

- **Date:** 2026-09-04
- **Status:** Accepted (executed)
- **Related:** Supersedes DEC-099, which queued the Pedestrian Lane drop alone. DEC-042 (floor 1,500), DEC-087 (review policy).

### Context

All three classes sat below DEC-042's floor — *"Per-Class Image Ceiling Set at 4,500 (Floor 1,500, 3:1 Ratio Invariant)"* — measured against reviewed labels with exclusions applied:

| class | images | shortfall |
|---|---|---|
| Stairs | 1,375 | 125 |
| Elevator | 1,351 | 149 |
| Pedestrian Lane | 1,099 | 401 |

Every other class cleared the floor comfortably; five sit over the 4,500 cap. Closing the gaps meant acquiring new sources for three classes whose existing sources the student had already deprioritised.

### Decision

Drop all three. Schema **16 -> 13**. Executed 2026-09-04, not queued.

### Rationale

The student accepted the trade explicitly, including that returning later costs a full retrain: a changed class count means a new detection head, not a fine-tune. Pedestrian Lane was already decided in DEC-099 after review of `revised_pedestrian_obstacle` cut it from 1,286 to 1,099 on quality grounds — a class propped up by images judged unusable was never really at its stated count.

### Consequences

**Re-index.** Only ids 0-4 survive unchanged; eight classes shift. `Shelf` 6->5, `Doors` 7->6, `Chairs` 8->7, `Tables` 9->8, `Tricycle` 10->9, `Potholes` 11->10, `Trash Bins` 12->11, `Bicycle` 15->12.

**Executed, in order** — promotion first, because migrating before promoting would copy un-migrated reviewed files over migrated ones:

1. `promote_reviews.py --all` — 4,592 files across 7 sources copied from `labels_reviewed/` to `labels/`. `cap_per_class.py` reads only `labels/` (cap_per_class.py:220), so until this ran no review was visible to the pipeline.
2. `drop_classes.py --drop Stairs Elevator "Pedestrian Lane"` — 53,206 of 124,045 files changed, 23,078 boxes removed, 16,529 files left with zero boxes. Stamp: `dataset/reports/class_migration_20260904_025230.json`.
3. `config/classes.yaml` — `nc` 16->13, `names`, `hailo_runtime_names` (14 incl. Background), and the three per-class blocks removed.
4. `scripts/utils/config_loader.py` — `CANONICAL_NAMES` and `EXPECTED_NC` -> 13.
5. `config/datasets.yaml` — four sources set `benched`, each 100% a dropped class and now contributing nothing: `elevator_awvus` (2,035 Elevator boxes), `stair_gaptw` (1,155 Stairs), `wtf_dwvgm` (540 Pedestrian Lane), `crosswalk_detector_lz3hc` (269 Pedestrian Lane).

**Verified:** a full scan of all 124,045 label files finds **no out-of-range class id**, and `load_classes()` validates at nc=13.

Empty files are deliberately not deleted — `cap_per_class.py` indexes by ids present, so a zero-box file is a candidate for nothing and never reaches `merged/`.

**Reversibility.** `dataset/backups/pre_class_drop_20260904_023304/` (4.1 GB) holds the complete 16-class state: config, reports, every `labels/` and `labels_reviewed/`, all referenced images including the `*_augmented_aside/` dirs, and all 18 FiftyOne review datasets. Verified self-contained — all 43,700 image references resolve inside it, `labels_reviewed/` byte-identical, and a FiftyOne round-trip import reproduced sample, box and tag counts exactly. `drop_classes.py` also wrote its own per-directory backups.

Restoring is a wholesale copy-back. There is **no inverse migration script**, so review work done under the 13-class schema would have to be reconciled with restored 16-class files by hand.

> **Correction added 2026-09-04 (DEC-105) — the backup is NOT self-sufficient.** The paragraph above says the backup holds "the complete 16-class state: config, ...". It holds the complete 16-class **data and config-file** state. It does **not** hold the **code** that defines the schema: `scripts/utils/config_loader.py` hardcodes `EXPECTED_NC` and `CANONICAL_NAMES`, and it is not in the backup. Restoring only the YAMLs makes `load_classes()` raise `classes.yaml 'nc' is 16, expected 13` immediately. See DEC-105 for the verified, complete revert procedure.

**Not yet run:** the cascade (`cap_per_class` -> `merge` -> `dedup` -> `split` -> `generate_yaml`). Everything under `dataset/merged/` and `dataset/final/` still reflects the 16-class pre-review state.

---

## DEC-101: `classes.yaml`'s Per-Class `id:` Fields Were Never Re-Indexed by DEC-100 — Silent Converter Corruption Averted

- **Date:** 2026-09-04
- **Status:** Accepted (executed)
- **Related:** DEC-100 (the incomplete migration), DEC-065 (which already documented `names:` as the authoritative field and the `classes:` block as separately-ordered metadata)

### Context

The first `cap_per_class.py` run after DEC-100 died immediately:

```
File "scripts/preprocess/cap_per_class.py", line 423, in run
    configured_cap = id_to_cap[class_id]
KeyError: 5
```

`config/classes.yaml` carries the schema **twice**: the authoritative `names:` map, and a parallel `classes:` block where each class repeats its own `id:` alongside its cap and provider documentation. DEC-100 updated `nc`, `names`, `hailo_runtime_names` and removed the three dropped blocks — but left the surviving blocks' `id:` fields on the **old 16-class numbering**. `names:` said Shelf was 5; the `shelf:` block still said `id: 6`. Bicycle still said `id: 15`, outside the valid range entirely.

`load_classes()` did not catch it because its validation checks `nc` against `EXPECTED_NC` and the length of `names` — never that the block ids agree with `names`.

### Decision

Re-index all eight stale `id:` fields to match `names:`: Shelf 6->5, Doors 7->6, Chairs 8->7, Tables 9->8, Tricycle 10->9, Potholes 11->10, Trash Bins 12->11, Bicycle 15->12. Ids 0-4 were already correct. Exactly the mapping DEC-100 recorded in prose but applied to only one of the two structures.

### Rationale

`names:` is authoritative (DEC-065 established this when `generate_yaml.py` had to choose between the two). The blocks are the copy that drifted, so the blocks are what gets corrected.

### Consequences

**The crash was the mild symptom.** Three converters read `entry["id"]` to build their canonical class-id map:

- `scripts/convert/yolo_to_intermediate.py:238` — every Roboflow source
- `scripts/acquire/acquire_exdark.py:163`
- `scripts/acquire/acquire_openimages.py:91`

Any re-conversion would have written **16-class ids into a 13-class dataset** with no error: Shelf boxes landing on Doors, Doors on Chairs, and Bicycle at an out-of-range 15. `cap_per_class.py` crashed loudly only because it happened to index a dict; the converters would have succeeded silently and produced a plausible-looking, wrong dataset.

This never fired because no converter had been re-run since DEC-100 — the labels on disk were migrated by `drop_classes.py` operating on label files directly, and DEC-100 verified no out-of-range id remained. Data on disk was never affected.

**Verified after the fix:** all 13 blocks agree with `names:`, `nc = 13`, block count 13, zero mismatches.

**Not done, deliberately:** no guard was added to `load_classes()` asserting block ids match `names:`. That is what would have caught this at import rather than mid-cascade, and it remains the obvious follow-up — left unimplemented because it is scope beyond the reported problem, pending the student's call.

**Generalisation worth keeping.** A schema stored in two places will desync, and the copy without validation is the one that rots. DEC-085 already found this exact shape once — `classes.yaml`'s per-class *provider* documentation had gone stale for 7 classes while the pipeline read `datasets.yaml` instead. That instance was harmless because nothing read it; this one was not, because three converters do.

---

## DEC-102: Post-Class-Drop Cascade Re-Run — Dedup Deliberately Not Re-Run, `split.py`'s Duplicate Over-Grouping Accepted (OPEN_QUESTIONS #13 Closed)

- **Date:** 2026-09-04
- **Status:** Accepted (executed)
- **Related:** DEC-100 (whose "Not yet run: the cascade" this completes), DEC-089 (the RunPod dedup result being preserved), DEC-090 (the false-positive review this rules on), DEC-101 (the bug that blocked step 1)

### Context

DEC-100 left everything under `dataset/merged/` and `dataset/final/` reflecting the 16-class, pre-review state. `docs/HANDOFF.md` prescribed a five-step cascade: `cap_per_class` -> `merge` -> `dedup` -> `split` -> `generate_yaml`, and asserted *"Nothing else is blocked on a decision."*

Two things were wrong with that.

**First, `docs/OPEN_QUESTIONS.md` #13 was genuinely open and gated step 4.** `near_duplicate_false_positives.json` — 3,829 pairs the student cleared by hand in DEC-090 — is read only by `build_dedup_keeplist.py`. Nothing under `scripts/build/` references it, so `split.py` was still treating every cleared pair as a duplicate.

**Second, step 3 was actively destructive.** `dedup.py` writes `dedup_report.json` unless `--limit` is passed (dedup.py:255). A bare local run would have replaced DEC-089's 66,907-image RunPod near-duplicate coverage with a 6,000-image stratified sample — a 91% coverage loss, irreversible without renting another pod.

### Decision

**Run four steps, not five. Skip `dedup.py` entirely.** `split.py:117-124` explicitly tolerates a report covering a superset pool, printing *"Expected when reusing a dedup run made against a larger pool... not a bug."* The existing report covers 67,109 images against a 36,875-image merged pool — exactly that case.

**OPEN_QUESTIONS #13: accept the over-grouping. No code change to `split.py`.**

### Rationale

The #13 decision was made on measurement, not preference. Duplicate groups chain transitively under `split.py`'s union-find, so the cleared pairs account for **3,844 of 12,715 edges (30.2%)**, welding 18,560 files into 5,868 clusters where subtraction would give 13,120 files in 4,255 clusters.

The student's stated concern was that subtracting false positives would delete pothole data, since potholes dominate them (`road_damage_detector` 977 + `pothole_voxrl` 630 + `pothole_detection` 227 = 1,834 of 3,829, 48%). **That premise does not hold:** `assign_splits()` uses duplicate groups only to union filenames so they move together, and every filename in the pool receives a split. Nothing is dropped under either option.

The real effect runs the other way — grouping *concentrates* a class rather than removing it. Measured per-class both ways:

| | val% | test% |
|---|---|---|
| Potholes, as-is | **17.7%** | 14.5% |
| Potholes, with subtraction | 15.0% | 15.0% |

Potholes is the most skewed class in the table and subtraction fixes precisely it. But as-is it still holds **472 val / 386 test** images — 2.7pp above target, not starved — and every other class sits within ±1.6pp. Editing `load_duplicate_groups()` means overriding an algorithm with a hand review on the one code path that prevents train/eval contamination; AGENTS.md ranks correctness above dataset quality, and 2.7pp on one class is a thin return for that risk.

**Cost accepted explicitly:** DEC-090's 3,829-pair review stays inert at split time. It retains its value as the record of why the 0.2 threshold was judged acceptable.

### Consequences

Executed in order, all verified against real files rather than script self-report:

1. `cap_per_class.py` — blocked by DEC-101, fixed, re-run. All 13 classes clear the 1,500 floor; **ratio invariant 2.64** (max Person 4,500 / min Trash Bins 1,702).
2. `merge.py` — **36,875 images** from 37,008 selected pairs, 133 removed by review `exclude` tags. Was 45,132 under the 16-class pre-review state.
3. `dedup.py` — **skipped, deliberately.** DEC-089's report preserved intact.
4. `split.py` — train 25,850 / val 5,527 / test 5,498 = **70.1/15.0/14.9**. `cross_split_duplicate_leakage: []`. 265 augmented-sibling groups and 8 cross-source groups held together. On-disk counts match the report exactly, images and labels paired 1:1 in all three splits.
5. `generate_yaml.py` — `dataset/final/data.yaml` at `nc: 13`.

**Verified against the real installed Ultralytics**, per AGENTS.md's standing instruction, using `check_det_dataset()` from an unrelated CWD: `nc=13`, 13 names in correct order, all three split paths resolved absolute and existing. This is DEC-066's lesson — the `path:` bug it caught would otherwise surface only on RunPod.

**Known cosmetic defect in `split_report.json`.** Its `duplicate_group_coverage_note` claims the near-duplicate check *"only cover[s] dedup_report.json's stratified sample."* Misleading: `split.py:131` computes `near_dup_full_scale = near_duplicate_sample_size >= images_checked`, which is `66,907 >= 67,109` = False. The gap exists because DEC-090's `dedup_extend_exact.py` grew `images_checked` to 67,109 while honestly freezing near-duplicate coverage at 66,907. The near-duplicate pass genuinely was full-scale over the pool it ran against. The flag feeds a report string only — no behaviour depends on it — so it was left alone rather than fixed blind.

**`docs/HANDOFF.md` gap, recorded for the next handoff author.** It states the schema is 13, reduced from 16, and gives the mechanical scale of the migration, but never names the three dropped classes and never gives the reason, deferring to DEC-095–100. A fresh session must infer the three from the benched-source list. The *what* was carried well; the *why* was not carried at all.

---

## DEC-103: `fiftyone_final_dataset.ipynb` Added — Whole-Dataset Browsing With `source` as a Real Field; `dataset/final/` Declared Read-Only

- **Date:** 2026-09-04
- **Status:** Accepted (executed)
- **Related:** DEC-102 (the cascade that produced the pool this browses), DEC-054 (notebook-per-stage convention), DEC-081 (`dataset.classes` required or the App blocks), DEC-087 (why sources are no longer single-class)

### Context

Student asked for a notebook to view the final dataset in its entirety, filterable by source. Checked before answering: no such thing existed.

`fiftyone_review_processed.ipynb` does accept `source_key = "final/train"`, but it falls short twice. It browses **one split at a time**, and its build cell never sets a `source` field — `sample["source"]` is assigned only in the mistakenness section (cell `27`), a separate cross-source mode. In `dataset/final/` the source survives purely as `merge.py`'s `<source>__` filename prefix, which the App sidebar cannot filter on.

### Decision

New notebook `notebooks/fiftyone_final_dataset.ipynb` (v1), created rather than extending the existing one — the student's explicit instruction, and independently the safer call: `fiftyone_review_processed.ipynb` is v40 with heavily interdependent cells and is routinely open in the student's IDE, where an edit would be overwritten by their buffer on save.

Loads every requested split into **one** dataset with `source`, `split` and `num_boxes` as top-level sample fields, all sidebar-filterable.

**`dataset/final/` is declared read-only.** The notebook has no write-back cell and must never get one.

### Rationale

`split.py` deletes and regenerates `dataset/final/` wholesale on every cascade run, so a box fixed there is destroyed by the next run with no warning. The only edit path that survives is upstream: `fiftyone_review_processed.ipynb` -> `labels_reviewed/` -> `promote_reviews.py` -> `labels/` -> cascade. Stating this in the notebook header is cheaper than discovering it after losing an afternoon's review — the same class of trap DEC-100 already had to warn about for `yolo_to_intermediate.py` silently deleting dlsu's hand-merged boxes.

### Consequences

Verified by executing every cell headlessly against the real 36,875-image pool, not by inspection:

- Loads in **~12s** at ~3.0K samples/s. Counts cross-check against `split_report.json` exactly (25,850 / 5,527 / 5,498) — a built-in guard that fails loudly if `dataset/final/` is not what `split.py` last wrote.
- Cross-checks `dataset/final/data.yaml` against `config/classes.yaml` and **raises** on disagreement. Given DEC-101 — where exactly this kind of two-copy schema drift went unnoticed — a viewer that silently labels boxes with stale names would be worse than no viewer.
- Sets `dataset.classes["ground_truth"]` (DEC-081), so the App does not block on its schema-import prompt.
- `persistent = False`: this holds no review work and is re-derived from disk in about a minute, so it must not accumulate alongside the 18 persistent review datasets.

**Immediately useful output** — the per-source class breakdown makes DEC-087's multi-class enrichment concrete and visible for the first time. `roboflow_pothole_voxrl` is not a pothole-only source (Potholes=665, but also Vehicle=140, Person=45, Motorcycle=14); `roboflow_cv_project_hovyc` carries Doors=1,256 plus nine other classes. Only `crowdhuman` (Person), `dataset_ninja_pothole_detection` and `dataset_ninja_road_damage_detector` (Potholes) remain genuinely single-class.

Registered in `notebooks/README.md`. Pre-existing gap noted but not fixed: `fiftyone_near_dup_inspection.ipynb` (DEC-090) is also absent from that index.

---

## DEC-104: Dataset Ninja Sources Measured for Unlabeled Objects — Reviewed Rather Than Dropped; `hide_duplicates` Would Have Hidden 73% of the Work

- **Date:** 2026-09-04
- **Status:** Accepted (review pending execution by the student)
- **Related:** DEC-085/086 (the floor risk that makes dropping expensive), DEC-087 (multi-class enrichment from review), DEC-090 (the false-positive review this depends on), DEC-102 (the cascade to re-run after)

### Context

Student's own observation while browsing the final dataset: *"a lot of the vehicles are not labeled"* in the Dataset Ninja sources. Both are Potholes-only and, per `docs/HANDOFF.md`, the only two active sources never reviewed.

Measured with COCO-pretrained `yolov8n` at conf >= 0.5 — any eligible detection is by definition unlabeled, since ground truth holds nothing but Potholes:

| source | images | >=1 unlabeled object | >=1 unlabeled vehicle | unlabeled boxes |
|---|---|---|---|---|
| `dataset_ninja_road_damage_detector` | 1,331 | **87.4%** | **55.1%** | 4,018 |
| `dataset_ninja_pothole_detection` | 665 | 23.3% | 19.5% | 378 |

`road_damage_detector` breaks down as Person 1,729 · Vehicle 1,188 · Motorcycle 1,077 · Bicycle 20 — roughly three unlabeled objects per image, with **Person the larger problem than Vehicle**.

The model has false positives, so the absolute numbers are soft. The 87.4% vs 23.3% gap between two similar pothole sources is not, and it has a control inside the project's own data: `roboflow_pothole_voxrl`, which *was* reviewed, carries Vehicle=140/Person=45 labels (DEC-087 enrichment). The unreviewed road-damage source carries zero on imagery where vehicles appear in 55% of frames.

### Decision

**Review both sources with the predictions overlay (DEC-079 workflow); do not drop them.** Student's call, option (a) of three offered.

### Rationale

An unlabeled car in a training image is not a neutral omission — it teaches the detector that cars are background, which is worse than the image being absent entirely. This is the structural cost of merging single-class datasets and the reason DEC-087's review pass added cross-class boxes at all.

Dropping was the obvious alternative and is not free. DEC-085 already established that *"Potholes' 2,661 total depends on `dataset_ninja_road_damage_detector` counting as active... without it, Potholes sits at 1,330, below the 1,500 floor."* Dropping the source drops the class, and DEC-100 established that re-adding a class later costs a full retrain rather than a fine-tune.

### Consequences

**A trap was found in the review setup and must not be re-encountered.** `fiftyone_review_processed.ipynb` defaults to `hide_duplicates = True`, which would have hidden:

| source | in pool | hidden | left to review |
|---|---|---|---|
| `road_damage_detector` | 1,331 | **977 (73.4%)** | 354 |
| `pothole_detection` | 665 | 227 (34.1%) | 438 |

**All 977 and all 227 are images the student had already cleared as false positives in DEC-090** — personally judged not to be duplicates. The default would have skipped 73% of exactly the images the review exists to fix, leaving them in training with unlabeled vehicles. **`hide_duplicates = False` is required for both sources**, and the general rule is that `hide_duplicates` is unsafe on any source whose flags were largely cleared as false positives.

Prerequisites verified on disk: both sources have 1,331/665 images and labels in `dataset/processed/`, both are fully present in the merged pool, and both have an empty `labels_reviewed/` — confirming neither has ever been through write-back.

**Expected downstream effect beyond Potholes.** Both sources become multi-class on promotion, changing candidate pools for Person, Vehicle and Motorcycle — precisely what `docs/OPEN_QUESTIONS.md` #16 anticipated. The cascade must be re-run after `promote_reviews.py --all`, still skipping `dedup.py` per DEC-102.

**Not measured, offered and not yet taken up:** the same unlabeled-object scan across all 12 sources, which would establish whether `road_damage_detector` is an outlier or whether other pools carry the same defect.

---

## DEC-105: The 16-Class Revert Procedure, Verified and Written Down — Backup Alone Is Insufficient

- **Date:** 2026-09-04
- **Status:** Accepted (procedure documented; **deliberately NOT executed**)
- **Related:** Corrects DEC-100's Reversibility section. DEC-101 (the block-id desync the backup predates)

### Context

Student asked whether the YAML config was included in `dataset/backups/pre_class_drop_20260904_023304/`, *"just in case that I want to go back to 16 classes."* Checked rather than trusted DEC-100's own claim — which had already proven incomplete once that day (DEC-101).

The YAMLs **are** there, and they are clean:

```
config/classes.yaml   nc: 16, 16 names, 16 per-class block ids -- all consistent
config/datasets.yaml  present (sources still active, pre-benching)
MANIFEST.json         class_count: 16
```

Worth recording: the backed-up `classes.yaml` is **more internally consistent than what DEC-100 left on disk**. Its 16 block ids all agree with its `names:` map. DEC-101's desync was introduced *by* the drop, not inherited from before it.

### The gap

`scripts/utils/config_loader.py` is **not in the backup**, and it hardcodes the schema:

```python
EXPECTED_NC: int = 13
CANONICAL_NAMES: list[str] = [...13 entries...]
```

Restoring the YAMLs alone fails immediately — `load_classes()` raises `classes.yaml 'nc' is 16, expected 13`. The backup covers data; git covers code; **neither alone is a revert.**

### Decision

Record the complete procedure now, while the facts are verified, rather than leaving it to be re-derived under pressure. **Not executed** — the student explicitly asked for it to be prepared and not run.

### The verified revert procedure

`b119543` is the commit that made the switch (`-EXPECTED_NC: int = 16` -> `+EXPECTED_NC: int = 13`, with `Stairs`/`Elevator`/`Pedestrian Lane` removed from `CANONICAL_NAMES`), confirmed by reading its diff. Its parent `2ac3cde` therefore holds the 16-class code.

```bash
B=dataset/backups/pre_class_drop_20260904_023304

# 1. Code -- from git, NOT the backup
git show 2ac3cde:scripts/utils/config_loader.py > scripts/utils/config_loader.py

# 2. Config -- from the backup
cp "$B/config/classes.yaml"  config/classes.yaml
cp "$B/config/datasets.yaml" config/datasets.yaml

# 3. Data -- ONLY the backup has this (dataset/processed/ is gitignored,
#    and labels_reviewed/ is irreplaceable hand work)
#    Wholesale copy-back of $B/processed/ over dataset/processed/

# 4. Re-run the cascade: cap_per_class -> merge -> split -> generate_yaml
#    (dedup still skipped, DEC-102)
```

### Consequences

- **The config is the cheap part; the review work is not.** DEC-100's warning stands and is the real cost: there is no inverse migration script, so any review done under the 13-class schema must be reconciled with restored 16-class files by hand. As of this entry that includes the DEC-102 cascade and any Dataset Ninja review (DEC-104).
- **Nine other files reference the dropped class names** — `cap_per_class.py`, `drop_classes.py`, `dedup_extend_exact.py`, `run_mistakenness.py`, `yolo_to_intermediate.py`, `config/datasets.yaml`, `docs/PROJECT.md`, `AGENTS.md`, `docs/OPEN_QUESTIONS.md`. None block a revert: they are either the drop tooling itself, doc strings (`run_mistakenness.py`'s `NO_COCO_ANALOG`), or documentation.
- **Generalisation.** A backup that captures data but not the code defining that data's schema is not a restore point. This project keeps the class schema in *three* places — `classes.yaml`'s `names:`, `classes.yaml`'s per-class `id:` blocks, and `config_loader.py`'s hardcoded constants. DEC-101 was two of them disagreeing; this is a backup covering only one. A guard asserting all three agree remains the unbuilt follow-up.

---

## DEC-106: Review Predictions Overlay Moved to `yolov8m` — Proxy Model Chosen on Measurement, Not Inherited Default

- **Date:** 2026-09-04
- **Status:** Accepted
- **Related:** DEC-060 (which introduced `yolov8n` as the proxy), DEC-074/079 (the overlay and auto-accept mechanism), DEC-104 (the review this was raised for)

### Context

`yolov8n` has been the COCO proxy model since DEC-060, never revisited. Student asked whether `yolov8m` or `yolov8l` would produce better labels, and stated the priority precisely: *"im especially keen towards ensuring that smaller instances in an image are caught."*

Two measurements were run rather than answering from published benchmarks.

**1. Recall on the target imagery** — 300 `dataset_ninja_road_damage_detector` images (seed 42, conf >= 0.5), which have no ground truth, so raw detection counts only:

| model | images with >=1 detection | boxes | Person | Motorcycle | Vehicle | speed |
|---|---|---|---|---|---|---|
| `yolov8n` | 265 (88.3%) | 956 | 419 | 241 | 290 | 11.5 img/s |
| `yolov8m` | 290 (96.7%) | 1,697 | 659 | **589** | 434 | 3.6 img/s |
| `yolov8l` | 288 (96.0%) | 1,817 | 727 | 615 | 451 | 1.8 img/s |

**2. Precision/recall against real ground truth.** The student's own remark — that `revised_pedestrian_obstacle` is *"as good as it gets in terms of labelling since I hand-reviewed it myself"* — makes it a trustworthy benchmark. 400 of its 2,088 hand-reviewed images, 2,015 eligible-class GT boxes, matched at same-class IoU >= 0.5:

| model | conf | precision | recall | F1 |
|---|---|---|---|---|
| `yolov8n` | 0.50 | **94.0%** | 48.6% | 64.1% |
| `yolov8n` | 0.25 | 76.0% | 70.2% | 73.0% (its peak) |
| `yolov8m` | 0.40 | 83.8% | 68.6% | **75.4%** (its peak) |
| `yolov8l` | 0.40 | 83.1% | 69.5% | **75.7%** (its peak) |

**Recall by object size at conf 0.5** (small < 0.33% of frame, medium < 3%, large >= 3% — COCO's convention mapped to relative area):

| model | small | medium | large |
|---|---|---|---|
| `yolov8n` | **12.9%** | 48.0% | 77.4% |
| `yolov8m` | **30.0%** | 63.7% | 84.4% |
| `yolov8l` | 32.0% | 65.1% | 86.3% |

### Decision

**Switch the review overlay to `yolov8m` with `AUTO_ACCEPT_CONFIDENCE = 0.7`.** `yolov8l` rejected.

### Rationale

**On small objects — the stated priority — `yolov8n` finds about one in eight. `yolov8m` finds nearly one in three**, a 133% improvement. `yolov8l` adds 2 points for double the runtime.

**The precision numbers invert the naive reading, and this is the substantive finding.** At conf 0.5 `yolov8n` scores 94.0% precision against `yolov8m`'s 88.4%, which looks like the smaller model is *more accurate*. It is not. Per-box precision rewards a model for declining to guess: `yolov8n` scores high because it only attempts large, easy objects and abstains on everything hard. F1 corrects for this — 73.0% peak for `n` against 75.4% for `m`.

**Model size and auto-accept threshold are independent knobs, and that resolves the student's actual goal of less hand-checking:**

| setting | auto-accepted | wrong | precision |
|---|---|---|---|
| `yolov8n` @ 0.60 | 823 | 20 | 97.6% |
| **`yolov8m` @ 0.70** | **908** | **41** | **95.5%** |
| `yolov8m` @ 0.80 | 587 | 12 | 98.0% |

`yolov8m` at 0.7 dominates `yolov8n` at 0.6 — 64 more correct boxes pre-tagged for 21 more errors to spot. And m's sub-threshold detections still render untagged in the App, so small distant objects are at least *visible* for manual acceptance. Under `yolov8n` they are not detected at all and can only be drawn by hand.

**Cost is not a constraint.** Full-source inference: `n` ~2 min, `m` ~6 min, `l` ~12 min for 1,331 images. DEC-079 already established the review cap was the student's time budget, never inference cost.

### Alternatives Considered

- **`yolov8l`.** Rejected on measurement: +2pp small-object recall and +0.3pp F1 over `m`, for 2x the runtime (299s vs 155s on 400 images). It also detected *fewer* images than `m` in the road-damage sample (288 vs 290).
- **Keep `yolov8n` for consistency with already-reviewed sources.** Rejected — an improvement is not a regression, and the sources most affected are re-runnable.

### Consequences

- Change is one string in `p0review3build`: `YOLO("yolov8n.pt")` -> `YOLO("yolov8m.pt")`. Left for the student to apply, since the notebook is open in their IDE and an edit from here would be overwritten on save.
- **The 7-class ceiling does not move.** COCO has 80 fixed classes regardless of model size, so `COCO_CROSSWALK` still covers only Person, Vehicle, Motorcycle, Bicycle, Animals, Chairs, Tables. Potholes, Doors, Pole, Shelf, Tricycle and Trash Bins get nothing from any model size — a bigger model finds *more of the same 7*, not new ones.
- **Measured precision is a lower bound.** A "false positive" here is any prediction with no matching GT box, which includes real objects the reviewer did not label. This most affects the *small* band, since tiny distant people are exactly what a human reasonably skips — so `yolov8m`'s 83.3% small-object precision is likely understated.
- **Student intends to re-run `dlsu_d_vehicle_type_detection`** with the new model (its own review was a quick pass). Well-founded: it is dense with Vehicle/Motorcycle/Person, the three classes `m` improves most. `revised_pedestrian_obstacle` is explicitly **not** being re-reviewed — it was used as the measuring stick and the student considers its labelling finished.
- **Not changed:** `run_mistakenness.py:157` and `final_merge_curation.py:91` still hardcode `yolov8n.pt` for Stage 5.5/5.7 mistakenness scoring. Those runs are complete and were not in scope here; if either is re-run, this decision is an argument for revisiting them too.
- **Benchmark scope, stated so it is not over-read:** 400 images / 2,015 boxes from a single source, one IoU threshold (0.5). `revised_pedestrian_obstacle` is street imagery similar to the road-damage target, so it should transfer, but this is not a general result.

---

## DEC-107: The Class Schema Lives in Five Places, Only One of Which Is Authoritative — Map of the Copies and How Each Fails

- **Date:** 2026-09-04
- **Status:** Accepted (finding; no code changed)
- **Related:** DEC-101 (copy 2 desynced), DEC-100 (the migration that desynced it), DEC-081 (which set copy 4), DEC-065 (which first identified `names:` as authoritative)

### Context

Three separate incidents in one session all traced to the same root shape: the class schema is duplicated across the project, and only one copy is validated. The fifth copy was identified by the student, from direct experience rather than from the code — *"every time i had to configure it myself... it can be the case that the schema in fiftyone is completely separate from the yaml configs."* That is correct, and verified below.

### The five copies

| # | Location | Kind | Validated? | How it has failed |
|---|---|---|---|---|
| 1 | `config/classes.yaml` `names:` | YAML | **authoritative** — `load_classes()` checks length vs `EXPECTED_NC` | — |
| 2 | `config/classes.yaml` per-class `id:` blocks | YAML | **no** | DEC-101: left on 16-class ids by DEC-100. Crashed `cap_per_class.py`; would have made three converters silently write 16-class ids into a 13-class dataset |
| 3 | `config_loader.py` `CANONICAL_NAMES` / `EXPECTED_NC` | Python constants | self-referential only | Served stale from a **bytecode cache** (`config_loader.cpython-314.pyc`, 2026-08-26) to the Python 3.14 notebook kernel while the 3.11 shell saw the correct 13 — rendered 2,657 Potholes as `Tricycle` |
| 4 | `dataset.classes` per FiftyOne dataset | MongoDB, per dataset | **no** | A snapshot taken at build time. Review datasets built before DEC-100 still hold 16 entries; `review_roboflow_pothole_vhmow` still holds `Escalator`, pre-DEC-083 |
| 5 | FiftyOne App annotation schema JSON | App-side state | **no** | Hand-pasted every session. Nothing derives it from 1–4 and nothing warns on drift |

Copy 6 exists but is safe by construction: `dataset/final/data.yaml` is generated from copy 1 by `generate_yaml.py`. Copy 7, `AGENTS.md`/`docs/PROJECT.md`, was stale for weeks and is what the student copied the 16-class annotation JSON from; both corrected 2026-09-04.

### Copy 5 verified, not assumed

Checked directly against FiftyOne 1.20.0 on a live review dataset:

```
dataset.classes            -> {'ground_truth': 13, 'predictions': 13}   correctly set
dataset.app_config         -> active_fields, color_scheme, media_fields,
                              sidebar_groups, plugins ...  no annotation schema
dataset.ontology           -> attribute does not exist
dataset.annotation_schema  -> attribute does not exist
```

`dataset.classes` is populated and correct, and **there is nowhere on the dataset that stores the annotate-tab schema**. It is App-side, which is exactly why it must be re-entered each session. DEC-081 fixed the sidebar/rendering path; it did not and cannot fix the annotation dropdown.

### Consequences

**Copy 5 cannot corrupt stored data by itself.** `Detection.label` is a free string, so the dropdown only matters at the moment a class is picked. This is also what rules copy 5 out as the cause of the Potholes-as-Tricycle incident — the labels were already wrong in MongoDB, written by copy 3. The narrow risk is picking a class from a stale dropdown: a stored `Stairs` would then hit write-back's guard.

**The mitigation for copy 5 is to generate rather than hand-keep it.** A snippet reading `get_canonical_names()` emits the JSON, so the pasted copy is always derived from copy 1, and its output doubles as a kernel-staleness check — a 16-entry result means copy 3 is stale before any review work begins.

**The `load_classes()` guard is now clearly worth building.** Proposed after DEC-101 and deferred as out of scope; two further incidents since. It would cover copies 1↔2 directly and, by failing loudly at import, would catch a stale copy 3 as well. Copies 4 and 5 need separate handling — 4 is fixed by rebuilding a dataset, 5 by generating the JSON.

**Generalisation:** every unvalidated copy of a schema is a place it can rot, and the copies that rot are the ones nothing checks. This project added copies for good local reasons — caps and provider docs alongside ids (2), import-time constants (3), App metadata (4) — and each was correct when written. The failure is not any single copy; it is that only copy 1 is ever verified.

---

## DEC-108: Pedestrian-Lane Sightings Recorded as Sample Tags, Exported to Disk — Boxes Deliberately Not Drawn

- **Date:** 2026-09-04
- **Status:** Accepted (executed — `scripts/preprocess/export_sample_tags.py` built and tested)
- **Related:** DEC-100 (dropped the class), DEC-105 (revert procedure), DEC-107 (why the App schema made this idea plausible), DEC-078 (the `exclude` tag precedent this mirrors)

### Context

While reviewing the Dataset Ninja sources the student noticed pedestrian lanes in the imagery and asked whether a class could be added to the FiftyOne schema **for those sources only**. It cannot — `data.yaml` carries one global `names:` list and YOLO applies it to every image.

But the underlying instinct was sound, and rested on DEC-107's finding: since the App's annotation schema is independent of the YAML, a class absent from the pipeline can still be offered in the dropdown. The stated goal was specific — *"so that if i were to come back and decided to re-include pedestrian lane, i wont have to hunt for those images that i already checked that has a pedestrian lane."*

**Measuring first changed how much this matters.** Pedestrian Lane was dropped at 1,099 images, but that count excluded `pedestrian_and_animal_crossing`, benched by DEC-083 as *superseded*, not failed. De-augmented totals across every source that carries the class:

| source | files | distinct photos | ratio |
|---|---|---|---|
| `pedestrian_and_animal_crossing` | 2,158 | 365 | 5.91x |
| `wtf_dwvgm` | 475 | 475 | 1.00x |
| `revised_pedestrian_obstacle` | 605 | 386 | 1.57x |
| `crosswalk_detector_lz3hc` | 202 | 202 | 1.00x |
| `cv_project_hovyc` | 4 | 4 | 1.00x |
| **total** | | **1,432** | |

**68 distinct images short of DEC-042's 1,500 floor** — not the 401 DEC-100 recorded, which was measured without the benched source. That is close enough that sightings found during an unrelated review could decide it.

### Decision

**Record sightings as a `has_pedestrian_lane` SAMPLE tag. Do not draw boxes.** New script `scripts/preprocess/export_sample_tags.py` exports the tags to `dataset/reports/<source>_tagged_<tag>.json`.

### Rationale

Drawing boxes was the obvious approach and is blocked by a deliberate gate. Write-back (cell `2343228f`) does not skip an unrecognised label — it **raises**:

```python
if det.label not in CANONICAL_NAMES:
    raise ValueError(f"{label_filename}: detection has label {det.label!r}, not one of the 16 canonical classes ...")
```

A single Pedestrian Lane box would abort the entire write-back, making the source unreviewable. Loud rather than silent, but fatal to the workflow.

Sample tags avoid this completely: write-back reads exactly one sample tag, `exclude` (verified by reading every tag reference in that cell). Every other tag is inert with respect to the pipeline.

**The tags are exported to disk rather than left in MongoDB, and that is the substantive part of this decision.** Tags live only in the live dataset until something writes them out. Rebuilding a review dataset drops them silently, with no error and nothing to recover from — and review datasets in this project have been destroyed twice (DEC-096) plus rebuilt repeatedly during this session alone. `labels_reviewed/` protects box edits; nothing protected tags before this script.

### Alternatives Considered

- **Sidecar file for real boxes** — relax the write-back guard to divert unknown labels into `labels_deferred/`. Preserves geometry and is the durable answer if the class is definitely returning. Deferred: it revises a deliberate hard gate for a class that may never come back, and the student's stated need is finding the images again, not the geometry.
- **Draw boxes, back up, delete them before write-back.** Rejected: one forgotten step yields either a crash or lost work.
- **Add the class back now.** Premature at 68 images short and with the largest contributor unreviewed.

### Consequences

- Script **accumulates by default**, mirroring `<source>_excluded.json` (DEC-078), so a half-finished session cannot erase marks recorded earlier. `--prune` opts into dropping entries whose tag was cleared in the App. Both paths tested for real: after untagging one of three samples, the default run held at 3 and `--prune` correctly dropped to 2. `--list` shows tag counts across every dataset without writing.
- Calls `dataset.reload()` before reading (DEC-097). Omitting it would return the kernel's stale snapshot and miss tags applied minutes earlier in the App — self-defeating for a script whose whole purpose is capturing them.
- **No pipeline stage reads these files.** They are a human record, not labels. Reinstating the class would still require the DEC-105-style migration, plus recovering the deleted boxes from the 16-class backup.
- **Appending Pedestrian Lane as id 13 would re-index nothing** — every existing class keeps its id and every label file stays valid, unlike the removal which shifted eight classes and rewrote 53,206 files. The migration back is far cheaper than the migration out; the cost is a full retrain (new detection head) and reviewing the 365-image `pedestrian_and_animal_crossing`, whose provenance DEC-053 flagged (real class name `==============================`, *"almost certainly a garbage name from a mislabeled annotation batch"*).

**Two unrelated findings surfaced by `--list`, recorded but not acted on:**

1. **A typo'd exclusion tag.** `review_roboflow_cv_project_hovyc` holds `{'exclude': 71, 'excluded': 1}` and `verify_roboflow_dlsu_d_vehicle_type_detection_reviewed` holds `{'excluded': 1}`. Write-back matches `exclude` exactly, so the `excluded` sample was **never excluded** despite being marked. One image, but it is a silent miss of exactly the kind DEC-078 exists to prevent.
2. **Large model weights are untracked and not ignored.** `notebooks/yolov8m.pt` (50M) and `notebooks/yolov8x.pt` (131M) now sit untracked, while `notebooks/yolov8n.pt` is **tracked in git**. `.gitignore` covers `models/weights/` but not `notebooks/*.pt`, so a careless `git add -A` would commit 181MB of downloadable weights.

---

## DEC-109: Cross-Class Alias Rule Made Per-Image and Containment-Based; 112 Wrongly Stacked Boxes Removed from dlsu

- **Date:** 2026-09-05
- **Status:** Accepted (executed)
- **Related:** Revises DEC-098's alias rule. DEC-106 (the bigger proxy model that would have amplified this), DEC-087 (dlsu as priority source for Vehicle/Motorcycle)

### Context

Student reported, from their own review: *"a lot of tricycle ground truths have vehicles/motorcycle autolabels that got accepted, which should really not be the case."*

Root cause is DEC-098's own documented blind spot. Its alias rule fires only on a source whose ground truth contains **no** box of the alias classes:

```python
alias_ok = {composite: set(aliases) for composite, aliases in GT_CLASS_ALIASES.items()
            if composite in present and not (set(aliases) & present)}
```

`dlsu_d_vehicle_type_detection` labels Vehicle, Motorcycle **and** Tricycle, so the rule switched itself off entirely — on the one source where tricycles are dense. A Vehicle prediction boxing a *piece* of a tricycle then matched neither branch: not same-class, and the alias branch was disabled. It survived, was auto-accepted, and write-back promoted it on top of the correct Tricycle box.

**Measured damage:** DEC-098 recorded 2 such boxes out of 13,434 before the review pass. After one pass: **76 at IoU >= 0.5 across 78 files** — a 38x increase.

### Decision

**The alias rule is now per-image and containment-based.** The per-source self-disable is removed; `ALIAS_CONTAINMENT = 0.8` added. Notebook is **v41**.

### Rationale

**Why the old guard existed, and why removing it is nonetheless safe.** dlsu labels **576 genuine Vehicle/Motorcycle boxes overlapping a Tricycle, 193 of them almost wholly inside one**. Suppressing on "any overlap" really would have destroyed real data, exactly as DEC-098 argued.

But those genuine boxes are **ground truth**. A prediction landing on one matches it same-class at `GT_MATCH_IOU` and is already tagged by the same-class branch, which runs first. Nothing is lost — the GT box stays and only the redundant prediction is suppressed, which is correct. The boxes that got through were precisely those with **no same-class GT to match**: a part of a tricycle the source never labelled.

**Containment, not IoU.** A tricycle's motorcycle half scores low IoU against the whole tricycle (the union is dominated by the tricycle) while sitting almost entirely inside it. Measured on dlsu's 107 wrongly-added boxes: **98 at containment >= 0.8, 90 at >= 0.9**, with 5 below 0.3.

**0.8 is chosen to protect the case the student cares about.** The 5 low-containment boxes are plausibly real vehicles merely clipping a tricycle's box — DEC-098's author objected to suppressing exactly those, and that objection still holds. 0.8 takes 91.6% of the bad boxes and leaves that tail alone.

Verified on four hand-constructed cases: motorcycle-half (containment 1.00, suppressed), sidecar cabin (1.00, suppressed), distant car clipping a corner (0.08, kept), separate motorcycle alongside (0.22, kept).

### The cleanup, and a wrong first attempt worth recording

`scripts/preprocess/strip_alias_stacked_boxes.py` removes what the old rule already wrote. A box goes only if it is a part class, sits >= 0.8 inside a composite box, **and is absent from a pre-review reference snapshot**.

**The first version used "has no same-class twin" as the safety condition and a dry run exposed it as wrong — it flagged 454 boxes where only ~107 were ever review-added.** A genuine part box the source labelled is usually the only one of its class in that image, so it has no twin either. The condition that protects a *prediction* (it matches an existing GT box) has no equivalent when inspecting GT boxes directly. Only a snapshot taken before the review pass separates them. `dataset/backups/pre_class_drop_20260904_023304/.../labels` serves: 9,104 files, **zero Person boxes**, so it provably predates every promotion. Its 16-class ids are compared by name, never by raw id.

Had that first version run, it would have destroyed the 193 genuine boxes this decision exists to protect. The dry run is what caught it.

### Consequences

Executed against `roboflow_dlsu_d_vehicle_type_detection`, both `labels/` and `labels_reviewed/` (byte-identical, promotion already applied), each backed up to `*_bak_alias_20260905_171115/` first:

- **112 boxes removed across 108 files** — Vehicle 32, Motorcycle 69, Bicycle 11. The Vehicle count matches the independent IoU-based measurement exactly.
- **Stacked boxes at IoU >= 0.5: 76 -> 8.** The residual 8 sit below the containment threshold, i.e. genuinely ambiguous rather than clearly parts.
- **741 part boxes still overlap a Tricycle** — the genuine population is preserved, as required.
- Class totals now: Vehicle 8,151 · Motorcycle 3,517 · Tricycle 1,827 · Person 2,011 · Bicycle 90 · Chairs 5 · Animals 4.

**`Person` is deliberately not an alias of `Tricycle`** and never has been. Riders are real objects and must keep being detected — DEC-098 established this on roitrikee, where 867 Person predictions are riders. The 2 Person boxes overlapping a Tricycle in dlsu were left untouched.

**Not re-run:** the cascade. `dataset/merged/` and `dataset/final/` still contain the 112 removed boxes and will until the next `cap_per_class` -> `merge` -> `split` -> `generate_yaml` pass.

**Applies to other sources on rebuild.** `roitrikee` and `augmented_tricycle` label only Tricycle, so the old rule was already active there and the new rule changes little; their alias suppressions now use containment rather than any-overlap, which is strictly more conservative.

---

## DEC-110: Loose Ground-Truth Boxes Let Redundant Predictions Through — Suppressed by Containment With an Area-Ratio Cap; Author Boxes Always Kept

- **Date:** 2026-09-05
- **Status:** Accepted (executed)
- **Related:** DEC-098 (`GT_MATCH_IOU`), DEC-109 (same containment technique, alias case), DEC-106 (the stronger model about to amplify this)

### Context

Student, from inspecting dlsu: *"sometimes its the case that the ground truth made by the authors of this dataset is too loose, and because of that the auto accept bounding boxes (which is sometimes actually better than the hand drawn boxes) still gets accepted."*

Exactly right. A loose author box scores **under** `GT_MATCH_IOU` against a tight prediction of the same object, so `dup_gt` never fires, the prediction is auto-accepted, and write-back promotes it — leaving two boxes on one object.

Measured against the pre-review snapshot (which separates author boxes from promoted ones): **191 cases, 8% of dlsu's 2,389 review-added boxes.** 185 of 188 host boxes contain exactly one added box, so multi-object confusion is rare. Area ratio of author box to prediction:

| ratio | count |
|---|---|
| 1.5–2x | 1 |
| **2–3x** | **132** |
| 3–5x | 19 |
| 5–10x | 6 |
| **>10x** | **33** |

Two clusters with a valley at 5–10x. The >10x group is not this phenomenon at all — a box 10–445x larger than the one inside it is a **different object** (a distant car within a truck's box) and must keep being detected.

### Decision

Extend the SAME-class branch of `dup_gt`: a prediction is redundant if it is **>= 0.8 contained** in a same-class ground_truth box **and** that box is **< 5x its area**. `LOOSE_GT_CONTAINMENT = 0.8`, `LOOSE_GT_MAX_RATIO = 5.0`. Notebook **v42**.

**The prediction is suppressed and the author's box kept.**

### Rationale

**Why not replace the author box with the better prediction.** A 2–3x ratio is equally consistent with two opposite situations, and no geometry separates them:

- the author drew loosely around a fully visible car — the prediction is better; or
- the car is **partially occluded**, the author boxed its full extent (standard YOLO convention), and the model boxed only the **visible part** — the author is better.

A visible half is roughly 40% of a full box, i.e. ~2.5x — dead centre of the observed cluster. So the very tightness that makes the cluster look clean also makes it ambiguous. Automatic replacement would silently truncate occluded vehicles, which is precisely the *"risk of actual value bounding boxes being deleted"* the student asked to be weighed. Defaulting to the human-drawn box is wrong only in the cheaper direction.

**The cap sits in the measured valley**, the same method DEC-098 used for `GT_MATCH_IOU` and DEC-109 for containment.

**Worth doing at all?** 191 pairs in ~15,600 boxes is 1.2%, and doing nothing was a defensible option. It was taken because a ~2-hour yolov8x pass over 6,099 images is about to generate thousands more auto-accepts, so the rate matters going forward more than the existing stock does.

### Consequences

`strip_alias_stacked_boxes.py` gained `--rule {alias,loose}` (default `alias`, so DEC-109's documented behaviour is unchanged). Run against dlsu, both dirs backed up to `*_bak_loose_20260905_181835/`:

- **152 boxes removed across 139 files** — Motorcycle 121, Vehicle 29, Bicycle 2. Exactly the 2–5x cluster.
- **Loose-GT duplicates: 152 -> 0.** Different-object cases (>5x): **37 preserved**, untouched by design.
- Class totals now: Vehicle 8,122 · Motorcycle 3,396 · Tricycle 1,827 · Person 2,011 · Bicycle 88 · Chairs 5 · Animals 4.

**Verified that no author box was deleted.** 36 reference boxes are absent from the current labels — but the count is **identical (36/13,252) before the alias cleanup, after it, and after this one**, so both scripts removed zero author boxes. Those 36 are the student's own review edits (deletions and box moves), which is what review is for. Worth recording because the raw number reads as data loss until staged that way; the check that settles it is comparing the *same* metric across the script's own backups rather than against the live state alone.

**Not re-run:** the cascade. `dataset/merged/` and `dataset/final/` still contain both DEC-109's 112 and this decision's 152 boxes.

---

## DEC-111: Area-Ratio Cap Added to the Alias Rule — Distant Objects Inside a Composite Box Are Not Its Parts

- **Date:** 2026-09-05
- **Status:** Accepted (executed)
- **Related:** Completes DEC-109, which introduced the alias containment rule without a size bound. DEC-110 (the same cap on the same-class branch), DEC-098 (whose objection this restores)

### Context

DEC-109 made the alias rule per-image and containment-based, but gave it **no bound on relative size** — unlike DEC-110's same-class branch, which caps at 5x. Auditing the 112 boxes DEC-109 removed from dlsu exposed the gap:

| Tricycle ÷ removed-box area | count |
|---|---|
| **<2x** | **70** |
| 2–3x | 11 |
| 3–5x | 14 |
| 5–10x | 7 |
| **>10x** | **10** |

median 1.3x, max **472x**

The 70 below 2x are the model calling a whole tricycle a "vehicle" — correctly suppressed. Real parts (motorcycle half, sidecar cabin) land at 2–5x. But **a box 472x smaller than the one containing it is not a part of that object**; it is a separate thing that happens to fall inside its bounding box, typically a distant vehicle.

That is precisely what DEC-098 declined to discard: *"a legitimately distant car in a tricycle photo would be suppressed, adding to the manual burden rather than reducing it"* — the student's own objection, restated in this session as the expensive case to get wrong.

### Decision

Add `ALIAS_MAX_RATIO = 10.0` to the alias branch, mirroring `LOOSE_GT_MAX_RATIO` on the same-class branch. Notebook **v43**.

### Rationale

**10x rather than 5x, deliberately.** A part is inherently larger relative to its composite than a loose box is to the object it wraps. Every measured real part sits under 5x, so 10x costs nothing genuine while clearly separating the distant-object cases, whose minimum is 10.9x. The gap between 5x and 10.9x in the observed data is empty of real parts.

**Direction of the change, since it inverts easily:** the cap makes the rule *more conservative*. It suppresses fewer predictions, and the ones it stops suppressing are exactly the small distant vehicles. Without it those were tagged `dup_gt`, had `accept` withdrawn, and were lost.

### Consequences

New `scripts/preprocess/restore_alias_overcut.py` repairs what the uncapped run deleted. It restores a box only when it was present in the pre-cleanup backup, absent now, is a part class, sits >= 0.8 inside a composite box, **and** that box is >= 10x its area — reconstructing exactly why the uncapped rule fired, so nothing removed for another reason (notably DEC-110's loose-GT cleanup, which ran afterwards) is disturbed. Lines are appended; existing ones are never rewritten.

Run against dlsu, both dirs backed up to `*_bak_prerestore_20260905_183941/`:

- **10 boxes restored across 8 files** — Vehicle 5, Motorcycle 3, Bicycle 2. Ratios 10.9x to 472.4x, median 44.6x.
- Final state: review-added boxes still suppressible under the capped rule **0**; distant objects kept **10**; dlsu's own **63** author part-boxes untouched throughout.
- Class totals: Vehicle 8,127 · Motorcycle 3,399 · Tricycle 1,827 · Person 2,011 · Bicycle 90 · Chairs 5 · Animals 4. No malformed or out-of-range lines.

**A verification lesson worth keeping, repeated from DEC-110.** The first check of this cleanup reported "63 still suppressible", which reads as a failure. It was counting dlsu's own author part-boxes alongside promoted ones. Any check on this source must separate author boxes from review-added ones using the pre-review reference; a metric that lumps them together will keep producing false alarms. Same shape as DEC-110's "36 author boxes missing", which turned out to be the student's own review edits.

**Net effect of DEC-109 through DEC-111 on dlsu:** 254 wrongly stacked boxes removed (112 alias + 152 loose, minus 10 restored), zero author boxes lost, and three rules that now all bound containment by relative size.

**Not re-run:** the cascade. `dataset/merged/` and `dataset/final/` still reflect the pre-cleanup state.

---

## Template for Future Decisions

```markdown
## DEC-XXX: [Decision Title]

- **Date:** YYYY-MM-DD
- **Status:** Accepted | Superseded | Deprecated

### Context
[What prompted this decision]

### Decision
[What was decided]

### Rationale
[Why this option was chosen]

### Alternatives Considered
[What else was evaluated and why it was rejected]

### Consequences
[Expected impact, tradeoffs, and follow-up actions]
```

---

## DEC-112: Post-dlsu-Review Cascade Re-Run — DEC-109/110/111 Validated On Real Data, `dataset/final/` Rebuilt At 36,877 Images

- **Date:** 2026-09-05
- **Status:** Accepted (executed)
- **Related:** DEC-102 (the identical cascade this repeats, whose skip-dedup ruling still holds), DEC-109/110/111 (the three suppression rules executing against real data for the first time), DEC-106 (the yolov8x proxy model), DEC-089 (the RunPod dedup coverage still being preserved)

### Context

The `roboflow_dlsu_d_vehicle_type_detection` review completed: yolov8x inference over 6,099 images produced 16,985 predictions, of which 12,261 cleared `AUTO_ACCEPT_CONFIDENCE = 0.5`. This was the first execution of DEC-109's containment-based alias rule, DEC-110's loose-ground-truth rule, and DEC-111's `ALIAS_MAX_RATIO` cap against real data — all three had been written and reasoned about but never run.

### What The Rules Actually Did

**12,637 of 16,985 predictions were tagged `dup_gt`**, dropping the promotion set from 12,261 to **2,096**. Without DEC-109/110/111, all 12,261 would have been promoted — the rules suppressed **10,165 redundant or wrongly-stacked boxes**, roughly 83% of what the confidence threshold alone would have accepted. This is the largest single correction any review rule has made in this project.

**DEC-111's area-ratio cap is confirmed load-bearing, not theoretical.** 41 promoted part-class boxes (Vehicle/Motorcycle/Bicycle) sit >= 0.8 inside a Tricycle box. Their area ratios run **min 5.1x, median 17.1x, max 330.4x** — 39 of 41 sit above the 10x cap, meaning they are distant vehicles inside a nearer tricycle's bounding box, exactly the case DEC-098's author refused to discard. An uncapped rule would have deleted all 41.

**Two residual sub-cap cases found, diagnosed, and deliberately not fixed.** Both surviving boxes below the 10x cap (5.1x and 8.4x) sit inside a Tricycle box that was **absent from pre-review `labels/`** — hand-drawn in the App during the review itself. `dup_gt` is computed once at dataset-build time, against ground truth as it stood before any editing, so a box drawn later cannot retroactively suppress a prediction. This is an ordering property of the design, not a rule defect. At 2 boxes in 2,096 promotions it does not justify re-running suppression after every edit.

### Write-Back Verification

Checked by reconciliation rather than script self-report, in both directions:

- FiftyOne `ground_truth`: **17,262 boxes**; on-disk `labels_reviewed/` for the same 6,099 files: **17,262 boxes**. Exact match.
- Pre-review `labels/` for those files held 15,169 boxes; net **+2,093** against 2,096 promotions, 4 hand-deletions across 3 files, and 1 hand-drawn addition.

An initial comparison of raw directory totals (`labels` 20,634 vs `labels_reviewed` 20,655, apparently only +21) was **misleading and should not be repeated**: `labels_reviewed/` accumulates across sessions and covered 7,880 files spanning several review scopes, against `labels/`'s 9,104. Only a stem-matched comparison scoped to the files a given write-back touched is meaningful.

### Cascade Result

`promote_reviews.py --all` promoted **3,094 files across 9 sources** — dlsu 1,605, plus two Dataset Ninja sources (`road_damage_detector` 1,304, `pothole_detection` 185) whose completed reviews had been sitting unpromoted from earlier sessions. Four steps then ran, `dedup.py` skipped again for DEC-102's unchanged reason.

| stage | result |
|---|---|
| `cap_per_class.py` | 13/13 classes clear the 1,500 floor; ratio invariant **2.64** (Person 4,500 / Trash Bins 1,702) |
| `merge.py` | **36,877 images** from 37,012 selected pairs, 135 removed by `exclude` tags |
| `dedup.py` | **skipped** — DEC-089's 66,907-image coverage preserved |
| `split.py` | train 25,606 / val 5,701 / test 5,570 = **69.4/15.5/15.1**; `cross_split_duplicate_leakage: []` |
| `generate_yaml.py` | `dataset/final/data.yaml` at `nc: 13` |

Verified independently of script self-report: on-disk image/label counts pair 1:1 in all three splits and match `split_report.json` exactly; every class id across all 122,264 final boxes falls in 0–12 with no strays; per-class box totals match `merge_report.json` exactly. `check_det_dataset()` from `/tmp` resolves `nc=13`, correct name order, and all three split paths absolute and existing — DEC-066's check, which matters because training runs on RunPod, not this machine.

### Consequences

`dataset/final/` is now training-ready and reflects every completed review. The pool grew by 2 images against DEC-102 (36,875 -> 36,877) but its **labels are materially different**: dlsu alone gained 2,093 net boxes, and the 10,165 suppressed duplicates are boxes DEC-102's pool would have carried had the review run under the old rules.

Not addressed here, and still open: `roboflow_cv_project_hovyc` (297 files), `door_detection_zqt59` (47), `revised_pedestrian_obstacle` (64) and `roitrikee` (39) have images in `labels/` with no `labels_reviewed/` entry — reviews narrower than their source, left alone by design (`promote_reviews.py` never deletes unreviewed files).

---

## DEC-113: FiftyOne's Sidebar Silently Truncates Unindexed Filter Values — Indexes Now Created At Build Time

- **Date:** 2026-09-05
- **Status:** Accepted (fixed)
- **Related:** DEC-103 (the notebook this fixes), DEC-112 (the pool being browsed), DEC-081 (`dataset.classes`, the *other* App-side setup step this notebook already does)

### Context

Browsing `dataset/final/` in `fiftyone_final_dataset.ipynb`, the App sidebar offered only **12 classes and 5 sources**. The pool genuinely holds **13 classes and 12 sources**. An earlier session hit the same thing and reported "11 classes / no roboflow" — investigated at the time, wrongly attributed to a stale App server, and left open. It is the same cause.

### Cause

FiftyOne 1.20 resolves a sidebar filter's value list with a **bounded scan when the field has no database index**, and returns a partial list rather than failing. The only signal is a small **"Incomplete search. create an index"** note under the dropdown. `final_dataset_browse` carried only the four automatic indexes (`id`, `filepath`, `created_at`, `last_modified_at`) — nothing on `source` or `ground_truth.detections.label`.

Verified the data was never at fault: `count_values("ground_truth.detections.label")` returns 13, `count_values("source")` returns 12, and all 122,264 label ids in `dataset/final/` fall in 0–12.

### Why This Is Worth An Entry

**What gets dropped is not predictable, and differs by field.** `source` kept the 5 alphabetically-first values (every `roboflow_*` invisible). `ground_truth.label` dropped **Doors from the middle of the alphabet** — the rarest class at 2,280 boxes. Neither list looks truncated on its own: a sidebar showing a plausible, alphabetically-ordered set of classes reads as complete. This misled two sessions into believing real data was missing, once far enough to question a completed cascade.

The general lesson: **the App sidebar is not a source of truth about dataset contents.** Verify with `count_values()` / `distinct()`, never by reading the filter dropdown.

### Decision

`fiftyone_final_dataset.ipynb`'s build cell now calls `create_index()` on `source`, `split`, and `ground_truth.detections.label` immediately after `add_samples()` (notebook v2). It must live in the build cell, not be run once by hand: the dataset is `persistent=False` and the cell drops and recreates it on every run, so the indexes go with it. Cost is ~0.1s per field on 36,877 samples.

Not applied to `fiftyone_review_processed.ipynb`, which browses one source at a time with far fewer distinct values and has not shown the symptom — left for whenever it actually does.

---

## DEC-114: Hailo Toolchain Validation Prepared — DFC 3.34 Can Downgrade Its Own HEF Version, Removing DFC 3.33 As The Only Fallback

- **Date:** 2026-09-06
- **Status:** Accepted — **RESOLVED 2026-09-06: PASS.** DFC 3.34.0 output loads on HailoRT 4.23.0; DFC 3.33 is not needed.
- **Related:** DEC-003/DEC-004 (repo split — RPi5 runtime lives in the separate `second-vision` repo), DEC-022/DEC-026 (the export path this validates), DEC-089 (RunPod operational lessons — SSH keys baked at boot, rsync over tar, Terminate not Stop), DEC-112 (`dataset/final/` at 36,877 images, the model this export path will eventually carry)

### Context

The export chain `DFC 3.34.0 -> HEF -> HailoRT 4.23.0` had never been run end to end. Ultralytics documents DFC **3.33** + HailoRT 4.23 as the validated pair; the wheel on hand is **3.34**, and HailoRT refuses HEFs newer than itself. If 3.34's output is rejected, everything downstream of training is blocked. The stated fallback was a login-gated ~500 MB download of DFC 3.33 from the Hailo Developer Zone.

### What Was Established Without Spending Anything

Read directly out of the wheel and the installed Ultralytics 8.4.118, before any pod was rented:

- **`LATEST_HEF_VERSION = 5`** (`hailo_sdk_common/versions.py:3`) — the concern is real; 3.34 does emit v5 by default.
- **DFC 3.34 can emit HEF v1 through v5.** `version_alias_dict` in `hailo_sdk_client/allocator/platform_params.py:34-49` maps `v1`-`v5`, wired to the model-script command `platform_param` (`commands.py:80`). The line `platform_param(hef_version=4)` downgrades the output.
- **Ultralytics exposes no hook for it.** The model script is hardcoded at `engine/exporter.py:1665-1716` and passed straight into `runner.load_model_script()`. Using the knob requires patching `exporter.py` or driving `ClientRunner` directly.
- **Expected failure signature**, from a documented case of the same mismatch class: `Unsupported hef version <N>` followed by `Failed parsing HEF file HAILO_INVALID_HEF(26)`.
- `exporter.py:622` asserts `LINUX and not ARM64` — compilation genuinely cannot happen on the student's Mac or on the Pi. A pod is unavoidable.

### Decision

**Compile two HEFs in one pod session — the default v5 and a patched v4 — rather than v5 alone.** A v5 rejection then still ends the session with a working answer instead of a gated download and a second sitting. DFC 3.33 becomes the fallback only if **both** are rejected.

### Two Corrections To The Handoff's Verified-Facts Section

Both claims were wrong against the installed source, and both were being treated as settled:

- **"Hailo export requires `data=`; it will not silently fall back."** It does fall back. `exporter.py:619-620` sets `self.args.data = TASK2CALIBRATIONDATA.get(model.task)`, which is `coco128.yaml` for detect. Passing `data=coco8.yaml` is still correct here, but for compile speed (8 images vs 128), not because it is mandatory.
- **"`quantize` must be `8` or `w8a16`."** True as a constraint, but it does not have to be passed. `exporter.py:611-617` auto-enables `quantize=8` with a warning. Only an explicit `quantize=32` raises.

The rest of the handoff's Ultralytics claims were verified accurate at the exact line numbers cited (646, 647, 611, 619, 1598, 1658), including that `name="hailo8"` must be passed or the export silently targets `hailo8l`, the 13-TOPS part rather than the 26-TOPS AI HAT+.

### The Pi Was Not Offline — It Moved

`192.168.1.19` (the address in the handoff) failed both ping and SSH, initially reading as a dead device. It was a DHCP lease change: the Pi answers at **192.168.1.20**, proven by an identical ED25519 host key, `SHA256:qCmOUTr/QBVvlyXqZmrzBYC0kqr3Hln65BVDR/mu37M`, recorded for `.19` in `known_hosts` and live at `.20`. `.19` is now a different device. Key auth from the Mac does not currently work for any tried user, so the Pi-side steps are written to be run at the device with a keyboard and monitor rather than over SSH, at the student's request.

### Rationale

The expensive failure mode here was spending the pod session to learn only "v5 is rejected", then needing a gated download and a second session. Reading the wheel first cost nothing and found the downgrade path the handoff had concluded did not exist ("could not be determined by reading the wheel"). Compiling both in one sitting converts a possible dead end into a decision.

### Consequences

- `runpod_hailo_validate_wizard.sh` (gitignored, ephemeral per the wizard skill, same as `runpod_dedup_wizard.sh` in DEC-089) drives 10 stages: pod deploy, connect, wheel upload, venv, TF-GPU check, both compiles, download, teardown.
- `docs/HANDOFF_hailo_pi_verification.md` holds the Pi-side steps, PASS/FAIL criteria, and the exact failure signature to look for.
- **The result is still unknown.** No pod has been rented and no HEF has been parsed. `hailo_validation_report.txt` will carry a `PENDING:` line until the Pi test runs.
- **The 30 GB network volume the handoff called for was dropped from this run**, on the student's challenge ("since I'm just doing a test run why do I need a 4090 right off the bat?") — a fair question that exposed a weaker premise underneath it. The DFC wheel is only needed to *export*, never to train, so it is used across a handful of sessions, not the many-session pattern a volume pays off for. The 4.7 GB dataset that would justify one is explicitly not being uploaded yet because the class schema may still change, so a volume created today would hold 524 MB in a 30 GB allocation at $2.10/month. Worse, attaching it would have **constrained the throwaway test pod's GPU choice to a single datacenter** in a session whose whole goal is renting the cheapest thing available. This is DEC-089's own reasoning for rejecting a volume for the one-time dedup run, applied to a job of the same shape.
- **This concerns the Network Volume only, not the pod's own disk.** RunPod calls both "storage" and they are easy to conflate (they were, in this session): the pod's persistent/volume disk at `/workspace` is set at deploy time, billed with the pod, dies with it, and is mandatory — every pod has one. Take the template's recommendation, typically 50 GB, which comfortably fits the 524 MB wheel plus the DFC's TensorFlow/JAX/CUDA install.
- **The datacenter decision is therefore deferred to training planning**, where it can be made against a settled schema, real dataset and checkpoint sizes, and actual training-GPU requirements. Accepted cost: one 524 MB re-upload of the wheel at export time. Accepted risk: a preferred datacenter may lack storage capacity when the volume is finally created — the same failure hit during DEC-089 — but that is resolved by picking a different datacenter, which is precisely the flexibility committing today would spend.
- The TF-GPU check is a warn-and-continue gate, not a hard stop: an 8-image calibration on CPU still answers the HEF-acceptance question. The real 36,877-image run must not proceed on a CPU-only image.

### Session Interrupted 2026-09-06 — Environment Proven, Compile Not Yet Run

Stopped for session limits after the environment was built and verified, before any HEF was compiled. **No `parse-hef` has run; the central question is still open.** Four real operational failures were hit and fixed, all now baked into the wizard:

- **`pip install -q` over a silent SSH channel died mid-install.** No keepalives, minutes of no traffic. Fixed with `ServerAliveInterval=30` on both remote helpers, and by running the install **detached on the pod** with its own log and a polling loop — a dropped connection now costs a reconnect, not the install. A guard prevents a re-run launching a second pip into the same venv.
- **The venv was being built on `/workspace`, which is network storage** (`fuseblk`, MooseFS at `mfs#euro-3.runpod.net:9421`), not local disk. Tens of thousands of small package files across FUSE is slow and fragile. Moved venv and work dir to `/root` (local overlay); the install then finished in ~100 seconds.
- **The blocker that would have survived every existing check: a CUDA major-version collision.** pip took the newest torch, `2.14.0+cu130`, onto a `570.195.03` driver capped at **CUDA 12.8**. TensorFlow was genuinely fine — a real matmul ran on the L4 — so the wizard's TF-only gate at stage 6 would have **passed straight through**. But importing torch initialised a CUDA 13 runtime that poisoned the process, and `import hailo_sdk_client` then died on `cudaGetDevice() failed`. `yolo export` imports both. Fixed by purging the `nvidia-*-cu13` stack and installing torch from the driver-matched index; the wizard now reads the CUDA ceiling from `nvidia-smi` and selects cu130/cu128/cu126/cu124/cu121/cu118 **before** ultralytics can pull a torch of its own choosing. This will recur on the real post-training export if not carried forward.
- **The DFC's first-run check wants OS packages** the image lacks: `python3-tk`, `graphviz`, `libgraphviz-dev`, and `column` (`bsdmainutils`). Non-fatal — the import succeeds regardless — but installed to avoid discovering it mid-compile. A residual `Cannot use graphviz, so no visualizations will be created` warning remains and is cosmetic (visualisations only).

Verified working state at the stop: `hailo_sdk_client 3.34.0` (LATEST_HEF_VERSION 5), `torch 2.11.0+cu128` with `cuda True`, `tensorflow 2.18.0` with 1 GPU, `ultralytics 8.4.118`, on an **NVIDIA L4** at $0.50/hr. The venv lives on the pod's local disk and does **not** survive termination — but it now rebuilds in about three minutes unattended, so terminating is the cheap choice.

### Resumed And Re-Stopped 2026-09-06 — What A RunPod Restart Actually Costs, Measured

The pod was left Running over an hour-long pause, then found unreachable: the host still answered ping, but the SSH port was closed. It had **restarted** — Running in the console, but with everything below re-initialised. Three concrete behaviours, measured rather than assumed, and worth not rediscovering:

- **The exposed TCP port changes on restart** (12592 → 10512). Any stored SSH command goes stale. This is why `.env` holding a previous pod's `RUNPOD_SSH_CMD` is actively dangerous — the wizard's `ask` offers it as an `[Enter keeps current]` default, so pressing Enter silently targets a dead pod.
- **`/root` (container overlay) is wiped; `/workspace` survives.** The entire venv — torch, TensorFlow, the DFC, the apt packages — was gone, while the 524 MB wheel was intact byte-for-byte. So the pod's *own* disk is not a place to keep anything, but the upload does not need repeating.
- **The host key changes**, so a restarted pod fails `BatchMode` SSH with "Host key verification failed". This exposed a real latent bug: the wizard's stage-3 connection test had no host-key policy, meaning it would have failed on **every freshly deployed pod** and reported the misleading "SSH key was added after boot" message. Fixed with `StrictHostKeyChecking=accept-new` on all SSH paths — which trusts a first-seen key but still refuses a *changed* one.

Full unattended rebuild from a wiped `/root` measured at **~180 seconds** (apt packages, venv, driver-matched torch, ultralytics, DFC). The student elected to **Stop** rather than Terminate, judging reclaim unlikely across a short window — a materially cheaper bet than DEC-089's, where a stopped pod held a multi-hour dataset upload. Here the only unrecoverable asset is ~100 seconds of wheel re-upload.

### RESOLVED — The Answer Is PASS

Both HEFs compiled (v5 in 298s, v4 in 334s on an L4) and **both parse cleanly on the Pi** under HailoRT 4.23.0, firmware 4.23.0, Board Hailo-8, HAILO8:

    Architecture HEF was compiled for: HAILO8
    Network group name: yolov8n, Single Context
    Input  yolov8n/input_layer1 UINT8, NHWC(640x640x3)
    Output yolov8n/yolov8_nms_postprocess FLOAT32, HAILO NMS BY CLASS
    Op YOLOV8, score th 0.250, IoU th 0.70, 640x640, 80 classes

`hailortcli run` on the v5 HEF: **1,596 frames, 318.84 FPS**, 3134.29 Mbit/s send.

**Consequences:**

- **The DFC 3.34 / HailoRT 4.23 version worry is closed.** Ultralytics documents 3.33 + 4.23 as the validated pair, but 3.34's v5 output is accepted as-is. The login-gated ~500 MB DFC 3.33 download is not needed and should not be pursued.
- **The v4 hedge was never required.** Both files parse identically. `parse-hef` prints no HEF version number, so whether `platform_param(hef_version=4)` actually took effect is still unconfirmed — now moot, but the knob and its wiring (`platform_params.py:34-49`, `commands.py:80`) remain documented here in case a future HailoRT/DFC pairing does reject one.
- **`name=hailo8` was correctly applied** — the HEF reports `HAILO8`, not the 13-TOPS `HAILO8L` that Ultralytics defaults to at `exporter.py:646`. That default remains the single easiest way to silently ship a wrong-accelerator binary.
- **318 FPS is not a prediction for this project.** It is stock yolov8n over COCO's 80 classes. YOLOv8s on 13 classes will differ. It does establish that the accelerator has large headroom over a smartglass obstacle-warning frame rate.
- The Pi-side silence that briefly looked like a driver fault was **a loose ribbon cable on the AI HAT+** — worth checking first next time, before `lsmod`/`lspci`/`dmesg` diagnosis.
- The export path is now unblocked end to end. The remaining Hailo work is the real export after training, which must reuse the driver-matched-torch fix recorded above.

## DEC-115: Schema 13 → 15 — Stairs and Bench Added From Open Images. Doors Top-Up Proposed Then Withdrawn. Queued, Not Yet Executed

- **Date:** 2026-09-06
- **Status:** Accepted (queued — no config, dataset, or code changed by this entry). **Amended same day — the Doors top-up was withdrawn by the student; see the amendment at the end of this entry. The Doors passages below are left as originally written so the reversal is legible.**
- **Related:** DEC-100 (dropped Stairs; this partially reverses it), DEC-083 (the propose-then-*look* precedent this followed), DEC-042 (floor/cap/ratio), DEC-107 (the five schema copies), DEC-101/DEC-105 (why appending beats re-indexing), DEC-052 (cross-class-folder box merging, which creates the Bench↔Tables conflict), DEC-102 (dedup deliberately skipped in the cascade). Full evidence: `docs/HANDOFF_replacement_classes.md`. **Numbering:** originally written as DEC-114 and renumbered to 115 — a concurrent session had independently taken 114 (Hailo toolchain validation), which is already referenced from `TASKS.md` and `docs/HANDOFF_hailo_pi_verification.md` while this entry had no references. Same resolution rule as DEC-083's collision.

### Context

Training was ready to start on the verified 13-class `dataset/final/` (DEC-112, 36,877 images). Before starting, the student asked whether any class should replace the three dropped by DEC-100. A planning pass enumerated **all 599 Open Images V7 detection classes**, measured every navigation-relevant and indoor-obstacle candidate for distinct-image count, boxes/image, median box area and small-box share, and visually inspected the shortlist with boxes drawn — Stairs (30 samples), Bench, Traffic sign, Traffic light, Door (24 each), Bed, Couch, Houseplant, Box (18 each).

The pass recommended **Stairs only**. The student decided on **Stairs + Bench + a Doors top-up**.

### Decision

Schema **13 → 15**, all three sourced from Open Images V7:

| change | class | id | Open Images label |
|---|---|---|---|
| new | Stairs | **13** | `"Stairs"` |
| new | Bench | **14** | `"Bench"` |
| top-up | Doors | 6 (unchanged) | `"Door"` |

**Appended, not inserted.** ids 0–12 are untouched, so none of `dataset/final/`'s 122,264 boxes is rewritten and no inverse of `drop_classes.py` is needed.

**Stairs is built from Open Images alone; the benched Roboflow stair sources stay benched.** `drop_classes.py` deleted every Stairs box from every source's `labels/`, `stair_gaptw` included, and DEC-100 records that no inverse migration script exists. Recovering those ~1,416 images would mean hand-restoring old-id-5 boxes from the 4.1 GB backup across six sources and remapping 5 → 13 with no tooling — DEC-101/DEC-105 territory. A fresh Open Images pull yields more images (2,630) and needs none of it.

**`doors:` must be restructured from `primary_providers` to `primary:`.** `get_openimages_targets()` (`acquire_openimages.py`) selects on `primary.source == "open_images"` and ignores `secondary_providers` entirely; `openimages_to_intermediate.py:53` imports that same function. An Open Images entry added under `secondary_providers` would be **silently ignored** — no error, no data. The two existing Roboflow door sources move to `secondary_providers`. Same restructure DEC-083 applied to Pole.

**Deliberate `cap:` divergence for the two new classes — Stairs 5500, Bench 8000.** `acquire_openimages.py` splits `ceil(cap × 1.35)` evenly across three splits, and both classes are train-heavy, so `cap: 4500` strands 450 Stairs and **1,544 Bench** images. Bench at 4,500 pulls only 2,039, which after exclusions lands near the 1,500 floor — the exact failure mode that killed the previous three classes. Verified that `cap_per_class.py` takes its ceiling from `--hard-cap` (default 4500, `cap_per_class.py:114`) and **never reads the per-class `cap` field**, so `cap` is purely an acquisition-buffer input and DEC-042's selection ceiling is still enforced at Stage 5.4. `cap: 5500`/`cap: 8000` pull each class in full (2,630 / 3,583), config-only. Each block **must** carry an inline `note:` saying so — `classes.yaml` documents `cap` as "hard cap 4500 for every class", and without the note a future reader will correctly-looking-ly revert it.

### Two measurements that corrected this session's own earlier claims

Both were first asserted from small visual samples and then measured against the full annotation set. Recording both, because the sampling bias behind them will recur.

1. **Open Images `"Door"` car-door contamination: 1.8%, not ~17%.** The handoff first flagged Door as unsafe because 4 of 24 rendered samples boxed car doors. Measured properly: **350 of 19,970 Door boxes (1.8%)** sit ≥80% inside a Car/Bus/Truck/Van/Taxi box; 237 images (1.7%) carry one; 199 (1.4%) are nothing but vehicle doors. The visual sample was drawn from locally cached images, which are the intersection with images pulled *for Person and Vehicle* — i.e. structurally car-biased. **The claim was wrong by an order of magnitude and the source is sound.**
2. **Bench ↔ Tables collision: 9.7% of boxes, not ~33%.** 685 of 7,038 Bench boxes sit at IoU ≥ 0.5 with a `Table`/`Coffee table`/`Kitchen & dining room table`/`Desk` box; 84 more (1.2%) against `Chair`; 13.8% of Bench images are affected. Real, an order of magnitude smaller than the 4-of-24 sample implied, and mitigable.

**Generalisation worth keeping:** the locally cached Open Images subset is *not* a random sample of Open Images — it is the residue of past per-class pulls, and it inherits their class bias. Use it to judge *annotation quality*; never to estimate a *rate*. Rates come from `detections.csv`.

### The Bench ↔ Tables conflict lands inside our own pool

Not merely a source quirk. Per DEC-052, `openimages_to_intermediate.py` merges boxes across class folders rather than dropping them, so once Bench is a target class, an image pulled for Tables that also carries a Bench box gets **both boxes on the same picnic table** — one `Tables`, one `Bench`. Contradictory supervision the model cannot resolve, and exactly the class of defect DEC-112 showed to be worth 10,165 suppressed boxes elsewhere.

Chosen mitigation: **filter at conversion** — drop the Bench box when IoU ≥ 0.5 with a same-image Tables or Chairs box, keep the Tables box. ~770 boxes. This is the **only code change** in the plan (a small addition to `openimages_to_intermediate.py`), and it mirrors DEC-109/110/111's existing containment logic rather than inventing a new rule.

### Rationale

Stairs is the highest-consequence obstacle in the schema — every other class causes a collision, a missed descending staircase causes a fall — and it was dropped purely on a 125-image shortfall that Open Images removes. Its Open Images boxes are proper axis-aligned rectangles, so DEC-031's polygon-derived-sliver defect against the old Roboflow stair sources does not reproduce; median box area is 16.6% of frame with only 5.6% of boxes under 1%, among the best measured, which is what makes it viable at 640px on Hailo-8 where Traffic sign and Traffic light are not.

Bench was recommended against and the student overrode it. The override is reasonable on the corrected 9.7% figure: street furniture is a genuine torso-height hazard, and the collision is a tenth of the class with a mechanical fix, not a third with no fix.

Doors is the schema's thinnest class (1,938 images / 2,280 boxes) and Open Images offers 13,910 at 1.8% contamination. Cheapest real quality gain available.

### Alternatives Considered

- **Insert Stairs at its old id 5**: Rejected — re-indexes eight classes and rewrites every label file for a purely cosmetic ordering gain, after this project has already paid twice for schema churn (DEC-101, DEC-107).
- **Revive Stairs by restoring the Roboflow sources from the backup**: Rejected — no inverse migration script exists (DEC-100), it needs a hand-written 5 → 13 remap across six sources, and it yields *fewer* images than a clean Open Images pull.
- **Leave both new classes at `cap: 4500`**: Rejected — strands 1,544 Bench images and leaves the class hovering at the floor with no recovery short of another acquisition round.
- **Add Bed instead of Bench**: Not chosen by the student. On the data it is the strongest indoor candidate measured (median box area 0.330, 0.8% tiny boxes, 17 of 18 samples clean, zero overlap) but a bed sits in the one room a blind user knows best — best data, weakest use case.
- **Accept the Bench/Tables double-labelling**: Rejected — DEC-112 measured what redundant stacked boxes cost when left in.

### Consequences

- **Full retrain, not a fine-tune.** 13 → 15 is a new detection head. Accepted knowingly, as in DEC-100.
- **All five DEC-107 schema copies move together**, plus `dataset/final/data.yaml` (generated) and the FiftyOne App annotation JSON (generate it from `get_canonical_names()`; do not hand-type it). Delete `__pycache__` before validating `config_loader.py` — DEC-107's bytecode-staleness incident.
- **The cascade re-runs**: `acquire_openimages.py --classes doors,stairs,bench` → `openimages_to_intermediate.py` → review → `promote_reviews.py` → `cap_per_class.py --hard-cap 4500` → `merge.py` → *skip `dedup.py`* (DEC-102) → `split.py` → `generate_yaml.py`. `openimages_to_intermediate.py` rebuilds the **whole** `dataset/processed/open_images/` pool, so verify its output against the report rather than trusting it — the DEC-050/DEC-085 stale-file pattern.
- **Expected end state**: Doors ~4,500 (hits the cap), Stairs ~2,100–2,500, Bench ~2,600–3,000, every other class unchanged. Ratio invariant stays **2.64** (Person 4,500 / Trash Bins 1,702) provided both new classes land above 1,702; Trash Bins remains the binding minimum.
- **Stairs will be instance-thin** — ~1.2 boxes/image, ~2,600–3,000 instances, against DEC-042's 10,000 target (6,000 for small/hard). Not disqualifying: `Doors` ships 2,280 boxes and `Trash Bins` 2,737. Expect Doors-like AP.
- **Training is delayed by roughly 3–5 days**, dominated by the review pass on three new pools, not by compute.
- **Housekeeping found while planning**: `cap_per_class.py`'s `CLASS_PRIORITY_SOURCES` still carries a stale `"Elevator"` key for a class dropped in DEC-100. Harmless — never looked up — but delete it while the file is open.
- **Not yet done**: everything above. This entry is the queued plan; a separate session executes it. `docs/HANDOFF_replacement_classes.md` §0 holds the step-by-step.

### Amendment, 2026-09-06 — Doors top-up withdrawn by the student

**The Doors top-up is cancelled. `doors:` is not restructured, Open Images `"Door"` is not pulled, and `doors:` keeps its existing `primary_providers` shape with the two Roboflow sources. Only Stairs (id 13) and Bench (id 14) proceed.**

The student's reasoning, recorded verbatim in substance: *if there is even a chance of car doors being in the pool, drop it.* This overrides the 1.8% measurement above, and it is a risk-tolerance judgment rather than a factual disagreement — the measurement is not in dispute, the acceptable level of contamination is. Noted and not re-argued.

Worth recording that a mitigation existed and was not taken: the same containment test used to *measure* the contamination (Door box ≥80% inside a Car/Bus/Truck/Van/Taxi box) would also **remove** it at conversion, dropping 350 boxes and the 199 images that are nothing but vehicle doors. The student chose to avoid the source rather than filter it. If Doors is ever revisited, that filter is the cheap path and this entry's measurements are the basis for it.

**Consequences of the withdrawal:**

- `Doors` stays at **1,938 images / 2,280 boxes** and remains the schema's thinnest class by image count. It is still **above** DEC-042's 1,500 floor, so nothing is out of compliance.
- The ratio invariant is unaffected: the binding minimum is `Trash Bins` at 1,702, not Doors, so max/min stays **4,500 / 1,702 = 2.64**.
- The plan loses its only `primary_providers` → `primary:` restructure. **The finding behind it still stands and is still worth knowing:** `get_openimages_targets()` selects on `primary.source == "open_images"` and ignores `secondary_providers` entirely, so any future attempt to add an Open Images provider to a class under `secondary_providers` will be **silently ignored** — no error, no data. This is a live trap for `doors:`, `potholes:`, `tricycle:` and every other `primary_providers`-shaped class.
- Acquisition narrows to `acquire_openimages.py --classes stairs,bench`.
- Expected end state revises to: Stairs ~2,100–2,500, Bench ~2,600–3,000, **Doors unchanged at 1,938**, every other class unchanged.
- `docs/OPEN_QUESTIONS.md` gains nothing new — Doors' thinness was never an open question, and it is not blocking.

### Amendment 2, 2026-09-06 — browse-first pull executed. The `est. pull` model was wrong for Stairs; `cap: 5500` would have failed the floor

Step 4 of the plan ("browse the pools before touching config") was run ahead of everything else, as a throwaway pull into two persistent FiftyOne datasets (`preview_oi_stairs`, `preview_oi_bench`) outside the pipeline. Nothing in `config/`, `scripts/` or `dataset/` was touched. It reproduced `acquire_openimages.py`'s real behaviour exactly — per-split `max_samples`, `shuffle=True`, `seed=42`, then DEC-043's `IsDepiction`/`IsGroupOf` filter.

**It caught a plan-breaking error before any schema copy was edited. This is why the browse-first step exists.**

| | predicted | **actual** |
|---|---|---|
| Stairs @ `cap: 5500` | 2,630 images | raw 2,643 → **1,514** carrying a clean Stairs box |
| Bench @ `cap: 8000` | 3,583 images | raw 3,614 → **3,555** carrying a clean Bench box |

**Stairs at `cap: 5500` yields 1,514 images — fourteen above DEC-042's 1,500 floor, before a single review exclusion.** At the 15–20% exclusion rate this session estimated, it lands at ~1,210–1,290, i.e. **below the floor** — the exact failure that killed it under DEC-100. `cap: 5500` is wrong and must not be executed.

**Why the model was wrong.** `est_pull` assumed the per-split budget is spent on images that survive DEC-043's filter. It is not: the Zoo's `classes=` selector picks images that contain the class *at any flag value*, then the filter runs afterwards. When a class has a high `IsGroupOf`/`IsDepiction` rate, most of the budget is spent on images that the filter then discards.

Measured clean rate — the share of a class's images that carry at least one box surviving DEC-043 — differs enormously by class:

| class | images with any box | with a clean box | clean rate | GroupOf/Depiction box rate |
|---|---|---|---|---|
| **Stairs** | 4,700 | 2,630 | **56.0%** | **48.8%** |
| Shelf | 6,797 | 5,915 | 87.0% | 8.9% |
| Bench | 3,644 | 3,583 | 98.3% | 3.0% |
| Street light | 11,364 | 11,323 | 99.6% | 2.3% |

**The Street light validation gave false confidence.** The model was checked against DEC-083's real Street light pull and matched to the image (2,159 exactly) — but Street light has a 99.6% clean rate, so the dilution the model ignores was invisible there. Validating on the easiest case is not validation. **Any `est. pull` figure in `docs/HANDOFF_replacement_classes.md` §3 and §6 is an upper bound, not an estimate, and is only trustworthy for classes with a high clean rate.**

### The real finding: DEC-043's blanket `IsGroupOf` filter is actively wrong for Stairs

**48.8% of Stairs boxes are flagged `IsGroupOf`** — five times Shelf's rate and twenty times Street light's. The cause is semantic: a flight of stairs reads to an annotator as *a group of steps*, so the flag fires on the ordinary case rather than the exceptional one.

**Those boxes were inspected (18 rendered at random from the real pool) and they are the *best* Stairs annotations in the class** — single, tight, correctly-framed boxes around one flight: a grand civic staircase, subway stairs, hillside steps, interior flights. They are not the sprawling multi-object cluster boxes `IsGroupOf` is meant to catch.

DEC-043's rationale for dropping `IsGroupOf` is specific and does not transfer: *"the on-device algorithm determines a group of a class by counting multiple individual instance detections within an area of interest"*, so a group-box would teach the model that a cluster is one object. That is correct for Person and Car. **For Stairs the desired output is exactly one detection per flight — nobody wants the device counting individual steps.** The rule is being applied to a class whose semantics invert it.

### Revised options for Stairs — the student's call

| option | cap | images | notes |
|---|---|---|---|
| ~~**A′.** as planned~~ | 5,500 | **1,514** | **Not viable.** 14 above the floor pre-exclusion. Do not execute. |
| **A. Keep DEC-043 as-is, raise the cap** | **~10,100** | **2,630** | Clears the floor with ~1,130 margin. No rule change, no code change. Downloads 4,700 images to keep 2,630. **Recommended.** |
| **B. Keep `IsGroupOf` for Stairs only** | ~10,100 | **4,700** | Best data — richer boxes, and the ones visually judged best. Requires a per-class filter in `acquire_openimages.py` (a code change) and an explicit amendment to DEC-043. |
| **C. Drop Stairs, ship Bench alone** | — | — | Schema 13 → 14. Honest fallback if neither the download nor the rule change is wanted. |

Recommended: **A**, with **B** flagged as the better-data option worth a deliberate decision rather than inheriting a blanket rule. Under A the `cap:` divergence noted in the main entry gets larger, not smaller — `cap: 10100` against a documented "hard cap 4500" — so the inline `note:` requirement is now mandatory, not advisory.

### Bench is confirmed, and this session's objection to it is withdrawn

**Bench came in at 3,555 clean images against 3,583 predicted — a 0.8% miss.** `cap: 8000` stands exactly as written.

More importantly, **24 images rendered at random from the real pool show this session's stated objection to Bench was a sampling artefact.** The original "picnic-table contamination, reject" verdict came from the locally cached subset, which is the residue of past Person/Vehicle/Tables pulls and therefore biased toward exactly the scenes that produce the confusion. On an unbiased draw the pool is overwhelmingly genuine outdoor street and park furniture — park benches, bus-shelter seating, riverside and playground benches — with roughly 2–3 of 24 questionable, matching the measured 9.7% collision rate rather than the 4-of-24 (17%) the biased sample implied.

**The student's choice of Bench was better than this session's recommendation against it.** The §0.5 IoU filter is still worth applying, but as tidy-up rather than rescue.

### Consequences of this amendment

- `cap: 5500` for Stairs is **withdrawn**. Execution must not proceed on the numbers in §0.4 of the handoff as originally written.
- Stairs' viability now depends on a decision the student has not yet made (A, B or C above). **This is the one open blocker in the plan.**
- Bench is unblocked and unchanged.
- The two preview datasets are persistent and browsable now, before anything is edited: `fo.load_dataset("preview_oi_stairs")` / `("preview_oi_bench")`, then `fo.launch_app(...)`. Per DEC-113, create an index on `ground_truth.detections.label` before trusting the sidebar's class list.
- The zoo cache at `~/fiftyone/open-images-v7/` grew by the pulled images. Harmless and reusable — a real acquisition run will find them already downloaded.
- **Method lesson, worth more than this entry's specific numbers:** a per-class count taken from `detections.csv` is a *ceiling*, not a *yield*. Yield depends on the class's clean rate and must be measured by pulling. This session asserted three numbers from static analysis (car-door rate, Bench collision rate, Stairs pull) and measurement corrected all three — twice in the pessimistic direction, once fatally in the optimistic direction.

---

### Handover note, 2026-09-06

Execution handed to the main session. `docs/HANDOFF_stairs_bench_execution.md` is the handover document.

Reasons, in order: (1) the `cap:` raise this plan depends on has already been discussed in the main session and this sub-session has no visibility into those concerns, so §4 of the handover states what the field mechanically does — every reader of it, verified by reading the code — rather than arguing a position; (2) DEC-114's Hailo session is live, has ~750 uncommitted lines in this file, and is validating a toolchain against `nc: 13` that this change would make `nc: 15` — sequencing belongs to whoever holds both threads; (3) the operative decision is scheduling, not technique.

**A third correction to this session's own claims, found while writing the handover.** This entry and the handoff both stated that `cap_per_class.py` "never reads the per-class `cap` field". **That is false** — it reads it at `cap_per_class.py:393` (`id_to_cap = {entry["id"]: entry["cap"] ...}`). The substantive claim survives: the value passed into `cap_class()` is `hard_cap_preset` from the CLI, never `configured_cap`, so selection is still governed by `--hard-cap` and DEC-042's ceiling is unaffected by a raised `cap`. But `cap` is not unread, and two further facts follow that matter for the decision:

- A raised `cap` is **visible at runtime, not silent** — `cap_per_class.py:426` prints `NOTE: <class>'s classes.yaml cap (N) overridden by --hard-cap=4500` on every run where they differ. That is a point in favour of the approach.
- `id_to_cap[class_id]` requires every id `0..nc-1` to exist in `classes.yaml` with a `cap`. A missing or mis-numbered per-class `id:` raises `KeyError` here — DEC-101's failure mode, and the first place a botched schema edit surfaces.

Tally for the record: this session asserted four things from static analysis — Door car-door rate (~17%), Bench/Tables collision (~17%), Stairs yield at `cap: 5500` (2,630), and `cap_per_class.py` not reading `cap`. Measurement corrected all four: 1.8%, 9.7%, 1,514, and false respectively. Three were harmless or pessimistic; one was plan-breaking. **The pattern, not the individual numbers, is the thing to carry forward: static analysis of `detections.csv` and of code produced confident wrong answers four times; running the thing produced right ones each time.**

**Concurrency, observed twice in one session.** This entry was renumbered 114 → 115 mid-session after the Hailo session took 114, and while appending this very note the same session appended **DEC-116**, so a blind append would have written it into DEC-116's body. `docs/DECISIONS.md` is being edited by two sessions concurrently right now: **locate your entry's own closing separator before appending, never `tail` the file.** This is the concrete recurrence of the hazard flagged in DEC-083's Consequences and `docs/OPEN_QUESTIONS.md`'s header.

---


## DEC-116: Torch Must Be Installed From a Driver-Matched CUDA Index Before Ultralytics, Or Hailo Export Breaks — And a TensorFlow GPU Check Will Not Catch It

- **Date:** 2026-09-06
- **Status:** Accepted
- **Related:** DEC-114 (the validation run where this was found), DEC-022/DEC-026 (the export path this protects), DEC-089 (RunPod operational lessons). **Numbering:** written as DEC-115 and renumbered to 116 — a concurrent session had independently taken 115 (schema 13 → 15). Same resolution rule as that entry used against DEC-114: the entry with no inbound references moves.

### Context

Found while validating the Hailo toolchain (DEC-114) on a RunPod pod with an NVIDIA L4. A plain `pip install ultralytics` resolves torch against the newest wheel on PyPI, with no reference to what the host's NVIDIA driver can actually run. On that pod it installed **torch 2.14.0+cu130** — a CUDA 13 build — onto driver **570.195.03**, which caps at **CUDA 12.8**.

### What Actually Breaks

Two failures, and the second is the dangerous one:

1. `torch.cuda.is_available()` returns `False`. Visible, easy to diagnose.
2. **Importing torch initialises a CUDA 13 runtime that poisons the process.** A subsequent `import hailo_sdk_client` then dies with `InternalError: cudaGetDevice() failed. CUDA driver version is insufficient for CUDA runtime version`. `yolo export ... format=hailo` imports both torch and the DFC in one process, so the collision is unavoidable once the stacks disagree.

The venv ends up holding a genuinely mixed stack — `nvidia-cublas 13.1.1.3`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13` sitting beside the CUDA 12 libraries TensorFlow uses.

### Why This Is Easy To Miss

**TensorFlow tested completely healthy throughout.** Not merely enumerating a device — a real `tf.matmul` ran on the L4, creating a 20,847 MB context at compute capability 8.9. So the natural gate, "does TensorFlow see the GPU?", **passes** while the export is already doomed. Any check built only around TF — which is the obvious one to write, since the DFC's heavy work is TF-side — gives false confidence here.

The failure also surfaces far from its cause: the error names `hailo_sdk_client`, not torch.

### Decision

**Install torch from a CUDA index matched to the driver, before ultralytics can resolve one of its own.** Read the ceiling from `nvidia-smi` and select accordingly:

    DRV=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -1 | awk '{print $3}')
    MAJ=${DRV%%.*}; MIN=${DRV##*.}
    if   [ "$MAJ" -ge 13 ]; then IDX=cu130
    elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 8 ]; then IDX=cu128
    elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 6 ]; then IDX=cu126
    elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 4 ]; then IDX=cu124
    elif [ "$MAJ" -eq 12 ]; then IDX=cu121
    else IDX=cu118
    fi
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/$IDX

Then install `ultralytics`, then the DFC wheel. Ordering matters: ultralytics accepts an already-satisfied torch, but left first it pulls whatever is newest.

**Verification must import the DFC, not just TensorFlow.** The gate is `import hailo_sdk_client` succeeding in the same process as `import torch` — nothing weaker proves anything.

### Consequences

- `runpod_hailo_validate_wizard.sh` does this automatically; it was the fix that unblocked DEC-114.
- **This recurs on the real post-training export.** Nothing about it was specific to yolov8n or to the L4 — any pod whose driver is older than the newest torch reproduces it, and driver ages vary across RunPod hosts and datacenters.
- Recovery, if hit anyway: purge every `nvidia-*` package plus torch/torchvision, then reinstall from the matched index. Measured at ~80 seconds.
- A related trap worth pairing with this: the DFC's first-run check wants OS packages the stock image lacks — `python3-tk`, `graphviz`, `libgraphviz-dev`, and `column` (`bsdmainutils`). Non-fatal, but it prints red `Error` lines that read like failure.

---

## DEC-117: Schema 13 → 15 Executed — Stairs and Bench Added, `cap` Divergence Justified By Measurement, Bench↔Furniture Collision Filter Built

- **Date:** 2026-09-06
- **Status:** Accepted (executing)
- **Related:** DEC-115 (the plan this executes, which was explicitly "queued, not yet executed"), DEC-100 (dropped Stairs; this partially reverses it), DEC-042 (floor/cap/ratio), DEC-043 (the `IsGroupOf`/`IsDepiction` filter at the centre of the Stairs problem), DEC-052 (cross-class-folder merging, which creates the Bench↔Tables conflict), DEC-109/110/111 (the same wrong-box-on-right-box pattern), DEC-101/105/107 (why appending beats re-indexing, and the five schema copies)

### Context

DEC-115 left one question unresolved: Stairs is viable only at a `cap:` well above the 4,500 convention, and the student had raised a concern in the main session that raising it would break DEC-042's ratio invariant. The student also chose to **skip the manual review pass** for both new classes and trust Open Images' annotations, on time grounds — which removes the safety net that would otherwise catch bad boxes, and raises the stakes on getting the measurements right first.

### Decision

**Option A.** Stairs at `cap: 10100`, Bench at `cap: 8000`, DEC-043 unchanged, no per-class filter exemption. Schema **13 → 15**, ids appended: **Stairs 13, Bench 14**.

Option B (allowing `IsGroupOf` for Stairs only) was rejected: it rests on 18 inspected boxes and would amend a project-wide rule at the same moment the review safety net is being removed. Option C (Bench alone at `nc: 14`) was not needed once Stairs measured viable.

### The ratio concern, resolved by measurement

The concern does not hold, for two independent reasons.

**Mechanically**, `cap:` controls download volume only — read at `acquire_openimages.py:93,234` as `max_samples = ceil(cap × 1.35)` and nowhere else that changes behaviour. `cap_per_class.py:393` reads it *only* to print an override NOTE. The 4,500 selection ceiling, 1,500 floor and 3:1 invariant are enforced separately by `cap_per_class.py --hard-cap 4500`.

**Empirically**, and this is the decisive part: **Open Images contains only 2,633 usable Stairs images in total.** `cap: 10100` therefore means "download everything that exists," which is still far below the 4,500 selection ceiling. Stairs cannot become the max class and cannot move the invariant. Person stays at 4,500, Trash Bins at 1,702, ratio **2.64**.

### Measurements taken before executing (all three Open Images splits, from `detections.csv`)

**Stairs** — 6,179 raw boxes across 4,700 images; after DEC-043, **3,173 boxes / 2,633 usable images** (51.4% box rate, 56.0% image rate). Confirms DEC-115's 2,630 estimate. The class is an outlier: 48.8% of its boxes carry `IsGroupOf`, three times the next-worst class in the schema, because a flight of stairs reads to an annotator as a group of steps.

Why the cap had to move: at `cap: 4500` the pull yields ~1,228 usable images — **below the floor**, the exact failure that killed this class under DEC-100. At `cap: 5500` (DEC-115's original plan, withdrawn) it yields 1,514, fourteen above the floor.

**Bench** — after DEC-043: 3,587 images / 7,042 boxes. Applying the new collision filter:

| IoU | boxes dropped | images losing last Bench box | usable images |
|---:|---:|---:|---:|
| 0.3 | 897 | 351 | 3,236 |
| **0.5** | **758 (10.8%)** | **307** | **3,280** |
| 0.7 | 555 | 214 | 3,373 |

Viability is not threshold-sensitive — every setting leaves Bench at roughly 2.2× the floor. **IoU 0.5** chosen to match DEC-109/110/111's existing precedent. Note the real collision rate is 10.8%, slightly worse than DEC-115's 9.7% estimate.

### Executed

1. **`config/classes.yaml`** — `nc: 15`; `names` 13/14 = Stairs/Bench; `hailo_runtime_names` 14/15 (Background still at 0, so runtime ids remain +1); two new class blocks. Verified: block ids are exactly 0–14 with no gaps, and every id agrees with `names`.
2. **`scripts/utils/config_loader.py`** — `EXPECTED_NC = 15`, `CANONICAL_NAMES` appended. All `__pycache__` cleared before validating (DEC-107's bytecode-staleness incident, which once made every Pothole render as a Tricycle).
3. **`scripts/convert/openimages_to_intermediate.py`** — new `drop_bench_on_furniture()` removing Bench boxes at IoU ≥ 0.5 against same-image Tables/Chairs boxes, called after the cross-folder merge and before `audit_class_counts()`. Class ids resolved via `get_class_id()`, never hand-copied (AGENTS.md:105-111). The **Bench** box is always the one dropped; Tables and Chairs predate this class and their counts must not move.

**Append-only is the point.** Ids 13 and 14 sit after Bicycle (12), so no existing id shifts. This is categorically safer than DEC-100's 16→13, which moved eight classes and produced DEC-101's near-miss where three converters would have silently written stale ids into a re-indexed dataset.

### The `cap` self-documentation hazard, handled

`classes.yaml`'s own header comment describes `cap` as "hard cap 4500 for every class per DEC-042", which a value of 10,100 contradicts on its face — the DEC-107 pattern of a field doing double duty with only one meaning documented. Mitigated with a long inline `note:` on the Stairs block stating that `cap` is download volume only, naming the two call sites, and giving the measured yields at 4,500 / 5,500 / 10,100 so a future reader can see why "fixing" it back would silently re-break the class.

### Consequences and accepted costs

- **No manual review pass** for either class — a deliberate, time-constrained choice by the student. Consequence: the Bench↔furniture filter is the *only* defense for those 758 boxes, and Open Images' annotation quality is trusted as-is. Recorded as a limitation, not a defect.
- **`openimages_to_intermediate.py` rebuilds the entire pool** (31,011 label files). Existing classes' labels are regenerated, and images previously pulled for Tables/Chairs will gain Bench boxes. Output must be verified against the report rather than trusted (DEC-050/085 stale-file pattern).
- **`dataset/final/` is rebuilt**, so the split is redrawn. The frozen-evaluation-set protocol for the training ablation must therefore be pinned to the *new* split, after this cascade — not the DEC-112 one.
- The imbalance figures measured for the training plan (image ratio 5.07, box ratio 11.91 at cap 4500) are recomputed at 15 classes. Both new classes land above Doors' 2,280 boxes, so Doors should remain the box-count minimum and the ratio should barely move — to be confirmed, not assumed.
- Backup at `dataset/backups/pre_schema15_20260906_205738/` holds the pre-change `classes.yaml`, `config_loader.py`, `openimages_to_intermediate.py`, `data.yaml` and all three cascade reports.

### Executed result (appended 2026-09-06, same day)

Cascade complete. `dedup.py` skipped per DEC-102.

| stage | result |
|---|---|
| `acquire_openimages.py --classes stairs,bench` | Stairs 3,980 images pulled, Bench 3,585. Requested 13,635 / 10,800 — both classes exhausted at source, as predicted |
| `openimages_to_intermediate.py` | **37,106 images / 95,659 boxes** (was 31,011). Verified on disk: file count and box count match the report exactly, images and labels pair 1:1 with **zero orphans**, no id outside 0–14 |
| `cap_per_class.py --hard-cap 4500` | 15/15 clear the floor. Stairs 2,630, Bench 3,545, both `stop=all_candidates_included`. **Ratio invariant 2.64, unchanged.** Both override NOTEs printed as designed |
| `merge.py` | **42,988 images** from 43,123 selected, 135 removed by `exclude` tags |
| `split.py` | train 29,949 / val 6,664 / test 6,375 = **69.7/15.5/14.8**; `cross_split_duplicate_leakage: []`, `missing_labels: []` |
| `generate_yaml.py` | `nc: 15`; `check_det_dataset()` from `/tmp` resolves 15 names in correct order, all three split paths existing |

**Final pool: 42,988 images / 132,434 boxes / nc=15.** All ids 0–14, none out of range.

**Imbalance essentially unmoved**, as predicted: image ratio **5.07** (Person 8,630 / Trash Bins 1,702 — identical to the 13-class figure); box ratio **11.93** vs 11.91 before (Person 27,190 / Doors 2,280). Doors remains the box-count minimum; both new classes land above it. The training plan's ablation framing therefore carries over unchanged.

**The collision filter fired far less than the standalone measurement predicted — 30 boxes across 20 images, not 758.** This is correct behaviour, and the discrepancy is worth understanding rather than treating as a bug. The 758 figure counted collisions in Open Images' *complete* annotation set. The converter takes only each folder's own native classes (Bench boxes from the bench folder, Table boxes from the tables folder), so both boxes reach the same label file only when an image was pulled into *both* folders — true for 3,008 of 37,106 images overall. Where the Table box never entered the pool there is no contradictory supervision to remove. The filter remains correct and worth keeping: it is scoped to exactly the case that actually causes harm, and its cost is negligible. Verified by 8 synthetic unit cases covering drop, keep-below-threshold, keep-non-furniture, never-drop-Tables, and multi-bench partial-drop.

**Known gap, accepted:** the 6,175 new Open Images images (2,630 Stairs + 3,545 Bench) are absent from `dedup_report.json`, which predates them, so `split.py` had no duplicate-group coverage for them and near-duplicates among them could straddle splits. Risk is materially lower than for the Roboflow video-frame sources the dedup work targeted — Open Images is curated and internally deduplicated — but it is a real gap. Re-running `dedup.py` is not an option (DEC-102: it would destroy DEC-089's 66,907-image RunPod coverage). `scripts/preprocess/dedup_extend_exact.py` could close the *exact*-duplicate half cheaply without a GPU, as DEC-090 did for the crosswalk source; not run here, left as the student's call.

---

## DEC-118: Stairs Boxes Recovered From Two Active Sources DEC-100 Had Stripped — Roboflow Stairs Datasets Stay Benched

- **Date:** 2026-09-06
- **Status:** Accepted (executed)
- **Related:** DEC-117 (reinstated Stairs from Open Images; this supplements it), DEC-100 (deleted these boxes), DEC-083 (which first noticed `revised_pedestrian_obstacle` carried Stairs incidentally), DEC-082 (why the dedicated Roboflow stairs sources are benched), DEC-093 (the promote-order trap this avoids)

### Context

DEC-117 restored Stairs from Open Images only. But `drop_classes.py` (DEC-100) had deleted Stairs boxes from **every** label file, including sources that are still active and whose boxes had already been through this project's own review pass. The student asked whether the `revised_pedestrian_obstacle` boxes could be revived.

Measured in `dataset/backups/pre_class_drop_20260904_023304/`, 7,370 images still hold legacy-id-5 (Stairs) boxes. Most sit in the three **dedicated** Roboflow stairs sources — `escalator_stairs` (4,429), `stairs_i2yia` (1,559), `stair_gaptw` (967) — which are benched on quality grounds and which the student explicitly ruled out. Two **active, general-purpose** sources also carried Stairs incidentally: `revised_pedestrian_obstacle` (338 images / 350 boxes) and `cv_project_hovyc` (77 / 87).

### Decision

Restore Stairs boxes from the two active sources only. New `scripts/preprocess/restore_stairs_from_predrop.py`. The three benched stairs sources are **not** touched and remain excluded.

The distinction that makes this safe: these are not stairs datasets being un-benched. They are sources already in the merged pool for other classes, whose staircases happen to have been annotated and reviewed. **Zero new images enter the pool** — this adds supervision to images already present.

### Two traps the implementation had to handle

**All 415 files also exist in `labels_reviewed/`.** `promote_reviews.py` copies `labels_reviewed/` **over** `labels/`, so patching only `labels/` would have left a live landmine: the next promote run would copy the unpatched reviewed file back and silently delete every restored box, with no error. Both directories are written. This is DEC-093's failure mode in a new guise.

**The backup carries 16-class ids.** Only legacy-id-5 lines are read and they are remapped to the current Stairs id via `get_class_id()` — never a literal (AGENTS.md:105-111). Every other line in the backup is ignored, so stale 16-class ids for other classes cannot leak back in. `LEGACY_NAMES[5] == "Stairs"` is asserted rather than assumed. Boxes are deduplicated on `(class_id, cx, cy, w, h)` at 6dp, so the script is safe to re-run.

### Result

**437 unique Stairs boxes restored across 415 images**, 0 duplicates, 0 out-of-range ids introduced. Verified present in both `labels/` and `labels_reviewed/` for both sources. Backups at `labels{,_reviewed}_bak_stairsrestore_20260906_222009`.

Cascade re-run (`dedup.py` skipped per DEC-102):

| | before (DEC-117) | after |
|---|---|---|
| Stairs candidates | 2,630 | **3,045** |
| Stairs in `dataset/final/` | 2,630 img / 3,165 boxes | **3,004 img / 3,561 boxes** |
| merged pool | 42,988 | **43,171** |
| splits | 29,949 / 6,664 / 6,375 | **30,123 / 6,645 / 6,403** |
| total boxes | 132,434 | **132,903** |
| ratio invariant | 2.64 | **2.64** |
| leakage | `[]` | `[]` |

Stairs provenance in the final pool, verified from the labels themselves rather than from config: **open_images 2,630 img / 3,165 boxes · revised_pedestrian_obstacle 297 / 309 · cv_project_hovyc 77 / 87.** No dedicated Roboflow stairs source contributes anything, which was the student's requirement.

The 338→297 shrink for `revised_pedestrian_obstacle` is expected — `merge.py` applies review `exclude` tags and cap selection after the restore, so not every patched image reaches the final pool.

`check_det_dataset()` from `/tmp` re-verified at `nc=15`, correct name order, all three split paths present.

### Also settled this session

`dedup_extend_exact.py --source open_images` was run against the new pool: it hashed all 37,106 open_images files against the merged pool and found **0 new exact-duplicate groups**. The exact-duplicate coverage gap DEC-117 flagged is therefore measured as empty rather than merely assumed small, and `dedup_report.json` was left untouched — DEC-089's RunPod coverage intact. Near-duplicate coverage for the ~6,175 new Open Images images remains genuinely unknown and is accepted as such.

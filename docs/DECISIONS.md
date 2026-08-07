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

> **Superseded (partial):** ID 3 (`Cart`) was dropped and replaced with `Pole` per DEC-021. ID 10 (`Wet Floor Sign`) was dropped and replaced with `Tricycle` per DEC-015. The remaining 13 entries and the overall 15-class framing are still authoritative — see `config/classes.yaml` for the current canonical list.

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

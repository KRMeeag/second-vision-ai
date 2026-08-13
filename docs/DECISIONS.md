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

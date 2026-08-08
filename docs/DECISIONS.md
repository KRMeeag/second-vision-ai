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

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

# PROJECT.md — Second Vision AI

## Project Overview

**Second Vision** is an IoT-based smart glass for the visually impaired. The system provides real-time spatial awareness through two complementary feedback channels:

- **Audio (TTS)**: Announces semantically identified objects — *"person left"*, *"car center"*, *"bicycle right"*
- **Haptic (vibration motors)**: Proportional vibration for all obstacles detected by depth estimation

This repository focuses on the **AI model pipeline** — producing the YOLOv8 object detection model that powers the semantic audio channel.

---

## System Context

### Where This Repository Fits

```
┌─────────────────────────────────────────────────┐
│               THIS REPOSITORY                   │
│                                                 │
│  Dataset Curation → Training → ONNX Export      │
│                                                 │
└───────────────────────┬─────────────────────────┘
                        │ .onnx file
                        ▼
              ┌─────────────────────┐
              │   Hailo Toolchain   │
              │  DFC + Model Zoo    │
              └─────────┬───────────┘
                        │ .hef file
                        ▼
              ┌─────────────────────┐
              │  Production Repo    │
              │  (second-vision)    │
              │  RPi5 + Hailo-8    │
              └─────────────────────┘
```

### Production Hardware

| Component | Specification |
|-----------|--------------|
| SBC | Raspberry Pi 5 (8GB RAM) |
| AI Accelerator | Hailo AI Hat+ (Hailo-8, 26 TOPS) |
| Camera | OV2640 USB Camera Module |
| OS | Raspberry Pi OS Trixie (64-bit) |
| Audio | Bone-conduction earphones |
| Haptic | 3× ERM vibration motors (L/C/R) |
| Motor Controller | ESP32 via USB serial |

### Production Software Stack

| Layer | Technology |
|-------|-----------|
| Inference Runtime | HailoRT + GStreamer pipeline |
| Detection Model | YOLOv8s → HEF (this repo produces this) |
| Depth Model | SC-DepthV3 → HEF (separate) |
| Application | `second_vision` Python package |
| TTS Engine | pyttsx3 / espeak-ng |

---

## Use Case

This model is part of an **assistive navigation system for visually impaired users**.

The detection model provides **semantic object identification** that complements depth estimation. While depth estimation tells the user *something is there*, the detection model tells them *what it is* — enabling meaningful audio guidance rather than generic obstacle warnings.

### Why Semantic Detection Matters

| Scenario | Depth-Only Response | With Detection |
|----------|-------------------|----------------|
| Person approaching | "Obstacle ahead" | "Person ahead, moving left" |
| Pothole in the path | Not detected (flat) | "Pothole ahead" |
| Doorway vs wall | "Wall" | "Door ahead" (navigational anchor) |
| Empty chair | "Obstacle ahead" | "Chair ahead" (a waypoint, not a hazard) |
| Tricycle at the kerb | "Obstacle ahead" | "Tricycle ahead" (may move) |

> Rows for Pedestrian Lane, Escalator/Stairs and Elevator were removed when those classes left
> the schema (DEC-100/DEC-083). The scenarios remain valid arguments for reinstating them — see
> **Dropped — and recoverable** below.

---

## Target Classes

**13 classes**, selected from user research (survey data, Figures B.6–B.11) and reduced from 16
on 2026-09-04 (DEC-100):

| ID | Class | Category | Why It Matters |
|----|-------|----------|---------------|
| 0 | Person | Dynamic hazard | 80% struggle with unpredictable crowds |
| 1 | Vehicle | Life safety | High-mass kinetic threat during outdoor transit |
| 2 | Motorcycle | Life safety | Fast, high-speed hazard common in Philippine streets |
| 3 | Pole | Static obstacle | Utility poles as common street-level collision hazard |
| 4 | Animals | Dynamic hazard | Unpredictable, low-level living obstacles |
| 5 | Shelf | Obstacle | Common mall/retail/grocery obstacle (aisle end-caps, protruding shelving) — replaces Escalator, DEC-083 |
| 6 | Doors | Navigation | 84% struggle to locate exact doors |
| 7 | Chairs | Waypoint | 84% struggle finding empty seating |
| 8 | Tables | Obstacle + Navigation | 100% cite tables as path blockers |
| 9 | Tricycle | Life safety | Common Philippine motorized three-wheeler |
| 10 | Potholes | Safety (invisible to depth) | Shallow ground anomalies |
| 11 | Trash Bins | Obstacle | Frequently relocated barriers |
| 12 | Bicycle | Life safety | Slow, silent hazard — easy to miss without engine noise |

### Dropped — and recoverable

Three classes were removed for sitting below DEC-042's 1,500-image floor. **The user need they
answer has not gone away, and none of them is permanently closed:**

| Class | Was ID | Images at drop | Floor | Why It Mattered |
|-------|--------|----------------|-------|-----------------|
| Stairs | 5 | 1,375 | 1,500 | 84% struggle with elevation changes |
| Elevator | 13 | 1,351 | 1,500 | Multi-floor transit, specific interaction |
| Pedestrian Lane | 14 | 1,099 | 1,500 | Guides safe street crossing |

All three were short on **data**, not on justification — the shortfalls are 125, 149 and 401
images. A source that clears the floor is all any of them needs. **DEC-105** records the verified
revert procedure; the 16-class state is preserved at
`dataset/backups/pre_class_drop_20260904_023304/`, though `config_loader.py` must come from git
(`2ac3cde`) rather than that backup.

Two costs before reinstating any of them: a changed class count means a **full retrain**, not a
fine-tune (new detection head), and there is **no inverse migration script**, so review work done
under 13 classes must be reconciled with restored 16-class labels by hand.

`Escalator` is a different case — its slot was reused for `Shelf` (DEC-083), so bringing it back
would be adding a new class rather than restoring one.

---

## Deployment Pipeline

Every model produced by this repository must survive this pipeline:

```
1. YOLOv8 Training (Ultralytics, this repo)
   ↓
2. ONNX Export (this repo)
   ↓
3. Hailo Dataflow Compiler (DFC)
   - Uses val split as calibration dataset
   - Requires GPU for production-quality compilation
   ↓
4. Hailo Model Zoo (HMZ)
   - Model architecture must be HMZ-compatible
   ↓
5. HEF Generation
   - Background class (ID 0) injected automatically
   - All original class IDs shift by +1
   ↓
6. hailo-apps on Raspberry Pi 5
   - Custom label JSON required (not COCO defaults)
   - Inference via HailoRT + GStreamer
```

### Hailo Runtime Class Mapping

```
Training (this repo)          Runtime (after Hailo)
─────────────────             ─────────────────────
0: Person                     0: Background (injected)
1: Vehicle                    1: Person
2: Motorcycle                 2: Vehicle
...                           ...
12: Bicycle                   13: Bicycle
```

---

## Dataset Sources

| Source | Type | Expected Contribution |
|--------|------|----------------------|
| Open Images V7 | Large-scale annotated | Primary for Person, Vehicle, Motorcycle, Bicycle, Animals, Chairs, Tables, Trash Bins |
| CrowdHuman | Person-focused detection | Secondary (volume_topup) for Person |
| ExDark | Low-light imagery | Cross-cutting augmentation for 6 classes (DEC-014) |
| Roboflow Universe | Community curated | Doors, Tricycle, Potholes, Trash Bins, Vehicle/Motorcycle secondary |
| Dataset Ninja (2 datasets) | Pothole-specific | Primary + secondary for Potholes |

All sources are mapped into the single canonical 13-class schema. Roboflow sources are pulled locally via pinned SDK versions (DEC-018r).

> Four Roboflow sources were benched by DEC-100 because each was 100% a dropped class —
> `elevator_awvus`, `stair_gaptw`, `wtf_dwvgm`, `crosswalk_detector_lz3hc`. They are benched, not
> deleted: reinstating any of the three dropped classes starts with reactivating these.

> **Dropped sources:** MS COCO 2017, Mapillary Vistas (DEC-013), Objects365 (DEC-024), Custom-collected.

---

## Success Criteria

A model is considered deployment-ready when:

- [ ] mAP@0.5 meets project threshold on the test split
- [ ] Per-class precision/recall are balanced (no class is catastrophically weak)
- [ ] ONNX export completes without errors
- [ ] Hailo DFC compilation succeeds
- [ ] HEF model runs correctly on Hailo-8 via hailo-apps
- [ ] Inference latency is acceptable for real-time use
- [ ] Custom label JSON correctly maps all 13 classes

---

## Related Repositories

| Repository | Responsibility |
|-----------|---------------|
| **second-vision-ai** (this repo) | Dataset, training, export |
| **second-vision** (production) | RPi5 runtime, threading, serial, application logic |

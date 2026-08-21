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
| Wet floor sign | Not detected (flat) | "Wet floor sign ahead" |
| Pedestrian crossing | Not detected (flat) | "Pedestrian lane ahead" |
| Escalator vs stairs | "Obstacle ahead" | "Escalator ahead" (different interaction needed) |
| Elevator doors | "Wall" | "Elevator ahead" (navigational anchor) |

---

## Target Classes

16 classes (15 confirmed + 1 possible) selected from user research (survey data, Figures B.6–B.11):

| ID | Class | Category | Why It Matters |
|----|-------|----------|---------------|
| 0 | Person | Dynamic hazard | 80% struggle with unpredictable crowds |
| 1 | Vehicle | Life safety | High-mass kinetic threat during outdoor transit |
| 2 | Motorcycle | Life safety | Fast, high-speed hazard common in Philippine streets |
| 3 | Pole | Static obstacle | Utility poles as common street-level collision hazard |
| 4 | Animals | Dynamic hazard | Unpredictable, low-level living obstacles |
| 5 | Stairs | Elevation | 84% struggle with elevation changes |
| 6 | Shelf | Obstacle | Common mall/retail/grocery obstacle (aisle end-caps, protruding shelving) — replaces Escalator, DEC-083 |
| 7 | Doors | Navigation | 84% struggle to locate exact doors |
| 8 | Chairs | Waypoint | 84% struggle finding empty seating |
| 9 | Tables | Obstacle + Navigation | 100% cite tables as path blockers |
| 10 | Tricycle | Life safety (possible) | Common Philippine motorized three-wheeler |
| 11 | Potholes | Safety (invisible to depth) | Shallow ground anomalies |
| 12 | Trash Bins | Obstacle | Frequently relocated barriers |
| 13 | Elevator | Navigation + Elevation | Multi-floor transit, specific interaction |
| 14 | Pedestrian Lane | Safety (invisible to depth) | Guides safe street crossing |
| 15 | Bicycle | Life safety | Slow, silent hazard — easy to miss without engine noise |

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
15: Bicycle                   16: Bicycle
```

---

## Dataset Sources

| Source | Type | Expected Contribution |
|--------|------|----------------------|
| Open Images V7 | Large-scale annotated | Primary for Person, Vehicle, Motorcycle, Bicycle, Animals, Chairs, Tables, Trash Bins |
| CrowdHuman | Person-focused detection | Secondary (volume_topup) for Person |
| ExDark | Low-light imagery | Cross-cutting augmentation for 6 classes (DEC-014) |
| Roboflow Universe (14 projects) | Community curated | Stairs, Escalator, Doors, Elevator, Pedestrian Lane, Tricycle, Pole, Vehicle secondary |
| Dataset Ninja (2 datasets) | Pothole-specific | Primary + secondary for Potholes |

All sources are mapped into the single canonical 16-class schema. Roboflow sources are pulled locally via pinned SDK versions (DEC-018r).

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
- [ ] Custom label JSON correctly maps all 16 classes

---

## Related Repositories

| Repository | Responsibility |
|-----------|---------------|
| **second-vision-ai** (this repo) | Dataset, training, export |
| **second-vision** (production) | RPi5 runtime, threading, serial, application logic |

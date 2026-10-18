#!/usr/bin/env python3
"""
Assemble models/current/deployment_kit_v2_cap4500/ -- the self-contained input to
the Hailo-8 HEF compile (docs/HANDOFF_v2_hailo_compile_runpod.md, Phase 0).

WHY A KIT AND NOT "JUST POINT AT THE REPO"
------------------------------------------
The compile runs on a rented x86 Linux pod that has none of this repo. Everything
the export touches -- weights, class schema, calibration images -- has to travel in
one directory, and the schema has to travel *with* the weights or the DEC-107
failure mode is live: v1 is nc=15 with Shelf at id 5, v2 is nc=14, so every id from
5 up differs by one and NOTHING downstream validates class indices against a name
list. A kit that carries its own data.yaml and classes.yaml.frozen is how that stops
being possible.

THREE THINGS THIS SCRIPT EXISTS TO GET RIGHT
--------------------------------------------
1. **The calibration images are symlinks, and tar does not follow them by default.**
   dataset/final_v2b/val/images holds 6,644 *absolute* symlinks into
   dataset/final/val/images. `tar czf` archives the links, not the bytes; on the pod
   they resolve to /Users/luna/... which does not exist, and the export then
   calibrates INT8 quantization on zero images without saying so. Every image here
   is read through its resolved target and written as a real file.

2. **640-bounded re-encode, not a raw copy (DEC-119 Decision 2).** ultralytics
   `data/base.py:257-261` resizes every image's long side to `imgsz` at load time,
   before any augmentation, and Hailo calibration goes through that same loader
   (`exporter.py:976,988`). The pod will therefore never see above 640 px, so
   shipping originals uploads ~1.0 GB that gets thrown away: raw the split is
   1,677 MiB, 640-bounded it is ~694 MiB. The arithmetic below is copied from
   scripts/train/prepare_runpod_bundle.py:173-191 so this kit and the v1 kit are
   produced by identical maths -- verified: v1's crowdhuman sample is 1200x675 in
   the source and 640x360 in the kit, which is exactly what this computes.

   DOWNSCALE ONLY. load_image() would upscale a sub-640 image; baking that in would
   inflate the kit for nothing.

3. **v2 labels, not v1's.** The DEC-127 rebuild changed most of them -- of 400 val
   label files compared, 364 differ from the v1 kit's copy. Labels come from
   dataset/final_v2b/val/labels and every class id is asserted to be in 0..13.

The kit mirrors the v1 layout file-for-file so RESTORE.md's verification snippet and
the CHECKSUM format stay interchangeable between the two.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]

RUN = REPO / "runs/detect/cap4500_yolov8s_v2-2"
SPLIT = REPO / "dataset/final_v2b"
CLASSES = REPO / "config/classes.yaml"
KIT = REPO / "models/current/deployment_kit_v2_cap4500"

# must match DEC-119 / prepare_runpod_bundle.py
IMGSZ = 640
JPEG_QUALITY = 95

NC = 14
EXPECTED_WEIGHTS_SHA = "ab3678b6995b2b023d2cceb1370b7f7cf2fca86dede89eac1b96acbb6d327f73"

# copied verbatim from the run directory
RUN_FILES = (
    "args.yaml",
    "provenance.json",
    "pip_freeze.txt",
    "results.csv",
    "eval_fp32_test_per_class.csv",
    "eval_fp32_test_summary.json",
)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _names_from_yaml(p: Path) -> dict[int, str]:
    """Minimal `names:` / `nc:` reader -- avoids a pyyaml dependency for two keys."""
    names: dict[int, str] = {}
    in_names = False
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not line.startswith((" ", "\t")):
                break                                    # dedent ends the block
            key, _, val = line.strip().partition(":")
            if not key.strip().isdigit():
                break
            names[int(key)] = val.strip().strip('"').strip("'")
    return names


def preflight() -> dict:
    """Fail loudly before writing anything. Returns facts the build needs."""
    problems: list[str] = []

    weights = RUN / "weights/best.pt"
    if not weights.is_file():
        problems.append(f"missing weights: {weights}")
    else:
        got = sha256_file(weights)
        if got != EXPECTED_WEIGHTS_SHA:
            problems.append(f"weights sha256 mismatch\n  expected {EXPECTED_WEIGHTS_SHA}\n  got      {got}")

    for f in RUN_FILES:
        if not (RUN / f).is_file():
            problems.append(f"missing run artefact: {RUN / f}")

    # schema must agree in all three places, or DEC-107 is live
    schemas = {}
    split_yaml = SPLIT / "data.yaml"
    if split_yaml.is_file():
        schemas["dataset/final_v2b/data.yaml"] = _names_from_yaml(split_yaml)
    if CLASSES.is_file():
        schemas["config/classes.yaml"] = _names_from_yaml(CLASSES)
    prov_path = RUN / "provenance.json"
    if prov_path.is_file():
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        pn = prov.get("data_yaml_contents", {}).get("names", {})
        schemas["run provenance.json"] = {int(k): v for k, v in pn.items()}

    ref_key = "run provenance.json"
    if ref_key in schemas:
        ref = schemas[ref_key]
        if len(ref) != NC:
            problems.append(f"{ref_key} has nc={len(ref)}, expected {NC}")
        for where, got in schemas.items():
            if got != ref:
                diff = {k: (ref.get(k), got.get(k)) for k in set(ref) | set(got) if ref.get(k) != got.get(k)}
                problems.append(f"class schema in {where} disagrees with {ref_key}: {diff}")
    else:
        problems.append("cannot read run provenance.json -- no schema reference to check against")

    # images: resolve every symlink, pair every image with a label
    img_dir, lbl_dir = SPLIT / "val/images", SPLIT / "val/labels"
    pairs: list[tuple[Path, Path, str]] = []
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        problems.append(f"missing split dirs under {SPLIT / 'val'}")
    else:
        dangling, unlabelled = [], []
        for entry in sorted(img_dir.iterdir()):
            if entry.name.startswith("."):
                continue                                  # .DS_Store / AppleDouble
            target = entry.resolve()
            if not target.is_file():
                dangling.append(entry.name)
                continue
            label = lbl_dir / (entry.stem + ".txt")
            if not label.is_file():
                unlabelled.append(entry.name)
                continue
            pairs.append((target, entry, entry.name))
        if dangling:
            problems.append(f"{len(dangling)} image symlinks do not resolve, e.g. {dangling[:3]}")
        if unlabelled:
            problems.append(f"{len(unlabelled)} images have no label file, e.g. {unlabelled[:3]}")

        # class ids must be in range -- a stray 14 means a v1 label leaked in
        bad: list[str] = []
        for _t, _e, name in pairs:
            lp = lbl_dir / (Path(name).stem + ".txt")
            for ln, line in enumerate(lp.read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                cid = line.split()[0]
                if not cid.isdigit() or not (0 <= int(cid) < NC):
                    bad.append(f"{lp.name}:{ln} class id {cid!r}")
                    break
            if len(bad) >= 5:
                break
        if bad:
            problems.append(f"label class ids outside 0..{NC - 1}: {bad}")

    if problems:
        print("PREFLIGHT FAILED\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    raw_bytes = sum(t.stat().st_size for t, _e, _n in pairs)
    print(f"preflight OK")
    print(f"  weights      {weights.relative_to(REPO)}  sha256 {EXPECTED_WEIGHTS_SHA[:16]}...")
    print(f"  schema       nc={NC}, identical across data.yaml / classes.yaml / provenance.json")
    print(f"  calibration  {len(pairs)} images + labels, {raw_bytes / 1048576:.0f} MiB raw before the 640 bound")
    return {"pairs": pairs, "raw_bytes": raw_bytes}


# ---------------------------------------------------------------------------
# image copy / downscale  (DEC-119; mirrors prepare_runpod_bundle.py:173-191)
# ---------------------------------------------------------------------------

def _resize_one(job: tuple[str, str]) -> tuple[str, int, int]:
    src_path, dst_path = job
    sp, dp = Path(src_path), Path(dst_path)
    with Image.open(sp) as im:
        w, h = im.size
        if max(w, h) <= IMGSZ:
            dp.write_bytes(sp.read_bytes())              # byte-identical, no re-encode
            return ("copied", w, h)
        r = IMGSZ / max(w, h)
        nw = min(math.ceil(w * r), IMGSZ)
        nh = min(math.ceil(h * r), IMGSZ)
        out = im.resize((nw, nh), Image.BILINEAR)        # cv2.INTER_LINEAR equivalent
        if dp.suffix.lower() in (".jpg", ".jpeg"):
            if out.mode not in ("RGB", "L"):
                out = out.convert("RGB")
            out.save(dp, "JPEG", quality=JPEG_QUALITY, optimize=True)
        else:
            out.save(dp)                                  # PNG stays PNG; mode preserved
    return ("resized", nw, nh)


# ---------------------------------------------------------------------------
# generated files
# ---------------------------------------------------------------------------

DATA_YAML = """\
# Self-contained data.yaml for HEF export from this kit.
# nc/names are byte-identical to what cap4500_yolov8s_v2-2 was trained with
# (lifted from provenance.json). No 'path' key -- DEC-066: ultralytics resolves
# splits relative to this file's own directory only when absent.
#
# v2 schema: Shelf dropped, nc=15 -> 14 (DEC-126). Every id from 5 up is one
# lower than in the v1 kit. Do not cross them.
train: calibration/images
val: calibration/images
test: calibration/images
nc: @@NC@@
names:
@@NAMES@@
"""

RESTORE_MD = """\
# Deployment kit — cap4500_yolov8s_v2-2 (v2)

Everything needed to compile this model to a Hailo-8 HEF, with no dependency on the
repo, the RunPod volume, or the dataset. Self-contained on purpose: if this directory
survives, the deployed model can be rebuilt.

Built by `scripts/train/build_deployment_kit_v2.py` on @@BUILT@@.

## What this model is

| | |
|---|---|
| Weights | `best.pt` — epoch @@EPOCH@@ of 100, selected by fitness (`0.1·mAP50 + 0.9·mAP50-95`) |
| Schema | **nc=14**, order fixed in `data.yaml` and `classes.yaml.frozen` |
| Test mAP@0.5 | **@@MAP50@@** (frozen test split, @@TESTIMG@@ images / @@TESTINST@@ instances) |
| Test mAP@0.5:0.95 | @@MAP95@@ |
| Trained at | 640 px, batch 32, 100 epochs, early stopping off, seed 42, yolov8s.pt |
| Git SHA | see `provenance.json` |

Per-class results are in `eval_fp32_test_per_class.csv`; the full training curve is in
`results.csv`.

**Read the verdict honestly.** `eval_fp32_test_summary.json` records
`"verdict": "FAIL"` — the primary target passed (@@MAP50@@ ≥ 0.70) but the per-class
floors did not: Person reached @@PERSON@@ against a 0.75 floor. Pole and Potholes were
expected weak going in. The kit is still the right thing to compile; the floor miss is
a result to report, not a reason to block deployment.

> **This kit is the 14-class v2 model.** The v1 kit
> (`deployment_kit_v1_cap4500/`) is nc=15 with Shelf at id 5, so **every id from 5 up
> differs by one**. The HEF emits class indices and nothing downstream validates them
> against a name list, so mixing a v1 HEF with a v2 label schema is **silent** — this
> is the DEC-107 failure mode at deployment. Never cross them.

## Environment

`pip_freeze.txt` is the authoritative record (DEC-120) — it captures the pod template's
CUDA-matched torch alongside everything pip resolved. `requirements-train.txt` in the
repo is intent, not record. The compile step needs a **separate venv**: the Hailo
Dataflow Compiler pins TF 2.18 / numpy 1.26.4 / protobuf 3.20.3, which cannot coexist
with a modern torch.

Torch must come from a **driver-matched CUDA index before ultralytics** (DEC-116), or
the DFC import dies with `cudaGetDevice() failed ... driver version is insufficient`
while **TensorFlow still tests completely healthy**. The only check that proves the
environment is `import torch` and `import hailo_sdk_client` succeeding in one process.

## Producing the two HEFs

Two are required because **conf/iou are baked into the on-chip NMS at compile time and
cannot be changed at runtime** (DEC-120).

```bash
# 1. evaluation HEF -- low threshold, for measuring the INT8 quantization gap
yolo export model=best.pt format=hailo data=data.yaml name=hailo8 \\
     imgsz=640 conf=0.001 fraction=1.0

# 2. deployment HEF -- the thresholds that actually ship
yolo export model=best.pt format=hailo data=data.yaml name=hailo8 \\
     imgsz=640 conf=0.05 iou=0.7 fraction=1.0
```

Four things that are easy to get wrong:

- **`name=hailo8` is mandatory.** The ultralytics default is `hailo8l`
  (`exporter.py:646`) — the 13-TOPS part, not the 26-TOPS AI HAT+.
- **`fraction=1.0`, never a bare fraction.** A partial fraction slices the *sorted*
  head of the split, which here is exdark + road-damage + pothole images and **zero
  open_images** — calibrating quantization on a distribution that does not resemble
  deployment (DEC-004, DEC-120).
- **`imgsz` must match training (640).** The HEF compiles for one fixed input size;
  running a different size on-device reduces accuracy.
- **Bake `conf=0.05`, not 0.25.** You can always discard a detection in software; you
  can never recover one the chip never emitted. Per-class F2-optimal thresholds run
  from 0.092 (Tables) to 0.301 (Doors), and the haptic and speech channels want
  different thresholds for the same class — a baked 0.25 destroys all of that.
  Apply per-class thresholds in the Pi-side Python instead.

`calibration/` holds all @@NIMG@@ val images the export needs, and `data.yaml` already
points at it, so the commands above work from inside this directory.

## About `calibration/`

These are **640-bounded re-encodes** of the val split, not the originals (DEC-119
Decision 2): ultralytics `data/base.py:257-261` resizes every image's long side to
`imgsz` before returning it, and Hailo calibration uses that same loader
(`exporter.py:976,988`), so the pod never sees above 640 px. @@NRESIZED@@ images were
downscaled (BILINEAR, JPEG q95; PNG kept as PNG) and @@NCOPIED@@ were already within
640 and copied byte-for-byte. Raw the split is @@RAWMIB@@ MiB; here it is
@@KITMIB@@ MiB. Lossless *for this pipeline* — and note the source tree stores these
images as absolute symlinks, so a kit built with plain `tar` instead of this script
would ship dangling links and silently calibrate on nothing.

## Validated toolchain

DFC 3.34.0 → HEF v5 → HailoRT 4.23.0 on real Hailo-8 hardware (DEC-114/116). DFC 3.33
is not needed — 3.34 can downgrade its own HEF version via
`platform_param(hef_version=4)`.

## Contents

| file | |
|---|---|
| `best.pt` | 22 MB, the weights |
| `data.yaml` | self-contained, points at `calibration/` |
| `classes.yaml.frozen` | the exact 14-class schema at training time |
| `calibration/` | @@NIMG@@ val images + labels for INT8 quantization |
| `provenance.json`, `pip_freeze.txt`, `args.yaml` | reproducibility record |
| `eval_fp32_test_*.csv/json`, `results.csv` | measured results |
| `CHECKSUM` | sha256 over path+size of every file |

## Verifying this kit

```bash
python3 - <<'PY'
import hashlib
from pathlib import Path
D = Path(".")
h = hashlib.sha256(); n = 0
for p in sorted(D.rglob("*")):
    if p.is_file() and p.name not in ("CHECKSUM", "RESTORE.md"):
        h.update(str(p.relative_to(D)).encode()); h.update(str(p.stat().st_size).encode()); n += 1
print(h.hexdigest(), n)
PY
```

Must match the digest and file count in `CHECKSUM`. On macOS, opening this directory in
Finder can add a `.DS_Store` that changes the digest — delete it before verifying.
"""


def build(facts: dict, jobs: int, make_tar: bool) -> None:
    pairs = facts["pairs"]
    KIT.mkdir(parents=True, exist_ok=True)
    img_out = KIT / "calibration/images"
    lbl_out = KIT / "calibration/labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # 1. weights + run artefacts
    shutil.copy2(RUN / "weights/best.pt", KIT / "best.pt")
    for f in RUN_FILES:
        shutil.copy2(RUN / f, KIT / f)
    shutil.copy2(CLASSES, KIT / "classes.yaml.frozen")
    print(f"  copied weights, {len(RUN_FILES)} run artefacts, classes.yaml.frozen")

    # 2. data.yaml, from the run's own schema
    prov = json.loads((RUN / "provenance.json").read_text(encoding="utf-8"))
    names = {int(k): v for k, v in prov["data_yaml_contents"]["names"].items()}
    (KIT / "data.yaml").write_text(
        DATA_YAML.replace("@@NC@@", str(NC)).replace(
            "@@NAMES@@", "\n".join(f"  {i}: {names[i]}" for i in sorted(names))),
        encoding="utf-8")
    print(f"  wrote data.yaml (nc={NC}, self-contained)")

    # 3. labels
    src_lbl = SPLIT / "val/labels"
    for _t, _e, name in pairs:
        stem = Path(name).stem + ".txt"
        shutil.copy2(src_lbl / stem, lbl_out / stem)
    print(f"  copied {len(pairs)} labels")

    # 4. images -- resolved through the symlinks, 640-bounded
    img_jobs = [(str(t), str(img_out / name)) for t, _e, name in pairs]
    tally = {"resized": 0, "copied": 0}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for status, _w, _h in ex.map(_resize_one, img_jobs, chunksize=64):
            tally[status] += 1
    kit_bytes = sum(p.stat().st_size for p in img_out.iterdir())
    print(f"  images: {tally['resized']} downscaled, {tally['copied']} copied unchanged "
          f"-- {facts['raw_bytes'] / 1048576:.0f} MiB -> {kit_bytes / 1048576:.0f} MiB")

    # 5. RESTORE.md
    ev = json.loads((RUN / "eval_fp32_test_summary.json").read_text(encoding="utf-8"))
    person = next((c["map50"] for c in ev.get("per_class_floors", []) if c["class"] == "Person"), None)
    n_test = len(list((SPLIT / "test/images").iterdir())) if (SPLIT / "test/images").is_dir() else 0
    subs = {
        "@@BUILT@@": datetime.now().strftime("%Y-%m-%d"),
        "@@EPOCH@@": str(facts.get("best_epoch", "?")),
        "@@MAP50@@": f"{ev['map50']:.4f}",
        "@@MAP95@@": f"{ev['map50_95']:.4f}",
        "@@PERSON@@": f"{person:.4f}" if person is not None else "see per-class CSV",
        "@@TESTIMG@@": f"{n_test:,}",
        "@@TESTINST@@": f"{ev.get('ground_truth_instances', 0):,}",
        "@@NIMG@@": f"{len(pairs):,}",
        "@@NRESIZED@@": f"{tally['resized']:,}",
        "@@NCOPIED@@": f"{tally['copied']:,}",
        "@@RAWMIB@@": f"{facts['raw_bytes'] / 1048576:,.0f}",
        "@@KITMIB@@": f"{kit_bytes / 1048576:,.0f}",
    }
    md = RESTORE_MD
    for k, v in subs.items():
        md = md.replace(k, v)
    (KIT / "RESTORE.md").write_text(md, encoding="utf-8")
    print("  wrote RESTORE.md")

    # 6. CHECKSUM -- same shape as the v1 kit so the snippets are interchangeable
    stray = [str(p.relative_to(KIT)) for p in KIT.rglob("*")
             if p.is_file() and p.name.startswith(".")]
    if stray:
        print(f"\nABORT: dotfiles in the kit would corrupt the digest: {stray[:5]}", file=sys.stderr)
        sys.exit(1)
    h = hashlib.sha256()
    n = 0
    for p in sorted(KIT.rglob("*")):
        if p.is_file() and p.name not in ("CHECKSUM", "RESTORE.md"):
            h.update(str(p.relative_to(KIT)).encode())
            h.update(str(p.stat().st_size).encode())
            n += 1
    (KIT / "CHECKSUM").write_text(
        f"{h.hexdigest()}  path+size digest over {n} files (excludes CHECKSUM, RESTORE.md)\n",
        encoding="utf-8")
    print(f"  CHECKSUM over {n} files: {h.hexdigest()[:16]}...")

    total = sum(p.stat().st_size for p in KIT.rglob("*") if p.is_file())
    print(f"\nkit built: {KIT.relative_to(REPO)}  ({total / 1048576:.0f} MiB, {n + 2} files)")

    # 7. optional tar -- written with tarfile, which unlike macOS tar never emits
    #    AppleDouble ._* companions that `tar tzf` on macOS then refuses to show you
    if make_tar:
        tar_path = REPO / "v2_kit.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(KIT, arcname="v2_kit", recursive=True)
        print(f"tar: {tar_path.relative_to(REPO)}  ({tar_path.stat().st_size / 1048576:.0f} MiB, root 'v2_kit/')")


def best_epoch() -> int | None:
    import csv
    p = RUN / "results.csv"
    if not p.is_file():
        return None
    rows = list(csv.DictReader(p.open()))
    if not rows:
        return None
    k50 = next(c for c in rows[0] if "mAP50(B)" in c)
    k95 = next(c for c in rows[0] if "mAP50-95(B)" in c)
    best = max(rows, key=lambda r: 0.1 * float(r[k50]) + 0.9 * float(r[k95]))
    return int(float(best["epoch"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="delete and rebuild an existing kit")
    ap.add_argument("--dry-run", action="store_true", help="run preflight only, write nothing")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--tar", action="store_true", help="also write v2_kit.tar.gz for upload")
    args = ap.parse_args()

    facts = preflight()
    facts["best_epoch"] = best_epoch()
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    if KIT.exists():
        if not args.force:
            print(f"\n{KIT.relative_to(REPO)} already exists -- pass --force to rebuild", file=sys.stderr)
            sys.exit(1)
        shutil.rmtree(KIT)
        print(f"  removed existing {KIT.relative_to(REPO)}")

    print(f"\nbuilding with {args.jobs} workers")
    build(facts, args.jobs, args.tar)


if __name__ == "__main__":
    main()

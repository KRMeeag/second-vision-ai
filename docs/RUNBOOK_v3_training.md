# Runbook — v3 three-arm training on RunPod

Headless, resumable, survives losing your laptop. Order is **permissive → loose**, with
**strict held back** until Comet tells you whether it is worth 3 more GPU-hours.

---

## 1. Start a GPU pod

* **RTX 4090** (v2's GPU — 100 epochs in 3.02 h at these settings)
* Attach the **EU-RO-1 network volume** at `/workspace`
* A PyTorch image; the pre-flight will verify the exact stack

Everything the run needs is already on the volume. Nothing to upload.

## 2. Install the pinned stack

```bash
pip install 'ultralytics==8.4.118' comet_ml
```

The pin is not cosmetic. v2 was trained on 8.4.118; a different version changes
augmentation and NMS defaults, and the three arms would no longer be comparable to
the 0.7118 result the whole experiment is anchored to. Pre-flight **fails** on a
mismatch rather than letting you discover it 9 hours later.

## 3. Comet (optional but this is how you will check in)

```bash
export COMET_API_KEY=...            # from .env.runpod on your Mac
export COMET_PROJECT_NAME=...
```

Without these the run still works; it just will not stream metrics, and you would
have to read `--status` instead. Pre-flight warns rather than failing.

> This key has not been rotated and a fragment leaked into diagnostic output in an
> earlier session. That is a known, accepted risk — recorded here, not re-litigated.

## 4. Launch

```bash
/workspace/scripts_v3/run_v3.sh
```

Prints `Detached. Safe to disconnect.` and returns. **Close the laptop.**

The script re-execs itself under `setsid`, leaving your SSH session's process group,
so a dropped connection cannot SIGHUP it. All output goes to `/workspace`, never the
container disk — a pod death costs the current arm, not the run.

Runs, in order:

| phase | what | time |
|---|---|---|
| pre-flight | version pin, datasets resolve, val/test identical across arms, 3 GiB writable | ~2 min |
| baseline | existing v2 weights scored on the new test labels | ~2 min |
| `v3p` | **PERMISSIVE** — 118,279 train boxes | ~3.0 h |
| `v3l` | **LOOSE** — 140,801 train boxes | ~3.0 h |
| compare | summary table | instant |

## 5. Check in

```bash
/workspace/scripts_v3/run_v3.sh --status
```

Phases done, results so far, and whether a training process is alive. Safe at any
time — it only reads.

```bash
tail -f /workspace/runs_v3/logs/driver.log   # live
tail -f /workspace/runs_v3/logs/v3p.log      # one arm
```

## 6. Decide on strict

After permissive and loose, compare in Comet or via `--status`. If strict still looks
worth it:

```bash
/workspace/scripts_v3/run_v3.sh v3s
```

Finished phases are skipped via their `.done` markers, so this only runs strict.

## 7. Retrieve before tearing the pod down

```bash
# on your Mac
scp -r -P <port> -i ~/.ssh/id_ed25519 \
    root@<host>:/workspace/runs_v3 ./runs/
```

Results are on the volume and survive the pod, but pulling them locally is the
durable copy — the volume has been cut off once before (DEC-137).

---

## If something goes wrong

**Pre-flight fails.** It names the check. Nothing has been spent. Most likely the
ultralytics pin.

**An arm fails mid-run.** The driver logs it and moves to the next arm rather than
dying. Re-running picks up only what is missing.

**You want to start an arm over.** Delete its marker and its run directory:

```bash
rm /workspace/runs_v3/v3p.done
rm -rf /workspace/runs_v3/v3_v3p
```

**Volume fills.** Pre-flight checks 3 GiB up front precisely because an exhausted
quota on this MooseFS volume writes **0-byte files instead of failing** (DEC-142).
If it happens mid-run, free space and re-run; completed arms are skipped.

---

## What is being compared

Identical in every arm: model (`yolov8s.pt`), 100 epochs, batch 32, imgsz 640,
seed 42, deterministic, `close_mosaic=10`, lr0 0.01, and **byte-identical val and
test labels**. The only variable is the training labels.

| arm | train boxes | rule |
|---|---:|---|
| PERMISSIVE | 118,279 | tiers A, C + tier D partial agreement + tier E strict agreement (v2_conf ≥ 0.25) |
| LOOSE | 140,801 | every proposal surviving the conflict filters — **no v2 corroboration at all** |
| STRICT | 113,965 | tiers A, C only |

The question loose answers: **22,522 extra boxes with no second opinion — does the
volume outweigh the noise?** If loose scores below permissive, it does not.

Five classes — Doors, Tricycle, Potholes, Trash Bins, Stairs — are **identical in all
three arms**. No COCO class maps to them, so yolov8x could never propose them. Any
difference between arms comes from the other eight.

## Caveat on the baseline number

v2 is a **14-class** model; the arms are 13-class. It is scored on its own schema over
the same images, so it is a **reference point, not a number you can subtract** from an
arm's mAP. The script states this in its output and in `baseline_result.json`.

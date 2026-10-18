# Runbook — v3 training on RunPod (permissive first, loose and strict optional)

Headless, resumable, survives losing your laptop.

**The default run is PERMISSIVE ALONE — one arm, ~3.2 h.** Loose and strict are added
later if the timeline allows, and they cost exactly their own 3 h each: all three arms
share **byte-identical val and test labels**, which pre-flight verifies across every
staged arm whether or not it is being trained that day. So an arm trained next month is
directly comparable to one trained today, and nothing is re-run to add it.

Permissive first is not a fallback. It is the arm the experiment is actually about —
the agreement boundary, where tier D/E boxes that v2 corroborated get included. Strict
and loose bracket it on either side. If only one run happens, this is the one worth
having.

---

## Run the wizard

```bash
scripts/train/runpod_v3_wizard.sh
```

Ten stages, each one screen, with gates that refuse to advance until the check
passes. It opens every RunPod and Comet URL, captures the pod IP and your Comet key
into `.env.runpod`, uploads and checksum-verifies all three arms, runs
`setup_pod_v3.sh` and launches the run — then, after training, pulls the results down
and **refuses to let you terminate the pod until they are verified on this laptop**.

```bash
START_AT=9 scripts/train/runpod_v3_wizard.sh    # come back for retrieve + terminate
ARM=v3l START_AT=3 scripts/train/runpod_v3_wizard.sh   # add an arm on a new pod
```

The rest of this document is the same procedure by hand, and the reference for when a
gate fails. **Prefer the wizard** — a runbook can tell you to pull results before
terminating a pod; only a gate can stop you.

---

## 0. Before you start a pod — upload the arms

The three arms are tiny (5.1–5.6 MB each): a flat `labels/` directory plus three text
files of image paths. The 109,034 images they point at are already on the volume and
are shared through a single symlink, because the MooseFS quota counts **directory
entries**, so materialising images per-arm would cost ~4 GiB each (DEC-142).

```bash
# on your Mac
python3 scripts/train/stage_v3_arms.py --apply        # writes dataset/staging/*.tar.gz
source scripts/train/podenv.sh
podssh "mkdir -p /workspace/staging_v3"
scp -P "$GPU_POD_PORT" -i "$SSH_KEY" dataset/staging/v3?.tar.gz \
    root@"$GPU_POD_HOST":/workspace/staging_v3/
```

`stage_v3_arms.py` verifies each archive after writing it — it re-opens the tarball,
parses the `data.yaml` inside, and checks the label count and every split listing.
That check exists because an earlier version built the `data.yaml` by slicing the
source file's lines and **dropped the `names:` key itself**, emitting something that
was not valid YAML at all. All three arms were unloadable and nothing downstream
could have trained on them.

Also commit and push first. The pod gets its code with `git pull`, so anything
uncommitted does not exist there.

## 1. Start the GPU pod

* **RTX 4090** (v2's GPU — 100 epochs in 3.02 h at these settings)
* Attach the **EU-RO-1 network volume** at `/workspace`
* A PyTorch template (it ships a CUDA-matched torch — this matters, see step 2)
* Expose TCP 22, and HTTP 6006 if you want live TensorBoard

## 2. Two commands

```bash
ssh -p <port> -i ~/.ssh/id_ed25519 root@<host>
export COMET_API_KEY=...            # from .env.runpod on your Mac
export COMET_PROJECT_NAME=...
/workspace/second-vision-ai/scripts/train/run_v3.sh
```

That is it. The driver runs setup in the foreground so you see it pass or fail, then
prints `Detached. Safe to disconnect.` and returns. **Close the laptop.**

| phase | what | time | on failure |
|---|---|---|---|
| 0 setup | `setup_pod_v3.sh` — volume check, `requirements-train.txt`, `YOLO_CONFIG_DIR`, pinned `runs_dir`, CUDA + version gate, arms laid out, stale caches moved aside | ~2 min | stops, foreground, nothing detached |
| 1 pre-flight | 12 sections, **every staged arm** verified; full label scan | ~4 min | stops, nothing spent |
| 2 smoke | `train.py --smoke` — 2 epochs on 2%, then a real test-split eval | ~4 min | **stops** — the same path runs for 3 h next |
| 3 baseline | v2 weights scored on the new test labels, remapped into v2's schema | ~2 min | continues (reference point, not a gate) |
| 4 `v3p` | **PERMISSIVE** — 118,279 train boxes | ~3.0 h | logs and stops |
| 5 compare | summary table, and what is still untrained | instant | — |

Pre-flight verifies all three staged arms even though only one trains. That is
deliberate: check #7 (val/test label contents identical **across** arms) is
meaningless with one arm in hand, and it is exactly what licenses comparing a later
run to this one. Scanning the other two costs about a minute.

Each phase drops a `.done` marker under `/workspace/runs_v3/`. Re-running skips what
finished, so a dead arm costs that arm, not the run.

The script re-execs itself under `setsid`, leaving your SSH session's process group,
so a dropped connection cannot SIGHUP it. All output goes to `/workspace`, never the
container disk — a pod death costs the current arm, not the run.

### Why setup is its own phase and not a comment in this file

It is stage 12 of `scripts/train/runpod_wizard.sh`, the procedure both previous runs
used, lifted out so it can run unattended. A runbook can tell you to pin `runs_dir`;
only a script can stop you training without it. In particular:

* **`requirements-train.txt` never names torch.** The template ships a CUDA-matched
  build; letting pip resolve torch usually swaps in a CPU-only wheel, and training
  then runs ~50× slower and **reports no error at all**. Setup installs that file and
  then separately gates on `torch.cuda.is_available()`.
* **`runs_dir` defaults to the relative string `runs`.** Launched from anywhere but
  the repo root, six hours of weights land on the container disk, which is destroyed
  with the pod. Setup pins it to the volume and puts `YOLO_CONFIG_DIR` there too, so
  the setting survives a pod restart.
* **`ultralytics==8.4.118` exactly.** v2 was trained on it. A minor bump can move
  resize, EXIF or NMS behaviour partway through an ablation.
* **A surviving `labels.cache` is a correctness bug, not a stale file.** Ultralytics
  keys that cache on the image+label **path list**, not on contents. Re-staging an arm
  rewrites label contents under identical filenames, so the cache still looks valid
  and would serve the previous arm's boxes for the whole run. Setup renames any it
  finds (`.cache.stale-<timestamp>`) and pre-flight **fails** if one is present.

Setup is idempotent — re-running it after a pod restart, which wipes the container
disk and takes pip and `~/.bashrc` with it, is the intended way to use it.

## 3. Check in

```bash
/workspace/second-vision-ai/scripts/train/run_v3.sh --status
```

Phases done, the results table so far, whether a training process is alive, and GPU
utilisation. Safe at any time — it only reads.

```bash
tail -f /workspace/runs_v3/logs/driver.log   # the whole run
tail -f /workspace/runs_v3/logs/v3p.log      # one arm
tail -f /workspace/runs_v3/logs/smoke.log    # the smoke run's output
```

Comet is the easier check-in. `train.py` prints `logger comet: ARMED` or `NOT ARMED`
at the start of every run, and it authenticates the key for real rather than testing
that the variable is set — a typo'd or revoked key otherwise reports armed and logs
nothing for six hours.

> The Comet key has not been rotated and a fragment leaked into diagnostic output in
> an earlier session. That is a known, accepted risk — recorded here, not re-litigated.

## 4. Adding an arm later

```bash
/workspace/second-vision-ai/scripts/train/run_v3.sh v3l       # loose,  ~3.0 h
/workspace/second-vision-ai/scripts/train/run_v3.sh v3s       # strict, ~3.0 h
/workspace/second-vision-ai/scripts/train/run_v3.sh v3l v3s   # both,   ~6.0 h
```

Finished phases are skipped via their markers, so this runs only the new arm. Setup
re-runs (two minutes, idempotent — necessary if the pod was recreated, since the
container disk takes pip and `~/.bashrc` with it) and the smoke phase is already done.

`compare_v3.py` prints the arms that have run plus the ones that have not, with the
exact command to add them, so `--status` always says where the experiment stands.

**Which to add first, if only one more fits.** `v3l` (loose). Permissive-vs-loose is
the question with a real unknown: 22,522 extra boxes carrying **no second opinion at
all**. Strict-vs-permissive is the narrower question of 4,314 boxes at the agreement
boundary, and permissive already includes them — if permissive beats the v2 baseline,
that half is substantially answered. Loose is the one that could go either way.

The decision is yours to make from Comet, not from this file. But that is the ordering
the numbers support.

## 5. Retrieve before tearing the pod down

```bash
# on your Mac
source scripts/train/podenv.sh
scp -r -P "$GPU_POD_PORT" -i "$SSH_KEY" root@"$GPU_POD_HOST":/workspace/runs_v3 ./runs/
scp -r -P "$GPU_POD_PORT" -i "$SSH_KEY" \
    root@"$GPU_POD_HOST":/workspace/second-vision-ai/runs/detect ./runs/
```

Two locations on purpose: `/workspace/runs_v3/` holds the driver's markers, logs and
result JSONs; `runs/detect/<name>/` holds what ultralytics wrote — weights,
`results.csv`, `args.yaml`, `provenance.json`, `pip_freeze.txt` and the per-class eval
CSV. The second is the one the paper cites.

Results survive the pod on the volume, but pulling them locally is the durable copy —
the volume has been cut off once before (DEC-137).

---

## What pre-flight actually checks

Each of these corresponds to something that has already gone wrong here.

| # | check | the failure it catches |
|---|---|---|
| 1 | `/workspace` is not an overlay filesystem | volume not attached; weights die with the pod |
| 2 | `ultralytics==8.4.118`, torch sees the GPU | CPU-only wheel, ~50× slower, no error |
| 3 | `runs_dir`/`weights_dir`/`datasets_dir` under `/workspace` | output on the container disk |
| 4 | `config/classes.yaml` loads at nc=13 | — |
| 5 | per arm: valid YAML, `names` present, schema matches, no `path` key, `check_det_dataset` resolves **from /tmp**, exact split counts, sampled images and derived label paths exist, `images` symlink resolves, 43,170 label files | the dropped-`names:` bug; DEC-066; DEC-135 |
| 6 | every label line: 5 fields, parseable, class id in 0–12, coords in [0,1], w/h > 0 | malformed or truncated labels |
| 7 | **val and test label CONTENTS hashed per arm and identical; train contents all distinct** | the evaluation yardstick moving between arms |
| 8 | train box counts equal 113,965 / 118,279 / 140,801 | the figures in this file and in DECISIONS.md being wrong |
| 9 | no `*.cache` under any arm | ultralytics serving another arm's boxes |
| 10 | 3 GiB write probe | DEC-142 — an exhausted quota writes **0-byte files** instead of failing, and `df` reports the whole 2.2 P cluster, never the quota |
| 11 | v2 args reference and v2 weights present | parity and baseline silently skipped |
| 12 | Comet key **authenticates** | a present-but-dead key logging nothing |

#7 is the one that matters most. The experiment is "same yardstick, different training
labels"; a listing-only check would not notice the boxes changing inside identically
named files.

Checks 1–5 and 9–12 take seconds. The full label scan (6–8) reads 129,510 files on
network storage and takes a few minutes; `--quick` samples the *validation* of train
labels while still hashing every file, because the digest is the part that proves
comparability.

## If something goes wrong

**Setup fails.** It names the check and nothing was detached. Most likely the volume
is not attached, or the repo was not pushed.

**Pre-flight fails.** It names the check. Nothing has been spent.

**The smoke run fails.** This stops the driver on purpose. The smoke run exercises
the entire path — dataloader, AMP, both loggers, checkpoint writing, plots, the schema
gate, `provenance.json`, `pip_freeze.txt`, `evaluate.py`, and the v2 hyperparameter
parity check — in about four minutes. A failure there is the same failure three hours
in.

**An arm fails mid-run.** The driver logs it and moves to the next arm rather than
dying. Re-running picks up only what is missing.

**You want to start an arm over.** Delete its marker and its run directory:

```bash
rm /workspace/runs_v3/v3p.done
rm -rf /workspace/second-vision-ai/runs/detect/v3_v3p
```

**Resume a partially trained arm instead.** Checkpoints are every 10 epochs:

```bash
cd /workspace/second-vision-ai
python3 scripts/train/train.py --resume runs/detect/v3_v3p/weights/last.pt
```

**Volume fills.** Pre-flight's 3 GiB probe runs up front for this reason. If it
happens mid-run, free space and re-run; completed arms are skipped.

**You re-staged an arm after pre-flight already passed.** Pre-flight is skipped on a
later run because `preflight.done` exists, so a rebuilt arm would go in unverified —
and a `labels.cache` written under the old labels would survive. Clear the marker:

```bash
rm /workspace/runs_v3/preflight.done
/workspace/second-vision-ai/scripts/train/run_v3.sh v3l
```

This is the one case where re-running pre-flight is not optional. Its four minutes are
the only thing standing between a re-stage and an arm scored on the wrong boxes.

---

## What is being compared

Identical in every arm: model (`yolov8s.pt`), 100 epochs, batch 32, imgsz 640, seed 42,
deterministic, `close_mosaic=10`, lr0 0.01, patience effectively disabled, and
**byte-identical val and test labels**. The only variable is the training labels.

| arm | train boxes | rule |
|---|---:|---|
| PERMISSIVE | 118,279 | tiers A, C + tier D partial agreement + tier E strict agreement (v2_conf ≥ 0.25) |
| LOOSE | 140,801 | every proposal surviving the conflict filters — **no v2 corroboration at all** |
| STRICT | 113,965 | tiers A, C only |

The question loose answers: **22,522 extra boxes with no second opinion — does the
volume outweigh the noise?** If loose scores below permissive, it does not.

With permissive alone, the comparison is against the v2 baseline rather than against
another arm: did adding 4,314 agreement-boundary boxes and dropping Pole move the model
at all? That is a complete, reportable result on its own. The arms exist to explain
*why*, not to make the first number meaningful.

Five classes — Doors, Tricycle, Potholes, Trash Bins, Stairs — are **identical in all
three arms**. No COCO class maps to them, so yolov8x could never propose them. Any
difference between arms comes from the other eight.

### Patience, and why it is not 20

`config/training.yaml` says `patience: 20`. It never reaches the run: `train.py`
defaults `--patience 0`, which maps to 100000 — above `epochs`, so it never fires.
That is deliberate. `close_mosaic=10` disables mosaic for the last 10 epochs and
reliably buys a mAP bump. An arm that early-stopped at epoch 62 would never get that
phase while an arm running to 100 does, so the arms would differ in **schedule** as
well as in data, and the ablation would stop measuring what it claims to.

`config/training.yaml` also says `batch: 16`; `train.py --batch` defaults to 32, which
is what v2 used. **Read the config alone and you would get both of these wrong.** That
is why parity is checked rather than asserted — see below.

### Hyperparameter parity with v2 is checked, not assumed

`scripts/train/check_args_parity.py` diffs each run's `args.yaml` against
`scripts/train/v2_args_reference.yaml`, which is v2's own `args.yaml` committed
verbatim. It runs automatically after every arm and after the smoke run, and its
verdict lands in `<arm>_result.json` as `args_parity_with_v2`.

Most of the resolved set — `workers`, `deterministic`, `close_mosaic`, `lr0`, `mosaic`,
`erasing` — is specified **nowhere in this repo**. It is ultralytics 8.4.118's
defaults. A version bump that changes one of them changes the experiment with no diff
to review. This is the check that turns the pin from a hope into a test.

It is reported, never fatal: a recorded, intentional change should not abort a
finished run. But `compare_v3.py` marks the arm with `!!` so a schedule difference is
never written up as a label effect.

## The baseline number

v2 is a **14-class** model; the arms are 13-class. Handing v2's weights a 13-class
`data.yaml` does not fail — it misaligns every class above the dropped Pole, so every
Pothole in the test set would be scored against v2's Tricycle output. That is DEC-107
again, one layer out.

So `baseline_eval.py` builds a **shadow dataset**: the arms' test labels remapped up
into v2's own 14-class schema, **by name**, never by arithmetic. The remap is lossless
in this direction because the new labels contain no Pole boxes at all. A canonical
class name missing from v2's head aborts the script rather than guessing.

What remains, and is stated in the output and in `baseline_result.json`: v2's head
still carries a Pole output with no Pole ground truth, so its Pole predictions land in
the background column as false positives, and the overall mAP is averaged over 14
classes of which one cannot score. **It is a reference point biased downward, not a
number to subtract.** Compare per-class AP50 across the 13 shared classes instead.

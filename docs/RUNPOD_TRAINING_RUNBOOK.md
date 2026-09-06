# RunPod Training Runbook — cap4500 baseline

Operational companion to **DEC-119** (bundle construction), **DEC-120** (run
configuration) and **DEC-026** (training on RunPod).

> **Drive this with the wizard, not by hand:**
> ```bash
> scripts/train/runpod_wizard.sh          # 14 stages, Phases 1-4
> START_AT=6 scripts/train/runpod_wizard.sh   # resume (e.g. at the upload)
> ```
> The wizard opens each console page, captures pod IPs into `.env.runpod`, and
> **gates** the steps this document can only advise on — it will not offer to
> terminate the CPU pod until `verify_bundle.py` has passed, and will not let you
> start a 100-epoch run until `torch.cuda.is_available()` is true.
>
> This file stays the reference: read it when something breaks, and cite it in the
> thesis. Phase 5 (the multi-hour run itself) is manual by design — there is
> nothing to step through.

**What is being shipped**

| | |
|---|---|
| Archive | `dataset/bundle.tar` |
| Size | 5,464,991,744 bytes (5.09 GiB) |
| sha256 | `f71d355e7d394cbcc1edf5405a04ee2c2f549605b1ef8491c7b6c81cd5d249cc` |
| Contents | 54,514 images + 54,514 labels + 3 manifests + meta, `nc=15` |
| Transfer estimate | **~2.2 h** at the measured ~0.7 MB/s |
| First run | cap 4500 — train 30,123 / val 6,644 / test 6,403, box ratio 11.95 |

---

## Phase 0 — before you provision (local, costs nothing)

**0.1 Commit and push.** The pod clones the repo from GitHub, so anything
uncommitted does not exist as far as the pod is concerned. Outstanding right now:

```
.gitignore                        # bundle.tar exclusion (see 0.2)
docs/DECISIONS.md                 # DEC-119
docs/RUNPOD_TRAINING_RUNBOOK.md   # this file
requirements-train.txt            # pinned pod environment
scripts/train/verify_bundle.py    # REQUIRED on the pod
scripts/train/upload_bundle.sh
```

`verify_bundle.py` is the one that actually blocks: without it there is no way to
prove the transfer landed intact.

**0.2 Why `.gitignore` changed.** `dataset/bundle/` was already ignored, but
`dataset/bundle.tar` was not. A 5.1 GB file is far over GitHub's 100 MB push
limit, so a stray `git add .` would have built a commit that cannot be pushed and
has to be unpicked from history — the same failure the `*.pt` rule already guards
against.

**0.3 Optional but recommended — install GNU rsync.**

```bash
brew install rsync
```

macOS ships **openrsync**, which announces itself as "rsync version 2.6.9
compatible" but rejects `--append-verify` and `--info=progress2`. Both work for
this transfer, but they resume differently: GNU rsync appends from the partial's
end, while openrsync delta-transfers against it (correct, just more I/O per
retry). `upload_bundle.sh` detects which you have and adjusts — this step only
makes retries cheaper.

---

## Phase 1 — provision

**1.1 Pick the datacenter FIRST.** Network volumes are datacenter-scoped and a pod
can only mount a volume in its own datacenter. Creating the volume somewhere with
no RTX 4090 stock means deleting it and re-uploading. Confirm 4090 availability in
the region *before* creating anything.

**1.2 Create a 30 GB network volume** in that datacenter (~$2.10/month at the
$3.50/50 GB rate you checked).

<details>
<summary>Where the 30 GB goes</summary>

| item | size |
|---|---:|
| `bundle.tar`, kept as backup | 5.1 GB |
| extracted `bundle/` | 5.0 GB |
| repo clone | <0.1 GB |
| `final_cap4500/` (hardlinks — no image bytes) | ~0.02 GB |
| `runs/` — best + last + 10 periodic checkpoints, ×3 arms | ~0.8 GB |
| ONNX + HAR + HEF, ×2 thresholds | ~0.3 GB |
| **steady state** | **~11.4 GB** |

20 GB would fit, but 30 GB leaves room for all three ablation arms plus exports
for $0.70/month more. Python packages do **not** count — they live in the
container image, not the volume.
</details>

**1.3 Deploy a pod with the volume attached.** The volume mounts at `/workspace`.

Use the **cheapest CPU pod** your datacenter offers for the upload, not the 4090.
Two hours of transfer on a GPU pod burns ~$0.75–0.97 of idle GPU; a CPU pod is
~$0.22–0.44. If CPU pods are not offered there, use the GPU pod — it is under a
dollar, not worth blocking on.

**The partial transfer is safe either way.** It lands on the network volume, which
survives pod termination. A reclaimed or stopped pod costs you a new pod, not the
bytes already uploaded.

**1.4 Add your SSH key BEFORE starting the pod.** RunPod injects public keys at pod
*start*. Adding the key afterwards requires a restart.

```bash
cat ~/.ssh/id_ed25519.pub    # paste into RunPod → Settings → SSH Public Keys
```

**1.5 Use "SSH over exposed TCP", not the proxy.** The pod's Connect panel offers
two forms. The proxy form (`ssh <id>@ssh.runpod.io`) is a restricted shell with no
rsync or scp on the far side — it cannot receive this transfer. You need the
direct form, `ssh root@<IP> -p <PORT>`, which requires the template to expose TCP
port 22. Note the IP and port.

---

## Phase 2 — transfer (~2.2 h, unattended)

```bash
scripts/train/upload_bundle.sh --host <IP> --port <PORT>
```

It retries with backoff until the transfer completes, so it can be left running
through a dropped connection, and it compares the remote sha256 against the local
one before reporting success. Stop only when it prints **CHECKSUM MATCH**.

If it cannot connect, the cause is almost always 1.4 (key added after pod start)
or 1.5 (proxy host instead of direct TCP).

> **Do not rebuild `dataset/bundle.tar` while a transfer is in flight.** Both
> resume modes assume the source file is immutable.

---

## Phase 3 — land it and prove it

On the pod:

```bash
cd /workspace
sha256sum bundle.tar          # must equal f71d355e...  (upload_bundle.sh already
                              # checked this; costs 30 s to be certain)
tar -xf bundle.tar

git clone https://github.com/KRMeeag/second-vision-ai.git
python3 second-vision-ai/scripts/train/verify_bundle.py --bundle /workspace/bundle
```

Expected: `PASS - bundle is intact and every manifest resolves.` (~1 s).

**Keep `bundle.tar`.** It is 5.1 GB of a 30 GB volume and it is the only copy on
that side of a 2.2-hour uplink. If `bundle/` is ever damaged, extracting again
costs seconds instead of an afternoon.

**Clone into `/workspace`, never `/root` or `~`.** Only `/workspace` is the network
volume. A repo — and therefore a `runs/` directory full of trained weights — placed
anywhere else is on the container disk and is destroyed when the pod is terminated.

**3.1 Materialise the condition here, on the cheap pod.**

```bash
python3 second-vision-ai/scripts/train/materialize_condition.py \
  --bundle /workspace/bundle --cap 4500
```

`verify_bundle.py` and `materialize_condition.py` are both **stdlib-only** — argparse,
hashlib, json, os, shutil, sys, pathlib — so the CPU pod needs nothing installed.

This creates **~86,000 hardlinks** (43,170 images × image + label). It is pure
metadata I/O on network-attached storage, the slowest kind, and it needs no GPU at
all. Doing it here rather than on the 4090 means the GPU pod starts with the dataset
already laid out. Verifying here also means that if something is wrong, you debug it
on the cheap pod.

**3.2 TERMINATE the CPU pod — do not Stop it.**

A *stopped* pod keeps billing for its container disk. Terminate destroys the container
disk; the network volume, the archive, the bundle, the repo clone and `final_cap4500/`
all persist. When you deploy the 4090, attach the **existing** volume in the **same
datacenter** — it mounts at `/workspace` again, so every path below still resolves.

Installed pip packages do **not** survive, which is expected: the CPU pod needed none,
and `requirements-train.txt` is installed on the GPU pod against its CUDA-matched torch.

---

## Phase 4 — GPU pod and the smoke gate

**4.1** Deploy an **RTX 4090** pod, PyTorch template, same datacenter, same volume.

**4.2 Install the pinned environment.**

```bash
cd /workspace/second-vision-ai
pip install -r requirements-train.txt
python3 -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
```

Must print `True` and `8.4.118`. `requirements-train.txt` deliberately does **not**
name torch: the template ships a CUDA-matched build, and letting pip resolve torch
usually swaps in a CPU-only wheel — training then runs on CPU with no error message
at all, and you find out from the epoch timer.

**4.3 The condition is already materialised** (step 3.1, on the CPU pod). Confirm it
survived termination — it lives on the volume, so it should:

```bash
ls /workspace/final_cap4500/{train,val,test}/images | head
python3 -c "import pathlib; print({s: len(list(pathlib.Path(f'/workspace/final_cap4500/{s}/images').iterdir())) for s in ('train','val','test')})"
```

Expect `{'train': 30123, 'val': 6644, 'test': 6403}`. If you skipped 3.1, run it now:

```bash
python3 scripts/train/materialize_condition.py --bundle /workspace/bundle --cap 4500
```

**4.4 Resolve the data.yaml from an unrelated directory.** This is the check that
DEC-066 exists for — a `data.yaml` that only resolves from its own directory works
locally and fails on the pod:

```bash
cd /tmp && python3 -c "from ultralytics.data.utils import check_det_dataset as c; \
d=c('/workspace/final_cap4500/data.yaml'); print(d['nc'], d['names'])"
```

Must print `15` and the canonical name order.

**4.5 Smoke run — 2 epochs on 2% of the data, ~2 minutes.**

```bash
cd /workspace/second-vision-ai
python3 scripts/train/train.py --data /workspace/final_cap4500/data.yaml --device 0 --smoke
```

This exercises the dataloader, AMP, checkpoint writing, plots and both loggers
before you commit to a multi-hour run. Confirm in its output:

- `schema check: data.yaml agrees with classes.yaml at nc=15`
- `logger tensorboard: ARMED` (and Comet, if you exported `COMET_API_KEY`)
- weights actually written under `runs/detect/..._smoke/weights/`

**Multiply the smoke run's epoch time by ~50** for a rough full-epoch estimate — it
trains on 2% of the data — then by 100 epochs. Expect roughly 5–7 h total.

---

## Phase 5 — the baseline run

```bash
cd /workspace/second-vision-ai
python3 scripts/train/train.py \
  --data /workspace/final_cap4500/data.yaml \
  --device 0 --batch 32 --name cap4500_yolov8s
```

Defaults that matter, and why:

- **`--patience 0`** (the default) maps to an effectively infinite patience. Early
  stopping would give one ablation arm the `close_mosaic` phase and deny it to
  another, so the arms would differ in *schedule* as well as in data. `best.pt` is
  still selected by fitness, so this costs GPU time, not model quality.
- **`--batch 32`** must be **identical** across all three arms.
- **`--save-period 10`** checkpoints every 10 epochs, so a dead pod costs at most
  10 epochs. Resume with `--resume runs/detect/cap4500_yolov8s/weights/last.pt`.

Run it under `tmux` or `nohup` — an SSH drop otherwise kills training.

`provenance.json` and `pip_freeze.txt` are written into the run directory at the
end; those, not `requirements-train.txt`, are what the paper should cite.

**Before deleting the volume, pull the results down.** From your laptop:

```bash
rsync -avP -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" \
  root@<IP>:/workspace/second-vision-ai/runs/detect/cap4500_yolov8s ./runs/detect/
```

---

## Gotcha index

| symptom | cause | fix |
|---|---|---|
| `rsync: --append-verify: unknown option` | macOS openrsync | `brew install rsync`, or let `upload_bundle.sh` pick flags |
| ssh works, rsync/scp does not | using the `ssh.runpod.io` proxy | use "SSH over exposed TCP" (1.5) |
| `Permission denied (publickey)` | key added after pod start | restart the pod (1.4) |
| Training is ~50× slower than expected | pip replaced CUDA torch with a CPU wheel | check `torch.cuda.is_available()` (4.2) |
| `runs/` gone after terminating the pod | repo cloned outside `/workspace` | clone into `/workspace` (Phase 3) |
| Still being billed for a pod you finished with | pod was **Stopped**, not **Terminated** — stopped pods keep billing for container disk | Terminate it (3.2) |
| `nc=13` or wrong class order | stale clone or stale `data.yaml` | `git pull`, re-run 4.3, then 4.4 |
| Early stopping fired mid-ablation | ultralytics invoked directly with `cfg=config/training.yaml`, which still carries `patience: 20` and `batch: 16` | always go through `scripts/train/train.py` |

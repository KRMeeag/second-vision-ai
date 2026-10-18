#!/usr/bin/env bash
# GPU-pod setup for the v3 three-arm run. Run this ONCE on the pod, before run_v3.sh.
#
#     ssh -p <port> -i ~/.ssh/id_ed25519 root@<host>
#     bash /workspace/second-vision-ai/scripts/train/setup_pod_v3.sh
#
# This is stage 12 of scripts/train/runpod_wizard.sh, lifted out so it can run
# unattended. Every step below exists because skipping it has cost this project a
# run before, and all of them are CHEAP -- the whole script is about two minutes
# against the ~6.5 GPU-hours it protects.
#
# IDEMPOTENT. Re-running after a pod restart is the expected way to use it: a
# terminated pod loses its container disk, so pip packages and ~/.bashrc are gone
# while /workspace survives. Nothing here is destructive -- a stale labels.cache is
# RENAMED, never deleted (its contents are the only record of what a prior run
# actually loaded).
set -uo pipefail

ROOT=/workspace
REPO=$ROOT/second-vision-ai
STAGING=$ROOT/staging_v3
PIN=8.4.118
ARMS=(v3p v3l v3s)

fail=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=$((fail+1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

head_ "1. /workspace must be the NETWORK VOLUME, not the container disk"
# Getting this wrong is silent and total: training writes six hours of weights to
# a disk that is destroyed with the pod.
if mountpoint -q "$ROOT" 2>/dev/null; then
    ok "/workspace is a mount point"
elif df -h "$ROOT" | tail -1 | grep -q overlay; then
    bad "/workspace is an OVERLAY filesystem -- the volume is NOT attached. Stop and redeploy the pod with it mounted."
else
    ok "/workspace is not overlay ($(df -h "$ROOT" | tail -1 | awk '{print $1}'))"
fi
df -h "$ROOT" | sed 's/^/        /'

head_ "2. Repo present and up to date"
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only 2>&1 | sed 's/^/        /' || warn "git pull failed -- continuing with the checkout as-is"
    ok "repo at $REPO ($(git -C "$REPO" rev-parse --short HEAD))"
else
    bad "no repo at $REPO -- clone it there (and ONLY there; anywhere else dies with the pod)"
fi
# The run is driven from the repo, so a checkout that predates these scripts would
# fail in confusing ways later. Check for the files this run actually needs.
for f in scripts/train/train.py scripts/train/preflight_v3.py scripts/train/run_v3.sh \
         scripts/train/train_one_arm.py scripts/train/check_args_parity.py \
         requirements-train.txt config/classes.yaml config/training.yaml; do
    [ -f "$REPO/$f" ] && ok "present: $f" || bad "missing: $f -- push it and re-run"
done

head_ "3. Pinned environment"
# requirements-train.txt deliberately does not name torch: the RunPod template
# ships a CUDA-matched build and letting pip resolve torch usually swaps in a
# CPU-only wheel. Training then runs ~50x slower and reports NO error at all --
# you find out from the epoch timer, hours in.
if [ -f "$REPO/requirements-train.txt" ]; then
    python3 -m pip install --break-system-packages --root-user-action=ignore -q \
        -r "$REPO/requirements-train.txt" >/tmp/pip_v3.log 2>&1
    rc=$?
    tail -5 /tmp/pip_v3.log | sed 's/^/        /'
    [ "$rc" -eq 0 ] && ok "pip install -r requirements-train.txt completed" \
                    || bad "pip install failed (rc=$rc) -- see /tmp/pip_v3.log"
else
    bad "cannot install: requirements-train.txt missing"
fi

head_ "4. Pin ultralytics' output paths to the VOLUME"
# The default runs_dir is the RELATIVE string 'runs', so training launched from
# anywhere but the repo root writes weights onto the container disk. YOLO_CONFIG_DIR
# on the volume makes the setting itself survive a pod restart.
mkdir -p "$ROOT/.ultralytics"
export YOLO_CONFIG_DIR="$ROOT/.ultralytics"
grep -q YOLO_CONFIG_DIR /root/.bashrc 2>/dev/null \
    || echo "export YOLO_CONFIG_DIR=$ROOT/.ultralytics" >> /root/.bashrc
python3 - >/tmp/settings_v3.log 2>&1 <<PY
from ultralytics.utils import SETTINGS
SETTINGS.update({'runs_dir': '$REPO/runs',
                 'weights_dir': '$REPO/weights',
                 'datasets_dir': '$ROOT'})
print('runs_dir     ', SETTINGS['runs_dir'])
print('weights_dir  ', SETTINGS['weights_dir'])
print('datasets_dir ', SETTINGS['datasets_dir'])
PY
rc=$?
sed 's/^/        /' /tmp/settings_v3.log
if [ "$rc" -eq 0 ]; then ok "runs_dir pinned under $REPO (on the volume)"
else bad "could not pin runs_dir -- do not launch until this works"; fi

head_ "5. CUDA survived the install, and the ultralytics pin holds"
python3 - <<PY
import sys, torch, ultralytics
print(f"        torch        {torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"        ultralytics  {ultralytics.__version__}  (pin $PIN)")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"        gpu          {p.name}  {p.total_memory // 2**20} MiB")
sys.exit(0 if torch.cuda.is_available() and ultralytics.__version__ == "$PIN" else 1)
PY
rc=$?
if [ "$rc" -eq 0 ]; then ok "torch sees the GPU and ultralytics is $PIN"
else bad "CUDA missing or ultralytics is not $PIN -- the arms would not be comparable to v2"; fi

head_ "6. Lay out the three arms on the volume"
# Each arm is a symlink+txt dataset root (DEC-142: MooseFS counts directory
# entries, so materialising 43,170 images per arm would cost ~4 GiB each).
for arm in "${ARMS[@]}"; do
    if [ ! -d "$ROOT/$arm" ]; then
        if [ -f "$STAGING/$arm.tar.gz" ]; then
            tar xzf "$STAGING/$arm.tar.gz" -C "$ROOT" && ok "$arm: extracted from $STAGING/$arm.tar.gz"
        else
            bad "$arm: neither $ROOT/$arm nor $STAGING/$arm.tar.gz exists -- upload the tarball"
            continue
        fi
    else
        ok "$arm: already laid out"
    fi

    # The symlink is created HERE and never archived. DEC-135 forbade symlinks
    # because `tar czf` archived the LINKS, which resolved to non-existent
    # /Users/luna/... paths on the pod and silently calibrated INT8 on zero
    # images. That does not apply to a link made on the pod pointing at an
    # absolute path on the same volume.
    if [ ! -e "$ROOT/$arm/images" ]; then
        ln -s "$ROOT/bundle/images" "$ROOT/$arm/images" && ok "$arm: images -> $ROOT/bundle/images"
    fi
    tgt=$(readlink -f "$ROOT/$arm/images" 2>/dev/null)
    if [ "$tgt" = "$ROOT/bundle/images" ] && [ -d "$tgt" ]; then
        ok "$arm: images symlink resolves"
    else
        bad "$arm: images symlink resolves to '$tgt' -- expected $ROOT/bundle/images"
    fi

    # A labels.cache keyed on FILE PATHS, not contents. Ultralytics hashes the
    # image+label PATH LIST, so rewriting label CONTENTS under the same filenames
    # -- exactly what re-staging an arm does -- leaves the cache looking valid.
    # It would then train on the previous arm's boxes. Renamed, never deleted:
    # the file is the only record of what a prior run actually loaded.
    for c in "$ROOT/$arm"/*.cache; do
        [ -f "$c" ] || continue
        mv "$c" "$c.stale-$(date -u +%Y%m%dT%H%M%SZ)" \
            && warn "$arm: moved stale $(basename "$c") aside (label contents changed under identical filenames)"
    done
done

head_ "7. Comet"
if [ -n "${COMET_API_KEY:-}" ]; then
    grep -q COMET_API_KEY /root/.bashrc 2>/dev/null \
        || echo "export COMET_API_KEY=$COMET_API_KEY" >> /root/.bashrc
    ok "COMET_API_KEY set (${#COMET_API_KEY} chars) and persisted to /root/.bashrc"
    [ -n "${COMET_PROJECT_NAME:-}" ] && ok "COMET_PROJECT_NAME=$COMET_PROJECT_NAME" \
        || warn "COMET_PROJECT_NAME unset -- runs land in Comet's default project"
else
    warn "COMET_API_KEY not set. Training still works and TensorBoard still logs,"
    warn "but you would have to read --status instead of checking in from Comet."
    warn "  export COMET_API_KEY=... ; export COMET_PROJECT_NAME=... ; then re-run this script"
fi

head_ "Result"
if [ "$fail" -eq 0 ]; then
    printf '  \033[32mSETUP OK\033[0m  — next:  %s/scripts/train/run_v3.sh\n\n' "$REPO"
    printf '  That runs pre-flight and a smoke run before it spends a single real\n'
    printf '  GPU-hour, then detaches. Nothing below here needs your laptop awake.\n\n'
    exit 0
else
    printf '  \033[31mSETUP FAILED\033[0m — %d check(s) above. Nothing has been spent; fix and re-run.\n\n' "$fail"
    exit 1
fi

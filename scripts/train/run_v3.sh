#!/usr/bin/env bash
# Headless driver for the v3 arm comparison.
#
# Survives losing your laptop. The launcher re-execs itself under `setsid` so the
# work is detached from the SSH session's process group -- closing the terminal, or
# the connection dropping, sends SIGHUP to a group this process is no longer in.
#
# Everything is written to /workspace (the network volume), never the container disk,
# so a pod death loses nothing but the time since the last checkpoint.
#
# RESUMABLE: each phase drops a .done marker. Re-running skips finished phases, so if
# an arm dies you lose that arm, not the run.
#
# DEFAULT IS PERMISSIVE ALONE -- ONE ARM, ~3.2 h.
#
# Not a reduced ambition: it is the only order that survives an uncertain timeline.
# Permissive is the arm the experiment is actually about (the agreement boundary), and
# every later arm is comparable to it because all three share BYTE-IDENTICAL val and
# test labels -- pre-flight verifies that across every staged arm whether or not it is
# being trained today. So adding loose or strict next week costs exactly its own 3 h
# and nothing is re-run.
#
# THE ORDER OF THE PHASES IS THE POINT. Setup and pre-flight are free; the smoke run
# is two minutes. All of it runs BEFORE the first GPU-hour, because every failure they
# catch is one this project has already paid for once.
#
#   ./run_v3.sh                 # setup + preflight + smoke + baseline + v3p   (~3.2 h)
#   ./run_v3.sh v3l             # loose later, if the timeline allows
#   ./run_v3.sh v3s             # strict later, if Comet says it is worth it
#   ./run_v3.sh v3p v3l v3s     # all three in one go (~9.5 h)
#   ./run_v3.sh --status        # print progress and exit (safe any time)
#   ./run_v3.sh --setup-only    # just run the pod setup, in the foreground
#   ./run_v3.sh --no-smoke      # skip the smoke run (only if one already passed)
#
set -uo pipefail

ROOT=/workspace
REPO=$ROOT/second-vision-ai
PROJECT=$ROOT/runs_v3
STATUS=$PROJECT/STATUS
LOGS=$PROJECT/logs
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$PROJECT" "$LOGS"

say() { printf '%s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS"; }

DO_SETUP=1
DO_SMOKE=1
POSITIONAL=()
for a in "$@"; do
    case "$a" in
        --no-setup)   DO_SETUP=0 ;;
        --no-smoke)   DO_SMOKE=0 ;;
        --setup-only) DO_SETUP=2 ;;
        --status)     POSITIONAL=("--status") ;;
        -*)           echo "unknown flag: $a" >&2; exit 2 ;;
        *)            POSITIONAL+=("$a") ;;
    esac
done

# ---- --status: read-only, safe at any time -------------------------------------
if [ "${POSITIONAL[0]:-}" = "--status" ]; then
    echo "=== STATUS ==="; [ -f "$STATUS" ] && tail -40 "$STATUS" || echo "(not started)"
    echo; echo "=== markers ==="; ls -1 "$PROJECT"/*.done 2>/dev/null || echo "(none)"
    echo; echo "=== results ==="
    python3 "$HERE/compare_v3.py" "$PROJECT" 2>/dev/null || echo "(none yet)"
    echo; echo "=== running? ==="
    # A PID FILE, not a pgrep pattern. Two reasons pattern matching is wrong here:
    #
    #   1. SELF-MATCH. pgrep -f matches full command lines, and this process's own
    #      command line is ".../run_v3.sh --status" -- so a pattern containing
    #      run_v3.sh reports the status check itself as a live run. The same
    #      mistake once killed an SSH session on this project via `pkill -f`.
    #   2. PLATFORM. `pgrep -af` means "list the full command line" on Linux
    #      procps and is not a valid option on macOS, so the behaviour of any
    #      pattern here cannot be reasoned about from a laptop.
    #
    # The detaching parent records the driver's PID below. `kill -0` asks the
    # kernel directly, and /proc confirms the PID was not recycled onto something
    # unrelated. No patterns, no ambiguity.
    if [ -f "$PROJECT/driver.pid" ]; then
        dpid=$(cat "$PROJECT/driver.pid" 2>/dev/null | tr -d '[:space:]')
        if [ -n "$dpid" ] && kill -0 "$dpid" 2>/dev/null; then
            dcmd=$(tr '\0' ' ' < "/proc/$dpid/cmdline" 2>/dev/null || echo "?")
            case "$dcmd" in
                *run_v3*) echo "  driver ALIVE  pid=$dpid  $dcmd" ;;
                *)        echo "  pid $dpid is alive but is NOT the driver (recycled): $dcmd"
                          echo "  -> treat the run as stopped" ;;
            esac
        else
            echo "  driver not running (pid file names ${dpid:-<empty>}, which is gone)"
        fi
    else
        echo "  no driver.pid -- never launched, or launched before this was added"
    fi
    echo; echo "=== anything holding the GPU? ==="
    # The authoritative check. utilization can read 0% between batches; a process
    # holding VRAM cannot hide here.
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
            || true
        nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . \
            || echo "  no process holding GPU memory"
        echo; echo "=== GPU ==="
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
                   --format=csv,noheader
    else
        echo "  (no nvidia-smi)"
    fi
    exit 0
fi

# ---- phase 0: pod setup, in the FOREGROUND -------------------------------------
# Deliberately not detached: a failure here means nothing can work, and you want to
# see it on your terminal rather than find it in a log after walking away. It is
# idempotent and takes about two minutes, so re-running it after a pod restart --
# which wipes the container disk, taking pip and ~/.bashrc with it -- is the
# intended way to use it.
if [ "${V3_DETACHED:-}" != "1" ] && [ "$DO_SETUP" -ne 0 ]; then
    echo "=== phase 0: pod setup (foreground) ==="
    bash "$HERE/setup_pod_v3.sh" 2>&1 | tee -a "$LOGS/setup.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo
        echo "Setup failed (rc=$rc). Nothing has been spent and nothing was detached."
        exit 1
    fi
    if [ "$DO_SETUP" -eq 2 ]; then
        echo "--setup-only: stopping here. Launch the run with:  $0"
        exit 0
    fi
fi

# ---- re-exec detached ----------------------------------------------------------
if [ "${V3_DETACHED:-}" != "1" ]; then
    export V3_DETACHED=1
    say "launching detached; follow with: tail -f $LOGS/driver.log"
    # NOT `$!`. setsid FORKS when it is already a process-group leader -- which
    # is exactly the case for a background job under job control -- so `$!` is the
    # PID of a setsid wrapper that exits immediately, and `kill -0` on it reports
    # the driver as dead one second after launch. The detached child records its
    # own $$ instead (below), which is correct however setsid behaves.
    setsid nohup "$0" --no-setup "$@" >>"$LOGS/driver.log" 2>&1 </dev/null &
    sleep 1
    echo
    echo "Detached. Safe to disconnect."
    echo "  progress : $0 --status"
    echo "  live log : tail -f $LOGS/driver.log"
    exit 0
fi

# From here on we ARE the detached driver, so it owns the pid file for its whole
# life. One trap covers every exit path -- normal completion, a failed pre-flight,
# a failed smoke run -- so a stopped run never reports as alive. This must not be
# set in the launcher: that process exits a second after detaching and would
# delete the file it had just written.
printf '%s\n' "$$" > "$PROJECT/driver.pid"
trap 'rm -f "$PROJECT/driver.pid"' EXIT

ARMS=("${POSITIONAL[@]}")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(v3p)

say "=== v3 run starting: arms = ${ARMS[*]} ==="

# ---- phase 1: pre-flight -------------------------------------------------------
# Every check corresponds to something that has already gone wrong here: a wrong
# ultralytics version, a data.yaml that was not valid YAML, label contents that
# differed on the evaluation split, a stale labels.cache serving another arm's
# boxes, and a volume whose exhausted quota wrote 0-byte files instead of failing.
if [ ! -f "$PROJECT/preflight.done" ]; then
    # --train names what is being trained TODAY; pre-flight verifies every arm
    # staged on the volume regardless. That is deliberate: the val/test-identical
    # check is only meaningful ACROSS arms, and it is what makes an arm trained
    # next week comparable to one trained today.
    say "[preflight] starting (verifies every staged arm; full label scan, a few minutes on MooseFS)"
    python3 "$HERE/preflight_v3.py" --train "${ARMS[*]}" 2>&1 | tee -a "$LOGS/preflight.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then say "[preflight] FAILED (rc=$rc) -- stopping, nothing spent"; exit 1; fi
    touch "$PROJECT/preflight.done"; say "[preflight] OK"
else
    say "[preflight] already done, skipping"
fi

# ---- phase 2: smoke run --------------------------------------------------------
# 2 epochs on 2% of the data, then a real evaluation on the full test split. This
# exercises the dataloader, AMP, both loggers, checkpoint writing, plots, the
# schema gate, provenance.json, pip_freeze.txt, evaluate.py and the v2
# hyperparameter parity check -- the entire path -- for about two minutes.
SMOKE_ARM=${ARMS[0]}
if [ "$DO_SMOKE" -eq 1 ] && [ ! -f "$PROJECT/smoke.done" ]; then
    say "[smoke] 2 epochs on 2% of $SMOKE_ARM (~2 min)"
    python3 "$HERE/train_one_arm.py" "$SMOKE_ARM" "$PROJECT" --smoke 2>&1 | tee -a "$LOGS/smoke.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        say "[smoke] FAILED (rc=$rc) -- stopping. A smoke failure is a real failure:"
        say "        the same path runs for three hours in the next phase."
        exit 1
    fi
    touch "$PROJECT/smoke.done"; say "[smoke] OK"
    # Surface the two lines worth reading, so --status shows them without the log.
    grep -E "schema check:|logger (comet|tensorboard):|PARITY (OK|FAILED)" \
        "$LOGS/smoke.log" | tail -6 | sed 's/^/        /' | tee -a "$STATUS"
elif [ "$DO_SMOKE" -eq 0 ]; then
    say "[smoke] skipped (--no-smoke)"
else
    say "[smoke] already done, skipping"
fi

# ---- phase 3: v2 re-baseline ---------------------------------------------------
# Scores the EXISTING v2 weights against the new shared test labels. Two minutes,
# and without it there is no reference point: the arms would be comparable only to
# each other, not to where the project already stands.
if [ ! -f "$PROJECT/baseline.done" ]; then
    say "[baseline] scoring v2 weights on the new test labels"
    python3 "$HERE/baseline_eval.py" "$PROJECT" 2>&1 | tee -a "$LOGS/baseline.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then say "[baseline] FAILED (rc=$rc) -- continuing anyway, it is a reference point not a gate"
    else touch "$PROJECT/baseline.done"; say "[baseline] OK"; fi
else
    say "[baseline] already done, skipping"
fi

# ---- phases 4..n: the arms -----------------------------------------------------
for arm in "${ARMS[@]}"; do
    if [ -f "$PROJECT/$arm.done" ]; then say "[$arm] already done, skipping"; continue; fi
    if [ ! -f "$ROOT/$arm/data.yaml" ]; then say "[$arm] MISSING $ROOT/$arm/data.yaml -- skipping"; continue; fi
    say "[$arm] training started (~3.0 h on a 4090)"
    python3 "$HERE/train_one_arm.py" "$arm" "$PROJECT" 2>&1 | tee -a "$LOGS/$arm.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then say "[$arm] FAILED (rc=$rc) -- moving to the next arm"; continue; fi
    touch "$PROJECT/$arm.done"
    r=$(python3 -c "
import json
d = json.load(open('$PROJECT/${arm}_result.json')); t = d['test']
print(f\"mAP50={t['mAP50']:.4f} mAP50-95={t['mAP50_95']:.4f} \"
      f\"P={t['precision']} R={t['recall']} ({d['train_hours']}h)\"
      + ('' if d['args_parity_with_v2'] else '  !! ARGS DIFFER FROM v2'))" 2>/dev/null)
    say "[$arm] DONE  $r"
done

# ---- final comparison ----------------------------------------------------------
say "[compare] building summary"
python3 "$HERE/compare_v3.py" "$PROJECT" 2>&1 | tee -a "$LOGS/compare.log" | tee -a "$STATUS"
say "=== v3 run complete ==="
# With one arm done, say plainly how to add another -- nothing above is re-run.
for a in v3p v3l v3s; do
    [ -f "$PROJECT/$a.done" ] || say "  not yet trained: $a   ->   $0 $a"
done
say "Pull the results to your laptop BEFORE tearing the pod down -- the volume has"
say "been cut off once before (DEC-137):"
say "  scp -r -P <port> -i ~/.ssh/id_ed25519 root@<host>:$PROJECT ./runs/"
say "  scp -r -P <port> -i ~/.ssh/id_ed25519 root@<host>:$REPO/runs/detect ./runs/"

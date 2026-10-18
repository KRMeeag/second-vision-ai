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
# ORDER IS PERMISSIVE -> LOOSE -> STRICT on purpose: strict is gated on what the first
# two show in Comet, so it must not run first. Default runs the first two only.
#
#   ./run_v3.sh                 # baseline + permissive + loose   (the planned ~6.5 h)
#   ./run_v3.sh v3s             # strict later, if Comet says it is worth it
#   ./run_v3.sh v3p v3l v3s     # all three in one go (~9.5 h)
#   ./run_v3.sh --status        # print progress and exit (safe any time)
#
set -uo pipefail

ROOT=/workspace
PROJECT=$ROOT/runs_v3
STATUS=$PROJECT/STATUS
LOGS=$PROJECT/logs
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$PROJECT" "$LOGS"

say() { printf '%s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS"; }

if [ "${1:-}" = "--status" ]; then
    echo "=== STATUS ==="; [ -f "$STATUS" ] && tail -40 "$STATUS" || echo "(not started)"
    echo; echo "=== markers ==="; ls -1 "$PROJECT"/*.done 2>/dev/null || echo "(none)"
    echo; echo "=== results ==="
    for f in "$PROJECT"/*_result.json; do
        [ -e "$f" ] || continue
        python3 - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
t=d["test"]
print(f"  {d['arm']:<5} mAP50={t['mAP50']:.4f}  mAP50-95={t['mAP50_95']:.4f}  "
      f"P={t['precision']:.4f}  R={t['recall']:.4f}  ({d['train_hours']}h)")
PY
    done
    echo; echo "=== running? ==="
    pgrep -af "train_one_arm.py" || echo "  no training process"
    exit 0
fi

# Re-exec detached unless already detached.
if [ "${V3_DETACHED:-}" != "1" ]; then
    export V3_DETACHED=1
    say "launching detached; follow with: tail -f $LOGS/driver.log"
    setsid nohup "$0" "$@" >>"$LOGS/driver.log" 2>&1 </dev/null &
    sleep 1
    echo "Detached. Safe to disconnect."
    echo "  progress : $0 --status"
    echo "  live log : tail -f $LOGS/driver.log"
    exit 0
fi

ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(v3p v3l)

say "=== v3 run starting: arms = ${ARMS[*]} ==="

# ---- phase 0: pre-flight -------------------------------------------------------
# Checked BEFORE burning GPU hours, because every one of these has bitten this
# project before: a wrong ultralytics version, a dataset whose paths do not resolve,
# and a volume that silently produced 0-byte files when its quota ran out.
if [ ! -f "$PROJECT/preflight.done" ]; then
    say "[preflight] starting"
    python3 "$HERE/preflight_v3.py" "${ARMS[@]}" 2>&1 | tee -a "$LOGS/preflight.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then say "[preflight] FAILED (rc=$rc) -- stopping"; exit 1; fi
    touch "$PROJECT/preflight.done"; say "[preflight] OK"
else
    say "[preflight] already done, skipping"
fi

# ---- phase 1: v2 re-baseline ---------------------------------------------------
# Scores the EXISTING v2 weights against the new shared test labels. Two minutes, and
# without it there is no reference point: the three arms would only be comparable to
# each other, not to where the project already stands.
if [ ! -f "$PROJECT/baseline.done" ]; then
    say "[baseline] scoring v2 weights on the new test labels"
    python3 "$HERE/baseline_eval.py" "$PROJECT" 2>&1 | tee -a "$LOGS/baseline.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then say "[baseline] FAILED (rc=$rc) -- continuing anyway"
    else touch "$PROJECT/baseline.done"; say "[baseline] OK"; fi
else
    say "[baseline] already done, skipping"
fi

# ---- phases 2..n: the arms -----------------------------------------------------
for arm in "${ARMS[@]}"; do
    if [ -f "$PROJECT/$arm.done" ]; then say "[$arm] already done, skipping"; continue; fi
    if [ ! -f "$ROOT/$arm/data.yaml" ]; then say "[$arm] MISSING $ROOT/$arm/data.yaml -- skipping"; continue; fi
    say "[$arm] training started (~3.0 h on a 4090)"
    python3 "$HERE/train_one_arm.py" "$arm" "$PROJECT" 2>&1 | tee -a "$LOGS/$arm.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then say "[$arm] FAILED (rc=$rc) -- moving to next arm"; continue; fi
    touch "$PROJECT/$arm.done"
    r=$(python3 -c "
import json;d=json.load(open('$PROJECT/${arm}_result.json'));t=d['test']
print(f\"mAP50={t['mAP50']:.4f} mAP50-95={t['mAP50_95']:.4f} P={t['precision']:.4f} R={t['recall']:.4f}\")" 2>/dev/null)
    say "[$arm] DONE  $r"
done

# ---- final comparison ----------------------------------------------------------
say "[compare] building summary"
python3 "$HERE/compare_v3.py" "$PROJECT" 2>&1 | tee -a "$LOGS/compare.log" | tee -a "$STATUS"
say "=== v3 run complete ==="

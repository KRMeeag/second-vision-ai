# Source this in any new terminal to talk to the pods:
#
#     source scripts/train/podenv.sh
#
# Defines podssh / cpussh and loads .env.runpod. Both are shell FUNCTIONS, not
# scripts, so they exist only in the shell that sourced this file — a new
# terminal tab needs to source it again. That is also why they are functions and
# not a POD="-p ... -i ..." variable: in bash an unquoted $POD word-splits into
# separate arguments, but zsh passes it as ONE argument and ssh then reads
# everything after -p as the port ("Bad port"). A function passes "$@" through
# cleanly in both shells. scripts/train/runpod_wizard.sh:246 uses the same shape.

_here="${${(%):-%x}:A:h:h:h}"       # zsh: this file's repo root
[ -n "$_here" ] || _here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

set -a
# shellcheck disable=SC1090
. "$_here/.env.runpod"
set +a
unset _here

# RunPod paths lose packets -- a euro-3 pod measured 36.7% loss on 2026-10-18
# with a clean path MTU and an idle pod at the far end. A dropped key-exchange
# packet parks ssh in the handshake with no application timeout, which reads as
# "Connection timed out" even though the pod is healthy. ConnectTimeout does not
# cover that (it bounds the TCP connect only); ServerAliveInterval and a retry
# on rc 255 -- ssh's own transport failures, never the remote command's status --
# do. Same settings as scripts/train/runpod_v3_wizard.sh; keep them in step.
SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=15
  -o ConnectionAttempts=3
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=3
)
SSH_TRIES="${SSH_TRIES:-5}"

_ssh_retry() {
  local tries="$1"; shift
  local n=1 rc=0
  while :; do
    "$@" && return 0
    rc=$?
    [ "$rc" -ne 255 ] && return "$rc"
    [ "$n" -ge "$tries" ] && return "$rc"
    printf 'ssh transport failed (%s/%s) — retrying in %ss\n' \
      "$n" "$tries" "$((n * 3))" >&2
    sleep "$((n * 3))"
    n=$((n + 1))
  done
}

podssh() {
  _ssh_retry "$SSH_TRIES" \
    ssh -p "$GPU_POD_PORT" -i "$SSH_KEY" "${SSH_OPTS[@]}" \
        root@"$GPU_POD_HOST" "$@"
}

cpussh() {
  _ssh_retry "$SSH_TRIES" \
    ssh -p "$CPU_POD_PORT" -i "$SSH_KEY" "${SSH_OPTS[@]}" \
        root@"$CPU_POD_HOST" "$@"
}

# podsync SRC... DEST — resumable transfer either direction. Use instead of scp:
# scp restarts from byte zero on a drop, which on this link can lose more than
# it moves. --partial-dir keeps the partial as a delta basis for the next try.
podsync() {
  local n=1
  while :; do
    rsync -az --partial --partial-dir=.rsync-partial --timeout=60 \
          -e "ssh -p $GPU_POD_PORT -i $SSH_KEY ${SSH_OPTS[*]}" "$@" && return 0
    [ "$n" -ge "$SSH_TRIES" ] && return 1
    printf 'transfer interrupted (%s/%s) — resuming in %ss\n' \
      "$n" "$SSH_TRIES" "$((n * 5))" >&2
    sleep "$((n * 5))"
    n=$((n + 1))
  done
}

printf 'pod env loaded — GPU %s:%s\n' "$GPU_POD_HOST" "$GPU_POD_PORT"
printf '  podssh "<command>"   run on the GPU pod   (retries lost connections)\n'
printf '  cpussh "<command>"   run on the CPU pod\n'
printf '  podsync SRC DEST     resumable file copy, either direction\n'
[ -n "$COMET_API_KEY" ] \
  && printf '  COMET_API_KEY loaded (%d chars)\n' "${#COMET_API_KEY}" \
  || printf '  COMET_API_KEY NOT set — training will log to TensorBoard only\n'

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
  # The payloads are already-compressed bytes (.tar.gz up, .pt down), so SSH
  # compression spends CPU to shrink nothing. upload_bundle.sh sets this for
  # the same reason on the same link.
  -o Compression=no
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
# Higher than SSH_TRIES on purpose: a transfer holds the link open much longer
# than a one-shot command, and every attempt makes forward progress because the
# partial survives. A 12 MB upload took more than 3 attempts on 2026-10-18.
SYNC_TRIES="${SYNC_TRIES:-12}"

# Pick GNU rsync explicitly. THIS KNOWLEDGE IS NOT NEW -- it is lifted from
# scripts/train/upload_bundle.sh, which already probes these three paths and
# documents why: macOS /usr/bin/rsync is openrsync, which reports "rsync version
# 2.6.9 compatible" but REJECTS --append-verify and --info=progress2. Calling a
# bare `rsync` worked only because PATH happened to find Homebrew's GNU build; on
# a machine without it, podsync would have failed on flags rather than on the link.
RSYNC_BIN=""
_openrsync_fallback=""
for _cand in /opt/homebrew/bin/rsync /usr/local/bin/rsync /usr/bin/rsync; do
  [ -x "$_cand" ] || continue
  if "$_cand" --version 2>&1 | head -1 | grep -qi openrsync; then
    [ -n "$_openrsync_fallback" ] || _openrsync_fallback="$_cand"
  else
    RSYNC_BIN="$_cand"; break
  fi
done
[ -n "$RSYNC_BIN" ] || RSYNC_BIN="$_openrsync_fallback"
[ -n "$RSYNC_BIN" ] || RSYNC_BIN="rsync"
unset _cand _openrsync_fallback

# NOT -a. `-a` implies -o and -g (preserve owner and group), and setting those
# fails on this MooseFS volume: "chown ... Operation not permitted", rsync exit
# 23. Exit 23 means "the DATA moved, some attributes did not" -- so every attempt
# transferred the whole tarball and then reported failure, and the loop re-sent a
# file that had already landed, with nothing in the message to distinguish it
# from packet loss. Observed 2026-10-18; the sha256 gate confirmed all three arms
# were in fact intact on the pod.
#
# No -z either. Every payload here is already compressed -- .tar.gz going up,
# .pt coming down -- so compressing it again spends CPU to save nothing. Same
# reason upload_bundle.sh sets ssh Compression=no, which SSH_OPTS now does too.
#
# -rlt keeps recursion, symlinks and mtimes, and requests no ownership at all.
RSYNC_OPTS=(-rlt --partial --partial-dir=.rsync-partial --timeout=60)

# Exit codes that will NEVER succeed on retry -- usage, protocol mismatch, bad
# file selection. Retrying these just hides the message that says what is wrong.
_rsync_fatal() { case "$1" in 1|2|3|4|6) return 0 ;; *) return 1 ;; esac; }

podsync() {
  local n=1 delay rc
  while :; do
    "$RSYNC_BIN" "${RSYNC_OPTS[@]}" \
          -e "ssh -p $GPU_POD_PORT -i $SSH_KEY ${SSH_OPTS[*]}" "$@" && return 0
    rc=$?
    if _rsync_fatal "$rc"; then
      printf 'rsync exit %s — not a transient failure, not retrying\n' "$rc" >&2
      return "$rc"
    fi
    [ "$n" -ge "$SYNC_TRIES" ] && return "$rc"
    delay=$(( n * 5 )); [ "$delay" -gt 30 ] && delay=30
    # The exit code goes in the message. Without it a non-network fault looks
    # exactly like packet loss and you watch it fail identically twelve times.
    printf 'transfer interrupted (rsync exit %s, %s/%s) — resuming in %ss\n' \
      "$rc" "$n" "$SYNC_TRIES" "$delay" >&2
    sleep "$delay"
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

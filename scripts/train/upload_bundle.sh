#!/usr/bin/env bash
# Resumable upload of dataset/bundle.tar to a RunPod pod, with end-to-end verification.
#
# WHY A SCRIPT AND NOT A ONE-LINE rsync
# -------------------------------------
# The archive is ~5.1 GB and the measured uplink is ~0.7 MB/s, so this runs for
# roughly two hours. Three things go wrong over that window, and all three are
# handled here rather than discovered at hour two:
#
#   1. macOS no longer ships GNU rsync. `/usr/bin/rsync` is openrsync, which
#      reports "rsync version 2.6.9 compatible" but REJECTS --append-verify and
#      --info=progress2. A copy-pasted GNU rsync command fails instantly; worse,
#      a command that silently drops the resume flag restarts from zero on every
#      retry. This script detects which binary it has and picks flags to match.
#   2. Home links drop. A bare rsync exits non-zero and leaves you to notice.
#      This retries with backoff until the transfer completes, so it can be left
#      running unattended.
#   3. "It finished" is not the same as "it arrived intact". The remote sha256 is
#      compared against the local one before this reports success.
#
# RESUME BEHAVIOUR DIFFERS BY BINARY, AND BOTH ARE FINE:
#   * GNU rsync   -- --append-verify appends from the partial's end and
#                    re-checksums the existing prefix. Cheapest resume.
#   * openrsync   -- --partial keeps the partial; the next run delta-transfers
#                    against it and sends only the missing blocks. Costs a local
#                    + remote read of the partial per retry, but does not re-send
#                    bytes that already landed.
#
# Both are safe ONLY because the source file is immutable. Do not rebuild
# dataset/bundle.tar while a transfer is in flight.
#
# Usage
# -----
#   scripts/train/upload_bundle.sh --host 213.x.x.x --port 22014
#   scripts/train/upload_bundle.sh --host 213.x.x.x --port 22014 --remote-dir /workspace
#
# Get --host and --port from the pod's Connect panel, "SSH over exposed TCP".
# The proxy form (ssh.runpod.io) will NOT work -- it is a restricted shell with
# no rsync/scp on the far side.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAR="${REPO_ROOT}/dataset/bundle.tar"
HOST=""; PORT=""; KEY="${HOME}/.ssh/id_ed25519"; REMOTE_DIR="/workspace"
USER_AT="root"; MAX_RETRIES=100

while [ $# -gt 0 ]; do
  case "$1" in
    --host)       HOST="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --key)        KEY="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --user)       USER_AT="$2"; shift 2 ;;
    --file)       TAR="$2"; shift 2 ;;
    --retries)    MAX_RETRIES="$2"; shift 2 ;;
    -h|--help)    sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$HOST" ] && [ -n "$PORT" ] || { echo "ERROR: --host and --port are required." >&2; exit 2; }
[ -f "$TAR" ] || { echo "ERROR: archive not found: $TAR" >&2; exit 2; }
[ -f "$KEY" ] || { echo "ERROR: ssh key not found: $KEY" >&2; exit 2; }

# ---- local checksum (reuse the sidecar if it is present; computing it costs ~30 s)
SHA_FILE="${TAR}.sha256"
if [ -f "$SHA_FILE" ]; then
  LOCAL_SHA="$(awk '{print $1}' "$SHA_FILE")"
else
  echo "computing local sha256 (no sidecar found) ..."
  LOCAL_SHA="$(shasum -a 256 "$TAR" | awk '{print $1}')"
fi

# ---- pick an rsync and matching flags
RSYNC=""; OPEN_FALLBACK=""
for cand in /opt/homebrew/bin/rsync /usr/local/bin/rsync /usr/bin/rsync; do
  [ -x "$cand" ] || continue
  if "$cand" --version 2>&1 | head -1 | grep -qi openrsync; then
    [ -n "$OPEN_FALLBACK" ] || OPEN_FALLBACK="$cand"
  else
    RSYNC="$cand"; break
  fi
done
if [ -n "$RSYNC" ]; then
  FLAGS=(-a --partial --progress --append-verify)
  RMODE="GNU rsync ($("$RSYNC" --version 2>&1 | head -1 | awk '{print $3}')) — append-verify resume"
else
  RSYNC="$OPEN_FALLBACK"
  FLAGS=(-a --partial --progress)
  RMODE="openrsync — delta resume (install GNU rsync with 'brew install rsync' for cheaper retries)"
fi
[ -n "$RSYNC" ] || { echo "ERROR: no rsync found." >&2; exit 2; }

SSH_OPTS="-p ${PORT} -i ${KEY} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
SSH_OPTS="${SSH_OPTS} -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
# The payload is JPEG. SSH compression would spend CPU to shrink already-compressed
# bytes and typically slows the transfer rather than speeding it up.
SSH_OPTS="${SSH_OPTS} -o Compression=no"

BYTES=$(stat -f %z "$TAR" 2>/dev/null || stat -c %s "$TAR")
printf 'archive   : %s\n' "$TAR"
printf 'size      : %s bytes (%.2f GiB)\n' "$BYTES" "$(echo "$BYTES" | awk '{print $1/1073741824}')"
printf 'sha256    : %s\n' "$LOCAL_SHA"
printf 'target    : %s@%s:%s (port %s)\n' "$USER_AT" "$HOST" "$REMOTE_DIR" "$PORT"
printf 'transport : %s\n' "$RMODE"
printf 'estimate  : ~%.1f h at 0.7 MB/s\n\n' "$(echo "$BYTES" | awk '{print $1/700000/3600}')"

ssh $SSH_OPTS "${USER_AT}@${HOST}" "mkdir -p '${REMOTE_DIR}'" || {
  echo "ERROR: cannot ssh to the pod. Check that you used the 'SSH over exposed TCP'" >&2
  echo "       host/port (not ssh.runpod.io) and that ${KEY}.pub is in your RunPod" >&2
  echo "       account settings AND the pod was started AFTER you added it." >&2
  exit 1
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "=== attempt ${attempt} — $(date '+%H:%M:%S') ==="
  if "$RSYNC" "${FLAGS[@]}" -e "ssh $SSH_OPTS" "$TAR" "${USER_AT}@${HOST}:${REMOTE_DIR}/"; then
    echo "transfer reported success."
    break
  fi
  if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    echo "ERROR: giving up after ${attempt} attempts. The partial file is kept on the" >&2
    echo "       pod — re-running this script resumes from it." >&2
    exit 1
  fi
  backoff=$(( attempt < 6 ? attempt * 10 : 60 ))
  echo "  dropped; retrying in ${backoff}s (partial file kept, this resumes)"
  sleep "$backoff"
done

echo
echo "verifying the archive ON THE POD ..."
REMOTE_SHA="$(ssh $SSH_OPTS "${USER_AT}@${HOST}" \
  "sha256sum '${REMOTE_DIR}/$(basename "$TAR")' 2>/dev/null | cut -d' ' -f1")"
echo "  local  : ${LOCAL_SHA}"
echo "  remote : ${REMOTE_SHA:-<could not read>}"
if [ "$REMOTE_SHA" != "$LOCAL_SHA" ]; then
  echo
  echo "CHECKSUM MISMATCH — do not extract this archive." >&2
  echo "Delete it on the pod and re-run this script:" >&2
  echo "  ssh $SSH_OPTS ${USER_AT}@${HOST} 'rm ${REMOTE_DIR}/$(basename "$TAR")'" >&2
  exit 1
fi

cat <<NEXT

CHECKSUM MATCH — the archive arrived intact.

Next, on the pod:
  cd ${REMOTE_DIR}
  tar -xf $(basename "$TAR")          # KEEP the .tar — it is your only local backup
  git clone https://github.com/KRMeeag/second-vision-ai.git
  python3 second-vision-ai/scripts/train/verify_bundle.py --bundle ${REMOTE_DIR}/bundle
NEXT

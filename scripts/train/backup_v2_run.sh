#!/usr/bin/env bash
# backup_v2_run.sh — runs ON THE POD. Copies the finished v2 run OUT of the git
# checkout to a safe place on the network volume, with checksums.
#
# Why this is not paranoia: the run lives at
#   /workspace/second-vision-ai/runs/detect/cap4500_yolov8s_v2*/
# which is inside the pod's git checkout, and `.gitignore:71` ignores `runs/`.
# A single `git clean -xfd` in that directory deletes every weight file. DEC-120
# exists in this project because a plan was lost to exactly that command.
#
# Destination is /workspace/model_backups/ — same network volume (so it survives
# pod stop/terminate) but OUTSIDE any git working tree.
#
# This is a real copy, not `cp -al`. Hardlinks would protect against deletion but
# not against in-place modification, and cost nothing to avoid at ~45 MB.
#
# Safe to run at any time. It only ever creates; it never deletes.
#
# Launch (does NOT disturb the running post_training_v2.sh):
#   scp -P "$GPU_POD_PORT" -i "$SSH_KEY" scripts/train/backup_v2_run.sh \
#       root@"$GPU_POD_HOST":/workspace/
#   podssh "tmux new -d -s backup 'bash /workspace/backup_v2_run.sh 2>&1 | tee /workspace/backup.log'"

set -uo pipefail

REPO=/workspace/second-vision-ai
DEST=/workspace/model_backups
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

say() { printf '\n=== %s ===\n' "$*"; }

# ── wait for the evaluation pass, so the backup includes its artefacts ─────
if tmux has-session -t post 2>/dev/null; then
  say "waiting for the 'post' session (evaluation) to finish first"
  while tmux has-session -t post 2>/dev/null; do sleep 60; done
elif tmux has-session -t v2 2>/dev/null; then
  say "no 'post' session; waiting for training itself to finish"
  while tmux has-session -t v2 2>/dev/null; do sleep 60; done
fi

mkdir -p "$DEST"

# ── back up every finished v2 run, plus v1 for good measure ───────────────
for RUN in "$REPO"/runs/detect/cap4500_yolov8s_v2* "$REPO"/runs/detect/cap4500_yolov8s; do
  [ -d "$RUN" ] || continue
  [ -f "$RUN/weights/best.pt" ] || { echo "skip $(basename "$RUN") — no best.pt"; continue; }

  NAME=$(basename "$RUN")
  OUT="$DEST/${NAME}_${STAMP}"
  say "backing up $NAME -> $OUT"

  mkdir -p "$OUT"
  cp -a "$RUN"/. "$OUT"/ || { echo "COPY FAILED for $NAME"; continue; }

  # checksum manifest over the copy
  ( cd "$OUT" && find . -type f -exec sha256sum {} + | sort -k2 > SHA256SUMS ) 2>/dev/null

  # verify the two files that actually matter
  ok=1
  for f in weights/best.pt weights/last.pt results.csv args.yaml; do
    if [ -f "$RUN/$f" ]; then
      a=$(sha256sum "$RUN/$f" | awk '{print $1}')
      b=$(sha256sum "$OUT/$f" | awk '{print $1}')
      if [ "$a" = "$b" ]; then
        printf '  verified  %-18s %s\n' "$f" "${a:0:16}…"
      else
        printf '  MISMATCH  %-18s\n' "$f"; ok=0
      fi
    fi
  done
  [ "$ok" = 1 ] && echo "  OK" || echo "  !! VERIFICATION FAILED for $NAME"

  # a single tarball makes it one scp to pull home
  say "tarring $NAME"
  tar czf "$DEST/${NAME}_${STAMP}.tar.gz" -C "$DEST" "$(basename "$OUT")" \
    && echo "  $(du -h "$DEST/${NAME}_${STAMP}.tar.gz" | cut -f1)  $DEST/${NAME}_${STAMP}.tar.gz"
done

# ── keep the evaluation summary alongside the weights ─────────────────────
for f in /workspace/RESULTS.md /workspace/post.log /workspace/v2_train.log; do
  [ -f "$f" ] && cp -a "$f" "$DEST/" && echo "copied $(basename "$f")"
done

say "contents of $DEST"
ls -lh "$DEST"

say "DONE $STAMP"
echo "DONE $STAMP" > /workspace/backup.status

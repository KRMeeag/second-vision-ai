#!/usr/bin/env bash
# post_training_v2.sh — runs ON THE POD, unattended, after the v2 training finishes.
#
# Waits for the training tmux session to end, then does every step that would
# otherwise need a human at a keyboard:
#
#   1. verifies training actually completed (100 epochs, best.pt present)
#   2. evaluates v2 on the v2b test split
#   3. builds final_cap4500_v2a from v2a_labels.tar.gz
#   4. re-baselines v1 on the CURRENT v2a labels, on this pod (DEC-134)
#   5. writes /workspace/RESULTS.md with the like-for-like comparison
#
# It NEVER deletes a run directory, never terminates the pod, and never touches
# dataset/final_cap4500. Everything it writes is new.
#
# Launch (from the laptop, before shutting down):
#   scp -P "$GPU_POD_PORT" -i "$SSH_KEY" scripts/train/post_training_v2.sh \
#       root@"$GPU_POD_HOST":/workspace/
#   scp -P "$GPU_POD_PORT" -i "$SSH_KEY" dataset/v2a_labels.tar.gz \
#       root@"$GPU_POD_HOST":/workspace/
#   podssh "tmux new -d -s post 'bash /workspace/post_training_v2.sh 2>&1 | tee /workspace/post.log'"
#
# Check later:
#   podssh "cat /workspace/RESULTS.md"

set -uo pipefail

REPO=/workspace/second-vision-ai
WS=/workspace
export YOLO_CONFIG_DIR=/workspace/.ultralytics
STATUS="$WS/post.status"

say() { printf '\n=== %s ===\n' "$*"; }
fail() { printf '\nFAILED: %s\n' "$*"; echo "FAILED: $*" > "$STATUS"; exit 1; }

echo "started $(date -u +%FT%TZ)" > "$STATUS"

# ── 1. wait for training ──────────────────────────────────────────────────
say "waiting for the v2 training session to finish"
while tmux has-session -t v2 2>/dev/null; do sleep 60; done
say "training session ended $(date -u +%FT%TZ)"

# find the run dir: newest cap4500_yolov8s_v2* that has weights/best.pt
RUN=$(ls -dt "$REPO"/runs/detect/cap4500_yolov8s_v2* 2>/dev/null \
      | while read -r d; do [ -f "$d/weights/best.pt" ] && echo "$d"; done | head -1)
[ -n "$RUN" ] || fail "no cap4500_yolov8s_v2* run dir with weights/best.pt"
say "run dir: $RUN"

EPOCHS=$(($(wc -l < "$RUN/results.csv") - 1))
echo "epochs recorded: $EPOCHS"
[ "$EPOCHS" -ge 100 ] || echo "WARNING: only $EPOCHS epochs recorded — training may have stopped early"

# ── 2. evaluate v2 on the v2b test split ──────────────────────────────────
say "evaluating v2 on the v2b test split"
cd "$REPO" || fail "cannot cd $REPO"
python3 scripts/train/evaluate.py \
  --weights "$RUN/weights/best.pt" \
  --data "$WS/final_cap4500_v2/data.yaml" \
  --split test 2>&1 | tail -40 || echo "WARNING: evaluate.py returned non-zero"

# ── 3. build the v2a tree for the v1 re-baseline ──────────────────────────
if [ -f "$WS/v2a_labels.tar.gz" ]; then
  say "building final_cap4500_v2a (15-class labels, for the v1 re-baseline)"
  rm -rf "$WS/final_cap4500_v2a"
  cp -al "$WS/final_cap4500" "$WS/final_cap4500_v2a" || fail "cp -al failed"
  rm -rf "$WS/final_cap4500_v2a/train/labels" \
         "$WS/final_cap4500_v2a/val/labels" \
         "$WS/final_cap4500_v2a/test/labels" \
         "$WS/final_cap4500_v2a/data.yaml"
  tar xzf "$WS/v2a_labels.tar.gz" --no-same-owner -C "$WS/final_cap4500_v2a" \
    || fail "extract of v2a labels failed"
  echo "v2a label files: $(find "$WS/final_cap4500_v2a" -path '*/labels/*.txt' | wc -l)"
else
  say "v2a_labels.tar.gz not present — SKIPPING the v1 re-baseline"
fi

# ── 4 + 5. the like-for-like comparison ───────────────────────────────────
say "computing the 14-class comparison (DEC-134)"
python3 - <<'PY' 2>&1 | tee "$WS/RESULTS.md"
import os, json, statistics
from pathlib import Path
from ultralytics import YOLO

WS = Path("/workspace"); REPO = WS / "second-vision-ai"
runs = sorted((p for p in (REPO/"runs/detect").glob("cap4500_yolov8s_v2*")
               if (p/"weights/best.pt").is_file()),
              key=lambda p: p.stat().st_mtime, reverse=True)
v2_run = runs[0]
v1_w = REPO/"runs/detect/cap4500_yolov8s/weights/best.pt"

def per_class(weights, data):
    r = YOLO(str(weights)).val(data=str(data), split="test", iou=0.7, max_det=300, verbose=False)
    return ({r.names[c]: float(r.box.ap50[i]) for i, c in enumerate(r.box.ap_class_index)},
            float(r.box.map50))

print("# v2 results\n")
print(f"- v2 run: `{v2_run.name}`")

rows = {}
v2_pc, v2_map = per_class(v2_run/"weights/best.pt", WS/"final_cap4500_v2/data.yaml")
rows["v2 (14 classes, v2b labels)"] = (v2_pc, v2_map)

v2a = WS/"final_cap4500_v2a/data.yaml"
if v2a.is_file() and v1_w.is_file():
    v1_pc, v1_map = per_class(v1_w, v2a)
    rows["v1 (15 classes, current v2a labels)"] = (v1_pc, v1_map)
else:
    v1_pc = None
    print("\n> v1 re-baseline SKIPPED (missing v2a tree or v1 weights)")

print("\n## Per-class mAP@0.5\n")
names = sorted(set().union(*[d.keys() for d, _ in rows.values()]))
hdr = " | ".join(rows.keys())
print(f"| class | {hdr} |")
print("|---" * (len(rows) + 1) + "|")
for n in names:
    cells = " | ".join(f"{d.get(n, float('nan')):.4f}" for d, _ in rows.values())
    print(f"| {n} | {cells} |")

print("\n## Headline (DEC-134: compare 14-class to 14-class)\n")
v2_14 = statistics.mean([v for k, v in v2_pc.items() if k != "Shelf"])
print(f"- **v2, 14-class mean mAP@0.5 : {v2_14:.4f}**")
if v1_pc:
    v1_14 = statistics.mean([v for k, v in v1_pc.items() if k != "Shelf"])
    v1_15 = statistics.mean(list(v1_pc.values()))
    print(f"- v1 re-baselined, 14-class mean : {v1_14:.4f}   <-- the number v2 must beat")
    print(f"- v1 re-baselined, 15-class mean : {v1_15:.4f}   (includes Shelf; NOT comparable)")
    print(f"\n- **delta (v2 - v1, both 14-class) : {v2_14 - v1_14:+.4f}**")
    print(f"- Shelf's drag on the 15-class mean : {v1_14 - v1_15:+.4f} "
          f"(a class-drop effect, report separately from annotation completeness)")
print("\n- published v1 (15 classes, v1 labels, DEC-122): 0.6695 — inflated by missing annotations")
PY

say "DONE $(date -u +%FT%TZ)"
echo "DONE $(date -u +%FT%TZ)" > "$STATUS"

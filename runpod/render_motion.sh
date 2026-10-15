#!/usr/bin/env bash
# RENDER-MOTION-V1 — Stage 2b/3: rebuild her avatar from the SAME curated
# 8 views used for the PLY (/workspace/avatar_inputs) and render her
# performing one of the project's bundled motions, using the LHM++ app's
# own inference pipeline in-process (build_avatar.py). Output video:
# /workspace/avatar_out/avatar_motion_test.mp4 — David's judgment artifact
# (is this her, in motion?) and the Stage 3 pose-drive proof in one.
set -u
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_HOME=/usr/local/cuda
VPY=/workspace/avatar-venv/bin/python
REPO=/workspace/LHM-plusplus
REFS=/workspace/avatar_inputs
OUT=/workspace/avatar_out
LOG=/workspace/avatar_render.log
DRIVER=/workspace/build_avatar.py

exec > >(tee -a "$LOG") 2>&1
echo "=== RENDER-MOTION-V1 $(date) ==="

[ -x "$VPY" ] || { echo "RENDER-FAILED: avatar venv missing"; exit 1; }
[ -d "$REPO" ] || { echo "RENDER-FAILED: LHM-plusplus repo missing"; exit 1; }
[ -f "$DRIVER" ] || { echo "RENDER-FAILED: build_avatar.py missing at $DRIVER"; exit 1; }
[ -d "$REFS" ] || { echo "RENDER-FAILED: curated refs missing at $REFS (run build_avatar.sh first)"; exit 1; }
ls -la "$REFS"
cd "$REPO" || exit 1

# ---------- pick a bundled motion ----------
MOTION=""
for d in ./motion_video/*/; do
  name=$(basename "$d")
  if [ -f "${d}samurai_visualize.mp4" ]; then
    [ -f "${d}${name}.mp4" ] || cp "${d}samurai_visualize.mp4" "${d}${name}.mp4"
    MOTION="${d}${name}.mp4"
    break
  fi
done
[ -n "$MOTION" ] || { echo "RENDER-FAILED: no bundled motion found"; exit 1; }
echo "[motion] $MOTION"

# ---------- run ----------
# PYTHONPATH must name the repo: the driver imports the project's own
# packages (core, engine, scripts), and Python only auto-adds the
# driver's own directory (/workspace), not the repo root.
PYTHONPATH="$REPO" "$VPY" "$DRIVER" --refs "$REFS" --motion "$MOTION" --out "$OUT"
rc=$?
ls -la "$OUT" 2>/dev/null
if [ "$rc" -eq 0 ] && [ -s "$OUT/avatar_motion_test.mp4" ]; then
  echo "RENDER-DONE: $OUT/avatar_motion_test.mp4"
else
  echo "RENDER-FAILED (driver rc=$rc, see $LOG)"
fi
exit "$rc"

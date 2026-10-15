#!/usr/bin/env bash
# RENDER-V2-V1 — Stage 2d: rebuild + motion render from David's own picked
# stills. Noor uploads his 16 curated picks to the pod's ComfyUI/input as
# v2_01.* .. v2_16.* (slot order baked into the names: standing full-body
# fronts first). This script assembles them into /workspace/avatar_inputs_v2
# and runs the same driver + motion (BasketBall_I, 120f, gs backend) as the
# earlier renders, writing to a SEPARATE output dir so all three renders
# stay comparable: avatar_out (8-view), avatar_out16 (vid-frame 16),
# avatar_out_v2 (David's picks).
set -u
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_HOME=/usr/local/cuda
VPY=/workspace/avatar-venv/bin/python
REPO=/workspace/LHM-plusplus
SRC=/workspace/ComfyUI/input
REFS=/workspace/avatar_inputs_v2
OUT=/workspace/avatar_out_v2
LOG=/workspace/avatar_render_v2.log
DRIVER=/workspace/build_avatar.py

exec > >(tee -a "$LOG") 2>&1
echo "=== RENDER-V2-V1 $(date) ==="

fail() { echo "RENDER-V2-FAILED: $1"; exit 1; }

[ -x "$VPY" ] || fail "avatar venv missing"
[ -d "$REPO" ] || fail "LHM-plusplus repo missing"
[ -f "$DRIVER" ] || fail "build_avatar.py missing at $DRIVER"
grep -q 'default="gs"' "$DRIVER" || fail "build_avatar.py is not the gs-default version — re-download it first"

mkdir -p "$REFS" "$OUT"
rm -f "$REFS"/*
n=0
for f in "$SRC"/v2_*; do
  [ -f "$f" ] || continue
  n=$((n+1))
  cp "$f" "$REFS/$(basename "$f")"
  echo "ref $n: $f"
done
echo "picks found in ComfyUI/input: $n"
[ "$n" -eq 16 ] || fail "expected 16 v2_* picks in $SRC, found $n — uploads not complete"
ls -la "$REFS"

cd "$REPO" || exit 1
MOTION=""
for d in ./motion_video/*/; do
  name=$(basename "$d")
  if [ -f "${d}samurai_visualize.mp4" ]; then
    [ -f "${d}${name}.mp4" ] || cp "${d}samurai_visualize.mp4" "${d}${name}.mp4"
    MOTION="${d}${name}.mp4"
    break
  fi
done
[ -n "$MOTION" ] || fail "no bundled motion found"
echo "[motion] $MOTION"

PYTHONPATH="$REPO" "$VPY" "$DRIVER" --refs "$REFS" --motion "$MOTION" --out "$OUT" --backend gs
rc=$?
ls -la "$OUT" 2>/dev/null
if [ "$rc" -eq 0 ] && [ -s "$OUT/avatar_motion_test.mp4" ]; then
  echo "RENDER-V2-DONE: $OUT/avatar_motion_test.mp4"
else
  echo "RENDER-V2-FAILED (driver rc=$rc, see $LOG)"
fi
exit "$rc"

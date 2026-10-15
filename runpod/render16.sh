#!/usr/bin/env bash
# RENDER16-V1 — Stage 2c: identity-strengthened re-render. David's verdict
# on the 8-view render was "a little bit" her, so this rebuilds the avatar
# from a 16-view set (LHM++'s documented maximum): the same 4 full-body
# stills in slots 01-04, plus 12 of the 13 staged video frames in slots
# 05-16 — twice the real-pixel angles of her face/head/upper body feeding
# the build. Same motion (BasketBall_I, 120f) and same driver as the
# 8-view render, so the comparison is apples-to-apples. Output goes to a
# SEPARATE dir (/workspace/avatar_out16) — the 8-view clip is kept.
set -u
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_HOME=/usr/local/cuda
VPY=/workspace/avatar-venv/bin/python
REPO=/workspace/LHM-plusplus
SRC8=/workspace/avatar_inputs
ALL=/workspace/avatar_inputs_all
REFS=/workspace/avatar_inputs16
OUT=/workspace/avatar_out16
LOG=/workspace/avatar_render16.log
DRIVER=/workspace/build_avatar.py

exec > >(tee -a "$LOG") 2>&1
echo "=== RENDER16-V1 $(date) ==="

fail() { echo "RENDER16-FAILED: $1"; exit 1; }

[ -x "$VPY" ] || fail "avatar venv missing"
[ -d "$REPO" ] || fail "LHM-plusplus repo missing"
[ -f "$DRIVER" ] || fail "build_avatar.py missing at $DRIVER"
[ -d "$SRC8" ] || fail "curated 8-view set missing at $SRC8"
[ -d "$ALL" ] || fail "full set missing at $ALL"

# ---------- assemble the 16-view set ----------
mkdir -p "$REFS" "$OUT"
rm -f "$REFS"/*
slot=0
for f in "$SRC8"/0[1-4]_*; do
  [ -f "$f" ] || continue
  slot=$((slot+1))
  cp "$f" "$REFS/$(printf '%02d' $slot)_still_$(basename "$f" | sed 's/^[0-9]*_//')"
  echo "slot $slot (still): $f"
done
[ "$slot" -eq 4 ] || fail "expected 4 stills from $SRC8, got $slot"
VIDS=$(ls "$ALL"/*vid_t* 2>/dev/null | sort)
NVID=$(printf '%s\n' "$VIDS" | grep -c . || true)
echo "video frames available in full set: $NVID"
[ "$NVID" -ge 12 ] || fail "need >=12 vid_t frames in $ALL, found $NVID"
for pick in 1 2 3 4 6 7 8 9 10 11 12 13; do
  f=$(printf '%s\n' "$VIDS" | sed -n "${pick}p")
  [ -n "$f" ] || continue
  slot=$((slot+1))
  cp "$f" "$REFS/$(printf '%02d' $slot)_frame_$(basename "$f" | sed 's/^[0-9]*_//')"
  echo "slot $slot (frame pick $pick): $f"
done
NREF=$(ls "$REFS" | wc -l)
echo "16-view set assembled: $NREF views in $REFS"
ls -la "$REFS"
[ "$NREF" -eq 16 ] || fail "assembled $NREF views, expected 16"

# ---------- pick a bundled motion (same rule as render_motion.sh) ----------
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

# ---------- run ----------
PYTHONPATH="$REPO" "$VPY" "$DRIVER" --refs "$REFS" --motion "$MOTION" --out "$OUT" --backend gs
rc=$?
ls -la "$OUT" 2>/dev/null
if [ "$rc" -eq 0 ] && [ -s "$OUT/avatar_motion_test.mp4" ]; then
  echo "RENDER16-DONE: $OUT/avatar_motion_test.mp4"
else
  echo "RENDER16-FAILED (driver rc=$rc, see $LOG)"
fi
exit "$rc"

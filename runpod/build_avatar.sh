#!/bin/bash
# AVATAR-BUILD-V1 — Stage 2 of the 3D virtual-body program: build her
# canonical 3D Gaussian avatar (T-pose PLY) from the staged reference
# material on this pod, using the LHM++ stack installed by
# install_avatar.sh (marker AVATAR-INSTALL-V2, venv /workspace/avatar-venv,
# repo /workspace/LHM-plusplus).
#
# WHY THE INPUT SET IS CURATED, NOT DUMPED:
# scripts/inference/to_gs_ply.py resolves its input images with a hard
# cap of 8 views, taken in sorted filename order (verified in the
# upstream source: _list_images_from_dir / _resolve_image_paths slice
# [:8]). Feeding it the 13 video frames + the full-body stills in one
# folder would silently use only the first 8 sorted names and could
# drop every full-body still — the exact views that carry her lower
# body and proportions. This script therefore assembles a deliberate
# 8-view set with numeric prefixes that control the sort order:
#   slots 01-04: full-body stills (pod-side Studio/training photos)
#   slots 05-08: video frames spaced across her turn (face/head angles)
# The FULL uncurated set is also copied to /workspace/avatar_inputs_all/
# for the later animation stage, whose app path accepts more views.
#
# Safe to re-run: assembly is rebuilt from scratch each run; the export
# overwrites its own output PLY. Ends with AVATAR-BUILD-DONE on success.
set -u
LOG=/workspace/avatar_build.log
exec > >(tee -a "$LOG") 2>&1
echo "=== AVATAR-BUILD-V1 starting $(date) ==="

fail() { echo "AVATAR-BUILD-FAILED: $1"; exit 1; }

REPO=/workspace/LHM-plusplus
VPY=/workspace/avatar-venv/bin/python
MODEL="$REPO/pretrained_models/Damo_XR_Lab/LHMPP-700M-PixelShuffle"
ALL=/workspace/avatar_inputs_all
CUR=/workspace/avatar_inputs
OUT=/workspace/avatar_out
mkdir -p "$ALL" "$CUR" "$OUT"

# ---------- pre-checks ----------
[ -x "$VPY" ] || fail "avatar venv missing — install_avatar.sh has not completed"
[ -d "$REPO" ] || fail "LHM-plusplus repo missing at $REPO"
[ -f "$REPO/scripts/inference/to_gs_ply.py" ] || fail "to_gs_ply.py missing in repo"
# The weights landed in the downloader's cache layout
# (pretrained_models/models/Damo_XR_Lab--LHMPP-700M-PixelShuffle-DNA/
# snapshots/master). If the canonical path is absent, resolve the real
# snapshot dir and link it into place.
if [ ! -d "$MODEL" ]; then
  FOUND=$(find "$REPO/pretrained_models" -iname 'model.safetensors' -path '*PixelShuffle*' 2>/dev/null | head -1)
  [ -n "$FOUND" ] || fail "PixelShuffle weights not found anywhere under pretrained_models (install download step not finished?)"
  FOUND_DIR=$(dirname "$FOUND")
  echo "linking weights: $FOUND_DIR -> $MODEL"
  mkdir -p "$(dirname "$MODEL")"
  ln -sfn "$FOUND_DIR" "$MODEL"
fi
[ -f "$MODEL/model.safetensors" ] || fail "PixelShuffle weights missing at $MODEL (install download step not finished?)"
FREE_GB=$(df -BG /workspace | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "free space on /workspace: ${FREE_GB}G"
[ "${FREE_GB:-0}" -ge 3 ] || fail "need >=3G free on /workspace, have ${FREE_GB}G"

# ---------- discover candidate material ----------
# Known locations: the 13 video frames were staged into the pod's
# ComfyUI input folder as vid_t*.png; full-body stills live with the
# Studio job material / LoRA training set on the volume. Survey the
# plausible roots rather than trusting one path.
echo "--- surveying candidate images"
CANDIDATES=$(mktemp)
for root in /workspace/ComfyUI/input /workspace/lora_dataset /workspace/lora_data /workspace/dataset /workspace/data /workspace/studio /workspace/uploads; do
  [ -d "$root" ] && find "$root" -maxdepth 3 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) >> "$CANDIDATES" 2>/dev/null
done
# The local extraction set, if it was also copied onto the volume:
find /workspace -maxdepth 2 -type d -name 'refs_video' 2>/dev/null | while read -r d; do
  find "$d" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' \) >> "$CANDIDATES" 2>/dev/null
done
sort -u -o "$CANDIDATES" "$CANDIDATES"
echo "candidate images found: $(grep -c . "$CANDIDATES" || true)"
cat "$CANDIDATES"

VIDFRAMES=$(grep -i 'vid_t' "$CANDIDATES" || true)
STILLS=$(grep -v -i 'vid_t' "$CANDIDATES" || true)
[ -n "$VIDFRAMES" ] || fail "no vid_t* video frames found in any surveyed location — the staged frames are not where the main chat said they were; stop and report the survey list above"
[ -n "$STILLS" ] || echo "WARN: no non-video stills found — build will be waist-up only and proportions will be prior-driven. Report this before judging the result."

# ---------- assemble the full set (for later stages) ----------
rm -f "$ALL"/*
i=0
while IFS= read -r f; do
  [ -n "$f" ] || continue
  i=$((i+1))
  cp "$f" "$ALL/$(printf '%02d' "$i")_$(basename "$f")"
done < "$CANDIDATES"
echo "full set assembled in $ALL: $i images"

# ---------- curate the 8-view export set ----------
rm -f "$CUR"/*
slot=0
# Slots 1-4: full-body stills first, largest files first (the pod's
# stills are full-resolution photos; frames are video-resolution).
STILL_SORTED=$(mktemp)
echo "$STILLS" | tr '\n' '\0' | xargs -r -0 ls -S 2>/dev/null > "$STILL_SORTED" || echo "$STILLS" > "$STILL_SORTED"
head -4 "$STILL_SORTED" > "${STILL_SORTED}.top"
while IFS= read -r f; do
  [ -n "$f" ] || continue
  slot=$((slot+1))
  cp "$f" "$CUR/$(printf '%02d' "$slot")_still_$(basename "$f")"
  echo "slot $slot (still): $f"
done < "${STILL_SORTED}.top"
# Slots 5-8: video frames spaced across the turn. The frame files are
# named vid_t<sec>s_f<frame>.png in time order; take the 1st, 5th, 9th
# and 13th of the sorted list to span back-3/4 -> turn -> front -> close.
VID_SORTED=$(mktemp)
echo "$VIDFRAMES" | sort > "$VID_SORTED"
for pick in 1 5 9 13; do
  f=$(sed -n "${pick}p" "$VID_SORTED")
  [ -n "$f" ] || continue
  slot=$((slot+1))
  [ "$slot" -le 8 ] || break
  cp "$f" "$CUR/$(printf '%02d' "$slot")_frame_$(basename "$f")"
  echo "slot $slot (frame $pick): $f"
done
N_CUR=$(ls "$CUR" | wc -l)
echo "curated export set: $N_CUR views in $CUR"
ls -la "$CUR"
[ "$N_CUR" -ge 2 ] || fail "curated set too small ($N_CUR) — LHM++ needs at least 2 views to be meaningful"

# ---------- export the canonical avatar (T-pose Gaussian PLY) ----------
echo "--- exporting T-pose GS PLY (PixelShuffle, gs_render-supported model)"
cd "$REPO" || fail "cd repo"
"$VPY" scripts/inference/to_gs_ply.py \
  --model_name LHMPP-700M-PixelShuffle \
  --model_path "$MODEL" \
  --images_dir "$CUR" \
  --output "$OUT/her_avatar_tpose.ply" || fail "to_gs_ply export"

[ -s "$OUT/her_avatar_tpose.ply" ] || fail "export produced no PLY at $OUT/her_avatar_tpose.ply"
ls -la "$OUT"
echo "AVATAR-BUILD-DONE"
echo "Avatar asset: $OUT/her_avatar_tpose.ply"
echo "Next stage (separate step): turntable / multi-view render of this PLY for David's is-this-her judgment, then pose-driving. Stop the pod if no further pod work is queued — it bills \$3.99/hr while running."

#!/usr/bin/env bash
# MYAI GPU pod bootstrap — run ON a fresh RunPod pod (as root), once.
#
# Idempotent: safe to re-run. Existing clones and downloaded models are
# skipped, so a re-run after an interruption just fills the gaps.
#
# What it does:
#   1. Installs ComfyUI into the network-volume mount ($VOLUME, default
#      /workspace) so it — and the ~128GB of models — survive pod
#      termination. The container disk is wiped when a pod stops; the
#      volume is not.
#   2. Installs ComfyUI-Manager + the custom nodes the Studio workflows
#      need (VideoHelperSuite for video IO, KJNodes utilities).
#   3. Downloads every model in models.txt (next to this script) into
#      ComfyUI/models/<subdir>/ using hf_transfer for parallel speed.
#      Set HF_TOKEN in the environment for gated/faster downloads.
#
# After it finishes:
#   cd /workspace/ComfyUI && python3 main.py --listen 0.0.0.0 --port 8188
# and put the pod's proxy URL in ~/.nanobot/gpu.env on the Studio server
# (see README.md in this directory).

set -euo pipefail

VOLUME="${MYAI_VOLUME:-/workspace}"
COMFY="$VOLUME/ComfyUI"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_TXT="${MODELS_TXT:-$SCRIPT_DIR/models.txt}"


# Network volumes choke on git's lock-and-rename writes ("could not
# write config file ... No such file or directory" — seen live on
# RunPod US-CA-2, Oct 14 2026). Clone onto the local disk first,
# then copy the tree onto the volume.
git_clone_to() {
  local url="$1" dest="$2" stage
  stage="$(mktemp -d)"
  git clone --depth 1 "$url" "$stage/repo"
  rm -rf "$dest"
  mkdir -p "$(dirname "$dest")"
  cp -a "$stage/repo" "$dest"
  rm -rf "$stage"
}

if [ ! -d "$VOLUME" ]; then
  echo "ERROR: volume mount $VOLUME does not exist." >&2
  echo "Create the pod with a network volume mounted at /workspace." >&2
  exit 1
fi
if [ ! -f "$MODELS_TXT" ]; then
  echo "ERROR: models.txt not found at $MODELS_TXT" >&2
  exit 1
fi

echo "== system packages =="
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq || true
  apt-get install -y -qq git ffmpeg python3-pip python3-venv || true
fi

echo "== python deps (hf_transfer) =="
python3 -m pip install -q --upgrade huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "== ComfyUI =="
if [ ! -d "$COMFY/.git" ]; then
  git_clone_to https://github.com/comfyanonymous/ComfyUI.git "$COMFY"
else
  git -C "$COMFY" pull --ff-only || true
fi
python3 -m pip install -q -r "$COMFY/requirements.txt"

echo "== custom nodes =="
clone_node() {
  local url="$1" dir="$2"
  if [ ! -d "$COMFY/custom_nodes/$dir/.git" ]; then
    git_clone_to "$url" "$COMFY/custom_nodes/$dir"
  else
    git -C "$COMFY/custom_nodes/$dir" pull --ff-only || true
  fi
  if [ -f "$COMFY/custom_nodes/$dir/requirements.txt" ]; then
    python3 -m pip install -q -r "$COMFY/custom_nodes/$dir/requirements.txt" \
      || true
  fi
}
# Manager: install further nodes from the ComfyUI UI when needed.
clone_node https://github.com/ltdrdata/ComfyUI-Manager.git ComfyUI-Manager
# Video IO (VHS_LoadVideo / VHS_VideoCombine) — required by gpu_workflows.
clone_node https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
  ComfyUI-VideoHelperSuite
# General utility nodes used by many community workflows (escape hatch).
clone_node https://github.com/kijai/ComfyUI-KJNodes.git ComfyUI-KJNodes

echo "== model directories =="
for sub in diffusion_models text_encoders vae loras checkpoints \
           clip_vision controlnet upscale_models; do
  mkdir -p "$COMFY/models/$sub"
done

echo "== models (from $MODELS_TXT) =="
# Line format: <hf_repo> <path_in_repo> <target_subdir>
grep -v '^[[:space:]]*#' "$MODELS_TXT" | grep -v '^[[:space:]]*$' | \
while read -r repo path_in_repo subdir; do
  fname="$(basename "$path_in_repo")"
  dest="$COMFY/models/$subdir/$fname"
  if [ -s "$dest" ]; then
    echo "skip (present): $fname"
    continue
  fi
  echo "downloading: $repo :: $path_in_repo -> models/$subdir/$fname"
  # hf download stages into a local dir mirroring the repo path; move the
  # single file flat into models/<subdir>/ where ComfyUI's loaders look.
  stage="$(mktemp -d)"
  hf download "$repo" "$path_in_repo" --local-dir "$stage"
  mv "$stage/$path_in_repo" "$dest"
  rm -rf "$stage"
done

echo
echo "== done =="
du -sh "$COMFY/models" || true
echo "Start ComfyUI with:"
echo "  cd $COMFY && python3 main.py --listen 0.0.0.0 --port 8188"
echo "Then set GPU_BASE_URL on the Studio server (see runpod/README.md)."

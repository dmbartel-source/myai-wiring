#!/bin/bash
# BAKEOFF-PFLHM-V1 — PF-LHM (LHM++) view-count bake-off on a RunPod pod.
# Bakes avatars from 1 / 8 / 16 graded reference views, renders each with
# the same stock motion, and drops the three videos (plus a T-pose PLY of
# the 16-view avatar) into ComfyUI/output/bakeoff/ for pickup.
# Inputs arrive via ComfyUI/input/bakeoff/set{1,8,16}/ (uploaded through
# the ComfyUI API while this script installs). Idempotent: re-running
# skips stages whose outputs already exist.
set -u
export TORCH_CUDA_ARCH_LIST="9.0"
REPO=/workspace/LHM-plusplus
VENV=/workspace/lhmpp-venv
OUT=/workspace/ComfyUI/output/bakeoff
mkdir -p "$OUT"
rm -f "$OUT/BAKEOFF-DONE.txt" "$OUT/BAKEOFF-FAIL.txt"
LHMPP_SHA=906b5d9fb967ab42efb92f6fa55bf22cac86b653
mkdir -p "$OUT"
exec > >(tee -a /workspace/bakeoff.log) 2>&1
fail() { echo "BAKE-FAIL $1"; echo "BAKE-FAIL $1" > "$OUT/BAKEOFF-FAIL.txt"; exit 1; }
stage() { echo "BAKE-STAGE $1 OK"; }

echo "=== BAKEOFF-PFLHM-V1 start $(date -u +%FT%TZ) ==="

# ---- Stage 0: preflight -------------------------------------------------
nvidia-smi --query-gpu=name --format=csv,noheader || fail preflight-gpu
FREE_KB=$(df --output=avail -k /workspace | tail -1)
echo "volume free KB: $FREE_KB"
[ "$FREE_KB" -ge 6000000 ] || fail preflight-disk
if ! curl -s -m 3 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
  # ComfyUI's Python deps live in the pod's ephemeral system python;
  # a fresh pod needs them installed before ComfyUI can start.
  if ! python3 -c "import tqdm" 2>/dev/null; then
    echo "installing ComfyUI requirements into system python..."
    python3 -m pip install -q -r /workspace/ComfyUI/requirements.txt || fail preflight-comfyui-deps
  fi
  echo "starting ComfyUI..."
  (cd /workspace/ComfyUI && nohup python3 main.py --listen 0.0.0.0 --port 8188 >> /workspace/comfyui.log 2>&1 &)
  for i in $(seq 1 150); do
    curl -s -m 3 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && break
    sleep 2
  done
fi
curl -s -m 3 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 || fail preflight-comfyui
stage preflight

# ---- Stage 1: repo at pinned commit -------------------------------------
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/aigc3d/LHM-plusplus "$REPO" || fail clone
fi
git -C "$REPO" checkout -q "$LHMPP_SHA" || fail checkout
[ "$(git -C "$REPO" rev-parse HEAD)" = "$LHMPP_SHA" ] || fail checkout-verify
stage clone

# ---- Stage 2: python 3.10 venv + deps (uv end-to-end) -------------------
# uv performs every install via --python, so the venv never needs its own
# pip (uv venvs ship without one - that is what killed v1-v3).
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user -q uv || fail uv-install
fi
# A venv counts only if a python 3.10 actually runs from it.
if ! "$VENV/bin/python" -c "import sys; assert sys.version_info[:2] == (3, 10)" >/dev/null 2>&1; then
  rm -rf "$VENV"
  uv python install 3.10 || fail uv-python
  uv venv --seed --clear --python 3.10 "$VENV" || fail venv
fi
PY="$VENV/bin/python"
PIP() { uv pip install --python "$PY" "$@"; }
PIPB() { uv pip install --python "$PY" --no-build-isolation "$@"; }
PIP pip setuptools wheel || fail seed-tools
export TORCH_CUDA_ARCH_LIST="9.0"
if ! "$PY" -c "import torch, pytorch3d, pointops, gsplat, spconv, torch_scatter, xformers" 2>/dev/null; then
  PIP torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121 || fail torch
  PIP -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121 || fail xformers
  PIP ninja Cython || fail buildtools
  (cd "$REPO" && PIPB -r requirements.txt) || fail requirements
  PIP "rembg[cpu]" modelscope || fail rembg
  PIP fvcore iopath || fail fvcore
  PIP --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html || fail pytorch3d
  PIP spconv-cu121 || fail spconv
  PIP torch_scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html || fail torch-scatter
  PIPB git+https://github.com/ashawkey/diff-gaussian-rasterization/ || fail dgr
  PIPB git+https://github.com/camenduru/simple-knn/ || fail simple-knn
  PIPB gsplat==1.4.0 || fail gsplat
  (cd "$REPO" && PIPB ./lib/pointops/) || fail pointops
  # pointops' setup.py declares no packages - it only builds the pointops_cuda
  # extension. The importable package is the lib/pointops directory itself
  # (it carries __init__.py), so put lib/ on the venv path via a .pth file.
  echo "$REPO/lib" > "$VENV/lib/python3.10/site-packages/lhmpp_lib.pth"
fi
"$PY" -c "import torch, pytorch3d, pointops, gsplat, spconv, torch_scatter, xformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || fail imports
echo "BAKE-IMPORTS-OK"

# ---- Stage 3: weights + motion videos ------------------------------------
cd "$REPO" || fail cd
CKPT=./pretrained_models/Damo_XR_Lab/LHMPP-700M-PixelShuffle
if [ ! -d "$CKPT" ]; then
  "$PY" scripts/download_pretrained_models.py --prior || fail download-prior
  "$PY" -c "from modelscope import snapshot_download; snapshot_download(model_id='Damo_XR_Lab/LHMPP-700M-PixelShuffle', cache_dir='./pretrained_models')" || fail download-weights
fi
[ -d "$CKPT" ] || fail weights-missing
if [ ! -d ./motion_video ]; then
  "$PY" scripts/download_motion_video.py || fail download-motion
fi
MOTION=""
for cand in ./motion_video/Dance_I/Dance_I.mp4 ./motion_video/TAICHI/TAICHI.mp4; do
  [ -f "$cand" ] && MOTION="$cand" && break
done
if [ -z "$MOTION" ]; then
  MOTION=$(find ./motion_video -name "*.mp4" | head -1)
fi
[ -n "$MOTION" ] || fail motion-missing
echo "motion: $MOTION"
stage downloads

# ---- Stage 4: wait for the input image sets ------------------------------
declare -A WANT=( [set1]=1 [set8]=8 [set16]=16 )
for s in set1 set8 set16; do
  n=0
  for i in $(seq 1 120); do
    n=$(ls /workspace/ComfyUI/input/bakeoff/$s/*.jpg 2>/dev/null | wc -l)
    [ "$n" -ge "${WANT[$s]}" ] && break
    sleep 20
  done
  [ "$n" -ge "${WANT[$s]}" ] || fail "images-$s"
  mkdir -p "bake_inputs/$s"
  cp /workspace/ComfyUI/input/bakeoff/$s/*.jpg "bake_inputs/$s/"
  echo "$s: $(ls bake_inputs/$s | wc -l) images"
done
stage images

# ---- Stage 5: the three bakes ---------------------------------------------
for s in set1 set8 set16; do
  N=${WANT[$s]}
  if [ ! -f "$OUT/bakeoff_views$N.mp4" ]; then
    "$PY" scripts/test/test_app_case.py \
      --model_name LHMPP-700M-PixelShuffle \
      --model_path "$CKPT" \
      --image_glob "./bake_inputs/$s/*.jpg" \
      --motion_video "$MOTION" \
      --ref_view "$N" \
      --output_dir "./bake_outputs/$s" || fail "bake-$s"
    VID=$(find "./bake_outputs/$s" -name "*.mp4" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d" " -f2-)
    [ -n "$VID" ] || fail "bake-output-$s"
    cp "$VID" "$OUT/bakeoff_views$N.mp4" || fail "bake-copy-$s"
  fi
  echo "bake $s -> $OUT/bakeoff_views$N.mp4"
done
stage bakes

# ---- Stage 6: T-pose PLY of the 16-view avatar (asset; non-fatal) ---------
if [ ! -f "$OUT/avatar_16view_tpose.ply" ]; then
  "$PY" scripts/inference/to_gs_ply.py \
    --model_path "$CKPT" \
    --image_glob "./bake_inputs/set16/*.jpg" \
    --output "$OUT/avatar_16view_tpose.ply" || echo "PLY export failed (non-fatal)"
fi
stage ply

echo "BAKEOFF-DONE"
echo "BAKEOFF-DONE $(date -u +%FT%TZ)" > "$OUT/BAKEOFF-DONE.txt"

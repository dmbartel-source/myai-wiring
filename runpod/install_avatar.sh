#!/bin/bash
# AVATAR-INSTALL-V2 — install the LHM++ virtual-body avatar stack on the pod.
# Everything lands in a DEDICATED venv (/workspace/avatar-venv) and repo dir
# (/workspace/LHM-plusplus) so ComfyUI's own python environment is untouched.
# Safe to re-run: completed steps are skipped.
# Ends with AVATAR-INSTALL-DONE on success; log at /workspace/avatar_install.log
set -u
LOG=/workspace/avatar_install.log
exec > >(tee -a "$LOG") 2>&1
echo "=== AVATAR-INSTALL-V2 starting $(date) ==="

fail() { echo "AVATAR-INSTALL-FAILED: $1"; exit 1; }

# ---------- pre-checks ----------
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
FREE_GB=$(df -BG /workspace | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "free space on /workspace: ${FREE_GB}G"
[ "${FREE_GB:-0}" -ge 18 ] || fail "need >=18G free on /workspace, have ${FREE_GB}G"
python3 --version || fail "python3 missing"

# ---------- uv (provides a managed python 3.10 for the venv) ----------
if ! command -v uv >/dev/null 2>&1; then
  echo "--- installing uv"
  python3 -m pip install -q uv || curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv install"
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version || fail "uv not usable"

# ---------- venv ----------
VENV=/workspace/avatar-venv
if [ ! -x "$VENV/bin/python" ]; then
  echo "--- creating venv (python 3.10) at $VENV"
  uv venv --python 3.10 "$VENV" || fail "venv create"
fi
VPY="$VENV/bin/python"
"$VPY" --version
UPIP() { uv pip install --python "$VPY" "$@"; }
UPIP_NBI() { uv pip install --python "$VPY" --no-build-isolation "$@"; }
# uv venvs ship without pip/setuptools; several packages in this stack
# (chumpy, the CUDA extensions) build from source and import them.
echo "--- seeding pip/setuptools/wheel into the venv"
UPIP pip setuptools wheel || fail "seed pip/setuptools"

# CUDA_HOME for the compiled extensions (pointops / rasterizer / simple-knn)
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -d /usr/local/cuda ]; then export CUDA_HOME=/usr/local/cuda;
  elif [ -d /usr/local/cuda-12.1 ]; then export CUDA_HOME=/usr/local/cuda-12.1; fi
fi
echo "CUDA_HOME=${CUDA_HOME:-unset} ; nvcc: $(command -v nvcc || echo none)"

# ---------- torch stack (LHM++ pins torch 2.3.0 / cu121 / py310) ----------
echo "--- torch 2.3.0 + xformers"
UPIP torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121 || fail "torch"
UPIP -U "xformers==0.0.26.post1" --index-url https://download.pytorch.org/whl/cu121 || fail "xformers"
if ! command -v nvcc >/dev/null 2>&1 && [ -z "${CUDA_HOME:-}" ]; then
  echo "--- no nvcc found; installing pip nvcc for extension builds"
  UPIP "nvidia-cuda-nvcc-cu12==12.1.105" || fail "nvcc"
  export CUDA_HOME="$VENV/lib/python3.10/site-packages/nvidia/cuda_nvcc"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# ---------- clone LHM++ ----------
REPO=/workspace/LHM-plusplus
if [ ! -d "$REPO/.git" ]; then
  echo "--- cloning LHM-plusplus"
  git clone --depth 1 https://github.com/aigc3d/LHM-plusplus "$REPO" || fail "clone"
fi
cd "$REPO" || fail "cd repo"

# ---------- python deps ----------
echo "--- requirements.txt + extras"
UPIP_NBI -r requirements.txt || fail "requirements"
UPIP "rembg[cpu]" spconv-cu121 || fail "extras"
echo "--- prebuilt wheels (torch_scatter / pytorch3d / gsplat)"
UPIP "https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp310-cp310-linux_x86_64.whl" || fail "torch_scatter"
UPIP --index-url https://pypi.org/simple fvcore iopath "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl" || fail "pytorch3d"
UPIP "https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt23cu121-cp310-cp310-linux_x86_64.whl" || echo "WARN gsplat wheel failed (optional renderer dep)"

echo "--- compiled extensions"
if ! "$VPY" -c "import pointops" 2>/dev/null; then
  ( cd lib/pointops && "$VPY" setup.py install ) || fail "pointops build"
fi
UPIP_NBI "git+https://github.com/ashawkey/diff-gaussian-rasterization/" || fail "diff-gaussian-rasterization"
UPIP_NBI "git+https://github.com/camenduru/simple-knn/" || fail "simple-knn"

# ---------- weights (priors incl. SMPL-X human_model_files + LHMPP-700M) ----------
echo "--- downloading weights (one-click script; HF first, ModelScope fallback)"
"$VPY" scripts/download_all.py || fail "download_all"

# ---------- smoke test ----------
echo "--- smoke test"
"$VPY" - <<'EOF' || fail "smoke imports"
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
import pytorch3d, pointops
print("pytorch3d + pointops OK")
EOF
ls pretrained_models/ | head
echo "AVATAR-INSTALL-DONE"

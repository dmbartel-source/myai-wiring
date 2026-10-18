#!/bin/bash
# GATE-TEST-V2 — MYAI measured-body gate test (plan GATE_TEST_PLAN.md).
# Builds a dedicated venv, proves every precondition, then runs
# gate_fit.py: segment 7 generated-set candidates -> select 4 heroes
# by measured azimuth -> joint SMPL-X silhouette fit (shared shape +
# per-vertex offsets) -> metrics + GATE-VERDICT (pass/kill/middle).
# Run ON THE POD, detached:
#   setsid nohup bash /workspace/gate_test.sh > /workspace/gate_test.log 2>&1 < /dev/null &
set -u
echo "GATE-TEST-V2 starting $(date -u)"
cd /workspace
VENV=/workspace/gatefit-venv
WORK=/workspace/gate_test
IN=/workspace/ComfyUI/input/gate_candidates
set -x
( while :; do cp -f /workspace/gate_test.log /workspace/ComfyUI/output/gate_test_log_copy.txt 2>/dev/null; sleep 60; done ) &
MIRROR_PID=$!
trap 'kill $MIRROR_PID 2>/dev/null' EXIT
echo "== ENV DUMP"
date -u; hostname; nvidia-smi || true; df -h /workspace

echo "== PREFLIGHT"
FREE_GB=$(df -BG /workspace | tail -1 | awk '{print $4}' | tr -d 'G')
echo "volume free: ${FREE_GB}G"
if [ "$FREE_GB" -lt 10 ]; then echo "GATE-FAIL preflight: <10G free"; exit 1; fi
NT=$(ls "$IN" 2>/dev/null | wc -l)
echo "candidates=$NT (expect 7)"
if [ "$NT" -ne 7 ]; then echo "GATE-FAIL preflight: candidate uploads incomplete ($NT/7)"; exit 1; fi
if ! command -v uv > /dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.10 "$VENV"
fi
echo "-- installing fit stack (torch 2.3.0 cu121 + pytorch3d wheel + smplx + rembg)"
uv pip install --python "$VENV/bin/python" torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$VENV/bin/python" pytorch3d fvcore iopath \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html
uv pip install --python "$VENV/bin/python" smplx rembg "onnxruntime==1.20.1" scipy pillow numpy
echo "-- imports (the exact modules the test runs)"
if "$VENV/bin/python" -c "import torch, pytorch3d, smplx, rembg, scipy; print('IMPORT-OK torch', torch.__version__, 'cuda', torch.cuda.is_available())"; then
  echo "import path OK"
else
  echo "GATE-FAIL preflight: fit stack imports broken"
  exit 1
fi
echo "-- metric selftest (synthetic masks)"
if "$VENV/bin/python" /workspace/gate_fit.py selftest; then
  echo "selftest OK"
else
  echo "GATE-FAIL preflight: metric selftest failed"
  exit 1
fi
echo "-- SMPL-X model files on the volume"
SMPLX_DIR=$("$VENV/bin/python" -c "
import sys; sys.path.insert(0, '/workspace')
from gate_fit import find_smplx
print(find_smplx() or '')")
if [ -n "$SMPLX_DIR" ]; then
  echo "ok: SMPL-X models at $SMPLX_DIR"
else
  echo "GATE-FAIL preflight: SMPLX_FEMALE.npz not found anywhere on the volume"
  echo "RESOLUTION: SMPL-X files are registration-gated (free research account) — Noor resolves access before any rerun; no fit money spent."
  exit 1
fi
echo "PREFLIGHT-OK - every precondition verified"

echo "STAGE-BEGIN GATE RUN at $(date -u +%s) $(date -u)"
"$VENV/bin/python" /workspace/gate_fit.py run --input "$IN" --work "$WORK"
echo "GATE-TEST-SCRIPT-DONE at $(date -u)"

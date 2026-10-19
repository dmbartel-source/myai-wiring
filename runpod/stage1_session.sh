#!/bin/bash
# STAGE1-SESSION-V1 — MYAI measured-body STAGE-1 scaffold fit
# (design: STAGE1_FITTER_DESIGN.md; fitter: stage1_fit.py STAGE1-FIT-V1).
# Proves every precondition, then runs the fit:
#   preflight -> ingest (39 manifest views) -> S1 cameras -> S2 init
#   -> S3 joint fit -> outlier rejection -> S4 canonical offsets
#   -> LOVO profile holdout -> S5 export + geometry validation.
# Expects these files already fetched to /workspace by the launch:
#   stage1_fit.py, gate_fit.py (dependency, unmodified),
#   view_manifest.json, mesh_validate.py.
# Run ON THE POD, detached:
#   setsid nohup bash /workspace/stage1_session.sh > /workspace/stage1.log 2>&1 < /dev/null &
set -u
echo "STAGE1-SESSION-V1 starting $(date -u)"
cd /workspace
exec 9>/workspace/stage1.lock
if ! flock -n 9; then echo "STAGE1-ALREADY-RUNNING (lock held) — this launch exits"; exit 0; fi
VENV=/workspace/gatefit-venv
WORK=/workspace/stage1
IN=/workspace/ComfyUI/input/stage1_views
LOG=/workspace/stage1.log
set -x
( while :; do cp -f "$LOG" /workspace/ComfyUI/output/stage1_log_copy.txt 2>/dev/null; sleep 20; done ) &
MIRROR_PID=$!
trap 'kill $MIRROR_PID 2>/dev/null' EXIT
echo "== ENV DUMP"
date -u; hostname; nvidia-smi || true; df -h /workspace

echo "== PREFLIGHT"
FREE_GB=$(df -BG /workspace | tail -1 | awk '{print $4}' | tr -d 'G')
echo "volume free: ${FREE_GB}G"
if [ "$FREE_GB" -lt 10 ]; then echo "STAGE1-FAIL preflight: <10G free"; exit 1; fi
for f in /workspace/stage1_fit.py /workspace/gate_fit.py /workspace/view_manifest.json /workspace/mesh_validate.py; do
  if [ ! -f "$f" ]; then echo "STAGE1-FAIL preflight: missing $f (launch fetch incomplete)"; exit 1; fi
done
echo "launch files OK"
if [ ! -x "$VENV/bin/python" ]; then
  echo "STAGE1-FAIL preflight: gatefit venv missing at $VENV (the gate session builds it — run gate_test.sh setup first)"
  exit 1
fi
NT=$(ls "$IN" 2>/dev/null | wc -l)
echo "stage1 input files=$NT (expect 39 manifest views)"
if [ "$NT" -ne 39 ]; then echo "STAGE1-FAIL preflight: stage1 view uploads incomplete ($NT/39)"; exit 1; fi
echo "-- imports (the exact modules the fit runs)"
if "$VENV/bin/python" -c "import torch, pytorch3d, smplx, rembg, scipy, PIL, numpy; print('IMPORT-OK torch', torch.__version__, 'cuda', torch.cuda.is_available())"; then
  echo "import path OK"
else
  echo "STAGE1-FAIL preflight: fit stack imports broken"
  exit 1
fi
echo "-- fitter selftest (torch-free helpers, synthetic meshes)"
if "$VENV/bin/python" -u /workspace/stage1_fit.py selftest; then
  echo "selftest OK"
else
  echo "STAGE1-FAIL preflight: stage1_fit selftest failed"
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
  echo "STAGE1-FAIL preflight: SMPLX_FEMALE.npz not found anywhere on the volume"
  exit 1
fi
echo "-- optional sidecars (absence is reported by the fit, never fatal)"
for s in "$WORK/keypoints.json" "$WORK/init.json"; do
  if [ -f "$s" ]; then echo "sidecar present: $s"; else echo "sidecar absent: $s"; fi
done
for d in "$WORK/normals" "$WORK/pointmaps"; do
  if [ -d "$d" ]; then echo "evidence dir present: $d ($(ls "$d" | wc -l) files)"; else echo "evidence dir absent: $d"; fi
done
echo "PREFLIGHT-OK - every precondition verified"

echo "STAGE-BEGIN STAGE1 FIT at $(date -u +%s) $(date -u)"
"$VENV/bin/python" -u /workspace/stage1_fit.py run \
  --manifest /workspace/view_manifest.json \
  --input "$IN" \
  --work "$WORK" \
  --validator /workspace/mesh_validate.py
echo "stage1_fit exit: $?"
echo "STAGE1-SCRIPT-DONE at $(date -u)"

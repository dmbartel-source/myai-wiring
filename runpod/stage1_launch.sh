#!/bin/bash
# STAGE1-LAUNCH-V9 — stage-1 session launcher (runs DETACHED on the pod).
# Chain: fetch pinned files -> marker greps -> rtmlib install ->
# keypoint extraction -> SYNTH GATE (known-body recovery; tolerances
# below) -> setsid-launch stage1_session.sh. Any failed gate stops the
# chain with a *-FAIL line and the session is NOT started.
set -u
cd /workspace
LOG=/workspace/stage1_launch.log
( while :; do cp -f "$LOG" /workspace/ComfyUI/output/stage1_launch_log_copy.txt 2>/dev/null; sleep 20; done ) &
MIRROR_PID=$!
trap 'kill $MIRROR_PID 2>/dev/null' EXIT
echo "STAGE1-LAUNCH-V9 starting $(date -u)"
R=https://raw.githubusercontent.com/dmbartel-source/myai-wiring
dl() { curl -fsSL "$R/$1/runpod/$2" -o "/workspace/$2" || { echo "LAUNCH-FAIL fetch $2"; exit 1; }; }
dl 25016e10fea50c6eef8c4521481f3baa4d928603 stage1_fit.py
dl 885cc23d1fd28bd6facba5ca6c6ac33e2d1b5661 stage1_session.sh
dl c05725dad25962a1fad90bc509041a2d9418969c extract_keypoints.py
dl 7aad95e17262c1eea2626d927ca44df235bcfa3f view_manifest.json
dl b7d154c94ff5f6d63980c63129ca55a2bc91db01 mesh_validate.py
dl 096fbdc35ffb0334ba9058f805dd8cfba78d239e gate_fit.py
for pair in "stage1_fit.py STAGE1-FIT-V6" "stage1_session.sh STAGE1-SESSION-V1" \
            "extract_keypoints.py EXTRACT-KEYPOINTS-V1" "view_manifest.json STAGE1-VIEW-MANIFEST-V1" \
            "mesh_validate.py MESH-VALIDATE-V1" "gate_fit.py GATE-FIT-V8"; do
  set -- $pair
  grep -q "$2" "/workspace/$1" || { echo "LAUNCH-FAIL marker $2 missing in $1"; exit 1; }
done
echo "FETCH-OK all six files, markers verified"
mkdir -p /workspace/stage1
VENV=/workspace/gatefit-venv
echo "== rtmlib install"
if "$VENV/bin/python" -c "import rtmlib" 2>/dev/null; then
  echo "RTMLIB already present"
elif "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  "$VENV/bin/python" -m pip install -q rtmlib && echo "RTMLIB-INSTALL-VIA pip"
elif command -v uv >/dev/null 2>&1; then
  uv pip install --python "$VENV/bin/python" rtmlib && echo "RTMLIB-INSTALL-VIA uv"
else
  "$VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1
  "$VENV/bin/python" -m pip install -q rtmlib && echo "RTMLIB-INSTALL-VIA ensurepip+pip"
fi
"$VENV/bin/python" -c "import importlib.metadata as md; from rtmlib import Body; print('RTMLIB-OK', md.version('rtmlib'))" || { echo "LAUNCH-FAIL rtmlib unusable"; exit 1; }
echo "== keypoint extraction"
"$VENV/bin/python" -u /workspace/extract_keypoints.py \
  --input /workspace/ComfyUI/input/stage1_views \
  --out /workspace/stage1/keypoints.json --device cuda || { echo "LAUNCH-FAIL extractor crashed"; exit 1; }
NKP=$("$VENV/bin/python" -c "import json; print(len(json.load(open('/workspace/stage1/keypoints.json'))))")
echo "keypoints extracted for $NKP/39 views"
if [ "$NKP" -lt 30 ]; then echo "LAUNCH-FAIL keypoints too sparse ($NKP/39) — fit would run weakened; stopping for review"; exit 1; fi
if [ "$NKP" -lt 35 ]; then echo "KEYPOINTS-PARTIAL ($NKP/39) — proceeding, fit will report the gaps"; fi
echo "== dependency preflight (the session's import list, checked early, plus scipy top-up)"
"$VENV/bin/python" -c "import scipy" 2>/dev/null || uv pip install --python "$VENV/bin/python" scipy || "$VENV/bin/python" -m pip install -q scipy || true
if "$VENV/bin/python" -c "import torch, pytorch3d, smplx, rembg, scipy, PIL, numpy; print('IMPORT-OK torch', torch.__version__, 'cuda', torch.cuda.is_available())"; then
  echo "import path OK"
else
  echo "LAUNCH-FAIL dependency preflight: fit stack imports broken"; exit 1
fi
echo "== SYNTH GATE (known-body recovery before any real photo is fitted)"
( while :; do cp -f /workspace/stage1_synth.log /workspace/ComfyUI/output/stage1_synth_log_copy.txt 2>/dev/null; sleep 15; done ) &
SYNTH_MIRROR_PID=$!
"$VENV/bin/python" -u /workspace/stage1_fit.py synth \
  --manifest /workspace/view_manifest.json \
  --input /workspace/ComfyUI/input/stage1_views \
  --work /workspace/stage1 \
  --synth-steps 300 --validator /workspace/mesh_validate.py > /workspace/stage1_synth.log 2>&1
kill $SYNTH_MIRROR_PID 2>/dev/null
cp -f /workspace/stage1_synth.log /workspace/ComfyUI/output/stage1_synth_log_copy.txt 2>/dev/null
REC=$(grep "SYNTH-RECOVERY" /workspace/stage1_synth.log | tail -1)
echo "synth: $REC"
[ -n "$REC" ] || { echo "SYNTH-GATE-FAIL: no recovery line — synth log tail follows"; tail -60 /workspace/stage1_synth.log; exit 1; }
cp -f /workspace/stage1/synth_true.ply /workspace/ComfyUI/output/synth_true.ply 2>/dev/null
cp -f /workspace/stage1/synth_recovered.ply /workspace/ComfyUI/output/synth_recovered.ply 2>/dev/null
echo "SYNTH-PLY-COPIED to ComfyUI/output (synth_true.ply, synth_recovered.ply) — copied before the verdict so a FAIL is still inspectable"
"$VENV/bin/python" - "$REC" <<'PY' || { echo "SYNTH-GATE-FAIL: recovery outside tolerances (surface mean<=25mm, offset RMSE<=20mm, beta mean30<=0.15) — session NOT started"; exit 1; }
import re, sys
m = re.search(r"beta max\|err\| ([\d.]+) \| beta mean\|err\|30 ([\d.]+) \| offset RMSE ([\d.]+) mm \| chest circumference err ([\d.]+|None) cm \| surface mean ([\d.]+) mm", sys.argv[1])
assert m, "unparseable recovery line"
beta_max, beta_mean30, rmse, chest, surf = float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4), float(m.group(5))
ok = beta_mean30 <= 0.15 and rmse <= 20.0 and surf <= 25.0
print(f"SYNTH-GATE {'PASS' if ok else 'FAIL'} beta_mean30={beta_mean30} beta_max_diag={beta_max} rmse_mm={rmse} surf_mean_mm={surf} chest_cm={chest}")
sys.exit(0 if ok else 1)
PY
echo "== launching the stage-1 session (detached)"
setsid nohup bash /workspace/stage1_session.sh > /workspace/stage1.log 2>&1 < /dev/null &
echo "STAGE1-SESSION-LAUNCHED-DETACHED"
echo "LAUNCH-DONE"

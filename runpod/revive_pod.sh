#!/bin/bash
# REVIVE-POD-V1
# MYAI pod revive — run in the pod web terminal on any FRESH pod (or after
# an Edit-pod reset wiped the container):
#   cd /workspace && curl -fsSL -O https://raw.githubusercontent.com/dmbartel-source/myai-wiring/main/runpod/revive_pod.sh && grep -c REVIVE-POD-V1 revive_pod.sh && bash revive_pod.sh
# Reinstalls every Python package ComfyUI + its node packs need into the
# pod's system python, then (re)launches ComfyUI on 8188. All project data
# lives on the network volume — this only restores container-side packages.
# Safe to re-run.
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /workspace/ComfyUI || { echo "NO /workspace/ComfyUI — is the volume attached?"; exit 1; }
echo "== core requirements"
python3 -m pip install -r requirements.txt
echo "== extras (video nodes, ColorMatch)"
python3 -m pip install opencv-python-headless imageio imageio-ffmpeg color-matcher
echo "== custom node pack requirements"
for f in /workspace/ComfyUI/custom_nodes/*/requirements.txt; do
  [ -f "$f" ] || continue
  echo "-- $f"
  python3 -m pip install -r "$f" || echo "WARN: $f had failures (continuing)"
done
echo "== (re)launch ComfyUI"
pkill -f "python3 main.py" 2>/dev/null
sleep 3
nohup python3 main.py --listen 0.0.0.0 --port 8188 > /workspace/comfyui.log 2>&1 &
echo "REVIVE-DONE — ComfyUI starting, give it ~1 minute"

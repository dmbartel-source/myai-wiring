#!/bin/bash
# MYAI pod install batch — Oct 15, 2026 (run in the pod's web terminal)
# Adds, from the two research sweeps:
#   1. GAP pack nodes (long-video chunking, 1-6 identity binding)
#   2. SCAIL Auto Extend (long-video extension workflows)
#   3. SeedVR2 video upscaler nodes + 7B fp8 model + VAE (detail finish)
#   4. SCAIL-2 Relighting LoRA (lighting-match in the quality recipe)
# Git clones are staged on local disk first: the network volume
# breaks git writes (bootstrap lesson, Oct 14).
set -x
cd /tmp && rm -rf gapx aex sx
git clone --depth 1 https://github.com/GeekatplayStudio/wan-scail2-gap gapx
git clone --depth 1 https://github.com/brobert-in-aus/scail-auto-extend aex
git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler sx
cp -a gapx /workspace/ComfyUI/custom_nodes/wan-scail2-gap
cp -a aex /workspace/ComfyUI/custom_nodes/scail-auto-extend
cp -a sx /workspace/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/requirements.txt
hf download Comfy-Org/SCAIL-2 loras/wan2.1_SCAIL_2_relight_lora_bf16.safetensors --local-dir /tmp/relight
cp /tmp/relight/loras/wan2.1_SCAIL_2_relight_lora_bf16.safetensors /workspace/ComfyUI/models/loras/
mkdir -p /workspace/ComfyUI/models/SEEDVR2
hf download AEmotionStudio/SeedVR2-models seedvr2_ema_7b_fp8_e4m3fn.safetensors ema_vae_fp16.safetensors --local-dir /workspace/ComfyUI/models/SEEDVR2
ls /workspace/ComfyUI/custom_nodes/ | grep -i -E "gap|extend|seedvr"
ls -la /workspace/ComfyUI/models/loras/wan2.1_SCAIL_2_relight_lora_bf16.safetensors /workspace/ComfyUI/models/SEEDVR2/
echo INSTALL-BATCH-DONE
# Then restart ComfyUI:
#   pkill -f "python3 main.py" ; sleep 3
#   cd /workspace/ComfyUI && nohup python3 main.py --listen 0.0.0.0 --port 8188 > /workspace/comfyui.log 2>&1 &

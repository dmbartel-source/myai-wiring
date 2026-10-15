#!/usr/bin/env bash
# MYAI person-LoRA trainer setup (Oct 15, 2026)
#
# Prepares everything the SCAIL-2-Tuner needs on the pod:
#   1. The full-size SCAIL-2 DiT (fp16, ~33 GB) the trainer trains
#      against -- into ComfyUI's diffusion_models (also usable for
#      inference). Downloads go DIRECTLY to the volume: the file
#      is bigger than the 30 GB container disk, no /tmp staging.
#   2. The bf16 UMT5 text encoder (~11 GB) for latent/text caching,
#      into text_encoders (musubi cannot use the fp8-scaled file).
#   3. The SCAIL-2-Tuner fork (fengjia-guo) cloned via local-disk
#      staging (the network volume breaks git writes), with its
#      OWN venv on the volume so training survives pod restarts.
#   4. A non-interactive accelerate config (single GPU, bf16).
#
# Requires the volume expanded first (200 GB). Run in the pod web
# terminal:  bash install_lora_trainer.sh
# Ends with: LORA-TRAINER-READY
set -e
cd /workspace

echo "== 1/4 SCAIL-2 fp16 DiT (trainer checkpoint)"
hf download Comfy-Org/SCAIL-2 \
  diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors \
  --local-dir /workspace/ComfyUI/models

echo "== 2/4 UMT5 bf16 text encoder"
hf download Kijai/WanVideo_comfy umt5-xxl-enc-bf16.safetensors \
  --local-dir /workspace/ComfyUI/models/text_encoders

echo "== 3/4 SCAIL-2-Tuner (clone staged on local disk, venv on volume)"
rm -rf /tmp/scail2-tuner-stage /workspace/SCAIL-2-Tuner
git clone https://github.com/fengjia-guo/SCAIL-2-Tuner \
  /tmp/scail2-tuner-stage
cp -a /tmp/scail2-tuner-stage /workspace/SCAIL-2-Tuner
cd /workspace/SCAIL-2-Tuner
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install tensorboard matplotlib ascii-magic

echo "== 4/4 accelerate config (single GPU, bf16)"
mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'YAML'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML

ls -la /workspace/ComfyUI/models/diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors \
       /workspace/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors
echo LORA-TRAINER-READY

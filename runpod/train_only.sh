#!/bin/bash
# TRAIN-ONLY-V1
# MYAI person-LoRA: relaunch ONLY the training (+ install watcher) from
# stage 2, for when assemble/finalize/cache already completed — e.g. the
# first stage-2 run died at the training launch. The latent cache in
# /workspace/lora_cache is reused as-is.
# Run ON THE POD (web terminal):
#   cd /workspace && curl -fsSL -O https://raw.githubusercontent.com/dmbartel-source/myai-wiring/main/runpod/train_only.sh && grep -c TRAIN-ONLY-V1 train_only.sh && HF_HUB_ENABLE_HF_TRANSFER=0 nohup bash train_only.sh > /workspace/lora_trainonly.log 2>&1 &
set -e
export HF_HUB_ENABLE_HF_TRANSFER=0
TUNER=/workspace/SCAIL-2-Tuner
DIT=/workspace/ComfyUI/models/diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors
VAE=/workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors
CLIP=/workspace/trainer_models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
T5=/workspace/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors

if [ ! -d /workspace/lora_cache ]; then
  echo "NO CACHE at /workspace/lora_cache — run the full stage2_full.sh instead"
  exit 1
fi

echo "== LAUNCH TRAINING (fp16, dim 32, 16 epochs, nohup)"
cd "$TUNER"
nohup .venv/bin/accelerate launch --num_cpu_threads_per_process 1 \
  --mixed_precision fp16 \
  src/musubi_tuner/scail2_train_network.py \
  --cache_directory /workspace/lora_cache \
  --dit "$DIT" --vae "$VAE" --clip "$CLIP" --t5 "$T5" \
  --sdpa --mixed_precision fp16 --fp8_base \
  --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --network_module networks.lora_wan --network_dim 32 \
  --optimizer_type adamw8bit --learning_rate 2e-4 \
  --lr_scheduler cosine --lr_warmup_steps 20 \
  --timestep_sampling shift --discrete_flow_shift 3.0 \
  --guidance_scale 1.0 \
  --scail2_history_prob 0.5 --scail2_history_latent_frames 2 \
  --max_train_epochs 16 --save_every_n_epochs 4 \
  --seed 42 --output_dir /workspace/lora_out --output_name skswoman \
  --logging_dir /workspace/lora_out/logs --log_with tensorboard \
  > /workspace/lora_train.log 2>&1 &
echo "TRAINING LAUNCHED pid $! — watch /workspace/lora_train.log"
echo "Checkpoints land in /workspace/lora_out (skswoman*.safetensors)."

# Post-training install watcher (same as stage 2): once the trainer
# exits, install the finished LoRA into ComfyUI's loras folder.
(
  while pgrep -f scail2_train_network > /dev/null; do sleep 30; done
  f=/workspace/lora_out/skswoman.safetensors
  [ -f "$f" ] || f=$(ls -t /workspace/lora_out/skswoman*.safetensors 2>/dev/null | head -1)
  if [ -n "$f" ] && [ -f "$f" ]; then
    cp "$f" /workspace/ComfyUI/models/loras/skswoman.safetensors
    echo "LORA INSTALLED from $f" >> /workspace/lora_stage2.log
  else
    echo "LORA INSTALL FAILED: no checkpoint in /workspace/lora_out" >> /workspace/lora_stage2.log
  fi
) &
echo "TRAIN-ONLY-DONE"

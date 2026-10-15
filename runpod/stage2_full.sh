#!/bin/bash
# STAGE2-FULL-V1
# MYAI person-LoRA stage 2: assemble the dataset from the overnight
# ComfyUI outputs, finalize it, cache latents, and launch training.
# Run ON THE POD (web terminal):
#   cd /workspace && curl -fsSL -O https://raw.githubusercontent.com/dmbartel-source/myai-wiring/main/runpod/stage2_full.sh && grep -c STAGE2-FULL-V1 stage2_full.sh && HF_HUB_ENABLE_HF_TRANSFER=0 nohup bash stage2_full.sh > /workspace/lora_stage2.log 2>&1 &
# Assembly copies, per prepared sample (tree created by stage 1's
# prepare-image step):
#   output/lora_edits/sampleNNN_R_*.png -> samples/sampleNNN/reference/R.png
#   output/lora_edits/sampleNNN_Z_*.png -> samples/sampleNNN/edited/z0.png
#   output/lora_masks/sampleNNN_Rmask_*.png -> samples/sampleNNN/masks/R_mask.png
#   output/lora_masks/sampleNNN_Zmask_*.png -> samples/sampleNNN/masks/z_mask.png
set -e
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /workspace
TUNER=/workspace/SCAIL-2-Tuner
PY=$TUNER/.venv/bin/python
DS=/workspace/lora_dataset
OUT=/workspace/ComfyUI/output
DIT=/workspace/ComfyUI/models/diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors
VAE=/workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors
CLIP=/workspace/trainer_models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
T5=/workspace/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors

echo "== PREP EXTRAS (David's extra reference photos, if present)"
# Uploaded overnight to ComfyUI input/lora_extra/sample1NN.*; their
# edits + masks were generated into output/lora_edits|lora_masks with
# matching sample ids, so after prepare-image scaffolds them here,
# the generic ASSEMBLE loop below picks them up like the originals.
EX_IN=/workspace/ComfyUI/input/lora_extra
if [ -d "$EX_IN" ] && ls "$EX_IN"/* > /dev/null 2>&1; then
  mkdir -p /workspace/lora_extra_raw
  for f in "$EX_IN"/*; do
    b=$(basename "$f")
    cp "$f" "/workspace/lora_extra_raw/$b"
    echo "skswoman, a woman, photorealistic" \
      > "/workspace/lora_extra_raw/${b%.*}.txt"
  done
  $PY -m musubi_tuner.scail2_prepare_dataset prepare-image \
    --input_dir /workspace/lora_extra_raw --output_dir "$DS" \
    --mode replace --overwrite
  echo "EXTRAS PREPARED"
fi

echo "== ASSEMBLE edits + masks into the dataset tree"
placed=0; skipped=""
for d in "$DS"/samples/*/; do
  id=$(basename "$d")
  r=$(ls "$OUT"/lora_edits/${id}_R_*.png 2>/dev/null | sort | tail -1)
  z=$(ls "$OUT"/lora_edits/${id}_Z_*.png 2>/dev/null | sort | tail -1)
  rmf=$(ls "$OUT"/lora_masks/${id}_Rmask_*.png 2>/dev/null | sort | tail -1)
  zmf=$(ls "$OUT"/lora_masks/${id}_Zmask_*.png 2>/dev/null | sort | tail -1)
  if [ -n "$r" ] && [ -n "$z" ] && [ -n "$rmf" ] && [ -n "$zmf" ]; then
    cp "$r" "$d/reference/R.png"
    cp "$z" "$d/edited/z0.png"
    cp "$rmf" "$d/masks/R_mask.png"
    cp "$zmf" "$d/masks/z_mask.png"
    placed=$((placed+1))
  else
    skipped="$skipped $id"
  fi
done
echo "ASSEMBLED $placed samples; skipped:$skipped"
if [ "$placed" -lt 20 ]; then
  echo "ABORT: fewer than 20 complete samples — do not train yet."
  exit 1
fi

echo "== FINALIZE (validates files, writes the cache manifest)"
$PY -m musubi_tuner.scail2_prepare_dataset finalize --output_dir "$DS"

echo "== RESOLUTION (from sample001 target, snapped to /32)"
RES=$($PY -c "
from PIL import Image
w, h = Image.open('$DS/samples/sample001/target/y.png').size
W = (w // 32) * 32
H = int(round(h * (W / w) / 32.0)) * 32
print(f'{W},{H}')")
echo "cache resolution: $RES"

echo "== CACHE LATENTS (VAE + native CLIP + T5)"
$PY -m musubi_tuner.scail2_cache_latents \
  --manifest "$DS/manifests/scail2_cache_manifest.jsonl" \
  --cache_directory /workspace/lora_cache \
  --resolution "$RES" \
  --vae "$VAE" --clip "$CLIP" --t5 "$T5"
echo "CACHE-DONE"

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

# Post-training: once the trainer exits, install the finished LoRA
# into ComfyUI's loras folder so the Studio Person LoRA toggle can
# load it. Runs detached; logs to lora_stage2.log.
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
echo "STAGE2-DONE"

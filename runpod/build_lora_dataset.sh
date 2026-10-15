#!/usr/bin/env bash
# MYAI person-LoRA dataset, stage 1 (Oct 15, 2026)
#
# Run in the POD web terminal (after install_lora_trainer.sh):
#   cd /workspace && nohup bash build_lora_dataset.sh \
#     > /workspace/lora_stage1.log 2>&1 &
#
# What it does:
#   1. SMOKE TEST FIRST (fail fast): the tuner's dummy dataset ->
#      latent cache -> a 1-step training run. Proves the trainer,
#      checkpoints and arguments are wired before real data work.
#      Ends that section with SMOKE-OK.
#   2. Probes 4707.mp4 (fps / frames / size) from ComfyUI input.
#   3. Extracts every 10th frame as PNG stills.
#   4. Picks 28 evenly spaced frames (skipping head/tail 5%) as
#      training targets, writes trigger captions, and copies the
#      chosen frames into ComfyUI input for the edit stage.
#   5. Runs the tuner's prepare-image (mode replace) to build the
#      dataset skeleton (targets + captions + helper manifests).
# Ends with: DATASET-STAGE1-DONE
set -e
cd /workspace
# The pod image sets HF_HUB_ENABLE_HF_TRANSFER=1 but the tuner venv
# has no hf_transfer package, which makes every huggingface_hub
# call (hf download, tokenizer fetches) die with a ValueError.
# Force the plain HTTP path for this script and all its children.
export HF_HUB_ENABLE_HF_TRANSFER=0
TUNER=/workspace/SCAIL-2-Tuner
PY=$TUNER/.venv/bin/python
DIT=/workspace/ComfyUI/models/diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors
VAE=/workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors
# The tuner's CLIP loader is strict and expects the NATIVE Wan
# checkpoint (full XLM-Roberta CLIP, visual.+textual. keys). The
# ComfyUI clip_vision_h.safetensors (HF-style vision_model.* keys,
# vision tower only) fails with a state_dict RuntimeError -- proven
# in the first smoke run Oct 15. Native file per musubi's Wan docs:
CLIP=/workspace/trainer_models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
T5=/workspace/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors
SRC=/workspace/ComfyUI/input/4707.mp4

echo "== PREP native CLIP checkpoint for the tuner (~4.4 GB)"
mkdir -p /workspace/trainer_models
hf download Wan-AI/Wan2.1-I2V-14B-720P \
  models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --local-dir /workspace/trainer_models
df -h /workspace | tail -1

echo "== SMOKE 1/3 dummy dataset"
$PY -m musubi_tuner.scail2_create_dummy_dataset \
  --output_dir /workspace/lora_dummy \
  --width 256 --height 256 --video_length 17 --overwrite

echo "== SMOKE 2/3 dummy latent cache"
$PY -m musubi_tuner.scail2_cache_latents \
  --manifest /workspace/lora_dummy/manifests/scail2_cache_manifest.jsonl \
  --cache_directory /workspace/lora_dummy_cache \
  --vae "$VAE" --clip "$CLIP" --t5 "$T5"

echo "== SMOKE 3/3 one training step"
# SMOKE-TRAIN-FP16: the SCAIL-2 DiT checkpoint is fp16, and the
# trainer asserts mixed precision must be fp16 (or no) for fp16
# weights -- bf16 aborts the run (proven Oct 15). Flip the global
# accelerate config too so the real training run inherits fp16.
sed -i 's/^mixed_precision:.*/mixed_precision: fp16/' \
  ~/.cache/huggingface/accelerate/default_config.yaml
cd $TUNER
.venv/bin/accelerate launch --num_cpu_threads_per_process 1 \
  --mixed_precision fp16 \
  src/musubi_tuner/scail2_train_network.py \
  --cache_directory /workspace/lora_dummy_cache \
  --dit "$DIT" \
  --sdpa --mixed_precision fp16 --fp8_base \
  --optimizer_type adamw8bit --learning_rate 2e-4 \
  --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --network_module networks.lora_wan --network_dim 8 \
  --timestep_sampling shift --discrete_flow_shift 3.0 \
  --max_train_steps 1 --seed 42 \
  --output_dir /workspace/lora_smoke_out --output_name smoke
cd /workspace
echo SMOKE-OK

echo "== PROBE 4707.mp4"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 "$SRC"

echo "== EXTRACT every 10th frame"
rm -rf /workspace/lora_frames /workspace/lora_raw
mkdir -p /workspace/lora_frames /workspace/lora_raw
ffmpeg -v error -i "$SRC" \
  -vf "select='not(mod(n\,10))'" -vsync vfr \
  /workspace/lora_frames/f%05d.png
ls /workspace/lora_frames | wc -l

echo "== PICK 28 targets + captions + stage inputs for the edit stage"
mapfile -t FRAMES < <(ls /workspace/lora_frames)
N=${#FRAMES[@]}
LO=$(( N * 5 / 100 ))
HI=$(( N - LO ))
SPAN=$(( HI - LO ))
mkdir -p /workspace/ComfyUI/input/lora_targets
for i in $(seq 0 27); do
  IDX=$(( LO + SPAN * i / 27 ))
  SRCF=${FRAMES[$IDX]}
  ID=$(printf "sample%03d" $(( i + 1 )))
  cp "/workspace/lora_frames/$SRCF" "/workspace/lora_raw/$ID.png"
  printf "skswoman, a woman at the beach in swimwear, full body\n" \
    > "/workspace/lora_raw/$ID.txt"
  cp "/workspace/lora_frames/$SRCF" \
    "/workspace/ComfyUI/input/lora_targets/$ID.png"
  echo "$ID <- $SRCF"
done

echo "== PREPARE dataset skeleton (mode replace)"
$PY -m musubi_tuner.scail2_prepare_dataset prepare-image \
  --input_dir /workspace/lora_raw \
  --output_dir /workspace/lora_dataset \
  --mode replace
ls /workspace/lora_dataset/samples | wc -l
echo DATASET-STAGE1-DONE

#!/bin/bash
# STAGE2-V3-V1
# MYAI person-LoRA v3, SELF-PAIR PROBE. Built from the v2 record
# (runpod/stage2_v2.sh, STAGE2-V2-V7) with ONLY these deltas:
#   1. Dataset = 55 images from David's graded set (server
#      /home/nanobot/lora_v3_dataset/core, uploaded to the pod as
#      ComfyUI/input/lora_targets_v3). sample001 = 5766.png, his
#      primary full-body reference, so the cache resolution comes
#      from it (v2's rule: resolution from sample001's target).
#   2. SELF-PAIR samples: reference/R.png and edited/z0.png are the
#      target image itself. NO generated edits anywhere (v2's edit
#      set carried blended/drifted images; this run removes that
#      variable entirely).
#   3. Masks come from the tuner vendor's own colored-mask runner
#      (workflows/scail2_colored_mask.py, SAM3.1, prompt "woman"),
#      the documented stage between prepare-image and finalize.
#   4. --network_alpha 32 (v1/v2 shipped alpha 1 against dim 32 --
#      the one confirmed mechanical defect; alpha/dim = 1 is the
#      standard convention this restores).
#   5. PROBE size: 4 epochs, save every 2 (220 steps). Everything
#      else in the train command is byte-identical to v2: dim 32,
#      lr 2e-4, cosine, warmup 20, adamw8bit, fp16 + --fp8_base,
#      discrete_flow_shift 3.0, history 0.5/2, seed 42.
# Output: skswoman_v3.safetensors. v1/v2 are NOT touched.
#
# PREDICTIONS (written before the run, Oct 18 2026):
#   P1 preflight passes; SAM3.1 checkpoint found on the volume
#      (UNVERIFIED until the find runs -- the gate exposes a miss
#      in the first minutes, before any training money burns).
#   P2 scaffold yields 55/55 sample dirs.
#   P3 self-pair places 55 R.png + 55 z0.png, md5-equal to targets.
#   P4 mask stage yields 55/55 mask pairs in ~10-25 min (sequential
#      SAM3.1 runs, 1-frame driving videos).
#   P5 finalize prints "Finalized 55 samples".
#   P6 cache resolution = 832x1248 (from 5766.png); cache completes.
#   P7 train runs 220 steps at ~3.3 s/step (~12-18 min); final
#      avr_loss in the 0.04-0.09 band (4 epochs vs v2's 16);
#      skswoman_v3.safetensors ~= 718 MB (same dim as v2).
#   P8 install lands in ComfyUI/models/loras and is verified there.
# Run ON THE POD, detached:
#   setsid nohup bash /workspace/stage2_v3.sh > /workspace/lora_stage2_v3.log 2>&1 < /dev/null &
set -u
echo "STAGE2-V3-V1 starting $(date -u)"
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /workspace
TUNER=/workspace/SCAIL-2-Tuner
export PYTHONPATH="$TUNER/src${PYTHONPATH:+:$PYTHONPATH}"
PY=$TUNER/.venv/bin/python
DS=/workspace/lora_dataset_v3
RAW=/workspace/lora_raw_v3
DIT=/workspace/ComfyUI/models/diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors
VAE=/workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors
CLIP=/workspace/trainer_models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
T5=/workspace/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors

# ---- EXTREME LOGGING (David's instruction, Oct 17) -----------------
set -x
( while :; do cp -f /workspace/lora_stage2_v3.log /workspace/ComfyUI/output/stage2_v3_log_copy.txt 2>/dev/null; sleep 60; done ) &
MIRROR_PID=$!
trap 'kill $MIRROR_PID 2>/dev/null' EXIT
echo "== ENV DUMP"
date -u
hostname
python3 --version
$PY --version
nvidia-smi || true
df -h /workspace
echo "tuner: $TUNER"
echo "dataset: $DS  cache: /workspace/lora_cache_v3  out: /workspace/lora_out_v3"

echo "== PREFLIGHT: rebuild the tuner venv whole, prove every precondition"
$PY -m ensurepip --upgrade > /dev/null 2>&1 || true
if ! $PY -m pip --version > /dev/null 2>&1; then
  echo "PREFLIGHT-FAIL: venv has no pip even after ensurepip"
  exit 1
fi
cd "$TUNER"
$PY -m pip install -e . av torchvision torchaudio tensorboard matplotlib ascii-magic
cd /workspace
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
echo "-- venv package inventory (pip freeze)"
$PY -m pip freeze
echo "-- model files"
for f in "$DIT" "$VAE" "$CLIP" "$T5"; do
  if [ -f "$f" ]; then echo "ok: $f ($(stat -c%s "$f") bytes)"; else echo "PREFLIGHT-FAIL: missing $f"; exit 1; fi
done
echo "-- mask stage files (vendor runner + workflow + SAM3.1 checkpoint)"
if [ -f "$TUNER/workflows/scail2_colored_mask.py" ] && [ -f "$TUNER/workflows/SCAIL-2_Colored_Mask_api.json" ]; then
  echo "ok: colored-mask runner + workflow JSON present"
else
  echo "PREFLIGHT-FAIL: vendor colored-mask runner or workflow JSON missing in $TUNER/workflows"
  exit 1
fi
SAM3=$(find /workspace/ComfyUI/models -name 'sam3.1_multiplex_fp16.safetensors' 2>/dev/null | head -1)
if [ -n "$SAM3" ]; then echo "ok: SAM3.1 checkpoint at $SAM3"; else echo "PREFLIGHT-FAIL: sam3.1_multiplex_fp16.safetensors not found under ComfyUI/models"; exit 1; fi
if command -v ffmpeg > /dev/null 2>&1; then echo "ok: ffmpeg on PATH"; else echo "PREFLIGHT-FAIL: ffmpeg not on PATH"; exit 1; fi
echo "-- ComfyUI API (the mask stage talks to it)"
if curl -fsS --max-time 10 http://127.0.0.1:8188/system_stats > /dev/null; then
  echo "ok: ComfyUI API up on 127.0.0.1:8188"
else
  echo "PREFLIGHT-FAIL: ComfyUI API not answering on 127.0.0.1:8188 (revive_pod.sh should have started it)"
  exit 1
fi
echo "-- data counts"
NT=$(ls /workspace/ComfyUI/input/lora_targets_v3 2>/dev/null | wc -l)
echo "targets=$NT (expect 55)"
if [ "$NT" -lt 55 ]; then
  echo "PREFLIGHT-FAIL: v3 target uploads incomplete"
  exit 1
fi
echo "-- disk"
FREE_GB=$(df -BG /workspace | tail -1 | awk '{print $4}' | tr -d 'G')
echo "volume free: ${FREE_GB}G"
if [ "$FREE_GB" -lt 10 ]; then
  echo "PREFLIGHT-FAIL: less than 10G free on the volume"
  exit 1
fi
echo "-- imports (the exact modules the build runs)"
if $PY -c "import av, torch, torchvision, requests; import musubi_tuner.scail2_prepare_dataset; import musubi_tuner.scail2_cache_latents; import musubi_tuner.scail2_train_network; print('IMPORT-OK torch', torch.__version__, 'torchvision', torchvision.__version__)"; then
  echo "import path OK"
else
  echo "PREFLIGHT-FAIL: imports still broken after venv rebuild"
  exit 1
fi
if $PY -m py_compile "$TUNER/workflows/scail2_colored_mask.py"; then
  echo "mask runner parses OK"
else
  echo "PREFLIGHT-FAIL: colored-mask runner does not parse"
  exit 1
fi
if [ -x "$TUNER/.venv/bin/accelerate" ]; then
  echo "accelerate present: $($TUNER/.venv/bin/accelerate --version | head -1)"
else
  echo "PREFLIGHT-FAIL: no accelerate in the tuner venv"
  exit 1
fi
echo "PREFLIGHT-OK - every precondition verified"

echo "STAGE-BEGIN SCAFFOLD v3 dataset at $(date -u +%s) $(date -u)"
echo "== SCAFFOLD v3 dataset (sample001 = 5766 primary still, then rest sorted)"
rm -rf "$RAW" "$DS" /workspace/lora_cache_v3 /workspace/lora_out_v3
mkdir -p "$RAW"
FILES=(5766.png 5760.png 5761.png facecand_5785.png faceref_5785_neutral.jpeg lib_4700-1_f_0001.jpg lib_4700-1_f_0003.jpg lib_4700-1_f_0006.jpg lib_4700-1_f_0007.jpg lib_4700-1_f_0008.jpg lib_4700-1_f_0009.jpg lib_4700-1_f_0010.jpg lib_4701_f_0000.jpg lib_4701_f_0003.jpg lib_4701_f_0010.jpg lib_4701_f_0015.jpg lib_4707_f_0000.jpg lib_4707_f_0003.jpg lib_4707_f_0005.jpg lib_4707_f_0009.jpg lib_5765_f_0001.jpg lib_5765_f_0005.jpg lib_5765_f_0006.jpg lib_5765_f_0007.jpg lib_5765_f_0012.jpg lib_5765_f_0013.jpg lib_5784-3_f_0000.jpg lib_5784-3_f_0004.jpg lib_5784-3_f_0007.jpg lib_5784-3_f_0008.jpg lib_5784-3_f_0009.jpg lib_5784-3_f_0010.jpg lib_5784-3_f_0011.jpg lib_5784-3_f_0012.jpg lib_5784-3_f_0016.jpg lib_5784-3_f_0019.jpg lib_5817_f_0000.jpg lib_5817_f_0001.jpg lib_5817_f_0002.jpg lib_5817_f_0003.jpg lib_5817_f_0005.jpg lib_5817_f_0008.jpg lib_5817_f_0009.jpg lib_5817_f_0011.jpg lib_5817_f_0013.jpg lib_5817_f_0015.jpg lib_5819_f_0000.jpg lib_5819_f_0003.jpg lib_5819_f_0005.jpg lib_5819_f_0006.jpg lib_5819_f_0007.jpg lib_5819_f_0008.jpg lib_5819_f_0010.jpg lib_5819_f_0011.jpg lib_5819_f_0012.jpg)
i=0
for f in "${FILES[@]}"; do
  i=$((i+1))
  ID=$(printf "sample%03d" $i)
  cp "/workspace/ComfyUI/input/lora_targets_v3/$f" "$RAW/$ID.${f##*.}"
  printf "skswoman, a woman, photorealistic\n" > "$RAW/$ID.txt"
  echo "$ID <- $f"
done
$PY -m musubi_tuner.scail2_prepare_dataset prepare-image \
  --input_dir "$RAW" --output_dir "$DS" --mode replace
NSAMP=$(ls "$DS/samples" 2>/dev/null | wc -l)
echo "SCAFFOLD-DONE samples: $NSAMP"
if [ "$NSAMP" -ne 55 ]; then
  echo "SCAFFOLD-FAILED: $NSAMP sample dirs, expected exactly 55 - aborting"
  exit 1
fi

echo "STAGE-BEGIN SELF-PAIR at $(date -u +%s) $(date -u)"
echo "== SELF-PAIR: R.png and z0.png are the target itself (no generated edits)"
placed=0
for d in "$DS"/samples/*/; do
  [ -d "$d" ] || continue
  cp "$d/target/y.png" "$d/reference/R.png"
  cp "$d/target/y.png" "$d/edited/z0.png"
  if [ "$(md5sum < "$d/target/y.png")" = "$(md5sum < "$d/reference/R.png")" ] && [ "$(md5sum < "$d/target/y.png")" = "$(md5sum < "$d/edited/z0.png")" ]; then
    placed=$((placed+1))
  else
    echo "SELF-PAIR-FAIL: md5 mismatch in $d"
    exit 1
  fi
done
echo "SELF-PAIR-DONE $placed samples (md5-verified against targets)"
if [ "$placed" -ne 55 ]; then
  echo "SELF-PAIR-FAILED: $placed/55 - aborting"
  exit 1
fi

echo "STAGE-BEGIN MASKS (vendor colored-mask runner, SAM3.1) at $(date -u +%s) $(date -u)"
echo "== MASKS: scail2_colored_mask.py --dataset-dir, prompt woman"
$PY "$TUNER/workflows/scail2_colored_mask.py" \
  --dataset-dir "$DS" --prompt woman --ref-prompt woman
masked=0
for d in "$DS"/samples/*/; do
  [ -d "$d" ] || continue
  if [ -s "$d/masks/R_mask.png" ] && [ -s "$d/masks/z_mask.png" ]; then
    masked=$((masked+1))
  else
    echo "mask missing in $d"
  fi
done
echo "MASKS-DONE $masked/55 samples with both masks"
if [ "$masked" -ne 55 ]; then
  echo "MASK-FAILED: only $masked/55 complete - aborting before finalize"
  exit 1
fi

echo "== FINALIZE"
$PY -m musubi_tuner.scail2_prepare_dataset finalize --output_dir "$DS"

echo "== RESOLUTION (from sample001 target, snapped to /32)"
RES=$($PY -c "
from PIL import Image
w, h = Image.open('$DS/samples/sample001/target/y.png').size
W = (w // 32) * 32
H = int(round(h * (W / w) / 32.0)) * 32
print(f'{W},{H}')")
echo "cache resolution: $RES"

echo "STAGE-BEGIN CACHE LATENTS at $(date -u +%s) $(date -u)"
echo "== CACHE LATENTS"
$PY -m musubi_tuner.scail2_cache_latents \
  --manifest "$DS/manifests/scail2_cache_manifest.jsonl" \
  --cache_directory /workspace/lora_cache_v3 \
  --resolution "$RES" \
  --vae "$VAE" --clip "$CLIP" --t5 "$T5"
echo "CACHE-DONE"

echo "STAGE-BEGIN TRAIN (fp16, alpha 32, PROBE 4 epochs) at $(date -u +%s) $(date -u)"
echo "== TRAIN (fp16, dim 32, alpha 32, 4 epochs probe, output skswoman_v3)"
cd "$TUNER"
.venv/bin/accelerate launch --num_cpu_threads_per_process 1 \
  --mixed_precision fp16 \
  src/musubi_tuner/scail2_train_network.py \
  --cache_directory /workspace/lora_cache_v3 \
  --dit "$DIT" --vae "$VAE" --clip "$CLIP" --t5 "$T5" \
  --sdpa --mixed_precision fp16 --fp8_base \
  --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --network_module networks.lora_wan --network_dim 32 --network_alpha 32 \
  --optimizer_type adamw8bit --learning_rate 2e-4 \
  --lr_scheduler cosine --lr_warmup_steps 20 \
  --timestep_sampling shift --discrete_flow_shift 3.0 \
  --guidance_scale 1.0 \
  --scail2_history_prob 0.5 --scail2_history_latent_frames 2 \
  --max_train_epochs 4 --save_every_n_epochs 2 \
  --seed 42 --output_dir /workspace/lora_out_v3 --output_name skswoman_v3 \
  --logging_dir /workspace/lora_out_v3/logs --log_with tensorboard
echo "TRAIN-DONE"

echo "== INSTALL"
f=/workspace/lora_out_v3/skswoman_v3.safetensors
[ -f "$f" ] || f=$(ls -t /workspace/lora_out_v3/skswoman_v3*.safetensors 2>/dev/null | head -1)
if [ -n "$f" ] && [ -f "$f" ]; then
  cp "$f" /workspace/ComfyUI/models/loras/skswoman_v3.safetensors
  echo "LORA-V3-INSTALLED from $f"
else
  echo "LORA INSTALL FAILED: no checkpoint in /workspace/lora_out_v3"
  exit 1
fi
echo "== FINAL INVENTORY"
ls -la /workspace/lora_out_v3/
echo "-- installed LoRA"
ls -la /workspace/ComfyUI/models/loras/
sha256sum /workspace/ComfyUI/models/loras/skswoman_v3.safetensors || true
echo "-- training loss tail (from the trainer's own log lines above)"
grep -a "avr_loss" /workspace/lora_stage2_v3.log | tail -5 || true
echo "BUILD-FINISHED at $(date -u)"
echo "STAGE2-V3-DONE"

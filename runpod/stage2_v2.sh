#!/bin/bash
# STAGE2-V2-V7
# MYAI person-LoRA v2, CORRECTED stage 2. The Oct 15 run died in its
# first minute: 'python -m musubi_tuner...' -> ModuleNotFoundError on
# the API-deployed raw-image pod (the module was importable on the old
# console-template pod where v1 ran). Fixes in this version:
#   - PYTHONPATH points at the tuner src explicitly for every call
#   - an import self-check runs first and aborts loudly if it fails
#   - the scaffold VERIFIES the sample dirs exist before waiting
#   - the assemble loop skips the unexpanded-glob phantom entry
# The 82 edit/mask outputs from Oct 15 already sit in ComfyUI/output,
# so the wait below completes immediately. Then: assemble, finalize,
# cache latents, train (output_name skswoman_v2), install into
# ComfyUI/models/loras/skswoman_v2.safetensors. v1 is NOT touched.
# Run ON THE POD, detached:
#   (setsid bash stage2_v2.sh < /dev/null > /workspace/lora_stage2_v2.log 2>&1 &)
set -u
echo "STAGE2-V2-V7 starting $(date -u)"
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /workspace
TUNER=/workspace/SCAIL-2-Tuner
export PYTHONPATH="$TUNER/src${PYTHONPATH:+:$PYTHONPATH}"
PY=$TUNER/.venv/bin/python
DS=/workspace/lora_dataset_v2
RAW=/workspace/lora_raw_v2
OUT=/workspace/ComfyUI/output
DIT=/workspace/ComfyUI/models/diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors
VAE=/workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors
CLIP=/workspace/trainer_models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
T5=/workspace/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors

# ---- EXTREME LOGGING (David's instruction, Oct 17) -----------------
# Full command trace from here on; a mirror of this log is copied into
# ComfyUI/output every 60s so Noor can read the build's own record
# remotely instead of inferring state from indirect signals.
set -x
( while :; do cp -f /workspace/lora_stage2_v2.log /workspace/ComfyUI/output/stage2_v2_log_copy.txt 2>/dev/null; sleep 60; done ) &
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
echo "dataset: $DS  cache: /workspace/lora_cache_v2  out: /workspace/lora_out_v2"

echo "== PREFLIGHT: rebuild the tuner venv whole, prove every precondition"
# The Oct 15 venv was built on the console-template pod and silently
# relied on that environment (importable musubi_tuner, PyAV, even pip
# itself). This preflight makes the venv self-contained on THIS image
# and verifies every file, count and import the build touches BEFORE
# the build starts. Any failure aborts loudly in the first minutes.
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
echo "-- data counts"
NT=$(ls /workspace/ComfyUI/input/lora_targets_v2 2>/dev/null | wc -l)
NE=$(ls "$OUT"/lora_edits/*.png 2>/dev/null | wc -l)
NM=$(ls "$OUT"/lora_masks/*.png 2>/dev/null | wc -l)
echo "targets=$NT edits=$NE masks=$NM (expect 41 / 82 / 82)"
if [ "$NT" -lt 41 ] || [ "$NE" -lt 82 ] || [ "$NM" -lt 82 ]; then
  echo "PREFLIGHT-FAIL: input data incomplete"
  exit 1
fi
echo "-- disk"
FREE_GB=$(df -BG /workspace | tail -1 | awk '{print $4}' | tr -d 'G')
echo "volume free: ${FREE_GB}G"
# Disk floor set from the build's measured footprint (Oct 17): latent
# cache <1G + 5 checkpoints ~2.7G + logs. 10G floor = >2.5x headroom.
# (The original 30G floor was set before measuring and blocked a
# healthy 24G-free volume.)
if [ "$FREE_GB" -lt 10 ]; then
  echo "PREFLIGHT-FAIL: less than 10G free on the volume"
  exit 1
fi
echo "-- imports (the exact modules the build runs)"
if $PY -c "import av, torch, torchvision; import musubi_tuner.scail2_prepare_dataset; import musubi_tuner.scail2_cache_latents; import musubi_tuner.scail2_train_network; print('IMPORT-OK torch', torch.__version__, 'torchvision', torchvision.__version__)"; then
  echo "import path OK"
else
  echo "PREFLIGHT-FAIL: imports still broken after venv rebuild"
  exit 1
fi
if [ -x "$TUNER/.venv/bin/accelerate" ]; then
  echo "accelerate present: $($TUNER/.venv/bin/accelerate --version | head -1)"
else
  echo "PREFLIGHT-FAIL: no accelerate in the tuner venv"
  exit 1
fi
echo "PREFLIGHT-OK - every precondition verified"

echo "STAGE-BEGIN SCAFFOLD v2 dataset at $(date -u +%s) $(date -u)"
echo "== SCAFFOLD v2 dataset (frames first, then stills)"
rm -rf "$RAW" "$DS" /workspace/lora_cache_v2 /workspace/lora_out_v2
mkdir -p "$RAW"
FILES=(img18.png img19.png img20.png img21.png img22.png img23.png img24.png img25.png img26.png img27.png img28.png img29.png img30.png img31.png img32.png img33.png img34.png img35.png img36.png img37.png img38.png img39.png img40.png img41.png img01.png img02.jpg img03.png img04.jpg img05.jpg img06.jpg img07.jpg img08.jpg img09.jpg img10.jpg img11.jpg img12.jpg img13.jpg img14.jpg img15.jpg img16.jpg img17.jpg)
i=0
for f in "${FILES[@]}"; do
  i=$((i+1))
  ID=$(printf "sample%03d" $i)
  cp "/workspace/ComfyUI/input/lora_targets_v2/$f" "$RAW/$ID.${f##*.}"
  printf "skswoman, a woman, photorealistic\n" > "$RAW/$ID.txt"
  echo "$ID <- $f"
done
$PY -m musubi_tuner.scail2_prepare_dataset prepare-image \
  --input_dir "$RAW" --output_dir "$DS" --mode replace
NSAMP=$(ls "$DS/samples" 2>/dev/null | wc -l)
echo "SCAFFOLD-DONE samples: $NSAMP"
if [ "$NSAMP" -lt 40 ]; then
  echo "SCAFFOLD-FAILED: only $NSAMP sample dirs - aborting"
  exit 1
fi

echo "== WAIT for edits + masks (41 samples x R/Z + 2 masks)"
DEADLINE=$(( $(date +%s) + 14400 ))
while : ; do
  complete=0
  for d in "$DS"/samples/*/; do
    [ -d "$d" ] || continue
    id=$(basename "$d")
    r=$(ls "$OUT"/lora_edits/${id}_R_*.png 2>/dev/null | head -1)
    z=$(ls "$OUT"/lora_edits/${id}_Z_*.png 2>/dev/null | head -1)
    rmf=$(ls "$OUT"/lora_masks/${id}_Rmask_*.png 2>/dev/null | head -1)
    zmf=$(ls "$OUT"/lora_masks/${id}_Zmask_*.png 2>/dev/null | head -1)
    [ -n "$r" ] && [ -n "$z" ] && [ -n "$rmf" ] && [ -n "$zmf" ] && complete=$((complete+1))
  done
  echo "wait: $complete/41 complete at $(date -u +%H:%M:%S)"
  [ "$complete" -ge 41 ] && break
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "WAIT-TIMEOUT at $complete/41"; break; fi
  sleep 60
done

echo "== ASSEMBLE edits + masks into the dataset tree"
placed=0; skipped=""
for d in "$DS"/samples/*/; do
  [ -d "$d" ] || continue
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
if [ "$placed" -lt 30 ]; then
  echo "ABORT: fewer than 30 complete samples - not training"
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
  --cache_directory /workspace/lora_cache_v2 \
  --resolution "$RES" \
  --vae "$VAE" --clip "$CLIP" --t5 "$T5"
echo "CACHE-DONE"

echo "STAGE-BEGIN TRAIN (fp16 at $(date -u +%s) $(date -u)"
echo "== TRAIN (fp16, dim 32, 16 epochs, output skswoman_v2)"
cd "$TUNER"
.venv/bin/accelerate launch --num_cpu_threads_per_process 1 \
  --mixed_precision fp16 \
  src/musubi_tuner/scail2_train_network.py \
  --cache_directory /workspace/lora_cache_v2 \
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
  --seed 42 --output_dir /workspace/lora_out_v2 --output_name skswoman_v2 \
  --logging_dir /workspace/lora_out_v2/logs --log_with tensorboard
echo "TRAIN-DONE"

echo "== INSTALL"
f=/workspace/lora_out_v2/skswoman_v2.safetensors
[ -f "$f" ] || f=$(ls -t /workspace/lora_out_v2/skswoman_v2*.safetensors 2>/dev/null | head -1)
if [ -n "$f" ] && [ -f "$f" ]; then
  cp "$f" /workspace/ComfyUI/models/loras/skswoman_v2.safetensors
  echo "LORA-V2-INSTALLED from $f"
else
  echo "LORA INSTALL FAILED: no checkpoint in /workspace/lora_out_v2"
  exit 1
fi
echo "== FINAL INVENTORY"
ls -la /workspace/lora_out_v2/
echo "-- installed LoRA"
ls -la /workspace/ComfyUI/models/loras/
sha256sum /workspace/ComfyUI/models/loras/skswoman_v2.safetensors || true
echo "-- training loss tail (from the trainer's own log lines above)"
grep -a "avr_loss" /workspace/lora_stage2_v2.log | tail -5 || true
echo "BUILD-FINISHED at $(date -u)"
echo "STAGE2-V2-DONE"

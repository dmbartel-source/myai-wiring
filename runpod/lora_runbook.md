# Person LoRA runbook (SCAIL-2-Tuner) — Oct 15, 2026

Goal: a COMPLETE person LoRA for David's reference model, trained
from MULTIPLE reference images (frames of 4707.mp4), plugged into
the SCAIL chain in gpu_workflows (new LoraLoaderModelOnly after
DPO/relight) once trained.

## Pod prerequisites (install_lora_trainer.sh, pushed 76358bc)
- DiT: /workspace/ComfyUI/models/diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors
- T5:  /workspace/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors
- VAE: /workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors (present)
- CLIP vision: /workspace/ComfyUI/models/clip_vision/clip_vision_h.safetensors (present)
- Tuner: /workspace/SCAIL-2-Tuner with .venv (activate before commands)
- Source video: 4707.mp4 in ComfyUI input (David uploads from server)

## Dataset design (image-sample route first)
Video samples additionally need a motion-transfer z.mp4 + mask
video per sample — much heavier. Image samples (1 frame each)
train identity (face/body) and are fully automatable with pod
tools we already run (Qwen-Image-Edit-2509 + SAM3). ~28 samples
from DISTINCT frames of 4707 (varied pose/angle/lighting; skip
head/tail 5%). Per sample, in samples/sampleNNN/:
- target/y.png        — the frame itself (her, as-is)
- reference/R.png     — Qwen EDIT of y: same person/view/bg/light,
                        different action/pose (must NOT be a copy)
- edited/z0.png       — Qwen EDIT of y: same pose/framing,
                        DIFFERENT person + background (driving)
- masks/R_mask.png    — SCAIL coloured mask of her in R (SAM3)
- masks/z_mask.png    — SCAIL coloured mask of the person in z0
- caption.txt         — describes y; include the trigger "skswoman"
                        + generic class ("skswoman, a woman ...")
Mode: "replace" for all samples.

Build steps on the pod:
1. Extract frames: VHS_LoadVideo workflow on 4707 (or the tuner's
   prepare-image route from a folder of frames + captions).
   Simplest controlled path: write frames with a small ComfyUI
   graph (LoadVideo -> ImageFromBatch at chosen indices ->
   SaveImage), then assemble sample dirs with a shell/python
   script run in the pod terminal OR via the tuner's
   scail2_prepare_dataset prepare-image --input_dir.
2. Qwen edits for R and z0 per sample (gpu_workflows.image_edit
   builder; two prompt templates: reference-edit / driving-edit).
3. Coloured masks: SAM3_Detect("woman"/"person") -> SCAIL palette
   colour for a single subject (blue); build masks as RGB images
   (mask helper workflow in the tuner repo is an alternative).
4. python -m musubi_tuner.scail2_prepare_dataset finalize \
     --output_dir /workspace/lora_dataset
   (validates files/frame counts -> manifests/scail2_cache_manifest.jsonl)

## Cache
.venv/bin/python -m musubi_tuner.scail2_cache_latents \
  --manifest /workspace/lora_dataset/manifests/scail2_cache_manifest.jsonl \
  --cache_directory /workspace/lora_cache \
  --resolution <W,H matching 4707 aspect, e.g. 704,512> \
  --vae /workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors \
  --clip /workspace/trainer_models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth (NATIVE Wan CLIP — the ComfyUI clip_vision_h file CANNOT load in the tuner; proven Oct 15) \
  --t5 /workspace/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors

## Train (fork's own example config)
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  src/musubi_tuner/scail2_train_network.py \
  --cache_directory /workspace/lora_cache \
  --dit /workspace/ComfyUI/models/diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors \
  --sdpa --mixed_precision fp16 --fp8_base \
  --optimizer_type adamw8bit --learning_rate 2e-4 \
  --gradient_checkpointing --max_data_loader_n_workers 2 \
  --persistent_data_loader_workers \
  --network_module networks.lora_wan --network_dim 32 \
  --timestep_sampling shift --discrete_flow_shift 3.0 \
  --scail2_history_prob 0.5 --scail2_history_latent_frames 2 \
  --max_train_epochs 16 --save_every_n_epochs 4 \
  --seed 42 --output_dir /workspace/lora_out --output_name skswoman
Run it with nohup in the pod terminal so it survives disconnects;
watch lora_out for epoch checkpoints. Est: ~28 samples x 16 epochs
= ~450 steps; measure step time on the first epoch, then quote
David the real total (budget approved ~$8-16 pod time).

## After training
- Copy the final LoRA to ComfyUI/models/loras/skswoman.safetensors
- gpu_workflows: MODELS["scail_person_lora"], LoraLoaderModelOnly
  node after relight (strength param, default 1.0), server knob +
  UI toggle default ON for scail/gap engines; A/B same clip.

## Acceptance test — "accurately replaces the model" (run right after install)
The LoRA ships as a Studio default ONLY if it clearly wins this A/B:
1. Studio GPU Replace, test clip (Referencevid_2.mp4), reference photo,
   Person LoRA checkbox OFF, Face Lock ON. Note the job's look.
2. Same clip, same reference, same settings, Person LoRA ON.
3. Judge both on David's four standing criteria: (a) the replacement
   reads as the reference woman — face AND body; (b) position/motion
   locked to the source performer; (c) the man present and unchanged;
   (d) border/halo around her.
Pass = ON is visibly truer to the reference than OFF on (a) without
losing (b) or (c). If it fails, the toggle stays but defaults change
and v2 (more real references) becomes the fix path.
Failure tree: "missing lora" validation error = the file is not in
ComfyUI/models/loras yet (training or the install watcher still
running). Couple swapped in the GAP engine = flip gap_swap.

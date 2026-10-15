#!/usr/bin/env python3
"""
ComfyUI workflow builders for MYAI Studio's self-hosted GPU path.

Each builder returns an API-format workflow dict (node_id -> {class_type,
inputs}) ready for gpu_client.queue_workflow(). Every meaningful knob is
a parameter — prompt, negative prompt, seed, steps, cfg, resolution,
frames, fps, denoise, reference strength — because the point of running
our own pod is full creative control: nothing is hard-coded that David
might want to change, and no prompt is ever altered.

Models are open-weights only, chosen for identity fidelity:
  - Qwen-Image-Edit-2509 (image_edit): instruction edits that keep the
    reference photo's subject — the true-to-reference image editor.
  - Wan 2.2 14B FP8, dual high/low-noise pass (image_to_video,
    text_to_video): the open Wan flagship.
  - Wan VACE 14B (video_replace_model): reference-to-video — drives a
    source video's motion/scene with a reference image's subject.

Model filenames live in MODELS below and MUST match runpod/models.txt
exactly (a test enforces it) — setup_pod.sh downloads precisely these
files into the pod's network volume.

Native ComfyUI core nodes are used wherever they exist; the only custom
node pack the video workflows need is ComfyUI-VideoHelperSuite
(VHS_LoadVideo / VHS_VideoCombine) for video IO.
"""

from __future__ import annotations

from typing import Optional

try:
    from . import gpu_client
except ImportError:  # tools/ used as plain scripts dir
    import gpu_client


# ------------------------------------------------------------- model files --
# Basenames as ComfyUI's loaders see them (models/<subdir>/<basename>).
# Keep in sync with runpod/models.txt — tests/test_gpu_workflows.py fails
# if the two drift apart.

MODELS = {
    # Qwen-Image-Edit-2509 (Comfy-Org/Qwen-Image-Edit_ComfyUI + Qwen-Image)
    "qwen_edit_unet": "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
    "qwen_clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "qwen_vae": "qwen_image_vae.safetensors",
    # Wan 2.2 14B FP8 dual-pass (Comfy-Org/Wan_2.2_ComfyUI_Repackaged)
    "wan_i2v_high": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    "wan_i2v_low": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    "wan_t2v_high": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    "wan_t2v_low": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
    # Wan shared pieces (Comfy-Org/Wan_2.1_ComfyUI_repackaged)
    "wan_clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "wan_vae": "wan_2.1_vae.safetensors",
    # Wan VACE 14B (Comfy-Org/Wan_2.1_ComfyUI_repackaged; fp16 is the
    # only 14B build Comfy-Org ships — the H100's 80GB takes it as-is)
    "vace_unet": "wan2.1_vace_14B_fp16.safetensors",
    # SAM3.1 person segmentation/tracking (Comfy-Org/sam3.1) — used
    # to mask the performer out of the source video for VACE swaps.
    "sam3_ckpt": "sam3.1_multiplex_fp16.safetensors",
    # SCAIL-2 character replacement (Comfy-Org/SCAIL-2) -- end-to-end
    # replacement: SAM3.1 coloured masks bind identities, no skeleton,
    # no erase-to-black. fp8 build (17.7 GB) fits beside the VACE set.
    "scail_unet": "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors",
    "scail_dpo_lora": "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors",
    "scail_lightx2v_lora":
        "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    "clip_vision_h": "clip_vision_h.safetensors",
}

DEFAULT_NEGATIVE = (
    "low quality, blurry, distorted, deformed, watermark, text, "
    "extra limbs, bad anatomy, flicker, morphing"
)

_WAN_SAMPLER = "euler"
_WAN_SCHEDULER = "simple"
_WAN_SHIFT = 8.0  # ModelSamplingSD3 shift used by the official templates


def _frames_4n1(frames: int) -> int:
    """Wan's VAE needs frame counts of the form 4n+1. Round up to fit."""
    frames = max(5, int(frames))
    rem = (frames - 1) % 4
    return frames if rem == 0 else frames + (4 - rem)


def queue_workflow(workflow: dict, opener=None) -> "gpu_client.GpuJob":
    """Raw escape hatch: queue ANY ComfyUI API-format workflow dict.

    Full creative control — a workflow built anywhere (the ComfyUI UI's
    "Save (API Format)", another tool, by hand) runs as-is.
    """
    return gpu_client.queue_workflow(workflow, opener=opener)


# ------------------------------------------------------------ image edit ----

def image_edit(
    reference_image: str,
    instruction: str,
    *,
    negative_prompt: str = "",
    seed: int = 0,
    steps: int = 20,
    cfg: float = 4.0,
    denoise: float = 1.0,
    sampler: str = "euler",
    scheduler: str = "simple",
    shift: float = 3.1,
    width: Optional[int] = None,
    height: Optional[int] = None,
    filename_prefix: str = "myai/edit",
) -> dict:
    """Edit a reference photo by instruction (Qwen-Image-Edit-2509).

    reference_image: ComfyUI input filename (from gpu_client.upload_image).
    instruction: the edit, passed to the model exactly as written.
    width/height: optional output size; omitted = the reference's own
    resolution (the identity-preserving default).
    """
    wf: dict = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": MODELS["qwen_edit_unet"],
                         "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": MODELS["qwen_clip"],
                         "type": "qwen_image"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": MODELS["qwen_vae"]}},
        "4": {"class_type": "LoadImage",
              "inputs": {"image": reference_image}},
        "5": {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": ["1", 0], "shift": shift}},
        "6": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0],
                         "image1": ["4", 0], "prompt": instruction}},
        "7": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0],
                         "image1": ["4", 0], "prompt": negative_prompt}},
        "8": {"class_type": "VAEEncode",
              "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["5", 0],
                         "positive": ["6", 0], "negative": ["7", 0],
                         "latent_image": ["8", 0],
                         "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": sampler, "scheduler": scheduler,
                         "denoise": denoise}},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0],
                          "filename_prefix": filename_prefix}},
    }
    if width and height:
        # Explicit output size: scale the reference first; encode and
        # conditioning both follow the scaled image.
        wf["12"] = {"class_type": "ImageScale",
                    "inputs": {"image": ["4", 0], "upscale_method": "lanczos",
                               "width": int(width), "height": int(height),
                               "crop": "center"}}
        wf["6"]["inputs"]["image1"] = ["12", 0]
        wf["7"]["inputs"]["image1"] = ["12", 0]
        wf["8"]["inputs"]["pixels"] = ["12", 0]
    return wf


# ------------------------------------------------------------- wan video ----

def _wan_dual_pass(
    *,
    high_unet: str,
    low_unet: str,
    latent_source: dict,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    shift: float,
    fps: float,
    filename_prefix: str,
    cond_from_latent_node: bool,
) -> dict:
    """Shared Wan 2.2 14B dual-noise sampling graph.

    latent_source is the node producing (positive, negative, latent) —
    WanImageToVideo for I2V, or a plain latent + raw text conditioning
    for T2V (cond_from_latent_node=False with text nodes added by the
    caller at ids 20/21/22).
    """
    wf: dict = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": high_unet, "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader",
              "inputs": {"unet_name": low_unet, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": MODELS["wan_clip"], "type": "wan"}},
        "4": {"class_type": "VAELoader",
              "inputs": {"vae_name": MODELS["wan_vae"]}},
        "5": {"class_type": "ModelSamplingSD3",
              "inputs": {"model": ["1", 0], "shift": shift}},
        "6": {"class_type": "ModelSamplingSD3",
              "inputs": {"model": ["2", 0], "shift": shift}},
    }
    wf.update(latent_source)
    if cond_from_latent_node:
        pos, neg, lat = ["10", 0], ["10", 1], ["10", 2]
    else:
        pos, neg, lat = ["20", 0], ["21", 0], ["22", 0]
    split = max(1, int(steps) // 2)
    wf["11"] = {"class_type": "KSamplerAdvanced",
                "inputs": {"model": ["5", 0], "positive": pos,
                           "negative": neg, "latent_image": lat,
                           "add_noise": "enable", "noise_seed": seed,
                           "steps": steps, "cfg": cfg,
                           "sampler_name": _WAN_SAMPLER,
                           "scheduler": _WAN_SCHEDULER,
                           "start_at_step": 0, "end_at_step": split,
                           "return_with_leftover_noise": "enable"}}
    wf["12"] = {"class_type": "KSamplerAdvanced",
                "inputs": {"model": ["6", 0], "positive": pos,
                           "negative": neg, "latent_image": ["11", 0],
                           "add_noise": "disable", "noise_seed": seed,
                           "steps": steps, "cfg": cfg,
                           "sampler_name": _WAN_SAMPLER,
                           "scheduler": _WAN_SCHEDULER,
                           "start_at_step": split, "end_at_step": steps,
                           "return_with_leftover_noise": "disable"}}
    wf["13"] = {"class_type": "VAEDecode",
                "inputs": {"samples": ["12", 0], "vae": ["4", 0]}}
    wf["14"] = {"class_type": "VHS_VideoCombine",
                "inputs": {"images": ["13", 0], "frame_rate": fps,
                           "loop_count": 0, "filename_prefix": filename_prefix,
                           "format": "video/h264-mp4", "pingpong": False,
                           "save_output": True}}
    return wf


def image_to_video(
    image: str,
    prompt: str,
    *,
    negative_prompt: str = DEFAULT_NEGATIVE,
    seed: int = 0,
    steps: int = 20,
    cfg: float = 5.0,
    width: int = 832,
    height: int = 480,
    frames: int = 81,
    fps: float = 16.0,
    shift: float = _WAN_SHIFT,
    filename_prefix: str = "myai/i2v",
) -> dict:
    """Animate a photo (Wan 2.2 14B I2V, FP8, dual high/low-noise pass).

    image: ComfyUI input filename (from gpu_client.upload_image).
    frames is normalized to Wan's 4n+1 grid (81 frames = ~5s at 16fps).
    """
    frames = _frames_4n1(frames)
    latent_source = {
        "7": {"class_type": "LoadImage", "inputs": {"image": image}},
        "8": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["3", 0], "text": prompt}},
        "9": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["3", 0], "text": negative_prompt}},
        "10": {"class_type": "WanImageToVideo",
               "inputs": {"positive": ["8", 0], "negative": ["9", 0],
                          "vae": ["4", 0], "width": int(width),
                          "height": int(height), "length": frames,
                          "batch_size": 1, "start_image": ["7", 0]}},
    }
    return _wan_dual_pass(
        high_unet=MODELS["wan_i2v_high"], low_unet=MODELS["wan_i2v_low"],
        latent_source=latent_source, prompt=prompt,
        negative_prompt=negative_prompt, seed=seed, steps=steps, cfg=cfg,
        shift=shift, fps=fps, filename_prefix=filename_prefix,
        cond_from_latent_node=True)


def text_to_video(
    prompt: str,
    *,
    negative_prompt: str = DEFAULT_NEGATIVE,
    seed: int = 0,
    steps: int = 20,
    cfg: float = 5.0,
    width: int = 832,
    height: int = 480,
    frames: int = 81,
    fps: float = 16.0,
    shift: float = _WAN_SHIFT,
    filename_prefix: str = "myai/t2v",
) -> dict:
    """Generate video from text alone (Wan 2.2 14B T2V, FP8 dual-pass)."""
    frames = _frames_4n1(frames)
    latent_source = {
        "20": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["3", 0], "text": prompt}},
        "21": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["3", 0], "text": negative_prompt}},
        "22": {"class_type": "EmptyHunyuanLatentVideo",
               "inputs": {"width": int(width), "height": int(height),
                          "length": frames, "batch_size": 1}},
    }
    return _wan_dual_pass(
        high_unet=MODELS["wan_t2v_high"], low_unet=MODELS["wan_t2v_low"],
        latent_source=latent_source, prompt=prompt,
        negative_prompt=negative_prompt, seed=seed, steps=steps, cfg=cfg,
        shift=shift, fps=fps, filename_prefix=filename_prefix,
        cond_from_latent_node=False)


# --------------------------------------------------------- model replace ----

def video_replace_model(
    source_video: str,
    reference_image: str,
    prompt: str = "",
    *,
    negative_prompt: str = DEFAULT_NEGATIVE,
    seed: int = 0,
    steps: int = 20,
    cfg: float = 5.0,
    width: int = 832,
    height: int = 480,
    frames: int = 81,
    fps: float = 16.0,
    denoise: float = 1.0,
    reference_strength: float = 1.0,
    mask_prompt: str = "woman",
    mask_expand: int = 20,
    mask_threshold: float = 0.5,
    composite: bool = True,
    composite_feather: int = 8,
    protect_contact: bool = True,
    partner_prompt: str = "man",
    partner_threshold: float = 0.55,
    contact_reach: int = 24,
    contact_pad: int = 16,
    contact_guard: int = 20,
    shift: float = _WAN_SHIFT,
    filename_prefix: str = "myai/replace",
) -> dict:
    """Swap the performer of a video for a reference model (Wan VACE 14B).

    Built on ComfyUI's official VACE inpainting recipe (the Oct 14 2026
    build without it silently returned the source video: VACE received
    the performer's untouched pixels as its control stream and simply
    reconstructed them, reference image flashing at the head):

    1. SAM3 tracks the performer (text-prompted, `mask_prompt`) through
       the source frames; the mask is grown by `mask_expand` px.
    2. ERASE: the masked region of the control video is painted black
       (inverted mask composited in) — the original performer never
       reaches the model, so it cannot be copied.
    3. WanVaceToVideo regenerates the blacked-out region as the
       reference subject (reference_image + prompt), driven by the
       mask's shape/motion and the untouched surroundings.
    4. TrimVideoLatent removes the reference segment after sampling,
       so the output is exactly `frames` long with no reference intro.
    5. COMPOSITE (default on): the regenerated frames are pasted back
       over the ORIGINAL source frames through the feathered mask, so
       everything outside the performer — room, other people, camera —
       is the original footage pixel for pixel, not a re-render. Only
       her region is synthesized (those pixels exist nowhere else).
    6. CONTACT PROTECTION (default on): the partner (`partner_prompt`)
       is tracked too; where the two bodies come within
       `contact_reach` px, that patch of her mask (grown by
       `contact_pad`) is subtracted from the replacement mask. The
       interlocked region — genitals/contact zone — is therefore
       never erased and never regenerated: in the output it is the
       source video itself. (Inside that patch, the visible anatomy
       remains the source performer's — that is what preserving it
       means. Identity replacement applies everywhere else.)

    source_video / reference_image: ComfyUI input filenames (from
    gpu_client.upload_video / upload_image). frames caps how much of
    the source video is processed (4n+1 grid).
    """
    frames = _frames_4n1(frames)
    wf: dict = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": MODELS["vace_unet"],
                         "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": MODELS["wan_clip"], "type": "wan"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": MODELS["wan_vae"]}},
        "4": {"class_type": "ModelSamplingSD3",
              "inputs": {"model": ["1", 0], "shift": shift}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": negative_prompt}},
        "7": {"class_type": "VHS_LoadVideo",
              "inputs": {"video": source_video, "force_rate": fps,
                         "custom_width": 0, "custom_height": 0,
                         "frame_load_cap": frames, "skip_first_frames": 0,
                         "select_every_nth": 1, "format": "None"}},
        "8": {"class_type": "LoadImage",
              "inputs": {"image": reference_image}},
        # Work at the render size everywhere downstream so frames,
        # masks and latents agree exactly.
        "9": {"class_type": "ImageScale",
              "inputs": {"image": ["7", 0], "upscale_method": "lanczos",
                         "width": int(width), "height": int(height),
                         "crop": "disabled"}},
        # --- SAM3 performer tracking -------------------------------
        "10": {"class_type": "CheckpointLoaderSimple",
               "inputs": {"ckpt_name": MODELS["sam3_ckpt"]}},
        "11": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["10", 1], "text": mask_prompt}},
        "12": {"class_type": "SAM3_VideoTrack",
               "inputs": {"images": ["9", 0], "model": ["10", 0],
                          "conditioning": ["11", 0],
                          "detection_threshold": mask_threshold,
                          "max_objects": 4, "detect_interval": 1}},
        "13": {"class_type": "SAM3_TrackToMask",
               "inputs": {"track_data": ["12", 0], "object_indices": ""}},
        "14": {"class_type": "GrowMask",
               "inputs": {"mask": ["13", 0], "expand": int(mask_expand),
                          "tapered_corners": True}},
        # --- erase the performer from the control video ------------
        "15": {"class_type": "InvertMask",
               "inputs": {"mask": ["14", 0]}},
        "16": {"class_type": "MaskToImage",
               "inputs": {"mask": ["15", 0]}},
        "17": {"class_type": "ImageCompositeMasked",
               "inputs": {"destination": ["9", 0], "source": ["16", 0],
                          "x": 0, "y": 0, "resize_source": False,
                          "mask": ["14", 0]}},
        # --- VACE regeneration --------------------------------------
        "18": {"class_type": "WanVaceToVideo",
               "inputs": {"positive": ["5", 0], "negative": ["6", 0],
                          "vae": ["3", 0], "width": int(width),
                          "height": int(height), "length": frames,
                          "batch_size": 1, "strength": reference_strength,
                          "control_video": ["17", 0],
                          "control_masks": ["14", 0],
                          "reference_image": ["8", 0]}},
        "19": {"class_type": "KSampler",
               "inputs": {"model": ["4", 0], "positive": ["18", 0],
                          "negative": ["18", 1], "latent_image": ["18", 2],
                          "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": _WAN_SAMPLER,
                          "scheduler": _WAN_SCHEDULER, "denoise": denoise}},
        "20": {"class_type": "TrimVideoLatent",
               "inputs": {"samples": ["19", 0],
                          "trim_amount": ["18", 3]}},
        "21": {"class_type": "VAEDecode",
               "inputs": {"samples": ["20", 0], "vae": ["3", 0]}},
        "22": {"class_type": "VHS_VideoCombine",
               "inputs": {"images": ["21", 0], "frame_rate": fps,
                          "loop_count": 0, "filename_prefix": filename_prefix,
                          "format": "video/h264-mp4", "pingpong": False,
                          "save_output": True}},
    }
    eff_mask = ["14", 0]
    if protect_contact:
        wf["25"] = {"class_type": "CLIPTextEncode",
                    "inputs": {"clip": ["10", 1], "text": partner_prompt}}
        wf["26"] = {"class_type": "SAM3_VideoTrack",
                    "inputs": {"images": ["9", 0], "model": ["10", 0],
                               "conditioning": ["25", 0],
                               "detection_threshold": partner_threshold,
                               "max_objects": 4, "detect_interval": 1}}
        wf["27"] = {"class_type": "SAM3_TrackToMask",
                    "inputs": {"track_data": ["26", 0],
                               "object_indices": ""}}
        wf["28"] = {"class_type": "GrowMask",
                    "inputs": {"mask": ["27", 0],
                               "expand": int(contact_reach),
                               "tapered_corners": True}}
        wf["29"] = {"class_type": "MaskComposite",
                    "inputs": {"destination": ["14", 0],
                               "source": ["28", 0], "x": 0, "y": 0,
                               "operation": "and"}}
        wf["30"] = {"class_type": "GrowMask",
                    "inputs": {"mask": ["29", 0],
                               "expand": int(contact_pad),
                               "tapered_corners": True}}
        wf["31"] = {"class_type": "MaskComposite",
                    "inputs": {"destination": ["14", 0],
                               "source": ["30", 0], "x": 0, "y": 0,
                               "operation": "subtract"}}
        eff_mask = ["31", 0]
        # Gut guard: no matter how wrong the partner track is, any
        # part of her farther than contact_guard px from the protected
        # core is ALWAYS regenerated -- the subtraction can never eat
        # the whole mask and silently return the source video.
        wf["32"] = {"class_type": "GrowMask",
                    "inputs": {"mask": ["29", 0],
                               "expand": int(contact_guard),
                               "tapered_corners": True}}
        wf["33"] = {"class_type": "MaskComposite",
                    "inputs": {"destination": ["14", 0],
                               "source": ["32", 0], "x": 0, "y": 0,
                               "operation": "subtract"}}
        wf["34"] = {"class_type": "MaskComposite",
                    "inputs": {"destination": ["31", 0],
                               "source": ["33", 0], "x": 0, "y": 0,
                               "operation": "add"}}
        eff_mask = ["34", 0]
        wf["15"]["inputs"]["mask"] = eff_mask
        wf["17"]["inputs"]["mask"] = eff_mask
        wf["18"]["inputs"]["control_masks"] = eff_mask
    if composite:
        wf["23"] = {"class_type": "FeatherMask",
                    "inputs": {"mask": eff_mask,
                               "left": int(composite_feather),
                               "top": int(composite_feather),
                               "right": int(composite_feather),
                               "bottom": int(composite_feather)}}
        wf["24"] = {"class_type": "ImageCompositeMasked",
                    "inputs": {"destination": ["9", 0],
                               "source": ["21", 0], "x": 0, "y": 0,
                               "resize_source": False,
                               "mask": ["23", 0]}}
        wf["22"]["inputs"]["images"] = ["24", 0]
    return wf



def video_replace_model_scail(
    source_video: str,
    reference_image: str,
    prompt: str = "",
    *,
    negative_prompt: str = DEFAULT_NEGATIVE,
    seed: int = 0,
    steps: int = 6,
    cfg: float = 1.0,
    width: int = 704,
    height: int = 544,
    frames: int = 81,
    fps: float = 16.0,
    denoise: float = 1.0,
    mask_prompt: str = "woman",
    mask_threshold: float = 0.4,
    ref_threshold: float = 0.5,
    sort_by: str = "left_to_right",
    pose_strength: float = 1.0,
    shift: float = 8.0,
    lightx2v_strength: float = 1.0,
    dpo_strength: float = 1.0,
    composite: bool = True,
    composite_expand: int = 12,
    composite_feather: int = 5,
    filename_prefix: str = "myai/scail_replace",
) -> dict:
    """SCAIL-2 end-to-end character replacement (single segment).

    The research pick over VACE inpainting: SAM3.1 tracks the performer
    in the driving video and the subject in the reference, SCAIL2Colored
    Mask binds the identity by colour, and WanSCAILToVideo regenerates
    her following the driving frames directly -- no pose skeleton and
    no erase-to-black silhouette leak (the two VACE hybrid sources).
    Defaults follow the measured pitfalls: distilled sampler (6 steps,
    cfg 1) with the LightX2V LoRA at full strength and sampling shift
    8 -- the popular 0.8/shift-5 settings measurably leak the original
    performer's details onto the replacement.

    With composite on (default), everything outside her tracked mask
    is pasted back from the (scaled) source frames, so the room and
    the partner stay literal source pixels; only she is generated.
    Long-video segment chaining (previous_frames) is a later layer --
    frames is one segment (keep 4n+1, <= 321).
    """
    frames = _frames_4n1(frames)
    wf = {
        "1": {"class_type": "VHS_LoadVideo",
              "inputs": {"video": source_video, "force_rate": float(fps),
                         "custom_width": 0, "custom_height": 0,
                         "frame_load_cap": int(frames),
                         "skip_first_frames": 0, "select_every_nth": 1,
                         "format": "AnimateDiff"}},
        "2": {"class_type": "ImageScale",
              "inputs": {"image": ["1", 0], "upscale_method": "lanczos",
                         "width": int(width), "height": int(height),
                         "crop": "disabled"}},
        "3": {"class_type": "LoadImage",
              "inputs": {"image": reference_image}},
        "4": {"class_type": "UNETLoader",
              "inputs": {"unet_name": MODELS["scail_unet"],
                         "weight_dtype": "default"}},
        "5": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["4", 0],
                         "lora_name": MODELS["scail_lightx2v_lora"],
                         "strength_model": float(lightx2v_strength)}},
        "6": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["5", 0],
                         "lora_name": MODELS["scail_dpo_lora"],
                         "strength_model": float(dpo_strength)}},
        "7": {"class_type": "ModelSamplingSD3",
              "inputs": {"model": ["6", 0], "shift": float(shift)}},
        "8": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": MODELS["wan_clip"], "type": "wan"}},
        "9": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["8", 0]}},
        "10": {"class_type": "CLIPTextEncode",
               "inputs": {"text": negative_prompt, "clip": ["8", 0]}},
        "11": {"class_type": "VAELoader",
               "inputs": {"vae_name": MODELS["wan_vae"]}},
        "12": {"class_type": "CheckpointLoaderSimple",
               "inputs": {"ckpt_name": MODELS["sam3_ckpt"]}},
        "13": {"class_type": "CLIPTextEncode",
               "inputs": {"text": mask_prompt, "clip": ["12", 1]}},
        "14": {"class_type": "SAM3_VideoTrack",
               "inputs": {"images": ["2", 0], "model": ["12", 0],
                          "conditioning": ["13", 0],
                          "detection_threshold": float(mask_threshold),
                          "max_objects": 1, "detect_interval": 1}},
        "15": {"class_type": "CLIPTextEncode",
               "inputs": {"text": mask_prompt, "clip": ["12", 1]}},
        "16": {"class_type": "SAM3_Detect",
               "inputs": {"model": ["12", 0], "image": ["3", 0],
                          "conditioning": ["15", 0],
                          "threshold": float(ref_threshold),
                          "refine_iterations": 2,
                          "individual_masks": False}},
        "17": {"class_type": "SCAIL2ColoredMask",
               "inputs": {"driving_track_data": ["14", 0],
                          "ref_track_data": ["16", 0],
                          "object_indices": "", "sort_by": sort_by,
                          "replacement_mode": True}},
        "18": {"class_type": "CLIPVisionLoader",
               "inputs": {"clip_name": MODELS["clip_vision_h"]}},
        "19": {"class_type": "CLIPVisionEncode",
               "inputs": {"clip_vision": ["18", 0], "image": ["3", 0],
                          "crop": "center"}},
        "20": {"class_type": "WanSCAILToVideo",
               "inputs": {"positive": ["9", 0], "negative": ["10", 0],
                          "vae": ["11", 0], "width": int(width),
                          "height": int(height), "length": int(frames),
                          "batch_size": 1,
                          "pose_strength": float(pose_strength),
                          "pose_start": 0.0, "pose_end": 1.0,
                          "video_frame_offset": 0,
                          "previous_frame_count": 5,
                          "pose_video": ["2", 0],
                          "pose_video_mask": ["17", 0],
                          "replacement_mode": True,
                          "reference_image": ["3", 0],
                          "reference_image_mask": ["17", 1],
                          "clip_vision_output": ["19", 0]}},
        "21": {"class_type": "KSamplerSelect",
               "inputs": {"sampler_name": _WAN_SAMPLER}},
        "22": {"class_type": "BasicScheduler",
               "inputs": {"model": ["7", 0], "scheduler": _WAN_SCHEDULER,
                          "steps": int(steps), "denoise": float(denoise)}},
        "23": {"class_type": "SamplerCustom",
               "inputs": {"model": ["7", 0], "add_noise": True,
                          "noise_seed": int(seed), "cfg": float(cfg),
                          "positive": ["20", 0], "negative": ["20", 1],
                          "sampler": ["21", 0], "sigmas": ["22", 0],
                          "latent_image": ["20", 2]}},
        "24": {"class_type": "VAEDecode",
               "inputs": {"samples": ["23", 0], "vae": ["11", 0]}},
    }
    final = ["24", 0]
    if composite:
        wf["25"] = {"class_type": "SAM3_TrackToMask",
                    "inputs": {"track_data": ["14", 0],
                               "object_indices": ""}}
        wf["26"] = {"class_type": "GrowMask",
                    "inputs": {"mask": ["25", 0],
                               "expand": int(composite_expand),
                               "tapered_corners": True}}
        wf["27"] = {"class_type": "FeatherMask",
                    "inputs": {"mask": ["26", 0],
                               "left": int(composite_feather),
                               "top": int(composite_feather),
                               "right": int(composite_feather),
                               "bottom": int(composite_feather)}}
        wf["28"] = {"class_type": "ImageCompositeMasked",
                    "inputs": {"destination": ["2", 0],
                               "source": ["24", 0], "x": 0, "y": 0,
                               "resize_source": False,
                               "mask": ["27", 0]}}
        final = ["28", 0]
    wf["29"] = {"class_type": "VHS_VideoCombine",
                "inputs": {"images": final, "frame_rate": float(fps),
                           "loop_count": 0,
                           "filename_prefix": filename_prefix,
                           "format": "video/h264-mp4", "pingpong": False,
                           "save_output": True}}
    return wf

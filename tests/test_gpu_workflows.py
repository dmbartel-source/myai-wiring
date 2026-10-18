"""Tests for tools/gpu_workflows.py — ComfyUI workflow builders.

Pure dict-shape tests: every builder must return a valid API-format
workflow with the caller's parameters actually injected, and the MODELS
constants must match runpod/models.txt exactly (bootstrap downloads).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import gpu_client
import gpu_workflows as gw


def assert_links_valid(wf):
    """Every [node_id, slot] link points at a node that exists."""
    for nid, node in wf.items():
        assert "class_type" in node, f"node {nid} has no class_type"
        assert "inputs" in node, f"node {nid} has no inputs"
        for key, val in node["inputs"].items():
            if isinstance(val, list) and len(val) == 2 \
                    and isinstance(val[0], str) and isinstance(val[1], int):
                assert val[0] in wf, \
                    f"node {nid} input {key} links to missing node {val[0]}"


def nodes_of(wf, class_type):
    return {nid: n for nid, n in wf.items() if n["class_type"] == class_type}


# ------------------------------------------------------------- model list --

def test_models_match_models_txt():
    txt = os.path.join(os.path.dirname(__file__), "..", "runpod",
                       "models.txt")
    basenames = set()
    with open(txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            assert len(parts) == 3, f"models.txt line malformed: {line}"
            repo, path_in_repo, subdir = parts
            assert subdir in ("diffusion_models", "text_encoders", "vae",
                              "loras", "checkpoints", "clip_vision",
                              "controlnet", "upscale_models", "SEEDVR2")
            basenames.add(os.path.basename(path_in_repo))
    # Locally produced artifacts are referenced by the builders but
    # cannot be downloaded by the bootstrap: the person LoRA is
    # trained on the pod and copied into loras/ (stage2_full.sh).
    local_only = {"skswoman.safetensors"}
    assert basenames == set(gw.MODELS.values()) - local_only


# -------------------------------------------------------------- image edit --

def test_image_edit_injects_parameters():
    wf = gw.image_edit("ref.png", "make the sky purple",
                       negative_prompt="no people", seed=42, steps=30,
                       cfg=3.5, denoise=0.8)
    assert_links_valid(wf)
    assert nodes_of(wf, "UNETLoader")["1"]["inputs"]["unet_name"] == \
        gw.MODELS["qwen_edit_unet"]
    assert nodes_of(wf, "CLIPLoader")["2"]["inputs"]["type"] == "qwen_image"
    assert nodes_of(wf, "LoadImage")["4"]["inputs"]["image"] == "ref.png"
    pos = nodes_of(wf, "TextEncodeQwenImageEditPlus")["6"]
    neg = nodes_of(wf, "TextEncodeQwenImageEditPlus")["7"]
    assert pos["inputs"]["prompt"] == "make the sky purple"  # verbatim
    assert neg["inputs"]["prompt"] == "no people"
    ks = nodes_of(wf, "KSampler")["9"]["inputs"]
    assert ks["seed"] == 42 and ks["steps"] == 30
    assert ks["cfg"] == 3.5 and ks["denoise"] == 0.8
    assert nodes_of(wf, "SaveImage"), "edit must save an image"
    # native resolution by default: no forced scale node
    assert not nodes_of(wf, "ImageScale")


def test_image_edit_explicit_size_scales_first():
    wf = gw.image_edit("ref.png", "edit", width=1024, height=1024)
    assert_links_valid(wf)
    scale = nodes_of(wf, "ImageScale")["12"]
    assert scale["inputs"]["width"] == 1024
    assert scale["inputs"]["height"] == 1024
    # VAE encode + conditioning follow the scaled image
    assert wf["8"]["inputs"]["pixels"] == ["12", 0]
    assert wf["6"]["inputs"]["image1"] == ["12", 0]


# --------------------------------------------------------------------- i2v --

def test_image_to_video_injects_parameters():
    wf = gw.image_to_video("start.jpg", "she turns and smiles",
                           seed=7, steps=24, cfg=6.0, width=1280,
                           height=720, frames=82, fps=24.0)
    assert_links_valid(wf)
    unets = nodes_of(wf, "UNETLoader")
    assert unets["1"]["inputs"]["unet_name"] == gw.MODELS["wan_i2v_high"]
    assert unets["2"]["inputs"]["unet_name"] == gw.MODELS["wan_i2v_low"]
    assert nodes_of(wf, "LoadImage")["7"]["inputs"]["image"] == "start.jpg"
    wan = nodes_of(wf, "WanImageToVideo")["10"]["inputs"]
    assert wan["width"] == 1280 and wan["height"] == 720
    assert wan["length"] == 85  # 82 normalized up to the 4n+1 grid
    assert wan["start_image"] == ["7", 0]
    texts = nodes_of(wf, "CLIPTextEncode")
    assert texts["8"]["inputs"]["text"] == "she turns and smiles"
    samplers = nodes_of(wf, "KSamplerAdvanced")
    hi, lo = samplers["11"]["inputs"], samplers["12"]["inputs"]
    assert hi["noise_seed"] == 7 and hi["cfg"] == 6.0
    assert hi["start_at_step"] == 0 and hi["end_at_step"] == 12
    assert lo["start_at_step"] == 12 and lo["end_at_step"] == 24
    assert hi["add_noise"] == "enable" and lo["add_noise"] == "disable"
    combine = nodes_of(wf, "VHS_VideoCombine")["14"]["inputs"]
    assert combine["frame_rate"] == 24.0
    assert combine["format"] == "video/h264-mp4"
    assert combine["save_output"] is True


def test_image_to_video_frames_grid():
    assert gw._frames_4n1(81) == 81
    assert gw._frames_4n1(82) == 85
    assert gw._frames_4n1(5) == 5
    assert gw._frames_4n1(1) == 5


# --------------------------------------------------------------------- t2v --

def test_text_to_video_injects_parameters():
    wf = gw.text_to_video("a lighthouse in a storm", seed=3, width=640,
                          height=640, frames=49, fps=16.0)
    assert_links_valid(wf)
    unets = nodes_of(wf, "UNETLoader")
    assert unets["1"]["inputs"]["unet_name"] == gw.MODELS["wan_t2v_high"]
    assert unets["2"]["inputs"]["unet_name"] == gw.MODELS["wan_t2v_low"]
    latent = nodes_of(wf, "EmptyHunyuanLatentVideo")["22"]["inputs"]
    assert latent["width"] == 640 and latent["height"] == 640
    assert latent["length"] == 49
    texts = nodes_of(wf, "CLIPTextEncode")
    assert texts["20"]["inputs"]["text"] == "a lighthouse in a storm"
    assert nodes_of(wf, "WanImageToVideo") == {}
    assert nodes_of(wf, "LoadImage") == {}  # t2v needs no input media


# ----------------------------------------------------------------- replace --

def test_video_replace_model_injects_parameters():
    wf = gw.video_replace_model("base.mp4", "newmodel.png",
                                prompt="same dance, new performer",
                                seed=9, steps=26, cfg=4.5, frames=45,
                                reference_strength=0.85)
    assert_links_valid(wf)
    assert nodes_of(wf, "UNETLoader")["1"]["inputs"]["unet_name"] == \
        gw.MODELS["vace_unet"]
    load = nodes_of(wf, "VHS_LoadVideo")["7"]["inputs"]
    assert load["video"] == "base.mp4"
    assert load["frame_load_cap"] == 45
    assert nodes_of(wf, "LoadImage")["8"]["inputs"]["image"] == "newmodel.png"
    vace = nodes_of(wf, "WanVaceToVideo")["18"]["inputs"]
    assert vace["strength"] == 0.85
    assert vace["reference_image"] == ["8", 0]
    assert vace["length"] == 45
    ks = nodes_of(wf, "KSampler")["19"]["inputs"]
    assert ks["seed"] == 9 and ks["steps"] == 26 and ks["cfg"] == 4.5
    assert ks["positive"] == ["18", 0] and ks["latent_image"] == ["18", 2]
    texts = nodes_of(wf, "CLIPTextEncode")
    assert texts["5"]["inputs"]["text"] == "same dance, new performer"


def test_video_replace_model_erases_performer_before_vace():
    """The Oct 14 2026 lesson: feeding VACE the untouched source makes
    it reconstruct the original performer. The workflow must track the
    performer with SAM3 and paint her black in the control video."""
    wf = gw.video_replace_model("base.mp4", "newmodel.png",
                                mask_prompt="the dancer", mask_expand=12)
    assert_links_valid(wf)
    ckpt = nodes_of(wf, "CheckpointLoaderSimple")["10"]["inputs"]
    assert ckpt["ckpt_name"] == gw.MODELS["sam3_ckpt"]
    detect_text = nodes_of(wf, "CLIPTextEncode")["11"]["inputs"]
    assert detect_text["text"] == "the dancer"
    assert detect_text["clip"] == ["10", 1]
    track = nodes_of(wf, "SAM3_VideoTrack")["12"]["inputs"]
    assert track["model"] == ["10", 0]
    assert track["images"] == ["9", 0]
    assert track["conditioning"] == ["11", 0]
    grow = nodes_of(wf, "GrowMask")["14"]["inputs"]
    assert grow["mask"] == ["13", 0] and grow["expand"] == 12
    comp = nodes_of(wf, "ImageCompositeMasked")["17"]["inputs"]
    assert comp["destination"] == ["9", 0]
    assert comp["source"] == ["16", 0]      # inverted mask as image
    assert comp["mask"] == ["34", 0]        # contact-protected mask (via gut guard)
    inv = nodes_of(wf, "InvertMask")["15"]["inputs"]
    assert inv["mask"] == ["34", 0]
    vace = nodes_of(wf, "WanVaceToVideo")["18"]["inputs"]
    assert vace["control_video"] == ["17", 0]   # the ERASED video
    assert vace["control_masks"] == ["34", 0]
    trim = nodes_of(wf, "TrimVideoLatent")["20"]["inputs"]
    assert trim["samples"] == ["19", 0]
    assert trim["trim_amount"] == ["18", 3]
    dec = nodes_of(wf, "VAEDecode")["21"]["inputs"]
    assert dec["samples"] == ["20", 0]
    # Composite: only the performer's region is pasted over the
    # ORIGINAL frames; everything else stays the source pixels.
    feather = nodes_of(wf, "FeatherMask")["23"]["inputs"]
    assert feather["mask"] == ["34", 0]
    paste = nodes_of(wf, "ImageCompositeMasked")["24"]["inputs"]
    assert paste["destination"] == ["9", 0]
    assert paste["source"] == ["21", 0]
    assert paste["mask"] == ["23", 0]
    combine = nodes_of(wf, "VHS_VideoCombine")["22"]["inputs"]
    assert combine["images"] == ["24", 0]


def test_video_replace_model_contact_zone_stays_source():
    """The partner is tracked; where the bodies interlock, her mask is
    cut back so the contact/genital region is never regenerated."""
    wf = gw.video_replace_model("base.mp4", "newmodel.png")
    assert_links_valid(wf)
    partner_text = nodes_of(wf, "CLIPTextEncode")["25"]["inputs"]
    assert partner_text["text"] == "man"
    track2 = nodes_of(wf, "SAM3_VideoTrack")["26"]["inputs"]
    assert track2["conditioning"] == ["25", 0]
    inter = nodes_of(wf, "MaskComposite")["29"]["inputs"]
    assert inter["operation"] == "and"
    assert inter["destination"] == ["14", 0]
    assert inter["source"] == ["28", 0]
    final = nodes_of(wf, "MaskComposite")["31"]["inputs"]
    assert final["operation"] == "subtract"
    assert final["destination"] == ["14", 0]
    assert final["source"] == ["30", 0]


def test_video_replace_model_contact_protection_can_be_disabled():
    wf = gw.video_replace_model("base.mp4", "newmodel.png",
                                protect_contact=False)
    assert_links_valid(wf)
    assert "31" not in wf and "26" not in wf
    vace = nodes_of(wf, "WanVaceToVideo")["18"]["inputs"]
    assert vace["control_masks"] == ["14", 0]


# -------------------------------------------------------------- escape hatch --

def test_queue_workflow_escape_hatch_delegates(monkeypatch):
    seen = {}
    sentinel = object()

    def fake_queue(workflow, opener=None):
        seen["workflow"] = workflow
        return sentinel

    monkeypatch.setattr(gpu_client, "queue_workflow", fake_queue)
    wf = {"99": {"class_type": "EmptyLatentImage",
                 "inputs": {"width": 64, "height": 64, "batch_size": 1}}}
    assert gw.queue_workflow(wf) is sentinel
    assert seen["workflow"] is wf


def test_contact_guard_and_partner_threshold():
    wf = gw.video_replace_model("v.mp4", "r.jpg")
    assert wf["26"]["inputs"]["detection_threshold"] == 0.55
    assert wf["12"]["inputs"]["detection_threshold"] == 0.5
    assert wf["34"]["inputs"]["operation"] == "add"
    assert wf["18"]["inputs"]["control_masks"] == ["34", 0]
    assert wf["15"]["inputs"]["mask"] == ["34", 0]


def test_video_replace_model_scail_graph():
    wf = gw.video_replace_model_scail("v.mp4", "r.jpg", "a woman",
                                      frames=30)
    assert_links_valid(wf)
    assert wf["20"]["class_type"] == "WanSCAILToVideo"
    assert wf["20"]["inputs"]["length"] == 33          # coerced to 4n+1
    assert wf["20"]["inputs"]["replacement_mode"] is True
    assert wf["17"]["inputs"]["replacement_mode"] is True
    assert wf["20"]["inputs"]["pose_video_mask"] == ["17", 0]
    assert wf["20"]["inputs"]["reference_image_mask"] == ["17", 1]
    assert wf["14"]["inputs"]["max_objects"] == 4
    # v2 two-identity + hero reference: the reference fed to SCAIL
    # and CLIP Vision is the hero frame (node 34), the reference
    # detection runs on it too, and the driving track sees everyone.
    assert wf["16"]["inputs"]["image"] == ["34", 0]
    assert wf["19"]["inputs"]["image"] == ["34", 0]
    assert wf["20"]["inputs"]["reference_image"] == ["34", 0]
    assert wf["34"]["class_type"] == "ImageCompositeMasked"
    assert wf["34"]["inputs"]["destination"] == ["30", 0]
    assert wf["33"]["inputs"]["image"] == ["30", 0]
    # Colour finish: generated frames are Reinhard-matched to the
    # source before the composite consumes them.
    assert wf["35"]["class_type"] == "ColorMatch"
    assert wf["35"]["inputs"]["image_target"] == ["24", 0]
    # Zoned finish (default): source is the destination, only her
    # zone (tracked-her minus the both-masks contact patch) is
    # generated; the union composite node is not built.
    assert "28" not in wf
    assert wf["49"]["inputs"]["destination"] == ["2", 0]
    assert wf["49"]["inputs"]["source"] == ["35", 0]
    assert wf["44"]["inputs"]["operation"] == "and"
    assert wf["47"]["inputs"]["operation"] == "subtract"
    assert wf["47"]["inputs"]["destination"] == ["46", 0]
    assert wf["47"]["inputs"]["source"] == ["45", 0]
    # SAM3 conditioning must come from the SAM3 checkpoint's own CLIP
    # output (node 12, slot 1) -- feeding it UMT5 (node 8) crashes the
    # track with a matmul shape error (pod-verified Oct 15).
    assert wf["13"]["inputs"]["clip"] == ["12", 1]
    assert wf["15"]["inputs"]["clip"] == ["12", 1]
    # Wan prompt conditioning stays on the UMT5 loader.
    assert wf["9"]["inputs"]["clip"] == ["8", 0]
    assert wf["10"]["inputs"]["clip"] == ["8", 0]
    # LoRA chain: UNET -> lightx2v (1.0) -> DPO -> sampling shift 8
    assert wf["5"]["inputs"]["model"] == ["4", 0]
    assert wf["5"]["inputs"]["strength_model"] == 1.0
    assert wf["6"]["inputs"]["model"] == ["5", 0]
    assert wf["7"]["inputs"]["shift"] == 8.0
    assert wf["23"]["inputs"]["cfg"] == 1.0
    assert wf["22"]["inputs"]["steps"] == 6
    # zoned composite default: output pastes generated her over
    # source frames inside her zone only
    assert wf["29"]["inputs"]["images"] == ["49", 0]
    assert wf["49"]["inputs"]["destination"] == ["2", 0]
    # anti-interference tuning (David's first SCAIL-2 run, Oct 15):
    # stricter reference detection keeps reference-background
    # colours out of the identity mask.
    assert wf["16"]["inputs"]["threshold"] == 0.5
    assert wf["46"]["inputs"]["expand"] == 12
    assert wf["48"]["inputs"]["left"] == 5


def test_video_replace_model_scail_relight_chain():
    wf = gw.video_replace_model_scail("v.mp4", "r.jpg")
    # Relight LoRA sits between DPO and the sampling node.
    assert wf["51"]["class_type"] == "LoraLoaderModelOnly"
    assert wf["51"]["inputs"]["model"] == ["6", 0]
    assert wf["7"]["inputs"]["model"] == ["51", 0]
    off = gw.video_replace_model_scail("v.mp4", "r.jpg",
                                       relight_strength=0.0)
    assert "51" not in off
    assert off["7"]["inputs"]["model"] == ["6", 0]


def test_video_replace_model_scail_person_lora_chain():
    # Default: no person LoRA node (file may not exist on the pod).
    base = gw.video_replace_model_scail("v.mp4", "r.jpg")
    assert "52" not in base
    # With relight on: person LoRA is last, after relight.
    wf = gw.video_replace_model_scail("v.mp4", "r.jpg",
                                      person_lora_strength=1.0)
    assert wf["52"]["class_type"] == "LoraLoaderModelOnly"
    assert wf["52"]["inputs"]["model"] == ["51", 0]
    assert wf["52"]["inputs"]["lora_name"] == "skswoman.safetensors"
    assert wf["7"]["inputs"]["model"] == ["52", 0]
    # With relight off: person LoRA hangs off DPO instead.
    wf2 = gw.video_replace_model_scail("v.mp4", "r.jpg",
                                       relight_strength=0.0,
                                       person_lora_strength=0.7)
    assert wf2["52"]["inputs"]["model"] == ["6", 0]
    assert wf2["52"]["inputs"]["strength_model"] == 0.7
    assert wf2["7"]["inputs"]["model"] == ["52", 0]
    # v2 selectable by version key; resolver falls back to v1.
    wf3 = gw.video_replace_model_scail("v.mp4", "r.jpg",
                                       person_lora_strength=1.0,
                                       person_lora_file="v2")
    assert wf3["52"]["inputs"]["lora_name"] == "skswoman_v2.safetensors"
    assert gw.person_lora_filename("v2") == "skswoman_v2.safetensors"
    assert gw.person_lora_filename("skswoman_v2.safetensors") == \
        "skswoman_v2.safetensors"
    assert gw.person_lora_filename("v1") == "skswoman.safetensors"
    assert gw.person_lora_filename("bogus") == "skswoman.safetensors"
    assert gw.person_lora_filename(None) == "skswoman.safetensors"


def test_video_replace_model_scail_gap_person_lora():
    wf = gw.video_replace_model_scail_gap(
        "v.mp4", "r.jpg", cache_id="job-9", person_lora_strength=1.0)
    assert_links_valid(wf)
    assert wf["52"]["inputs"]["lora_name"] == "skswoman.safetensors"
    assert wf["52"]["inputs"]["model"] == ["51", 0]
    assert wf["7"]["inputs"]["model"] == ["52", 0]
    off = gw.video_replace_model_scail_gap("v.mp4", "r.jpg",
                                           cache_id="job-9")
    assert "52" not in off
    v2 = gw.video_replace_model_scail_gap(
        "v.mp4", "r.jpg", cache_id="job-9", person_lora_strength=1.0,
        person_lora_file="v2")
    assert v2["52"]["inputs"]["lora_name"] == "skswoman_v2.safetensors"
    assert v2["52"]["inputs"]["model"] == ["51", 0]
    assert v2["7"]["inputs"]["model"] == ["52", 0]


def test_video_replace_model_scail_continuation():
    wf = gw.video_replace_model_scail(
        "v.mp4", "r.jpg", previous_frames_video="anchor.mp4",
        frame_offset=76)
    assert_links_valid(wf)
    assert wf["50"]["class_type"] == "VHS_LoadVideo"
    assert wf["20"]["inputs"]["previous_frames"] == ["50", 0]
    assert wf["20"]["inputs"]["video_frame_offset"] == 76
    assert wf["20"]["inputs"]["previous_frame_count"] == 5
    plain = gw.video_replace_model_scail("v.mp4", "r.jpg")
    assert "50" not in plain
    assert "previous_frames" not in plain["20"]["inputs"]


def test_video_replace_model_scail_union_mode():
    wf = gw.video_replace_model_scail("v.mp4", "r.jpg", zoned=False)
    assert_links_valid(wf)
    assert "49" not in wf
    assert wf["28"]["inputs"]["destination"] == ["2", 0]
    assert wf["29"]["inputs"]["images"] == ["28", 0]


def test_video_replace_model_scail_no_composite():
    wf = gw.video_replace_model_scail("v.mp4", "r.jpg", composite=False)
    assert "28" not in wf
    # No composite -- but the colour finish still applies.
    assert wf["29"]["inputs"]["images"] == ["35", 0]


def test_video_replace_model_scail_specific_prompts():
    # Pivot (Oct 18): single-target selection is by SAM3 text --
    # specific hero/ref terms must land in the graph's text inputs
    # (the generic "person" defaults were part of the fragility).
    wf = gw.video_replace_model_scail(
        "v.mp4", "r.jpg", mask_prompt="woman",
        hero_prompt="woman with long dark hair",
        ref_prompt="woman with long dark hair")
    assert_links_valid(wf)
    texts = [n["inputs"].get("text") for n in wf.values()
             if isinstance(n.get("inputs"), dict)]
    assert "woman with long dark hair" in texts
    assert "woman" in texts


def test_video_replace_model_scail_seedvr2_finish():
    plain = gw.video_replace_model_scail("v.mp4", "r.jpg", zoned=False)
    assert "62" not in plain
    wf = gw.video_replace_model_scail("v.mp4", "r.jpg", zoned=False,
                                      seedvr2_resolution=1024)
    assert_links_valid(wf)
    assert wf["60"]["class_type"] == "SeedVR2LoadDiTModel"
    assert wf["60"]["inputs"]["model"] == gw.MODELS["seedvr2_dit"]
    assert wf["61"]["class_type"] == "SeedVR2LoadVAEModel"
    assert wf["61"]["inputs"]["model"] == gw.MODELS["seedvr2_vae"]
    assert wf["62"]["class_type"] == "SeedVR2VideoUpscaler"
    # The pass runs over the finished (union-composited) frames.
    assert wf["62"]["inputs"]["image"] == ["28", 0]
    assert wf["62"]["inputs"]["resolution"] == 1024
    assert wf["29"]["inputs"]["images"] == ["62", 0]


def test_video_replace_model_scail_gap_graph():
    wf = gw.video_replace_model_scail_gap("v.mp4", "r.jpg",
                                          cache_id="job-9")
    assert_links_valid(wf)
    orch = wf["71"]
    assert orch["class_type"] == "GAPSCAIL2LongVideo"
    assert orch["inputs"]["pose_video"] == ["2", 0]
    assert orch["inputs"]["pose_video_mask"] == ["17", 0]
    assert orch["inputs"]["reference_image"] == ["70", 0]
    assert orch["inputs"]["reference_image_mask"] == ["70", 1]
    assert orch["inputs"]["clip_vision_output"] == ["19", 0]
    assert orch["inputs"]["character_count"] == ["70", 3]
    assert orch["inputs"]["cache_id"] == "job-9"
    assert orch["inputs"]["steps"] == 24
    assert orch["inputs"]["cfg"] == 5.0
    assert "{characters}" in orch["inputs"]["base_prompt"]
    lines = orch["inputs"]["character_prompts"].splitlines()
    assert lines[0].startswith("1: the woman")
    assert lines[1].startswith("2: the man")
    mcr = wf["70"]
    assert mcr["class_type"] == "GAPMultiCharacterReference"
    assert mcr["inputs"]["image_1"] == ["3", 0]
    assert mcr["inputs"]["mask_1"] == ["33", 0]
    assert mcr["inputs"]["image_2"] == ["30", 0]
    assert mcr["inputs"]["mask_2"] == ["37", 0]
    # CLIP vision encodes the composite primary, not the raw photo.
    assert wf["19"]["inputs"]["image"] == ["70", 2]
    # Union finish over the orchestrator output.
    assert wf["29"]["inputs"]["images"] == ["28", 0]
    assert wf["28"]["inputs"]["source"] == ["35", 0]


def test_video_replace_model_scail_gap_swap():
    wf = gw.video_replace_model_scail_gap("v.mp4", "r.jpg", gap_swap=True)
    assert wf["70"]["inputs"]["image_1"] == ["30", 0]
    assert wf["70"]["inputs"]["image_2"] == ["3", 0]
    lines = wf["71"]["inputs"]["character_prompts"].splitlines()
    assert lines[0].startswith("1: the man")
    assert lines[1].startswith("2: the woman")


def test_video_replace_model_scail_gap_zoned_finish():
    wf = gw.video_replace_model_scail_gap("v.mp4", "r.jpg",
                                          cache_id="job-9", zoned=True)
    assert_links_valid(wf)
    # Dedicated single-person tracks drive the zones, reusing the
    # reference conditionings (hers "woman" [32], his "man" [36]).
    assert wf["72"]["class_type"] == "SAM3_VideoTrack"
    assert wf["72"]["inputs"]["conditioning"] == ["32", 0]
    assert wf["72"]["inputs"]["max_objects"] == 1
    assert wf["73"]["inputs"]["conditioning"] == ["36", 0]
    assert wf["73"]["inputs"]["detection_threshold"] == 0.55
    # Contact patch = grown overlap of both tracks, padded, then
    # subtracted from her grown track.
    assert wf["78"]["inputs"]["operation"] == "and"
    assert wf["81"]["inputs"]["operation"] == "subtract"
    # Only her zone is generated: the composite destination is the
    # source frames and its source is the colour-matched generation.
    assert wf["83"]["class_type"] == "ImageCompositeMasked"
    assert wf["83"]["inputs"]["destination"] == ["2", 0]
    assert wf["83"]["inputs"]["source"] == ["35", 0]
    assert wf["29"]["inputs"]["images"] == ["83", 0]


def test_video_replace_model_scail_gap_qc_only():
    wf = gw.video_replace_model_scail_gap("v.mp4", "r.jpg", frames=81,
                                          qc_only=True)
    assert_links_valid(wf)
    # No sampling and no video combine: the QC run is tracking +
    # reference build only, so it costs seconds, not a render.
    assert "71" not in wf
    assert "29" not in wf
    saves = {n["inputs"]["filename_prefix"] for n in wf.values()
             if n["class_type"] == "SaveImage"}
    assert saves == {"myai/qc_mask_start", "myai/qc_mask_mid",
                     "myai/qc_mask_end", "myai/qc_src_start",
                     "myai/qc_src_mid", "myai/qc_src_end",
                     "myai/qc_ref_composite", "myai/qc_ref_mask"}
    # Mask stills come off the coloured-mask video at start/mid/end.
    batches = [n for n in wf.values()
               if n["class_type"] == "ImageFromBatch"
               and n["inputs"]["image"] == ["17", 0]]
    assert sorted(n["inputs"]["batch_index"] for n in batches) == [0, 40, 80]


def test_video_replace_model_scail_gap_qc_stills_in_full_graph():
    wf = gw.video_replace_model_scail_gap("v.mp4", "r.jpg", frames=81,
                                          qc_stills=True)
    assert_links_valid(wf)
    assert wf["71"]["class_type"] == "GAPSCAIL2LongVideo"
    prefixes = {n["inputs"]["filename_prefix"] for n in wf.values()
                if n["class_type"] == "SaveImage"}
    assert "myai/qc_mask_mid" in prefixes
    assert "myai/qc_ref_composite" in prefixes


def test_video_seedvr2_upscale_graph():
    wf = gw.video_seedvr2_upscale("v.mp4", resolution=720, fps=24.0)
    assert_links_valid(wf)
    assert wf["4"]["class_type"] == "SeedVR2VideoUpscaler"
    assert wf["4"]["inputs"]["image"] == ["1", 0]
    assert wf["4"]["inputs"]["resolution"] == 720
    assert wf["5"]["inputs"]["images"] == ["4", 0]
    assert wf["5"]["inputs"]["frame_rate"] == 24.0


def test_video_replace_model_scail_gap_closeup_extra_view():
    wf = gw.video_replace_model_scail_gap("v.mp4", "r.jpg",
                                          closeup_image="closeup.jpg")
    assert_links_valid(wf)
    assert wf["65"]["class_type"] == "LoadImage"
    assert wf["65"]["inputs"]["image"] == "closeup.jpg"
    assert wf["68"]["class_type"] == "GAPCharacterExtraView"
    assert wf["68"]["inputs"]["character"] == 1
    assert wf["68"]["inputs"]["reference_image"] == ["70", 0]
    assert wf["68"]["inputs"]["mask"] == ["67", 0]
    # The orchestrator consumes the extended reference batch; CLIP
    # vision still encodes the primary composite.
    assert wf["71"]["inputs"]["reference_image"] == ["68", 0]
    assert wf["71"]["inputs"]["reference_image_mask"] == ["68", 1]
    assert wf["19"]["inputs"]["image"] == ["70", 2]


def test_video_replace_model_scail_gap_closeup_follows_swap():
    wf = gw.video_replace_model_scail_gap("v.mp4", "r.jpg", gap_swap=True,
                                          closeup_image="closeup.jpg")
    assert_links_valid(wf)
    # With the slots flipped, her close-up belongs to slot 2.
    assert wf["68"]["inputs"]["character"] == 2


def test_video_replace_model_scail_gap_dit_default_fp8():
    wf = gw.video_replace_model_scail_gap("v.mp4", "r.jpg")
    assert wf["4"]["inputs"]["unet_name"] == gw.MODELS["scail_unet"]


def test_video_replace_model_scail_gap_dit_fp16_override():
    # Ladder-2 (Oct 18): the person LoRAs trained against the fp16
    # DiT; the inference default is the fp8_scaled build. The
    # override must land on the GAP graph's UNETLoader verbatim.
    wf = gw.video_replace_model_scail_gap(
        "v.mp4", "r.jpg",
        dit_file="wan2.1_14B_SCAIL_2_fp16.safetensors")
    assert_links_valid(wf)
    assert wf["4"]["inputs"]["unet_name"] == \
        "wan2.1_14B_SCAIL_2_fp16.safetensors"

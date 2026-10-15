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
                              "controlnet", "upscale_models")
            basenames.add(os.path.basename(path_in_repo))
    assert basenames == set(gw.MODELS.values())


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
    vace = nodes_of(wf, "WanVaceToVideo")["9"]["inputs"]
    assert vace["strength"] == 0.85
    assert vace["control_video"] == ["7", 0]
    assert vace["reference_image"] == ["8", 0]
    assert vace["length"] == 45
    ks = nodes_of(wf, "KSampler")["10"]["inputs"]
    assert ks["seed"] == 9 and ks["steps"] == 26 and ks["cfg"] == 4.5
    assert ks["positive"] == ["9", 0] and ks["latent_image"] == ["9", 2]
    texts = nodes_of(wf, "CLIPTextEncode")
    assert texts["5"]["inputs"]["text"] == "same dance, new performer"


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

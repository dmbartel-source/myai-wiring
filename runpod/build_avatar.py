# -*- coding: utf-8 -*-
"""Stage 2 headless driver for LHM++: build the avatar from reference images
and render a sample-motion video for judgment. AVATAR-BUILD-V1

Replicates the Gradio app's own core_fn pipeline in-process, using only the
project's supported modules (same setup as launch_gradio_app in app.py).
Run inside /workspace/LHM-plusplus with the avatar venv python.
"""
import argparse
import os
import sys
import traceback
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import numpy as np
import torch

torch._dynamo.config.disable = True

import imageio.v3 as iio
from accelerate import Accelerator
from PIL import Image

from core.datasets.data_utils import SrcImagePipeline
from core.utils.app_utils import get_motion_information, prepare_input_and_output
from core.utils.model_card import MODEL_CONFIG
from core.utils.model_download_utils import AutoModelQuery
from engine.pose_estimation.pose_estimator import PoseEstimator
from scripts.download_motion_video import motion_video_check
from scripts.inference.app_inference import (
    build_app_model,
    inference_results,
    parse_app_configs,
)
from scripts.inference.utils import easy_memory_manager

MODEL_NAME = "LHMPP-700M-PixelShuffle"
MOTION_SIZE = 120


def prior_model_check(save_dir: str = "./pretrained_models") -> None:
    """Mirror of app.py's prior check: skip if present, else download+link."""
    human_model_path = os.path.join(save_dir, "human_model_files")
    if os.path.exists(human_model_path):
        print("[build] prior models present")
        return
    if os.path.islink(human_model_path):
        try:
            os.unlink(human_model_path)
        except OSError:
            pass
    print("[build] prior models missing; downloading...")
    auto_query = AutoModelQuery(save_dir=save_dir)
    auto_query.download_all_prior_models()
    print("[build] prior models ready")


def load_ref_gallery(refs_dir: str):
    files = sorted(
        f for f in os.listdir(refs_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    if not files:
        raise SystemExit(f"No reference images found in {refs_dir}")
    gallery = []
    for f in files:
        path = os.path.join(refs_dir, f)
        arr = np.asarray(Image.open(path).convert("RGB"))
        gallery.append((arr, f))
        print(f"[build] ref: {f} {arr.shape}")
    print(f"[build] {len(gallery)} reference images")
    return gallery


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--motion", required=True, help="Path to bundled motion mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--motion-size", type=int, default=MOTION_SIZE)
    ap.add_argument("--backend", default="neural", choices=["neural", "gs"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    work_dir = os.path.join(args.out, "work")
    os.makedirs(work_dir, exist_ok=True)

    os.environ.update(
        {
            "APP_ENABLED": "1",
            "APP_MODEL_NAME": MODEL_NAME,
            "APP_TYPE": "infer.human_lrm_a4o",
            "NUMBA_THREADING_LAYER": "omp",
        }
    )

    prior_model_check("./pretrained_models")
    motion_video_check(save_dir=".")

    auto_query = AutoModelQuery(save_dir="./pretrained_models")
    model_path = auto_query.query(MODEL_NAME)
    print(f"[build] model_path: {model_path}")
    model_cards = {
        MODEL_NAME: {
            "model_path": model_path,
            "model_config": MODEL_CONFIG[MODEL_NAME],
        }
    }

    # The app creates an Accelerator before parsing configs; the model
    # code's logging requires the accelerate state to exist.
    accelerator = Accelerator()
    cfg, _ = parse_app_configs(model_cards)
    processing_list = [
        dict(
            name="PadRatioWithScale",
            target_ratio=5 / 3,
            tgt_max_size_list=[840],
            val=True,
        )
    ]
    dataset_pipeline = SrcImagePipeline(*processing_list)

    print("[build] building model...")
    lhmpp = build_app_model(cfg)
    lhmpp.to("cuda")

    pose_estimator = None
    if cfg.get("use_smplx_shape_estimator", True):
        pose_estimator = PoseEstimator(
            "./pretrained_models/human_model_files/", device="cpu"
        )
        pose_estimator.device = "cuda"

    gallery = load_ref_gallery(args.refs)
    ref_view = len(gallery)

    working = SimpleNamespace(name=work_dir)
    imgs, _sample_path, motion_path, _dump_img_dir, dump_video_path = (
        prepare_input_and_output(
            gallery, None, ref_view, args.motion, working, dataset_pipeline, cfg
        )
    )

    motion_name, motion_seqs = get_motion_information(
        motion_path, cfg, args.motion_size
    )
    video_size = len(motion_seqs["motion_seqs"])
    print(f"[build] motion: {motion_name}, frames: {video_size}")

    device = "cuda"
    dtype = torch.float32
    if pose_estimator is not None:
        with torch.no_grad():
            with easy_memory_manager(pose_estimator, device="cuda"):
                shape_pose = pose_estimator(imgs[0])
        assert shape_pose.is_full_body, (
            f"First reference image is not full-body per pose estimator: "
            f"{shape_pose.msg}"
        )
        print("[build] pose estimator: full-body OK, betas captured")

    img_np = np.stack(imgs) / 255.0
    ref_imgs_tensor = torch.from_numpy(img_np).permute(0, 3, 1, 2).float().to(device)
    smplx_params = motion_seqs["smplx_params"]
    if pose_estimator is not None:
        smplx_params["betas"] = torch.tensor(
            shape_pose.beta, dtype=dtype, device=device
        ).unsqueeze(0)

    print("[build] running inference (build + render)...")
    rgbs = inference_results(
        lhmpp,
        ref_imgs_tensor,
        smplx_params,
        motion_seqs,
        video_size=video_size,
        visualized_center=False,
        device=device,
        infer_output_renderer=args.backend,
    )

    iio.imwrite(
        dump_video_path,
        rgbs,
        fps=30,
        codec="libx264",
        pixelformat="yuv420p",
        bitrate="10M",
        macro_block_size=16,
    )
    final_path = os.path.join(args.out, "avatar_motion_test.mp4")
    os.replace(dump_video_path, final_path)
    print(f"[build] saved video: {final_path}")
    print("AVATAR-BUILD-DONE")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("AVATAR-BUILD-FAILED")
        sys.exit(1)

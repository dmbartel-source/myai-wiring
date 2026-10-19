#!/usr/bin/env python3
# ANCHOR-PRODUCER-V3 — MYAI stage-1 S2 anchor producer (SMPLest-X arm).
# Runs under the SMPLest-X venv python ON THE POD. For every image in
# the views dir: YOLOv8x person detection (ultralytics, conf 0.5,
# largest box) -> SMPLest-X regression through the pinned repo
# checkout -> per-view SMPL-X betas + body_pose, emitted as the
# fitter's init.json sidecar, plus a diagnostics JSON.
# Sidecar contract (verified from stage1_fit.py's stage_run S2 block;
# anchor_producer_research.md SS1): {file: {"betas": [<SMPL-X shape
# coefficients>], "body_pose": [exactly 63 axis-angle floats],
# "weight": <per-view scalar>, "source": "smplest-x-h"}} keyed by the
# manifest view's exact file string (the image basename here).
# Per-image failures are logged and skipped, never fatal.
# The stock SMPLest-X demo saves NO parameters (research SS2.3), so
# this driver wraps the repo's own model classes; the exact
# construction for the pinned checkout (fdebd88) is isolated in
# run_model() behind an ADAPT-AFTER-PROBE marker — run --probe-only
# first on the pod and adapt that one function to what it prints.
import argparse, json, os, sys

MARKER = "ANCHOR-PRODUCER-V3"
SOURCE_TAG = "smplest-x-h"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
WEIGHT_MIN, WEIGHT_MAX = 0.05, 1.0


def clamp_weight(conf):
    """Fusion weight from detector confidence: the fitter's fusion
    consumes ONE per-view scalar (research SS1 — there is no
    per-coefficient uncertainty channel), clamped to [0.05, 1.0] so
    a weak detection still counts a little and none dominates."""
    return min(WEIGHT_MAX, max(WEIGHT_MIN, float(conf)))


def _to_float_list(x):
    """Flatten a regressor output (list / tuple / numpy array /
    torch tensor, possibly batched with a leading 1) to a plain
    list of floats. Duck-typed: no torch import needed here."""
    if hasattr(x, "detach"):            # torch tensor
        x = x.detach().cpu().numpy()
    if hasattr(x, "tolist"):            # numpy array
        x = x.tolist()
    out = []

    def rec(v):
        if isinstance(v, (list, tuple)):
            for e in v:
                rec(e)
        else:
            out.append(float(v))

    rec(x)
    return out


def extract_params(out):
    """SMPLest-X test-mode forward dict -> (betas, body_pose63,
    betas_width). Keys verified from repo source (research SS2.3):
    out['smplx_shape'] (betas, standard SMPL-X shape basis) and
    out['smplx_body_pose'] (axis-angle; the repo's full-pose split
    is root 3 + body 63 + hands 45+45 + jaw 3, and smplx_body_pose
    itself is exactly the 63 body values the fitter consumes)."""
    if not isinstance(out, dict):
        raise ValueError(f"model output is not a dict: {type(out)}")
    for k in ("smplx_shape", "smplx_body_pose"):
        if k not in out:
            raise KeyError(f"model output missing {k!r}; keys: "
                           f"{sorted(out.keys())}")
    betas = _to_float_list(out["smplx_shape"])
    pose = _to_float_list(out["smplx_body_pose"])
    if len(pose) != 63:
        raise ValueError(f"smplx_body_pose has {len(pose)} values, "
                         f"contract needs exactly 63")
    if not betas:
        raise ValueError("smplx_shape is empty")
    return betas, pose, len(betas)


def write_init_json(entries, out_path):
    """Write the S2 sidecar, asserting the contract per entry:
    exactly the keys {betas, body_pose, weight, source}, a
    63-length body_pose, and a clamped weight."""
    for name, e in entries.items():
        assert set(e.keys()) == {"betas", "body_pose", "weight",
                                 "source"}, (name, sorted(e.keys()))
        assert len(e["body_pose"]) == 63, name
        assert WEIGHT_MIN <= e["weight"] <= WEIGHT_MAX, name
        assert e["source"] == SOURCE_TAG, name
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=1)
    return out_path


# ---------------- detection (YOLOv8x via ultralytics) ----------------

_YOLO = None


def detect_person(image_path, conf=0.5):
    """Largest person box (YOLOv8x via ultralytics; conf 0.5 and
    the yolov8x.pt weights are the SMPLest-X repo's own detector
    configuration, research SS2.3) -> (bbox_xyxy [x0, y0, x1, y1]
    floats, det_conf) or None when no person is detected."""
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO("yolov8x.pt")
    res = _YOLO.predict(source=image_path, conf=conf, classes=[0],
                        verbose=False)
    best = None
    for r in res:
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        cf = boxes.conf.cpu().numpy()
        for b, c in zip(xyxy, cf):
            area = float((b[2] - b[0]) * (b[3] - b[1]))
            if best is None or area > best[0]:
                best = (area, [float(v) for v in b], float(c))
    if best is None:
        return None
    return best[1], best[2]


# ---------------- the regressor (repo-wrapping core) ----------------

def run_model(repo, image_path, bbox):
    """Regress SMPL-X params for one image through the pinned
    SMPLest-X checkout at `repo`. Returns {"betas": [...],
    "body_pose": [63 floats], "betas_width": int} built by
    extract_params() from the model's test-mode output dict.

    Verified from repo source (research SS2.3): the parameters live
    in the model output dict under 'smplx_shape' /
    'smplx_body_pose'; main/inference.py builds the model from the
    repo config (configs/config_smplest_x_h.py — SMPLX layer over
    './human_models/human_model_files', already on our volume) and
    a Tester class, and the stock paths persist only overlay
    renders, never the parameters.

    # ADAPTED (Oct 19, 2026, Noor): wired from main/inference.py's
    # own code at fdebd88, read on the pod — Config.load_config on
    # pretrained_models/smplest_x_h/config_base.py + checkpoint
    # update, Tester(cfg)._make_model(), and the frame loop's exact
    # preprocessing (utils.data_utils process_bbox /
    # generate_patch_image, ToTensor()/255, model({'img': img},
    # {}, {}, 'test')). The bbox comes from this producer's own
    # YOLO pass (the repo's detector, same config).
    """
    return _run_model_impl(repo, image_path, bbox)


_MODEL_CACHE = {}


def _load_model(repo):
    """Construct the SMPLest-X Tester + model exactly as
    main/inference.py does (checkout fdebd88): Config.load_config
    on pretrained_models/smplest_x_h/config_base.py, update with
    the checkpoint path, Tester(cfg), tester._make_model().
    Cached: construction loads the 8.2 GB checkpoint once."""
    if "tester" in _MODEL_CACHE:
        return _MODEL_CACHE["tester"], _MODEL_CACHE["cfg"]
    import os
    os.chdir(repo)  # repo config resolves ./human_models etc.
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from main.config import Config
    from main.base import Tester
    cfg = Config.load_config(
        "./pretrained_models/smplest_x_h/config_base.py")
    cfg.update_config({
        "model": {"pretrained_model_path":
                  "./pretrained_models/smplest_x_h/smplest_x_h.pth.tar"},
        "log": {"exp_name": "anchor_producer",
                "log_dir": os.path.join(
                    repo, "outputs", "anchor_producer", "log")}})
    cfg.prepare_log()
    # human_models.SMPLX is a singleton whose FIRST construction
    # must pass the model path (human_models.py: assert config is
    # not None); main/inference.py constructs it before the Tester
    # for exactly this reason — the model's own layer deep-copies
    # the singleton's neutral layer.
    from human_models.human_models import SMPLX
    SMPLX(cfg.model.human_model_path)
    tester = Tester(cfg)
    tester._make_model()
    _MODEL_CACHE["tester"] = tester
    _MODEL_CACHE["cfg"] = cfg
    return tester, cfg


def _run_model_impl(repo, image_path, bbox):
    """Single-image regression mirroring main/inference.py's
    frame loop (fdebd88): load_img -> process_bbox (cfg ratio) ->
    generate_patch_image -> ToTensor()/255 -> tester.model(
    {'img': img}, {}, {}, 'test') -> extract_params(out)."""
    import numpy as np
    import torch
    import torchvision.transforms as transforms
    tester, cfg = _load_model(repo)
    from utils.data_utils import (load_img, process_bbox,
                                  generate_patch_image)
    original_img = load_img(image_path)
    img_h, img_w = original_img.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in bbox)
    xywh = np.zeros((4))
    xywh[0], xywh[1] = x0, y0
    xywh[2], xywh[3] = abs(x1 - x0), abs(y1 - y0)
    bb = process_bbox(bbox=xywh, img_width=img_w, img_height=img_h,
                      input_img_shape=cfg.model.input_img_shape,
                      ratio=getattr(cfg.data, "bbox_ratio", 1.25))
    img, _, _ = generate_patch_image(
        cvimg=original_img, bbox=bb, scale=1.0, rot=0.0,
        do_flip=False, out_shape=cfg.model.input_img_shape)
    img = transforms.ToTensor()(img.astype(np.float32)) / 255
    img = img.cuda()[None, :, :, :]
    with torch.no_grad():
        out = tester.model({"img": img}, {}, {}, "test")
    b, p, w = extract_params(out)
    return {"betas": b, "body_pose": p, "betas_width": w}


# ---------------- probe mode (on-pod adaptation aid) ----------------

def probe(repo):
    """Print the repo facts the on-pod adaptation of run_model()
    needs: main/inference.py top-level names (via ast — no import
    side effects), the Tester class signature via inspect when
    importable, and the config's human_model_path."""
    import ast, inspect
    print(f"PROBE repo={repo} marker={MARKER}")
    inf = os.path.join(repo, "main", "inference.py")
    if os.path.exists(inf):
        tree = ast.parse(open(inf).read())
        names = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(f"from {node.module} import "
                             + ", ".join(a.name for a in node.names))
        print(f"PROBE main/inference.py top-level: {names}")
    else:
        print(f"PROBE main/inference.py NOT FOUND at {inf}")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    found = False
    for modname in ("main.inference", "main.trainer"):
        try:
            mod = __import__(modname, fromlist=["Tester"])
            T = getattr(mod, "Tester", None)
            if T is not None:
                print(f"PROBE Tester: {modname}.Tester signature="
                      f"{inspect.signature(T.__init__)}")
                found = True
                break
        except Exception as e:
            print(f"PROBE Tester import via {modname} failed: {e}")
    if not found:
        print("PROBE Tester: not found in main.inference / "
              "main.trainer")
    for modname in ("configs.config_smplest_x_h", "main.config"):
        try:
            mod = __import__(modname, fromlist=["cfg"])
            cfg = getattr(mod, "cfg", None)
            hmp = None
            if cfg is not None:
                hmp = getattr(getattr(cfg, "model", None),
                              "human_model_path", None)
            print(f"PROBE config {modname}: cfg found="
                  f"{cfg is not None} human_model_path={hmp}")
            break
        except Exception as e:
            print(f"PROBE config {modname} failed: {e}")
    print("PROBE-DONE")


# ---------------- selftest (torch-free, repo-free) ----------------

def selftest():
    """Contract test with a FAKE regressor output — no repo, no
    ultralytics, no torch. Verifies extraction, the weight clamp,
    rejection of a malformed pose, and that the writer emits
    contract-valid init.json."""
    import tempfile
    fake_out = {"smplx_shape": [[0.11 * i for i in range(10)]],
                "smplx_body_pose": [[0.001 * i for i in range(63)]],
                "cam_trans": [[0.0, 0.0, 5.0]]}
    betas, pose, width = extract_params(fake_out)
    assert width == 10 and len(betas) == 10, (width, betas)
    assert len(pose) == 63, len(pose)
    assert clamp_weight(0.9) == 0.9
    assert clamp_weight(0.01) == WEIGHT_MIN
    assert clamp_weight(7.5) == WEIGHT_MAX
    entries = {"fake_view.jpg": {"betas": betas, "body_pose": pose,
                                 "weight": clamp_weight(0.9),
                                 "source": SOURCE_TAG}}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "init.json")
        write_init_json(entries, p)
        back = json.load(open(p))
    e = back["fake_view.jpg"]
    assert set(e.keys()) == {"betas", "body_pose", "weight", "source"}
    assert len(e["body_pose"]) == 63 and len(e["betas"]) == 10
    assert e["source"] == "smplest-x-h"
    try:
        extract_params({"smplx_shape": [[0.0] * 10],
                        "smplx_body_pose": [[0.0] * 10]})
        raise AssertionError("short body_pose was not rejected")
    except ValueError:
        pass
    print("ANCHOR-PRODUCER-SELFTEST-OK")


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/workspace/anchor/SMPLest-X")
    ap.add_argument("--views",
                    default="/workspace/ComfyUI/input/stage1_views")
    ap.add_argument("--out", default="/workspace/stage1/init.json")
    ap.add_argument("--diag", default="/workspace/stage1/init_diag.json")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.probe_only:
        probe(a.repo)
        return
    print(f"{MARKER} repo={a.repo} views={a.views}", flush=True)
    files = sorted(f for f in os.listdir(a.views)
                   if f.lower().endswith(IMAGE_EXTS))
    print(f"PRODUCER {len(files)} images", flush=True)
    entries, diag = {}, {}
    for name in files:
        path = os.path.join(a.views, name)
        try:
            det = detect_person(path)
            if det is None:
                diag[name] = {"betas_width": None, "det_conf": None,
                              "status": "no_detection"}
                print(f"PRODUCER-NODET {name}", flush=True)
                continue
            bbox, conf = det
            res = run_model(a.repo, path, bbox)
            w = clamp_weight(conf)
            entries[name] = {"betas": res["betas"],
                             "body_pose": res["body_pose"],
                             "weight": w, "source": SOURCE_TAG}
            diag[name] = {"betas_width": res["betas_width"],
                          "det_conf": float(conf), "status": "ok"}
            print(f"PRODUCER-OK {name} "
                  f"betas_width={res['betas_width']} "
                  f"det_conf={conf:.3f} weight={w:.3f}", flush=True)
        except Exception as e:
            diag[name] = {"betas_width": None, "det_conf": None,
                          "status": f"error: {e}"}
            print(f"PRODUCER-SKIP {name}: {e}", flush=True)
    write_init_json(entries, a.out)
    with open(a.diag, "w") as f:
        json.dump(diag, f, indent=1)
    print(f"PRODUCER-DONE ok={len(entries)} "
          f"skipped={len(files) - len(entries)} out={a.out} "
          f"diag={a.diag}", flush=True)


if __name__ == "__main__":
    main()

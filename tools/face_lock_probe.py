#!/usr/bin/env python3
"""Read-only Face Lock diagnostic probe (Oct 17, 2026).

Run on the Studio server with the Studio venv python:
    ~/myai-wiring/.venv/bin/python ~/myai-wiring/tools/face_lock_probe.py

Prints ONLY environment facts, file names, counts and shapes — never
image data — so its output can be pasted into chat safely. It answers,
in order, the only questions that can make _face_lock fail before any
video work starts:

  1. Does cv2 import? Does face_vfx import (studio_server sets fv=None
     on ImportError and every later lock attempt raises)?
  2. Which face detector would run (DNN vs Haar fallback), and do the
     DNN model files and the dlib landmark file exist?
  3. For every image in the Studio uploads dirs: can it be read, how
     many faces does detect_faces find, and does the server's own
     _ref_face_crop succeed on it (shape) or raise (exact error)?
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("python:", sys.version.split()[0])
try:
    import cv2
    print("cv2: OK", cv2.__version__)
except Exception as e:  # noqa: BLE001 - the probe reports, never dies
    print("cv2: FAIL", type(e).__name__, e)

fv = None
try:
    import face_vfx as fv
    print("face_vfx: OK")
    print("MODELS_DIR:", fv.MODELS_DIR)
    print("dnn proto exists:", os.path.isfile(fv._DNN_PROTO))
    print("dnn model exists:", os.path.isfile(fv._DNN_MODEL))
    print("dlib landmarks exists:", os.path.isfile(fv._DLIB_PREDICTOR))
    print("detection_method:", fv.detection_method())
except Exception as e:  # noqa: BLE001
    print("face_vfx: FAIL", type(e).__name__, e)

ss = None
try:
    import studio_server as ss
    print("studio_server import: OK | fv is None:", ss.fv is None)
except Exception as e:  # noqa: BLE001
    print("studio_server import: FAIL", type(e).__name__, e)

exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
seen = set()
for base in (os.path.expanduser("~/uploads"),
             os.path.expanduser("~/nanobot-uploads")):
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(base, "**", ext),
                           recursive=True)
    for path in sorted(files):
        if path in seen:
            continue
        seen.add(path)
        rel = os.path.relpath(path, os.path.expanduser("~"))
        try:
            import cv2 as _cv2
            img = _cv2.imread(path)
            if img is None:
                print(f"img {rel}: UNREADABLE")
                continue
            faces = fv.detect_faces(img) if fv is not None else []
            print(f"img {rel}: shape={img.shape} faces={len(faces)}")
            if ss is not None:
                try:
                    crop = ss._ref_face_crop(path)
                    print(f"    ref_face_crop: OK shape={crop.shape}")
                except Exception as e:  # noqa: BLE001
                    print(f"    ref_face_crop: FAIL "
                          f"{type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"img {rel}: ERROR {type(e).__name__}: {e}")
print("probe done")

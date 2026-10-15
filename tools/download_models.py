#!/usr/bin/env python3
"""Download optional face models for tools/face_vfx.py. NOT auto-run.

The pipeline works WITHOUT these (Haar cascade fallback). Run this once
on the VPS if you want the more accurate DNN detector and dlib's
68-point landmarks:

    python3 tools/download_models.py

Models (public research artifacts):
  1. OpenCV DNN face detector (Caffe, ~100 MB):
     prototxt: https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt
     caffemodel: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
  2. dlib 68-face-landmark shape predictor (bz2, ~100 MB download):
     http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Installs into ~/.myai/models (or $MYAI_MODELS_DIR).
"""

from __future__ import annotations

import bz2
import os
import shutil
import sys
import urllib.request

MODELS_DIR = os.environ.get(
    "MYAI_MODELS_DIR", os.path.join(os.path.expanduser("~"), ".myai", "models"))

FILES = [
    ("deploy.prototxt",
     "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt",
     False),
    ("res10_300x300_ssd_iter_140000.caffemodel",
     "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
     False),
    ("shape_predictor_68_face_landmarks.dat.bz2",
     "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
     True),
]


def _fetch(url: str, dest: str) -> None:
    print(f"downloading {url} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "MYAI/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 256)
    print(f"  saved {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def main() -> int:
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, url, is_bz2 in FILES:
        dest = os.path.join(MODELS_DIR, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            print(f"exists, skipping: {dest}")
            continue
        try:
            _fetch(url, dest)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"FAILED {name}: {e}", file=sys.stderr)
            if os.path.isfile(dest):
                os.remove(dest)
            continue
        if is_bz2:
            out = dest[: -len(".bz2")]
            print(f"decompressing -> {out} ...", flush=True)
            with bz2.open(dest, "rb") as fin, open(out, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=1024 * 256)
            print(f"  done ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(f"models in {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

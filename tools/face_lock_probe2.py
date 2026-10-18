#!/usr/bin/env python3
"""Face Lock probe, stage 2 (read-only): why does the detector see 0 faces?

Stage 1 (face_lock_probe.py) proved the reference crop fails with
"no face detected in reference photo" on every uploads image. This
stage separates the possible causes, printing only facts and counts:

- are the Haar cascade XMLs actually present in this cv2 install?
  (_detect_haar silently returns [] for every image if they are not)
- do the DNN model files exist, and how large are they?
  (a tiny file usually means a failed/HTML download)
- per sample image: face counts for auto / haar / dnn detection, and
  for auto detection after 90/180/270-degree rotation (phone photos
  whose EXIF orientation cv2.imread does not apply defeat frontal
  detectors when sideways).

Run with the Studio venv python on the server. No image data is
printed and nothing is written.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

import face_vfx as fv  # noqa: E402

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def sample_images(limit=8):
    out = []
    for base in (os.path.expanduser("~/uploads"),
                 os.path.expanduser("~/nanobot-uploads")):
        for ext in IMG_EXTS:
            out.extend(sorted(glob.glob(os.path.join(base, "**", ext),
                                        recursive=True)))
    return out[:limit]


def count(img, method):
    try:
        return len(fv.detect_faces(img, method=method))
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def main():
    print("cv2 version:", cv2.__version__)
    print("haarcascades dir:", cv2.data.haarcascades)
    for name in ("haarcascade_frontalface_default.xml",
                 "haarcascade_frontalface_alt2.xml"):
        p = os.path.join(cv2.data.haarcascades, name)
        print(f"cascade {name}: exists={os.path.isfile(p)}")
    for path in (fv._DNN_PROTO, fv._DNN_MODEL, fv._DLIB_PREDICTOR):
        print(f"model {path}: exists={os.path.isfile(path)} "
              f"bytes={os.path.getsize(path) if os.path.isfile(path) else 0}")
    print("detection_method():", fv.detection_method())
    imgs = sample_images()
    print(f"sample images: {len(imgs)}")
    for path in imgs:
        img = cv2.imread(path)
        if img is None:
            print(f"img {path}: unreadable")
            continue
        line = (f"img {path}: shape={img.shape} auto={count(img, 'auto')} "
                f"haar={count(img, 'haar')} dnn={count(img, 'dnn')}")
        for k, code in (("r90", cv2.ROTATE_90_CLOCKWISE),
                        ("r180", cv2.ROTATE_180),
                        ("r270", cv2.ROTATE_90_COUNTERCLOCKWISE)):
            line += f" {k}={count(cv2.rotate(img, code), 'auto')}"
        print(line)
    print("probe2 done")


if __name__ == "__main__":
    main()

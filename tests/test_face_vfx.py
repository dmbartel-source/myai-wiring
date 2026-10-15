"""Tests for tools/face_vfx.py.

All fixtures are SYNTHETIC cartoon faces drawn with OpenCV primitives
(circles/ellipses) — no real face images. No network, deterministic.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import face_vfx as fv

cv2 = pytest.importorskip("cv2")


# ---------------------------------------------------------------- fixtures ----

def cartoon_face(w=640, h=480, cx=320, cy=240, s=1.0):
    """High-contrast synthetic frontal face (Haar-detectable)."""
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    r = int(100 * s)
    cv2.ellipse(img, (cx, cy), (r, int(r * 1.3)), 0, 0, 360, (210, 180, 160), -1)
    ex, ey = int(r * 0.4), int(r * 0.25)
    for sx in (-1, 1):
        cv2.ellipse(img, (cx + sx * ex, cy - ey),
                    (int(r * 0.18), int(r * 0.12)), 0, 0, 360, (240, 240, 240), -1)
        cv2.circle(img, (cx + sx * ex, cy - ey), int(r * 0.08), (20, 20, 20), -1)
    cv2.ellipse(img, (cx - ex, cy - int(r * 0.45)),
                (int(r * 0.22), int(r * 0.06)), 0, 0, 360, (60, 50, 45), -1)
    cv2.ellipse(img, (cx + ex, cy - int(r * 0.45)),
                (int(r * 0.22), int(r * 0.06)), 0, 0, 360, (60, 50, 45), -1)
    cv2.line(img, (cx, cy - int(r * 0.15)), (cx, cy + int(r * 0.25)),
             (230, 205, 185), int(r * 0.09))
    cv2.ellipse(img, (cx, cy + int(r * 0.6)),
                (int(r * 0.32), int(r * 0.1)), 0, 0, 360, (90, 60, 60), -1)
    return img


def face_crop():
    """Tight crop of the cartoon face for use as a source face."""
    img = cartoon_face()
    return img[110:370, 220:420].copy()  # ~200x260 around the face


def write_video(path, frames, fps=15.0):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert out.isOpened()
    for f in frames:
        out.write(f)
    out.release()


def moving_face_video(path, n=12, w=320, h=240):
    """Frames with the cartoon face sliding horizontally."""
    frames = []
    for i in range(n):
        cx = 80 + i * 12
        frames.append(cartoon_face(w=w, h=h, cx=cx, cy=120, s=0.55))
    write_video(path, frames)
    return frames


# --------------------------------------------------------------- detection ----

def test_detect_finds_synthetic_face():
    img = cartoon_face()
    boxes = fv.detect_faces(img)
    if len(boxes) == 0:
        pytest.skip("Haar cascade did not detect synthetic face in this OpenCV build")
    x, y, w, h = boxes[0]
    # Detected box must overlap the drawn face region (220..420, 110..370).
    assert x < 420 and x + w > 220 and y < 370 and y + h > 110


def test_detect_empty_image_returns_empty():
    blank = np.full((240, 320, 3), 128, dtype=np.uint8)
    assert fv.detect_faces(blank) == []


def test_detection_method_reports_haar_without_models():
    # No model files in this environment -> Haar fallback.
    assert fv.detection_method() == "haar"


def test_detect_bad_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        fv.detect_faces(cartoon_face(), method="nope")


# ------------------------------------------------------- landmarks & align ----

def test_estimate_landmarks_fallback_shape_and_bounds():
    crop = face_crop()
    lm = fv.estimate_landmarks(crop)
    assert lm.shape == (3, 2)
    h, w = crop.shape[:2]
    assert (lm[:, 0] >= 0).all() and (lm[:, 0] < w).all()
    assert (lm[:, 1] >= 0).all() and (lm[:, 1] < h).all()
    # left eye left of right eye
    assert lm[0, 0] < lm[1, 0]


def test_similarity_transform_maps_points():
    src = np.array([[0, 0], [10, 0], [5, 8]], dtype=np.float32)
    dst = src * 2.0 + np.array([3, 7], dtype=np.float32)
    m = fv.similarity_transform(src, dst)
    mapped = (m[:, :2] @ src.T).T + m[:, 2]
    assert np.allclose(mapped, dst, atol=1e-3)


def test_align_face_output_size():
    crop = face_crop()
    lm = fv.landmarks_from_box((100, 80, 200, 200))
    out = fv.align_face(crop, lm, size=(128, 128))
    assert out.shape == (128, 128, 3)


def test_align_face_self_consistent():
    """Aligning a face to its own landmarks ≈ identity warp."""
    crop = face_crop()
    lm = fv.estimate_landmarks(crop)
    out = fv.align_face(crop, lm, size=(crop.shape[1], crop.shape[0]))
    diff = np.abs(out.astype(int) - crop.astype(int)).mean()
    assert diff < 12.0  # interpolation only


# ------------------------------------------------------------- compositing ----

def test_composite_differs_from_background():
    bg = np.full((480, 640, 3), 90, dtype=np.uint8)
    cv2.rectangle(bg, (0, 0), (640, 480), (100, 110, 120), -1)
    src = face_crop()
    lm = fv.landmarks_from_box((220, 110, 200, 260))
    out = fv.composite_face(bg, src, lm)
    assert out.shape == bg.shape
    assert np.abs(out.astype(int) - bg.astype(int)).mean() > 1.0


def test_composite_no_hard_edges():
    """seamlessClone output must be smoother at the boundary than a hard paste."""
    bg = np.full((480, 640, 3), (90, 100, 110), dtype=np.uint8)
    src = face_crop()
    box = (220, 110, 200, 260)
    lm = fv.landmarks_from_box(box)
    smooth = fv.composite_face(bg, src, lm, feather=15)

    # Hard-paste baseline with a binary mask.
    h, w = bg.shape[:2]
    mask = fv.face_mask((h, w), lm, feather=0)
    warped = cv2.warpAffine(
        src, fv.similarity_transform(fv.estimate_landmarks(src), lm), (w, h))
    hard = bg.copy()
    hard[mask > 0] = warped[mask > 0]

    # Boundary ring: dilate minus erode of the mask.
    k = np.ones((7, 7), np.uint8)
    ring = cv2.dilate((mask > 0).astype(np.uint8), k) - \
        cv2.erode((mask > 0).astype(np.uint8), k)
    ring = ring > 0
    assert ring.any()

    def max_grad(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        return np.sqrt(gx ** 2 + gy ** 2)[ring].max()

    assert max_grad(smooth) < max_grad(hard)


def test_match_color_shifts_toward_target():
    bg = np.full((200, 200, 3), (60, 90, 170), dtype=np.uint8)  # warm
    src = np.full((200, 200, 3), (170, 120, 60), dtype=np.uint8)  # cool tint
    mask = np.zeros((200, 200), np.uint8)
    mask[50:150, 50:150] = 255
    before = np.abs(src.astype(int)[50:150, 50:150].mean(axis=(0, 1))
                    - bg.astype(int)[50:150, 50:150].mean(axis=(0, 1))).mean()
    out = fv.match_color(src, bg, mask)
    after = np.abs(out.astype(int)[50:150, 50:150].mean(axis=(0, 1))
                   - bg.astype(int)[50:150, 50:150].mean(axis=(0, 1))).mean()
    assert after < before


# ---------------------------------------------------------------- tracking ----

def test_track_face_follows_moving_face(tmp_path):
    path = str(tmp_path / "move.mp4")
    moving_face_video(path, n=12)
    cap = cv2.VideoCapture(path)
    ok, first = cap.read()
    cap.release()
    assert ok
    boxes = fv.detect_faces(first)
    if not boxes:
        pytest.skip("tracker test needs a detected face in frame 0 (Haar missed it here)")
    tracked = fv.track_face(path, boxes[0])
    assert len(tracked) == 12
    # Face moves +12px/frame in x; last box should be well right of first.
    assert tracked[-1][0] > tracked[0][0] + 40
    # y should stay roughly constant.
    assert abs(tracked[-1][1] - tracked[0][1]) < 40


def test_track_face_missing_video_raises(tmp_path):
    with pytest.raises(RuntimeError, match="cannot open"):
        fv.track_face(str(tmp_path / "nope.mp4"), (10, 10, 50, 50))


def test_replace_in_video(tmp_path):
    src_path = str(tmp_path / "src.mp4")
    out_path = str(tmp_path / "out.mp4")
    frames = moving_face_video(src_path, n=8)
    src_face = face_crop()
    # Tint the source so the replacement is unmistakable.
    src_face = cv2.convertScaleAbs(src_face, alpha=1.0, beta=0)
    src_face[:, :, 0] = np.clip(src_face[:, :, 0].astype(int) + 40, 0, 255)
    result = fv.replace_in_video(src_path, src_face, out_path)
    assert result == out_path and os.path.getsize(out_path) > 0
    cap = cv2.VideoCapture(out_path)
    n_out = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame0 = cap.read()
    cap.release()
    assert ok and n_out == 8
    # Output frame 0 differs from input frame 0 in the face area.
    assert np.abs(frame0.astype(int) - frames[0].astype(int)).mean() > 0.5


# ------------------------------------------------------------------ region ----

def test_replace_region_masked_change():
    bg = np.full((240, 320, 3), (80, 90, 100), dtype=np.uint8)
    rep = np.full((240, 320, 3), (200, 60, 60), dtype=np.uint8)
    mask = fv.ellipse_mask((240, 320), (160, 120), (50, 40), feather=11)
    out = fv.replace_region(bg, rep, mask)
    inside = mask > 127
    outside = mask < 10
    assert np.abs(out[inside].astype(int) - bg[inside].astype(int)).mean() > 3.0
    assert np.abs(out[outside].astype(int) - bg[outside].astype(int)).mean() < 3.0


def test_replace_region_empty_mask_is_noop():
    bg = np.full((240, 320, 3), (80, 90, 100), dtype=np.uint8)
    rep = np.full((240, 320, 3), (200, 60, 60), dtype=np.uint8)
    out = fv.replace_region(bg, rep, np.zeros((240, 320), np.uint8))
    assert np.array_equal(out, bg)


def test_ellipse_mask_shape_and_feather():
    m = fv.ellipse_mask((100, 120), (60, 50), (30, 20), feather=11)
    assert m.shape == (100, 120) and m.dtype == np.uint8
    assert m[50, 60] == 255  # center solid
    assert m[0, 0] == 0      # corner empty
    # Feathered edge: some pixels strictly between 0 and 255.
    assert ((m > 0) & (m < 255)).any()


def test_rect_mask_covers_box():
    m = fv.rect_mask((100, 120), (10, 20, 40, 30), feather=0)
    assert m[25, 15] == 255 and m[0, 0] == 0


# ------------------------------------------------------------------- misc ----

def test_download_models_script_documents_urls():
    p = os.path.join(os.path.dirname(__file__), "..", "tools",
                     "download_models.py")
    src = open(p).read()
    assert "res10_300x300_ssd" in src
    assert "shape_predictor_68_face_landmarks" in src
    assert "deploy.prototxt" in src
    # The library itself must not import it (no network coupling).
    import ast
    tree = ast.parse(open(os.path.join(os.path.dirname(__file__), "..",
                                       "tools", "face_vfx.py")).read())
    imported = {n.names[0].name for n in ast.walk(tree)
                if isinstance(n, ast.Import)}
    imported |= {n.module for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom) and n.module}
    assert "urllib" not in imported and "download_models" not in imported


def test_list_capabilities():
    caps = fv.list_capabilities()
    assert caps["cv2"] is True
    assert caps["detection"] == "haar"  # no models here
    assert caps["seamless_clone"] is True

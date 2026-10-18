"""Face/body replacement VFX pipeline — local OpenCV compositing.

Professional head-replacement / face-replacement technique from film
post-production: detect -> align -> color-match -> Poisson-blend, with
per-frame tracking for video ("replace over a real movie").

Only use on people who have consented (yourself, collaborators, licensed
footage). Never create non-consensual intimate imagery or deceptive
content presenting real people.

Local-only: OpenCV + NumPy. No network calls in this module, ever.
Face detection prefers the OpenCV DNN detector when model files are
present (see download_models.py), otherwise falls back to Haar cascades.
Deterministic: no randomness anywhere; same inputs -> same outputs.

Conventions (match tools/local_effects.py):
  - Images are BGR uint8 NumPy arrays (OpenCV convention).
  - Boxes are (x, y, w, h) int tuples, top-left origin.
  - Landmarks are (N, 2) float32 arrays in image coordinates.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV required at runtime
    cv2 = None  # type: ignore


# ------------------------------------------------------------------ paths ----

def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not installed")


MODELS_DIR = os.environ.get(
    "MYAI_MODELS_DIR",
    os.path.join(os.path.expanduser("~"), ".myai", "models"),
)
_DNN_PROTO = os.path.join(MODELS_DIR, "deploy.prototxt")
_DNN_MODEL = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
_DLIB_PREDICTOR = os.path.join(
    MODELS_DIR, "shape_predictor_68_face_landmarks.dat")

_YUNET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")


# --------------------------------------------------------------- detection ----

def _detect_yunet(image: np.ndarray, conf: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """YuNet (ONNX) face detector via cv2.FaceDetectorYN.

    The detector OpenCV 5 supports: the Caffe importer was removed there,
    so the res10 Caffe pair can no longer load. Requires the YuNet model.
    """
    _require_cv2()
    if not os.path.isfile(_YUNET_MODEL):
        raise FileNotFoundError("YuNet face model not present; see download_models.py")
    create = getattr(cv2, "FaceDetectorYN_create", None) or getattr(
        cv2.FaceDetectorYN, "create", None)
    if create is None:
        raise RuntimeError("this OpenCV build has no FaceDetectorYN")
    h, w = image.shape[:2]
    det = create(_YUNET_MODEL, "", (w, h), conf, 0.3, 5000)
    det.setInputSize((w, h))
    _, faces = det.detect(image)
    boxes: List[Tuple[int, int, int, int]] = []
    if faces is not None:
        for row in faces:
            x, y, bw, bh = (int(round(float(v))) for v in row[:4])
            x, y = max(0, x), max(0, y)
            boxes.append((x, y, max(0, bw), max(0, bh)))
    return boxes


def _dnn_ready() -> bool:
    """True when a DNN-class detector can actually run: YuNet ONNX present,
    or the legacy Caffe pair present."""
    return os.path.isfile(_YUNET_MODEL) or (
        os.path.isfile(_DNN_PROTO) and os.path.isfile(_DNN_MODEL))


def _detect_dnn(image: np.ndarray, conf: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """OpenCV DNN face detector. Prefers YuNet (ONNX); falls back to the
    res10_300x300_ssd Caffe pair on OpenCV builds that still load Caffe."""
    _require_cv2()
    if os.path.isfile(_YUNET_MODEL):
        return _detect_yunet(image, conf=conf)
    if not (os.path.isfile(_DNN_PROTO) and os.path.isfile(_DNN_MODEL)):
        raise FileNotFoundError("DNN face model not present; see download_models.py")
    read_net = getattr(cv2.dnn, "readNet", None) or getattr(
        cv2.dnn, "readNetFromCaffe", None)
    if read_net is None:
        raise RuntimeError("this OpenCV build cannot load Caffe models")
    net = read_net(_DNN_MODEL, _DNN_PROTO)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    net.setInput(blob)
    det = net.forward()
    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(det.shape[2]):
        if float(det[0, 0, i, 2]) < conf:
            continue
        x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        boxes.append((x1, y1, max(0, x2 - x1), max(0, y2 - y1)))
    return boxes


def _detect_haar(image: np.ndarray, min_size: int = 60) -> List[Tuple[int, int, int, int]]:
    """Haar cascade fallback. Always available with stock OpenCV."""
    _require_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes: List[Tuple[int, int, int, int]] = []
    for name in ("haarcascade_frontalface_default.xml",
                 "haarcascade_frontalface_alt2.xml"):
        path = os.path.join(cv2.data.haarcascades, name)
        if not os.path.isfile(path):
            continue
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            continue
        for (x, y, w, h) in clf.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4,
                minSize=(min_size, min_size)):
            boxes.append((int(x), int(y), int(w), int(h)))
        if boxes:
            break
    return boxes


def detect_faces(image: np.ndarray, min_size: int = 60,
                 method: str = "auto") -> List[Tuple[int, int, int, int]]:
    """Detect faces -> [(x, y, w, h)].

    method="auto" (default): DNN when model files exist, else Haar.
    method="dnn" / "haar": force one; raises if unavailable.
    """
    _require_cv2()
    if method not in ("auto", "dnn", "haar"):
        raise ValueError(f"unknown method {method!r}")
    dnn_ready = _dnn_ready()
    if method == "dnn" or (method == "auto" and dnn_ready):
        try:
            return _detect_dnn(image)
        except (FileNotFoundError, RuntimeError):
            if method == "dnn":
                raise
    return _detect_haar(image, min_size=min_size)


def detection_method() -> str:
    """Which detector detect_faces(method='auto') would use right now."""
    dnn_ready = _dnn_ready()
    return "dnn" if dnn_ready else "haar"


# ------------------------------------------------------- landmarks & align ----

def _dlib_predictor():
    """Return dlib 68-landmark predictor, or None (dlib/model missing)."""
    if not os.path.isfile(_DLIB_PREDICTOR):
        return None
    try:
        import dlib  # type: ignore
    except ImportError:
        return None
    try:
        return dlib.shape_predictor(_DLIB_PREDICTOR)
    except Exception:
        return None


def estimate_landmarks(face_img: np.ndarray) -> np.ndarray:
    """Estimate 3 key landmarks: left eye, right eye, nose tip -> (3, 2).

    Uses the dlib 68-point predictor when available; otherwise a
    bounding-box proportion heuristic (robust, deterministic).
    face_img should be a tight face crop (BGR).
    """
    _require_cv2()
    h, w = face_img.shape[:2]
    predictor = _dlib_predictor()
    if predictor is not None:
        try:
            import dlib  # type: ignore
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            rect = dlib.rectangle(0, 0, w - 1, h - 1)
            shape = predictor(gray, rect)
            pts = np.array([(shape.part(i).x, shape.part(i).y)
                            for i in range(68)], dtype=np.float32)
            left_eye = pts[36:42].mean(axis=0)
            right_eye = pts[42:48].mean(axis=0)
            nose = pts[30]
            return np.stack([left_eye, right_eye, nose]).astype(np.float32)
        except Exception:
            pass  # fall through to heuristic
    # Heuristic: standard frontal-face proportions within the crop.
    return np.array([
        [0.35 * w, 0.40 * h],   # left eye
        [0.65 * w, 0.40 * h],   # right eye
        [0.50 * w, 0.58 * h],   # nose tip
    ], dtype=np.float32)


def similarity_transform(src_pts: np.ndarray,
                         dst_pts: np.ndarray) -> np.ndarray:
    """2x3 similarity (rotation+scale+translation) matrix src -> dst."""
    _require_cv2()
    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if m is None:  # degenerate: fall back to pure translation
        t = dst.mean(axis=0) - src.mean(axis=0)
        m = np.array([[1.0, 0.0, t[0]], [0.0, 1.0, t[1]]], dtype=np.float64)
    return m


def align_face(source_face: np.ndarray, target_landmarks: np.ndarray,
               size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Warp source_face so its landmarks land on target_landmarks.

    Returns the warped face at `size` (w, h). Deterministic.
    """
    _require_cv2()
    src_lm = estimate_landmarks(source_face)
    dst_lm = np.asarray(target_landmarks, dtype=np.float32)
    m = similarity_transform(src_lm, dst_lm)
    return cv2.warpAffine(source_face, m, size,
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def landmarks_from_box(box: Tuple[int, int, int, int]) -> np.ndarray:
    """Target landmarks from a face box, same proportions as the heuristic."""
    x, y, w, h = box
    return np.array([
        [x + 0.35 * w, y + 0.40 * h],
        [x + 0.65 * w, y + 0.40 * h],
        [x + 0.50 * w, y + 0.58 * h],
    ], dtype=np.float32)


# ------------------------------------------------------------- compositing ----

def _oval_from_landmarks(landmarks: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Face oval (center, axes) derived from eye/nose geometry.

    Works with the 3-point set (left eye, right eye, nose): the
    inter-eye distance scales a standard frontal-face oval covering
    forehead to chin.
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    leye, reye = pts[0], pts[1]
    eye_dist = float(np.linalg.norm(reye - leye)) + 1e-6
    mid = (leye + reye) / 2.0
    center = (int(mid[0]), int(mid[1] + 0.45 * eye_dist))
    axes = (int(1.15 * eye_dist), int(1.55 * eye_dist))
    return center, axes


def face_mask(shape: Tuple[int, int], landmarks: np.ndarray,
              feather: int = 15, expand: float = 1.15) -> np.ndarray:
    """Feathered uint8 face mask from landmarks.

    With dense landmarks (>= 6 points) uses their convex hull; with the
    sparse 3-point set (eyes + nose) fits a face oval from the eye
    geometry. The region is expanded slightly, filled, then
    Gaussian-feathered. `shape` is (h, w).
    """
    _require_cv2()
    h, w = shape
    pts = np.asarray(landmarks, dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) >= 6:
        hull = cv2.convexHull(pts.astype(np.int32))
        centroid = pts.mean(axis=0)
        expanded = (centroid + (hull.reshape(-1, 2).astype(np.float32)
                                - centroid) * expand).astype(np.int32)
        cv2.fillConvexPoly(mask, expanded, 255)
    else:
        (cx, cy), (ax, ay) = _oval_from_landmarks(pts)
        cv2.ellipse(mask, (cx, cy),
                    (int(ax * expand), int(ay * expand)),
                    0, 0, 360, 255, -1)
    if feather > 0:
        k = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def ellipse_mask(shape: Tuple[int, int], center: Tuple[int, int],
                 axes: Tuple[int, int], angle: float = 0.0,
                 feather: int = 11) -> np.ndarray:
    """Feathered elliptical uint8 mask. `shape` is (h, w)."""
    _require_cv2()
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    if feather > 0:
        k = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def rect_mask(shape: Tuple[int, int], box: Tuple[int, int, int, int],
              feather: int = 11) -> np.ndarray:
    """Feathered rectangular uint8 mask from an (x, y, w, h) box."""
    _require_cv2()
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, bw, bh = (int(v) for v in box)
    cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
    if feather > 0:
        k = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def match_color(source: np.ndarray, target: np.ndarray,
                mask: np.ndarray) -> np.ndarray:
    """Shift source's LAB mean/std toward target's inside the mask.

    Classic Reinhard color transfer, masked. Deterministic.
    """
    _require_cv2()
    m = (mask > 127)
    if not m.any():
        return source.copy()
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = src_lab.copy()
    for c in range(3):
        s = src_lab[:, :, c][m]
        t = tgt_lab[:, :, c][m]
        s_mean, s_std = s.mean(), s.std() + 1e-6
        t_mean, t_std = t.mean(), t.std()
        out[:, :, c][m] = np.clip((s - s_mean) * (t_std / s_std) + t_mean,
                                  0, 255)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2BGR)


def composite_face(background: np.ndarray, face_img: np.ndarray,
                   landmarks: np.ndarray, feather: int = 15) -> np.ndarray:
    """Blend a source face onto background at target landmarks.

    face_img: source face crop (BGR). landmarks: (N,2) target points in
    background coordinates. Steps: align -> color-match -> Poisson blend
    (cv2.seamlessClone) with a feathered convex-hull mask.
    """
    _require_cv2()
    bg = np.ascontiguousarray(background)
    h, w = bg.shape[:2]
    lm = np.asarray(landmarks, dtype=np.float32)
    warped = cv2.warpAffine(
        face_img, similarity_transform(estimate_landmarks(face_img), lm),
        (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = face_mask((h, w), lm, feather=feather)
    matched = match_color(warped, bg, mask)
    center = tuple(np.round(lm.mean(axis=0)).astype(int))
    # seamlessClone needs a non-empty mask and an interior center.
    if mask.max() == 0:
        return bg.copy()
    cx = int(np.clip(center[0], 1, w - 2))
    cy = int(np.clip(center[1], 1, h - 2))
    return cv2.seamlessClone(matched, bg, mask, (cx, cy), cv2.NORMAL_CLONE)


def replace_region(background: np.ndarray, replacement: np.ndarray,
                   mask: np.ndarray, feather: int = 11,
                   color_match: bool = True) -> np.ndarray:
    """Masked Poisson blend for body parts / objects.

    replacement is resized to background size; mask (uint8, 0-255) selects
    the region. Feathered internally. No-op when the mask is empty.
    """
    _require_cv2()
    bg = np.ascontiguousarray(background)
    h, w = bg.shape[:2]
    if mask.shape[:2] != (h, w):
        raise ValueError("mask must match background size")
    m = mask.astype(np.uint8)
    if feather > 0:
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    if m.max() == 0:
        return bg.copy()
    rep = cv2.resize(np.ascontiguousarray(replacement), (w, h),
                     interpolation=cv2.INTER_LINEAR)
    if color_match:
        rep = match_color(rep, bg, m)
    ys, xs = np.nonzero(m > 0)
    center = (int(xs.mean()), int(ys.mean()))
    cx = int(np.clip(center[0], 1, w - 2))
    cy = int(np.clip(center[1], 1, h - 2))
    return cv2.seamlessClone(rep, bg, m, (cx, cy), cv2.NORMAL_CLONE)


# ---------------------------------------------------------------- tracking ----

class _TemplateTracker:
    """Deterministic matchTemplate fallback when no contrib tracker exists."""

    def __init__(self) -> None:
        self.template: Optional[np.ndarray] = None
        self.box: Optional[Tuple[int, int, int, int]] = None

    def init(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = (int(v) for v in box)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[y:y + h, x:x + w].copy()
        self.box = (x, y, w, h)
        return self.template.size > 0

    def update(self, frame: np.ndarray):
        if self.template is None or self.box is None:
            return False, (0, 0, 0, 0)
        x, y, w, h = self.box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]
        # Search window: 2x box around last position, clamped.
        sx0, sy0 = max(0, x - w), max(0, y - h)
        sx1, sy1 = min(fw, x + 2 * w), min(fh, y + 2 * h)
        search = gray[sy0:sy1, sx0:sx1]
        th, tw = self.template.shape[:2]
        if search.shape[0] < th or search.shape[1] < tw:
            return False, self.box
        res = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, _, _, ml = cv2.minMaxLoc(res)
        nx, ny = sx0 + ml[0], sy0 + ml[1]
        self.box = (nx, ny, w, h)
        return True, self.box


def _make_tracker():
    """Best available OpenCV tracker, else the template fallback."""
    _require_cv2()
    for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
        ctor = getattr(cv2, name, None)
        if callable(ctor):
            try:
                return ctor()
            except Exception:
                continue
    return _TemplateTracker()


def _tracker_init(tr, frame, box) -> bool:
    ok = tr.init(frame, tuple(int(v) for v in box))
    return bool(ok)


def _tracker_update(tr, frame):
    ok, box = tr.update(frame)
    x, y, w, h = box
    return bool(ok), (int(x), int(y), int(w), int(h))


def track_face(video_path: str, face_box: Tuple[int, int, int, int],
               max_frames: Optional[int] = None) -> List[Tuple[int, int, int, int]]:
    """Track a face box through a video -> one (x, y, w, h) per frame.

    Uses the best available OpenCV tracker (CSRT > KCF > MIL), falling back
    to deterministic template matching. On tracker failure the last good
    box is repeated, so output length always equals frame count.
    """
    _require_cv2()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path!r}")
    boxes: List[Tuple[int, int, int, int]] = []
    tracker = None
    last = tuple(int(v) for v in face_box)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and n >= max_frames:
            break
        if tracker is None:
            tracker = _make_tracker()
            _tracker_init(tracker, frame, last)
            boxes.append(last)
        else:
            good, b = _tracker_update(tracker, frame)
            if good:
                last = b
            boxes.append(last)
        n += 1
    cap.release()
    return boxes


def replace_in_video(video_path: str, source_face_img: np.ndarray,
                     output_path: str, fps: Optional[float] = None,
                     face_box: Optional[Tuple[int, int, int, int]] = None,
                     feather: int = 15,
                     max_frames: Optional[int] = None,
                     redetect_every: int = 10) -> str:
    """Replace a face through a whole video ("replace over a real movie").

    Detects the largest face in the first frame (or uses face_box),
    tracks it, and composites the source face per frame with color
    matching + Poisson blending. Every `redetect_every` frames the
    face is re-detected and the tracker re-anchored when a detection
    lands near the tracked box — this stops the slow tracker drift
    that otherwise walks the composite off the face on longer clips.
    Audio is not carried (use ffmpeg to mux the original audio back
    if needed).
    """
    _require_cv2()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path!r}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = fps or src_fps
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and len(frames) >= max_frames:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {video_path!r}")

    if face_box is None:
        found = detect_faces(frames[0])
        if not found:
            h, w = frames[0].shape[:2]
            face_box = (w // 2 - 60, h // 2 - 60, 120, 120)
        else:
            face_box = max(found, key=lambda b: b[2] * b[3])

    tracker = _make_tracker()
    _tracker_init(tracker, frames[0], face_box)
    last = tuple(int(v) for v in face_box)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"cannot write video {output_path!r}")
    try:
        for i, frame in enumerate(frames):
            if i > 0:
                good, b = _tracker_update(tracker, frame)
                if good:
                    last = b
            # Periodic re-detection: snap the tracker back onto the
            # detected face when one is found near the tracked box.
            if redetect_every and i > 0 and i % redetect_every == 0:
                found = detect_faces(frame)
                if found:
                    cand = max(found, key=lambda bx: bx[2] * bx[3])
                    cx = cand[0] + cand[2] / 2.0
                    cy = cand[1] + cand[3] / 2.0
                    lx, ly, lw, lh = (float(v) for v in last)
                    if (lx - lw <= cx <= lx + 2 * lw
                            and ly - lh <= cy <= ly + 2 * lh):
                        last = tuple(int(v) for v in cand)
                        tracker = _make_tracker()
                        _tracker_init(tracker, frame, last)
            lm = landmarks_from_box(last)
            out.write(composite_face(frame, source_face_img, lm,
                                     feather=feather))
    finally:
        out.release()
    return output_path


# ------------------------------------------------------------------- misc ----

def list_capabilities() -> dict:
    """Report which backends are available in this OpenCV build."""
    return {
        "cv2": cv2 is not None,
        "cv2_version": getattr(cv2, "__version__", None),
        "detection": detection_method(),
        "dnn_models_present": os.path.isfile(_DNN_PROTO)
        and os.path.isfile(_DNN_MODEL),
        "dlib_landmarks": _dlib_predictor() is not None,
        "tracker": type(_make_tracker()).__name__ if cv2 else None,
        "seamless_clone": hasattr(cv2, "seamlessClone") if cv2 else False,
    }

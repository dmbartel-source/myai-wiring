"""Photoshop-style photo manipulation toolkit — local, deterministic, no API.

Pure PIL/Pillow + OpenCV + NumPy. Every op is a pure function:
PIL Image in -> new PIL Image out. No network, no unseeded randomness.

Sections:
  - Adjustments (PIL): brightness/contrast/saturation/sharpness, levels,
    curves, white balance, HSL, vibrance, clarity.
  - Filters (OpenCV): blurs, sharpening, artistic, film looks.
  - Blend modes + layered compositing (NumPy, exact math).
  - Utilities: thumbnails, contact sheets, text, geometry, EXIF-safe save.

Registry: PHOTO_OPS maps name -> {description, params, func}.
apply_op(name, image, params) runs one. suggest_ops(description) maps
plain English to candidate ops for the agent.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ArrayLike = Union[Image.Image, np.ndarray]


# ---------------------------------------------------------------- helpers ----

def _to_rgb(img: Image.Image) -> Image.Image:
    """Normalize to RGB (drop alpha, convert palette/gray)."""
    if img.mode == "RGB":
        return img
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def _arr(img: Image.Image) -> np.ndarray:
    """PIL RGB -> float32 numpy in [0, 1], shape (h, w, 3)."""
    return np.asarray(_to_rgb(img), dtype=np.float32) / 255.0


def _pil(a: np.ndarray) -> Image.Image:
    """float [0,1] or uint8 numpy -> PIL RGB."""
    if a.dtype != np.uint8:
        a = np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return Image.fromarray(a, "L").convert("RGB")
    return Image.fromarray(a[:, :, :3])


def _match_size(base: Image.Image, other: Image.Image) -> Image.Image:
    if other.size != base.size:
        return other.resize(base.size, Image.LANCZOS)
    return other


# -------------------------------------------------------------- adjustments ----

def brightness(img: Image.Image, factor: float = 1.0) -> Image.Image:
    """factor: 0.0 (black) .. 1.0 (identity) .. 2.0+ (brighter)."""
    return ImageEnhance.Brightness(_to_rgb(img)).enhance(max(0.0, factor))


def contrast(img: Image.Image, factor: float = 1.0) -> Image.Image:
    """factor: 0.0 (gray) .. 1.0 (identity) .. 2.0+ (more contrast)."""
    return ImageEnhance.Contrast(_to_rgb(img)).enhance(max(0.0, factor))


def saturation(img: Image.Image, factor: float = 1.0) -> Image.Image:
    """factor: 0.0 (grayscale) .. 1.0 (identity) .. 2.0+ (vivid)."""
    return ImageEnhance.Color(_to_rgb(img)).enhance(max(0.0, factor))


def sharpness(img: Image.Image, factor: float = 1.0) -> Image.Image:
    """factor: 0.0 (blurry) .. 1.0 (identity) .. 2.0+ (sharper)."""
    return ImageEnhance.Sharpness(_to_rgb(img)).enhance(max(0.0, factor))


def levels(img: Image.Image, black: int = 0, white: int = 255,
           gamma: float = 1.0) -> Image.Image:
    """Remap [black, white] -> [0, 255] with gamma. Per-image (all channels).

    black/white: 0-255 input range endpoints. gamma: >1 darkens midtones.
    """
    black = max(0, min(254, int(black)))
    white = max(black + 1, min(255, int(white)))
    gamma = max(0.1, min(5.0, float(gamma)))
    lut = []
    for i in range(256):
        x = (i - black) / (white - black)
        x = max(0.0, min(1.0, x)) ** (1.0 / gamma)
        lut.append(int(round(x * 255)))
    return _to_rgb(img).point(lut * 3)


def curves(img: Image.Image,
           points: list[tuple[float, float]] = ((0, 0), (1, 1))) -> Image.Image:
    """Apply a tone curve. points: list of (x, y) in [0,1], sorted by x.

    Linear interpolation between control points -> 256-entry LUT.
    """
    pts = sorted((max(0.0, min(1.0, float(x))),
                  max(0.0, min(1.0, float(y)))) for x, y in points)
    if len(pts) < 2:
        raise ValueError("curves needs at least 2 control points")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lut = []
    for i in range(256):
        x = i / 255.0
        if x <= xs[0]:
            y = ys[0]
        elif x >= xs[-1]:
            y = ys[-1]
        else:
            j = next(k for k in range(len(xs) - 1) if xs[k] <= x <= xs[k + 1])
            t = (x - xs[j]) / (xs[j + 1] - xs[j] + 1e-9)
            y = ys[j] + t * (ys[j + 1] - ys[j])
        lut.append(int(round(max(0.0, min(1.0, y)) * 255)))
    return _to_rgb(img).point(lut * 3)


def white_balance(img: Image.Image, temperature: float = 0.0,
                  tint: float = 0.0) -> Image.Image:
    """temperature: -1 (cool/blue) .. +1 (warm/yellow). tint: -1 (green) ..
    +1 (magenta)."""
    a = _arr(img)
    t = max(-1.0, min(1.0, float(temperature)))
    g = max(-1.0, min(1.0, float(tint)))
    # Temperature: shift red/blue in opposition.
    a[:, :, 0] = np.clip(a[:, :, 0] * (1.0 + 0.25 * t), 0, 1)
    a[:, :, 2] = np.clip(a[:, :, 2] * (1.0 - 0.25 * t), 0, 1)
    # Tint: shift green vs red+blue.
    a[:, :, 1] = np.clip(a[:, :, 1] * (1.0 - 0.15 * g), 0, 1)
    if g > 0:
        a[:, :, 0] = np.clip(a[:, :, 0] * (1.0 + 0.08 * g), 0, 1)
        a[:, :, 2] = np.clip(a[:, :, 2] * (1.0 + 0.08 * g), 0, 1)
    return _pil(a)


def _rgb_to_hsl(a: np.ndarray) -> np.ndarray:
    """float [0,1] RGB -> HSL (h in [0,1), s/l in [0,1]). Vectorized."""
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    l = (mx + mn) / 2.0
    d = mx - mn
    s = np.where(d == 0, 0.0,
                 np.where(l > 0.5, d / (2.0 - mx - mn + 1e-9),
                          d / (mx + mn + 1e-9)))
    h = np.zeros_like(l)
    mask = d != 0
    # Red is max.
    m = mask & (mx == r)
    h = np.where(m, ((g - b) / (d + 1e-9)) % 6.0, h)
    # Green is max.
    m = mask & (mx == g)
    h = np.where(m, (b - r) / (d + 1e-9) + 2.0, h)
    # Blue is max.
    m = mask & (mx == b)
    h = np.where(m, (r - g) / (d + 1e-9) + 4.0, h)
    h = (h / 6.0) % 1.0
    return np.stack([h, np.clip(s, 0, 1), np.clip(l, 0, 1)], axis=-1)


def _hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """float HSL -> [0,1] RGB. Vectorized."""
    h, s, l = hsl[:, :, 0], hsl[:, :, 1], hsl[:, :, 2]

    def hue2rgb(p: np.ndarray, q: np.ndarray, t: np.ndarray) -> np.ndarray:
        t = t % 1.0
        return np.where(t < 1 / 6, p + (q - p) * 6 * t,
               np.where(t < 1 / 2, q,
               np.where(t < 2 / 3, p + (q - p) * (2 / 3 - t) * 6, p)))

    q = np.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q
    r = hue2rgb(p, q, h + 1 / 3)
    g = hue2rgb(p, q, h)
    b = hue2rgb(p, q, h - 1 / 3)
    # Achromatic.
    ach = (s == 0)
    r = np.where(ach, l, r)
    g = np.where(ach, l, g)
    b = np.where(ach, l, b)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 1)


def hsl_adjust(img: Image.Image, hue_shift: float = 0.0,
               sat_shift: float = 0.0, light_shift: float = 0.0) -> Image.Image:
    """Shifts in HSL space. hue_shift: -1..1 (fraction of wheel).
    sat_shift/light_shift: -1..1 additive."""
    hs = float(hue_shift) % 1.0
    ss = float(sat_shift)
    ls = float(light_shift)
    if hs == 0.0 and ss == 0.0 and ls == 0.0:
        return _to_rgb(img).copy()
    hsl = _rgb_to_hsl(_arr(img))
    hsl[:, :, 0] = (hsl[:, :, 0] + hs) % 1.0
    hsl[:, :, 1] = np.clip(hsl[:, :, 1] + ss, 0, 1)
    hsl[:, :, 2] = np.clip(hsl[:, :, 2] + ls, 0, 1)
    return _pil(_hsl_to_rgb(hsl))


def vibrance(img: Image.Image, amount: float = 0.5) -> Image.Image:
    """Boost saturation more in muted areas, less where already vivid.

    amount: 0 (identity) .. 1 (strong)."""
    a = _arr(img)
    amount = max(0.0, min(1.0, float(amount)))
    if amount == 0:
        return _to_rgb(img).copy()
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)          # 0=gray .. 1=vivid
    boost = 1.0 + amount * 0.9 * (1.0 - sat)
    gray = a.mean(axis=2, keepdims=True)
    out = gray + (a - gray) * boost[:, :, None]
    return _pil(np.clip(out, 0, 1))


def clarity(img: Image.Image, amount: float = 0.5) -> Image.Image:
    """Local contrast (mid-tone punch). amount: -1..1 (negative softens)."""
    a = _arr(img)
    amount = max(-1.0, min(1.0, float(amount)))
    if amount == 0:
        return _to_rgb(img).copy()
    blur = cv2.GaussianBlur(a, (0, 0), sigmaX=3.0)
    detail = a - blur
    out = a + detail * (0.9 * amount)
    return _pil(np.clip(out, 0, 1))


# ----------------------------------------------------------------- filters ----

def _cv(img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR uint8."""
    return cv2.cvtColor(np.asarray(_to_rgb(img)), cv2.COLOR_RGB2BGR)


def _from_cv(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def gaussian_blur(img: Image.Image, radius: float = 3.0) -> Image.Image:
    """radius: sigma in px, 0 = identity."""
    radius = max(0.0, float(radius))
    if radius == 0:
        return _to_rgb(img).copy()
    k = max(1, int(radius * 2) * 2 + 1)
    return _from_cv(cv2.GaussianBlur(_cv(img), (k, k), radius))


def motion_blur(img: Image.Image, length: float = 15.0,
                angle: float = 0.0) -> Image.Image:
    """Directional blur. length: px. angle: degrees (0 = horizontal)."""
    length = max(1.0, float(length))
    k = max(3, int(length) | 1)  # odd
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    rad = math.radians(float(angle))
    m = cv2.getRotationMatrix2D((k / 2, k / 2), -float(angle), 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    kernel /= kernel.sum() + 1e-9
    return _from_cv(cv2.filter2D(_cv(img), -1, kernel))


def lens_blur(img: Image.Image, radius: float = 5.0) -> Image.Image:
    """Bokeh-ish blur: disk kernel instead of gaussian."""
    radius = max(0.5, float(radius))
    k = max(3, int(radius * 2) | 1)
    y, x = np.ogrid[-k // 2:k // 2 + 1, -k // 2:k // 2 + 1]
    kernel = (x * x + y * y <= (k // 2) ** 2).astype(np.float32)
    kernel /= kernel.sum()
    return _from_cv(cv2.filter2D(_cv(img), -1, kernel))


def sharpen(img: Image.Image, amount: float = 1.0) -> Image.Image:
    """amount: 0 = identity, 1 = standard sharpen."""
    amount = max(0.0, float(amount))
    if amount == 0:
        return _to_rgb(img).copy()
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
                      dtype=np.float32)
    base = _cv(img).astype(np.float32)
    sharp = cv2.filter2D(base, -1, kernel)
    out = base + (sharp - base) * amount
    return _from_cv(np.clip(out, 0, 255).astype(np.uint8))


def unsharp_mask(img: Image.Image, radius: float = 2.0, percent: float = 100.0,
                 threshold: int = 0) -> Image.Image:
    """Classic unsharp mask. percent: 0-500. threshold: 0-255."""
    return _pil(_unsharp_arr(_arr(img), radius, percent, threshold))


def _unsharp_arr(a: np.ndarray, radius: float, percent: float,
                 threshold: int) -> np.ndarray:
    radius = max(0.1, float(radius))
    blur = cv2.GaussianBlur(a, (0, 0), radius)
    diff = a - blur
    if threshold > 0:
        t = threshold / 255.0
        mask = (np.abs(diff).max(axis=2, keepdims=True) >= t).astype(np.float32)
        diff = diff * mask
    return np.clip(a + diff * (percent / 100.0), 0, 1)


def edge_enhance(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """Boost edges via Laplacian detail layer."""
    a = _arr(img)
    lap = cv2.Laplacian(a, cv2.CV_32F, ksize=3)
    out = a + lap * max(0.0, float(strength)) * 0.5
    return _pil(np.clip(out, 0, 1))


def oil_paint(img: Image.Image, size: int = 5,
              levels: int = 20) -> Image.Image:
    """Oil-paint effect: per-pixel mode filter over a neighborhood."""
    rgb = _to_rgb(img).resize(
        (max(1, img.width // 2), max(1, img.height // 2)), Image.LANCZOS)
    a = np.asarray(rgb, dtype=np.int32)
    size = max(3, int(size) | 1)
    levels = max(2, int(levels))
    pad = size // 2
    ap = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    h, w, _ = a.shape
    out = np.empty_like(a)
    # Vectorized histogram voting per intensity bin.
    gray = ap.mean(axis=2)
    for y in range(h):
        for x in range(w):
            win = gray[y:y + size, x:x + size]
            bins = np.clip((win / 256 * levels).astype(int), 0, levels - 1)
            counts = np.bincount(bins.ravel(), minlength=levels)
            best = int(np.argmax(counts))
            sel = (bins == best)
            patch = ap[y:y + size, x:x + size][sel]
            out[y, x] = patch.mean(axis=0) if len(patch) else a[y, x]
    small = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    return small.resize(img.size, Image.LANCZOS)


def watercolor(img: Image.Image, sigma: float = 3.0) -> Image.Image:
    """Watercolor: edge-preserving smooth + boosted saturation."""
    bgr = _cv(img)
    smooth = cv2.edgePreservingFilter(bgr, flags=2, sigma_s=60,
                                      sigma_r=max(0.1, sigma / 10))
    rgb = _from_cv(smooth)
    return saturation(rgb, 1.35)


def pencil_sketch(img: Image.Image) -> Image.Image:
    """Grayscale pencil sketch."""
    gray = np.asarray(_to_rgb(img).convert("L"), dtype=np.float32)
    inv = 255.0 - gray
    blur = cv2.GaussianBlur(inv, (0, 0), sigmaX=8.0)
    sketch = 255.0 - (255.0 - gray) * 255.0 / (255.0 - blur + 1e-6)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)
    return Image.fromarray(sketch, "L").convert("RGB")


def posterize(img: Image.Image, bits: int = 3) -> Image.Image:
    """Reduce to 2**bits levels per channel."""
    bits = max(1, min(7, int(bits)))
    return ImageOps.posterize(_to_rgb(img), bits)


def cross_process(img: Image.Image, strength: float = 0.7) -> Image.Image:
    """Cross-processed film: green shadows, yellow highlights."""
    a = _arr(img)
    s = max(0.0, min(1.0, float(strength)))
    # S-curves per channel, shifted.
    def curve(ch: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip(ch + s * (lo * (1 - ch) * ch * 4 * 0.25 +
                                 hi * (ch - 0.5)), 0, 1)
    a[:, :, 0] = curve(a[:, :, 0], -0.10, 0.12)   # red: warm highlights
    a[:, :, 1] = curve(a[:, :, 1], 0.10, 0.02)    # green: lifted shadows
    a[:, :, 2] = curve(a[:, :, 2], -0.14, -0.10)  # blue: crushed
    return _pil(a)


def bleach_bypass(img: Image.Image, strength: float = 0.7) -> Image.Image:
    """Bleach bypass: desaturated, high contrast, silvery."""
    s = max(0.0, min(1.0, float(strength)))
    desat = saturation(img, 1.0 - 0.55 * s)
    con = contrast(desat, 1.0 + 0.45 * s)
    a = _arr(con)
    # Silver tint: lift blacks slightly toward blue-gray.
    lift = s * 0.06
    a = a * (1 - lift) + lift * np.array([0.9, 0.95, 1.0], dtype=np.float32)
    return _pil(np.clip(a, 0, 1))


def faded_film(img: Image.Image, amount: float = 0.6) -> Image.Image:
    """Faded print: lifted blacks, lowered contrast, warm cast."""
    a = _arr(img)
    f = max(0.0, min(1.0, float(amount)))
    black_lift = 0.10 * f
    a = a * (1 - black_lift) + black_lift
    a = (a - 0.5) * (1 - 0.25 * f) + 0.5
    a[:, :, 0] = np.clip(a[:, :, 0] * (1 + 0.06 * f), 0, 1)
    a[:, :, 2] = np.clip(a[:, :, 2] * (1 - 0.08 * f), 0, 1)
    return _pil(np.clip(a, 0, 1))


# ------------------------------------------------------- blend modes ----

def _to_f(a: ArrayLike) -> np.ndarray:
    """Image or array -> float32 [0,1] (h, w, 3)."""
    if isinstance(a, Image.Image):
        return _arr(a)
    a = np.asarray(a, dtype=np.float32)
    if a.max() > 1.0:
        a = a / 255.0
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return np.clip(a[:, :, :3], 0, 1)


def _blend_normal(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return blend


def _blend_multiply(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return base * blend


def _blend_screen(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - base) * (1.0 - blend)


def _blend_overlay(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return np.where(base < 0.5, 2 * base * blend,
                    1.0 - 2 * (1.0 - base) * (1.0 - blend))


def _blend_soft_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    # Pegtop formulation.
    return np.where(blend < 0.5,
                    base - (1.0 - 2.0 * blend) * base * (1.0 - base),
                    base + (2.0 * blend - 1.0) *
                    (np.sqrt(np.clip(base, 0, 1)) - base))


def _blend_hard_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return _blend_overlay(blend, base)


def _blend_darken(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return np.minimum(base, blend)


def _blend_lighten(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return np.maximum(base, blend)


def _blend_difference(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return np.abs(base - blend)


def _blend_color_dodge(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return np.where(blend >= 1.0, 1.0,
                    np.minimum(1.0, base / (1.0 - blend + 1e-6)))


def _blend_color_burn(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return np.where(blend <= 0.0, 0.0,
                    np.maximum(0.0, 1.0 - (1.0 - base) / (blend + 1e-6)))


BLEND_MODES: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "normal": _blend_normal,
    "multiply": _blend_multiply,
    "screen": _blend_screen,
    "overlay": _blend_overlay,
    "soft_light": _blend_soft_light,
    "hard_light": _blend_hard_light,
    "darken": _blend_darken,
    "lighten": _blend_lighten,
    "difference": _blend_difference,
    "color_dodge": _blend_color_dodge,
    "color_burn": _blend_color_burn,
}


def blend(base: ArrayLike, blend_img: ArrayLike, mode: str = "normal",
          opacity: float = 1.0) -> Image.Image:
    """Blend two images. blend_img is resized to base. opacity: 0..1."""
    if mode not in BLEND_MODES:
        raise ValueError(f"unknown blend mode {mode!r}; "
                         f"expected one of {sorted(BLEND_MODES)}")
    b = _to_f(base)
    l = _to_f(blend_img)
    if l.shape[:2] != b.shape[:2]:
        l = np.asarray(_match_size(_pil(b), _pil(l)), dtype=np.float32) / 255.0
    mixed = BLEND_MODES[mode](b, l)
    o = max(0.0, min(1.0, float(opacity)))
    return _pil(np.clip(b * (1 - o) + mixed * o, 0, 1))


def composite_layers(layers: list[tuple]) -> Image.Image:
    """Composite a layer stack.

    layers: list of (image, blend_mode, opacity, mask) where mask is an
    optional PIL 'L' image (white = full effect) or None. Mask edges are
    feathered with a small gaussian for smooth blends.
    Returns a new PIL RGB image.
    """
    if not layers:
        raise ValueError("composite_layers needs at least one layer")
    base_img = _to_rgb(layers[0][0])
    result = _arr(base_img)
    for entry in layers[1:]:
        img, mode, opacity = entry[0], entry[1], entry[2]
        mask = entry[3] if len(entry) > 3 else None
        if mode not in BLEND_MODES:
            raise ValueError(f"unknown blend mode {mode!r}")
        lyr = _match_size(base_img, _to_rgb(img))
        l = _arr(lyr)
        mixed = BLEND_MODES[mode](result, l)
        o = max(0.0, min(1.0, float(opacity)))
        if mask is not None:
            m = mask.convert("L").resize(base_img.size, Image.LANCZOS)
            m = np.asarray(m, dtype=np.float32) / 255.0
            # Feather mask edges slightly for smooth transitions.
            m = cv2.GaussianBlur(m, (0, 0), sigmaX=1.0)
            w = (m * o)[:, :, None]
        else:
            w = o
        result = np.clip(result * (1 - w) + mixed * w, 0, 1)
    return _pil(result)


def feather_mask(mask: Image.Image, radius: float = 5.0) -> Image.Image:
    """Soften a mask's edges with a gaussian blur."""
    m = mask.convert("L")
    k = max(1, int(float(radius) * 2) * 2 + 1)
    blurred = cv2.GaussianBlur(np.asarray(m), (k, k), float(radius))
    return Image.fromarray(blurred, "L")


# --------------------------------------------------------------- utilities ----

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _find_font() -> Optional[str]:
    for fp in _FONT_PATHS:
        if os.path.exists(fp):
            return fp
    return None


def resize_image(img: Image.Image, width: int = 0, height: int = 0,
                 fit: str = "inside") -> Image.Image:
    """Resize. Give width and/or height; 0 keeps aspect. fit: inside/outside/exact."""
    img = _to_rgb(img)
    w, h = img.size
    if width <= 0 and height <= 0:
        return img.copy()
    if width <= 0:
        width = int(w * height / h)
    if height <= 0:
        height = int(h * width / w)
    if fit == "exact":
        return img.resize((width, height), Image.LANCZOS)
    scale = (min if fit == "inside" else max)(width / w, height / h)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                      Image.LANCZOS)


def crop_image(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Crop to (left, upper, right, lower), clamped to the image."""
    img = _to_rgb(img)
    l, u, r, b = box
    l = max(0, min(img.width, int(l)))
    u = max(0, min(img.height, int(u)))
    r = max(l + 1, min(img.width, int(r)))
    b = max(u + 1, min(img.height, int(b)))
    return img.crop((l, u, r, b))


def rotate_image(img: Image.Image, angle: float,
                 expand: bool = True) -> Image.Image:
    """Rotate counter-clockwise by angle degrees."""
    return _to_rgb(img).rotate(float(angle), resample=Image.BICUBIC,
                               expand=expand)


def straighten(img: Image.Image, angle: float) -> Image.Image:
    """Correct a tilted horizon. angle: degrees the horizon is off."""
    # Rotate then crop away the empty corners.
    rot = rotate_image(img, angle, expand=True)
    w, h = img.size
    # Inscribed crop: keep the central region that was fully covered.
    rad = math.radians(abs(float(angle)) % 90)
    if rad < 0.01:
        return rot.resize((w, h), Image.LANCZOS)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    # Conservative crop factor.
    cw = int(w * cos_a - h * sin_a)
    ch = int(h * cos_a - w * sin_a)
    cw, ch = max(1, cw), max(1, ch)
    rw, rh = rot.size
    l = (rw - cw) // 2
    u = (rh - ch) // 2
    return rot.crop((l, u, l + cw, u + ch)).resize((w, h), Image.LANCZOS)


def perspective_correct(img: Image.Image,
                        src_points: list[tuple[float, float]],
                        dst_size: tuple[int, int]) -> Image.Image:
    """Warp a quadrilateral to a rectangle. src_points: 4 (x, y) corners
    in order TL, TR, BR, BL. dst_size: (width, height) of output."""
    if len(src_points) != 4:
        raise ValueError("perspective_correct needs exactly 4 source points")
    w, h = dst_size
    src = np.array(src_points, dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(_cv(img), m, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return _from_cv(warped)


def _saliency_map(img: Image.Image) -> np.ndarray:
    """Face-agnostic interest map: edge density + saturation + center bias."""
    small = _to_rgb(img).resize((160, 160), Image.LANCZOS)
    a = np.asarray(small, dtype=np.float32) / 255.0
    gray = a.mean(axis=2)
    edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    edges = edges / (edges.max() + 1e-6)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    dist = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    center = np.clip(1.0 - dist * 1.2, 0, 1)
    score = 0.45 * edges + 0.30 * sat + 0.25 * center
    return cv2.GaussianBlur(score, (0, 0), sigmaX=5.0)


def thumbnail(img: Image.Image, size: tuple[int, int] = (320, 320),
              smart_crop: bool = True) -> Image.Image:
    """Thumbnail of exactly `size`. smart_crop picks the most interesting
    window (saliency-based, face-agnostic); otherwise center-crops."""
    img = _to_rgb(img)
    tw, th = size
    target_ratio = tw / th
    w, h = img.size
    if abs(w / h - target_ratio) > 0.01:
        if w / h > target_ratio:
            cw, ch = int(h * target_ratio), h
        else:
            cw, ch = w, int(w / target_ratio)
        if smart_crop and min(cw, ch) > 16:
            sal = _saliency_map(img)
            sh, sw = sal.shape
            # Search the best window on the small saliency map.
            bw = max(1, int(cw / w * sw))
            bh = max(1, int(ch / h * sh))
            best, bx, by = -1.0, 0, 0
            step = max(1, min(bw, bh) // 8)
            for yy in range(0, sh - bh + 1, step):
                for xx in range(0, sw - bw + 1, step):
                    s = float(sal[yy:yy + bh, xx:xx + bw].mean())
                    if s > best:
                        best, bx, by = s, xx, yy
            l = int(bx / sw * w)
            u = int(by / sh * h)
        else:
            l = (w - cw) // 2
            u = (h - ch) // 2
        img = img.crop((l, u, min(w, l + cw), min(h, u + ch)))
    return img.resize((tw, th), Image.LANCZOS)


def contact_sheet(images: list[Image.Image], cols: int = 4,
                  thumb: tuple[int, int] = (320, 200),
                  labels: Optional[list[str]] = None,
                  bg: tuple[int, int, int] = (18, 18, 18)) -> Image.Image:
    """Storyboard/reference sheet: grid of thumbnails with labels."""
    if not images:
        raise ValueError("contact_sheet needs at least one image")
    cols = max(1, int(cols))
    tw, th = thumb
    label_h = 28
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + label_h)), bg)
    draw = ImageDraw.Draw(sheet)
    font = None
    fp = _find_font()
    if fp:
        try:
            font = ImageFont.truetype(fp, 16)
        except Exception:
            font = None
    for i, im in enumerate(images):
        t = thumbnail(im, (tw, th), smart_crop=False)
        x, y = (i % cols) * tw, (i // cols) * (th + label_h)
        sheet.paste(t, (x, y))
        if labels and i < len(labels):
            draw.text((x + 6, y + th + 5), str(labels[i])[:40],
                      fill=(230, 230, 230), font=font)
    return sheet


def add_text(img: Image.Image, text: str, position: tuple[int, int] = (24, 24),
             size: int = 48, color: tuple[int, int, int] = (255, 255, 255),
             stroke: int = 2, bg_box: bool = False) -> Image.Image:
    """Burn text onto the image. Falls back gracefully if no font found."""
    img = _to_rgb(img).copy()
    draw = ImageDraw.Draw(img, "RGBA")
    font = None
    fp = _find_font()
    if fp:
        try:
            font = ImageFont.truetype(fp, int(size))
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()
    x, y = int(position[0]), int(position[1])
    if bg_box:
        bbox = draw.textbbox((x, y), text, font=font, stroke_width=stroke)
        draw.rectangle([bbox[0] - 8, bbox[1] - 6, bbox[2] + 8, bbox[3] + 6],
                       fill=(0, 0, 0, 160))
    draw.text((x, y), text, font=font, fill=color + (255,),
              stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    return img.convert("RGB")


def load_image(path: str) -> Image.Image:
    """Load any PIL-readable image as RGB."""
    with Image.open(path) as im:
        im.load()
        return _to_rgb(im)


def save_image(img: Image.Image, path: str, quality: int = 92,
               preserve_exif: bool = True,
               exif_source: Optional[Image.Image] = None) -> str:
    """Save with EXIF preservation and format conversion by extension.

    preserve_exif copies EXIF from exif_source (or keeps it if img has it).
    Supports .jpg/.jpeg/.png/.webp/.bmp/.tiff.
    """
    ext = os.path.splitext(path)[1].lower()
    out = _to_rgb(img)
    kwargs: dict[str, Any] = {}
    exif_bytes = None
    src = exif_source if exif_source is not None else img
    if preserve_exif:
        try:
            exif_bytes = src.info.get("exif")
        except Exception:
            exif_bytes = None
    if ext in (".jpg", ".jpeg"):
        kwargs = {"quality": max(1, min(100, int(quality))),
                  "optimize": True}
        if exif_bytes:
            kwargs["exif"] = exif_bytes
        out.save(path, "JPEG", **kwargs)
    elif ext == ".png":
        if exif_bytes:
            kwargs["exif"] = exif_bytes
        out.save(path, "PNG", **kwargs)
    elif ext == ".webp":
        kwargs = {"quality": max(1, min(100, int(quality)))}
        if exif_bytes:
            kwargs["exif"] = exif_bytes
        out.save(path, "WEBP", **kwargs)
    else:
        out.save(path)
    return path


# -------------------------------------------------------------- registry ----

def _p(name: str, typ: str, default: Any, help_: str) -> tuple[str, dict]:
    return name, {"type": typ, "default": default, "help": help_}


def _params(*ps: tuple[str, dict]) -> dict:
    return {n: v for n, v in ps}


PHOTO_OPS: dict[str, dict[str, Any]] = {
    # Adjustments
    "brightness": {
        "description": "Brightness. 1.0 = identity.",
        "params": _params(_p("factor", "float", 1.0, "0=black, 1=same, 2=double")),
        "func": brightness},
    "contrast": {
        "description": "Contrast. 1.0 = identity.",
        "params": _params(_p("factor", "float", 1.0, "0=gray, 1=same, 2=stronger")),
        "func": contrast},
    "saturation": {
        "description": "Saturation. 1.0 = identity, 0 = grayscale.",
        "params": _params(_p("factor", "float", 1.0, "0=gray, 1=same, 2=vivid")),
        "func": saturation},
    "sharpness": {
        "description": "Sharpness. 1.0 = identity.",
        "params": _params(_p("factor", "float", 1.0, "0=blurry, 1=same, 2=sharper")),
        "func": sharpness},
    "levels": {
        "description": "Levels: remap black/white points with gamma.",
        "params": _params(
            _p("black", "int", 0, "input black point, 0-255"),
            _p("white", "int", 255, "input white point, 0-255"),
            _p("gamma", "float", 1.0, ">1 darkens midtones")),
        "func": levels},
    "curves": {
        "description": "Tone curve from control points.",
        "params": _params(_p("points", "list", [[0, 0], [1, 1]],
                                 "list of [x, y] in 0-1, sorted by x")),
        "func": lambda img, points: curves(
            img, [tuple(p) for p in points])},
    "white_balance": {
        "description": "White balance: temperature + tint.",
        "params": _params(
            _p("temperature", "float", 0.0, "-1=cool, +1=warm"),
            _p("tint", "float", 0.0, "-1=green, +1=magenta")),
        "func": white_balance},
    "hsl_adjust": {
        "description": "Shift hue / saturation / lightness.",
        "params": _params(
            _p("hue_shift", "float", 0.0, "-1..1 fraction of color wheel"),
            _p("sat_shift", "float", 0.0, "-1..1 additive"),
            _p("light_shift", "float", 0.0, "-1..1 additive")),
        "func": hsl_adjust},
    "vibrance": {
        "description": "Smart saturation: boosts muted colors most.",
        "params": _params(_p("amount", "float", 0.5, "0..1")),
        "func": vibrance},
    "clarity": {
        "description": "Local contrast / midtone punch. Negative softens.",
        "params": _params(_p("amount", "float", 0.5, "-1..1")),
        "func": clarity},
    # Filters
    "gaussian_blur": {
        "description": "Soft gaussian blur.",
        "params": _params(_p("radius", "float", 3.0, "sigma in px, 0=none")),
        "func": gaussian_blur},
    "motion_blur": {
        "description": "Directional motion blur.",
        "params": _params(
            _p("length", "float", 15.0, "blur length in px"),
            _p("angle", "float", 0.0, "degrees, 0=horizontal")),
        "func": motion_blur},
    "lens_blur": {
        "description": "Bokeh-style disk blur.",
        "params": _params(_p("radius", "float", 5.0, "disk radius in px")),
        "func": lens_blur},
    "sharpen": {
        "description": "Sharpen detail.",
        "params": _params(_p("amount", "float", 1.0, "0=none")),
        "func": sharpen},
    "unsharp_mask": {
        "description": "Classic unsharp mask.",
        "params": _params(
            _p("radius", "float", 2.0, "blur radius px"),
            _p("percent", "float", 100.0, "0-500"),
            _p("threshold", "int", 0, "0-255")),
        "func": unsharp_mask},
    "edge_enhance": {
        "description": "Boost edges.",
        "params": _params(_p("strength", "float", 1.0, "0=none")),
        "func": edge_enhance},
    "oil_paint": {
        "description": "Oil-painting effect.",
        "params": _params(
            _p("size", "int", 5, "brush neighborhood"),
            _p("levels", "int", 20, "intensity bins")),
        "func": oil_paint},
    "watercolor": {
        "description": "Watercolor: smooth + vivid.",
        "params": _params(_p("sigma", "float", 3.0, "smoothing")),
        "func": watercolor},
    "pencil_sketch": {
        "description": "Grayscale pencil sketch.",
        "params": _params(),
        "func": pencil_sketch},
    "posterize": {
        "description": "Reduce color levels per channel.",
        "params": _params(_p("bits", "int", 3, "1-7 bits kept")),
        "func": posterize},
    "cross_process": {
        "description": "Cross-processed film look.",
        "params": _params(_p("strength", "float", 0.7, "0..1")),
        "func": cross_process},
    "bleach_bypass": {
        "description": "Bleach bypass: silvery, contrasty.",
        "params": _params(_p("strength", "float", 0.7, "0..1")),
        "func": bleach_bypass},
    "faded_film": {
        "description": "Faded print: lifted blacks, warm.",
        "params": _params(_p("amount", "float", 0.6, "0..1")),
        "func": faded_film},
    # Geometry
    "resize": {
        "description": "Resize keeping aspect unless exact.",
        "params": _params(
            _p("width", "int", 0, "0=auto"),
            _p("height", "int", 0, "0=auto"),
            _p("fit", "string", "inside", "inside/outside/exact")),
        "func": resize_image},
    "crop": {
        "description": "Crop to a box.",
        "params": _params(_p("box", "list", [0, 0, 100, 100],
                                 "[left, upper, right, lower]")),
        "func": lambda img, box: crop_image(img, tuple(box))},
    "rotate": {
        "description": "Rotate counter-clockwise.",
        "params": _params(
            _p("angle", "float", 90.0, "degrees"),
            _p("expand", "bool", True, "grow canvas to fit")),
        "func": rotate_image},
    "straighten": {
        "description": "Fix a tilted horizon, then re-crop.",
        "params": _params(_p("angle", "float", 2.0,
                             "degrees the horizon is off")),
        "func": straighten},
    "perspective_correct": {
        "description": "Warp a quadrilateral to a rectangle.",
        "params": _params(
            _p("src_points", "list", [[0, 0], [100, 0], [100, 100], [0, 100]],
               "4 corners TL,TR,BR,BL"),
            _p("dst_size", "list", [200, 200], "[width, height]")),
        "func": lambda img, src_points, dst_size: perspective_correct(
            img, [tuple(p) for p in src_points], tuple(dst_size))},
    # Compositing helpers (single-image ops; full stacks use composite_layers)
    "thumbnail": {
        "description": "Smart-cropped thumbnail of exact size.",
        "params": _params(
            _p("size", "list", [320, 320], "[width, height]"),
            _p("smart_crop", "bool", True, "saliency crop vs center")),
        "func": lambda img, size, smart_crop: thumbnail(
            img, tuple(size), smart_crop)},
    "add_text": {
        "description": "Burn text onto the image.",
        "params": _params(
            _p("text", "string", "TITLE", "text to draw"),
            _p("position", "list", [24, 24], "[x, y]"),
            _p("size", "int", 48, "font px"),
            _p("color", "list", [255, 255, 255], "[r, g, b]"),
            _p("stroke", "int", 2, "outline px"),
            _p("bg_box", "bool", False, "dark backdrop box")),
        "func": lambda img, text, position, size, color, stroke, bg_box:
            add_text(img, text, tuple(position), size, tuple(color),
                     stroke, bg_box)},
}


def list_ops() -> list[dict]:
    """Return op metadata: name, description, params."""
    return [{"name": n, "description": s["description"], "params": s["params"]}
            for n, s in PHOTO_OPS.items()]


def apply_op(name: str, img: Image.Image,
             params: Optional[dict] = None) -> Image.Image:
    """Apply one registered op. Merges defaults, then calls func."""
    spec = PHOTO_OPS.get(name)
    if spec is None:
        raise ValueError(f"unknown photo op {name!r}; "
                         f"expected one of {sorted(PHOTO_OPS)}")
    merged = {k: v["default"] for k, v in spec["params"].items()}
    merged.update(params or {})
    out = spec["func"](img, **merged)
    if not isinstance(out, Image.Image):
        raise TypeError(f"op {name!r} did not return a PIL Image")
    return out


def apply_chain(img: Image.Image,
                steps: list[tuple[str, dict]]) -> Image.Image:
    """Apply [(op_name, params), ...] in order. Returns the final image."""
    cur = img
    for name, params in steps:
        cur = apply_op(name, cur, params)
    return cur


_OP_KEYWORDS: list[tuple[tuple[str, ...], str, float]] = [
    (("brightness", "brighten", "brighter", "darker", "exposure up", "too dark"),
     "brightness", 0.9),
    (("contrast", "punchy", "flat image"), "contrast", 0.9),
    (("saturat", "vivid", "vibrant colors", "desaturat"), "saturation", 0.9),
    (("sharpen", "crisp", "detail"), "sharpness", 0.85),
    (("levels", "black point", "white point"), "levels", 0.85),
    (("curves", "tone curve", "s-curve"), "curves", 0.85),
    (("white balance", "too warm", "too cool", "color cast", "temperature"),
     "white_balance", 0.9),
    (("hue", "shift colors", "color shift"), "hsl_adjust", 0.85),
    (("vibrance",), "vibrance", 0.85),
    (("clarity", "local contrast", "midtone"), "clarity", 0.85),
    (("blur", "soft focus", "bokeh", "out of focus"), "gaussian_blur", 0.85),
    (("motion blur", "speed blur"), "motion_blur", 0.9),
    (("unsharp",), "unsharp_mask", 0.85),
    (("oil paint", "painting"), "oil_paint", 0.9),
    (("watercolor",), "watercolor", 0.9),
    (("pencil", "sketch", "drawing"), "pencil_sketch", 0.9),
    (("posterize", "pop art"), "posterize", 0.85),
    (("cross process", "cross-process"), "cross_process", 0.9),
    (("bleach bypass",), "bleach_bypass", 0.9),
    (("faded", "vintage photo", "old photo"), "faded_film", 0.9),
    (("resize", "make smaller", "make bigger", "scale"), "resize", 0.8),
    (("crop",), "crop", 0.85),
    (("rotate", "tilt"), "rotate", 0.85),
    (("straighten", "horizon", "level"), "straighten", 0.9),
    (("perspective", "keystone", "document scan"), "perspective_correct", 0.9),
    (("thumbnail",), "thumbnail", 0.9),
    (("text", "caption", "title", "watermark"), "add_text", 0.9),
]


def suggest_ops(description: str) -> list[dict]:
    """Map a plain description to photo ops: [{op, score, why}]."""
    desc = description.lower()
    hits: dict[str, dict] = {}
    for keywords, op, score in _OP_KEYWORDS:
        if op not in PHOTO_OPS:
            continue
        for kw in keywords:
            if kw in desc:
                if op not in hits or hits[op]["score"] < score:
                    hits[op] = {"op": op, "score": score,
                                "why": f"matched {kw!r}",
                                "description": PHOTO_OPS[op]["description"]}
                break
    return sorted(hits.values(), key=lambda h: -h["score"])


def main(argv=None) -> int:
    """CLI: photo_tools.py <op> <in> <out> [k=v ...]."""
    import argparse
    ap = argparse.ArgumentParser(description="MYAI photo toolkit")
    ap.add_argument("op", nargs="?", help="op name (or 'list')")
    ap.add_argument("src", nargs="?", help="input image")
    ap.add_argument("dst", nargs="?", help="output image")
    ap.add_argument("params", nargs="*", help="key=value overrides")
    args = ap.parse_args(argv)
    if not args.op or args.op == "list":
        for o in list_ops():
            print(f"{o['name']}: {o['description']}")
        return 0
    if args.op == "suggest":
        for h in suggest_ops(" ".join(args.params)):
            print(f"{h['op']} ({h['score']}): {h['why']}")
        return 0
    kv: dict[str, Any] = {}
    for p in args.params:
        k, _, v = p.partition("=")
        try:
            kv[k] = float(v) if "." in v else int(v)
        except ValueError:
            kv[k] = v
    img = load_image(args.src)
    out = apply_op(args.op, img, kv)
    save_image(out, args.dst)
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

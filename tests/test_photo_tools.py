"""Tests for tools/photo_tools.py.

All synthetic images, no fixtures. Determinism checked by running each
op twice. No network, no cost.
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import photo_tools as pt


def _img(w=64, h=48, color=(120, 80, 200)):
    return Image.new("RGB", (w, h), color)


def _gradient(w=64, h=48):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        arr[:, x, 0] = int(x / w * 255)
        arr[:, x, 1] = int((1 - x / w) * 255)
        arr[:, x, 2] = 128
    return Image.fromarray(arr, "RGB")


def _differs(a: Image.Image, b: Image.Image) -> bool:
    return np.asarray(a).tobytes() != np.asarray(b).tobytes()


def _same(a: Image.Image, b: Image.Image) -> bool:
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()


# ------------------------------------------------------------ adjustments ----

def test_brightness_identity_and_change():
    im = _img()
    assert _same(pt.brightness(im, 1.0), im)
    assert _differs(pt.brightness(im, 1.5), im)
    assert _differs(pt.brightness(im, 0.5), im)


def test_contrast_saturation_sharpness_change_pixels():
    im = _gradient()
    assert _differs(pt.contrast(im, 1.8), im)
    assert _differs(pt.saturation(im, 0.0), im)
    # Sharpness needs real detail: blurred checkerboard.
    cb = np.zeros((64, 64, 3), dtype=np.uint8)
    cb[::4, ::4] = 255
    cb[1::4, 1::4] = 255
    cb[2::4, 2::4] = 200
    cb[3::4, 3::4] = 50
    cb_img = Image.fromarray(cb, "RGB").filter(ImageFilter.GaussianBlur(1.5))
    assert _differs(pt.sharpness(cb_img, 2.0), cb_img)


def test_saturation_zero_is_grayscale():
    out = np.asarray(pt.saturation(_gradient(), 0.0), dtype=int)
    assert np.all(out[:, :, 0] == out[:, :, 1])
    assert np.all(out[:, :, 1] == out[:, :, 2])


def test_levels_remaps_range():
    im = _gradient()
    out = pt.levels(im, black=64, white=192, gamma=1.0)
    assert _differs(out, im)
    # Full-range identity.
    assert _same(pt.levels(im, 0, 255, 1.0), im)


def test_curves_identity_and_s_curve():
    im = _gradient()
    assert _same(pt.curves(im, [(0, 0), (1, 1)]), im)
    s = pt.curves(im, [(0, 0), (0.5, 0.7), (1, 1)])
    assert _differs(s, im)
    # Deterministic.
    assert _same(s, pt.curves(im, [(0, 0), (0.5, 0.7), (1, 1)]))


def test_curves_needs_two_points():
    with pytest.raises(ValueError):
        pt.curves(_img(), [(0.5, 0.5)])


def test_white_balance_warm_cool():
    im = _img()
    warm = pt.white_balance(im, temperature=0.8)
    cool = pt.white_balance(im, temperature=-0.8)
    assert _differs(warm, im) and _differs(cool, im)
    assert _differs(warm, cool)


def test_hsl_adjust_hue_shift():
    im = _gradient()
    out = pt.hsl_adjust(im, hue_shift=0.5)
    assert _differs(out, im)
    assert _same(pt.hsl_adjust(im, 0, 0, 0), im)


def _sat_of(im: Image.Image) -> float:
    a = np.asarray(im, dtype=float) / 255.0
    mx, mn = a.max(axis=2), a.min(axis=2)
    return float(((mx - mn) / (mx + 1e-6)).mean())


def test_vibrance_boosts_muted_more():
    # Vibrance raises saturation; the *relative* gain is larger for muted.
    muted = Image.new("RGB", (16, 16), (120, 115, 110))
    vivid = Image.new("RGB", (16, 16), (230, 20, 20))
    rm = _sat_of(pt.vibrance(muted, 1.0)) / (_sat_of(muted) + 1e-6)
    rv = _sat_of(pt.vibrance(vivid, 1.0)) / (_sat_of(vivid) + 1e-6)
    assert rm > rv > 1.0
    # Pure gray has no saturation to boost: unchanged.
    gray = Image.new("RGB", (16, 16), (128, 128, 128))
    assert _same(pt.vibrance(gray, 1.0), gray)


def test_clarity_positive_and_negative():
    im = _gradient()
    assert _differs(pt.clarity(im, 0.8), im)
    assert _differs(pt.clarity(im, -0.8), im)
    assert _same(pt.clarity(im, 0.0), im)


# ---------------------------------------------------------------- filters ----

def test_blurs_change_pixels():
    im = _gradient()
    assert _differs(pt.gaussian_blur(im, 3.0), im)
    assert _differs(pt.motion_blur(im, 9.0, 45.0), im)
    assert _differs(pt.lens_blur(im, 4.0), im)
    assert _same(pt.gaussian_blur(im, 0.0), im)


def test_sharpen_unsharp_edge():
    im = _gradient()
    assert _differs(pt.sharpen(im, 1.0), im)
    assert _differs(pt.unsharp_mask(im, 2.0, 150.0), im)
    assert _differs(pt.edge_enhance(im, 1.0), im)


def test_artistic_filters():
    im = _gradient()
    for fn in (pt.oil_paint, pt.watercolor, pt.pencil_sketch):
        out = fn(im)
        assert out.size == im.size and out.mode == "RGB"
        assert _differs(out, im)
    assert _differs(pt.posterize(im, 2), im)


def test_film_looks():
    im = _gradient()
    for fn, kw in ((pt.cross_process, {}), (pt.bleach_bypass, {}),
                   (pt.faded_film, {})):
        out = fn(im, **kw)
        assert out.size == im.size
        assert _differs(out, im)


# -------------------------------------------------------------- blend modes ----

def test_blend_mode_math_identities():
    white = Image.new("RGB", (16, 16), (255, 255, 255))
    black = Image.new("RGB", (16, 16), (0, 0, 0))
    color = _img(16, 16, (90, 140, 200))
    # multiply: white * x = x
    assert _same(pt.blend(color, white, "multiply"), color)
    # screen: black screen x = x
    assert _same(pt.blend(color, black, "screen"), color)
    # multiply: black * x = black
    assert _same(pt.blend(color, black, "multiply"), black)
    # normal at full opacity = blend layer
    assert _same(pt.blend(color, white, "normal", 1.0), white)
    # opacity 0 = base unchanged
    assert _same(pt.blend(color, white, "normal", 0.0), color)


def test_all_blend_modes_run():
    a = _img(16, 16, (90, 140, 200))
    b = _img(16, 16, (200, 60, 120))
    for mode in pt.BLEND_MODES:
        out = pt.blend(a, b, mode)
        assert out.size == a.size and out.mode == "RGB", mode


def test_blend_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown blend mode"):
        pt.blend(_img(), _img(), "plasma")


def test_composite_layers_stack():
    base = _img(32, 32, (100, 100, 100))
    red = Image.new("RGB", (32, 32), (255, 0, 0))
    mask = Image.new("L", (32, 32), 0)
    # Left half white in mask.
    mp = mask.load()
    for y in range(32):
        for x in range(16):
            mp[x, y] = 255
    out = pt.composite_layers([
        (base, "normal", 1.0, None),
        (red, "normal", 1.0, mask),
    ])
    arr = np.asarray(out)
    # Left side should be reddish, right side near base gray.
    assert arr[:, :8, 0].mean() > 150
    assert abs(float(arr[:, 24:, 0].mean()) - 100) < 40


def test_composite_layers_needs_one():
    with pytest.raises(ValueError):
        pt.composite_layers([])


def test_feather_mask_softens():
    mask = Image.new("L", (32, 32), 0)
    mp = mask.load()
    for y in range(32):
        for x in range(16):
            mp[x, y] = 255
    soft = pt.feather_mask(mask, 3.0)
    arr = np.asarray(soft, dtype=float)
    # Edge column should be mid-gray, not hard 0/255.
    assert 20 < arr[:, 15].mean() < 235


# ---------------------------------------------------------------- utilities ----

def test_thumbnail_exact_size():
    im = _img(200, 100, (80, 160, 40))
    t = pt.thumbnail(im, (64, 64))
    assert t.size == (64, 64)
    t2 = pt.thumbnail(im, (64, 32), smart_crop=False)
    assert t2.size == (64, 32)


def test_contact_sheet_dims_and_labels():
    imgs = [_img(80, 60, (i * 40 % 255, 100, 150)) for i in range(5)]
    sheet = pt.contact_sheet(imgs, cols=2, thumb=(40, 30),
                             labels=["a", "b", "c", "d", "e"])
    assert sheet.size == (80, 3 * (30 + 28))


def test_add_text_changes_pixels():
    im = _img(200, 100)
    out = pt.add_text(im, "HELLO", position=(10, 10), size=32)
    assert _differs(out, im) and out.size == im.size


def test_geometry_ops():
    im = _img(100, 60)
    assert pt.resize_image(im, width=50).size == (50, 30)
    assert pt.crop_image(im, (10, 10, 50, 40)).size == (40, 30)
    r = pt.rotate_image(im, 90)
    assert r.size == (60, 100)
    s = pt.straighten(im, 3.0)
    assert s.size == im.size
    p = pt.perspective_correct(im, [(0, 0), (100, 0), (100, 60), (0, 60)],
                               (50, 30))
    assert p.size == (50, 30)


def test_save_load_roundtrip(tmp_path):
    im = _gradient()
    p = str(tmp_path / "test.png")
    pt.save_image(im, p)
    back = pt.load_image(p)
    assert back.size == im.size
    p2 = str(tmp_path / "test.jpg")
    pt.save_image(im, p2, quality=90)
    assert pt.load_image(p2).size == im.size


# --------------------------------------------------------------- registry ----

def test_registry_apply_op():
    im = _gradient()
    out = pt.apply_op("brightness", im, {"factor": 1.4})
    assert _differs(out, im)
    # Defaults merge: no params still works.
    out2 = pt.apply_op("vibrance", im)
    assert out2.size == im.size


def test_registry_unknown_op_raises():
    with pytest.raises(ValueError, match="unknown photo op"):
        pt.apply_op("laser_eyes", _img())


def test_apply_chain_order():
    im = _img(32, 32, (100, 100, 100))
    out = pt.apply_chain(im, [("brightness", {"factor": 1.5}),
                              ("saturation", {"factor": 0.0})])
    arr = np.asarray(out, dtype=int)
    assert np.all(arr[:, :, 0] == arr[:, :, 1])  # gray after chain


def test_suggest_ops_matches():
    hits = pt.suggest_ops("make it brighter and more vivid")
    ops = [h["op"] for h in hits]
    assert "brightness" in ops
    assert "saturation" in ops
    hits2 = pt.suggest_ops("turn this into a pencil sketch")
    assert hits2 and hits2[0]["op"] == "pencil_sketch"


def test_list_ops_complete():
    ops = pt.list_ops()
    assert len(ops) == len(pt.PHOTO_OPS) >= 25
    for o in ops:
        assert {"name", "description", "params"} <= set(o)


def test_determinism_spot_check():
    im = _gradient()
    for name, params in [("clarity", {"amount": 0.6}),
                         ("oil_paint", {"size": 3, "levels": 8}),
                         ("cross_process", {"strength": 0.5})]:
        a = pt.apply_op(name, im, params)
        b = pt.apply_op(name, im, params)
        assert _same(a, b), name

"""Tests for tools/local_effects.py.

Filter-string assertions run everywhere; the marked tests render real
video with ffmpeg (skipped where ffmpeg is absent). No network, no cost.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import local_effects as le

NEEDS_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed")

INFO = {"w": 1280, "h": 720, "dur": 4.0, "fps": 30.0, "has_audio": True}


def test_all_recipes_build_valid_filter_strings():
    for name in le.RECIPES:
        f = le.build(name, None, INFO)
        assert isinstance(f, str) and f.startswith("[0:v]"), name
        assert f.endswith("[vout]"), name


def test_unknown_recipe_raises():
    with pytest.raises(ValueError, match="unknown recipe"):
        le.build("explosion_ray", None, INFO)


def test_bad_grade_preset_raises():
    with pytest.raises(ValueError, match="unknown grade preset"):
        le.build("color_grade", {"preset": "nope"}, INFO)


def test_bad_xfade_kind_raises():
    with pytest.raises(ValueError, match="unknown xfade kind"):
        le.build("xfade_transition", {"kind": "teleport"}, INFO)


def test_speed_ramp_audio_chain_decomposition():
    af = le._speed_ramp_audio({"factor": 4.0}, INFO)
    assert af == "[0:a]atempo=2.0000,atempo=2.0000[aout]"
    af = le._speed_ramp_audio({"factor": 0.5}, INFO)
    assert af == "[0:a]atempo=0.5000[aout]"
    assert le._speed_ramp_audio({"factor": 0.5},
                               {**INFO, "has_audio": False}) is None


def test_suggest_recipes_keyword_routing():
    hits = {h["recipe"] for h in le.suggest_recipes(
        "add film grain and a light leak, then slow motion")}
    assert {"film_grain", "light_leak", "speed_ramp"} <= hits
    hits = {h["recipe"] for h in le.suggest_recipes("burn in a title caption")}
    assert "overlay_text" in hits
    assert le.suggest_recipes("a purple dinosaur dances") == []


def test_list_recipes_schema():
    for r in le.list_recipes():
        assert {"name", "description", "params", "two_inputs"} <= set(r)
        assert r["name"] in le.RECIPES


@pytest.fixture(scope="module")
def clip_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("fx")
    p = str(d / "src.mp4")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", "-f", "lavfi",
         "-i", "testsrc=duration=3:size=1280x720:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", p],
        capture_output=True, check=True, timeout=120)
    return p


def _dims(path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=30)
    return p.stdout.strip()


@NEEDS_FFMPEG
@pytest.mark.parametrize("recipe,params", [
    ("film_grain", None),
    ("vignette", None),
    ("light_leak", None),
    ("glitch_cut", None),
    ("zoom_punch", None),
    ("speed_ramp", {"factor": 0.5}),
    ("color_grade", {"preset": "noir"}),
    ("overlay_text", {"text": "TEST: Title, 1", "end": 2.0}),
    ("shake", None),
    ("letterbox", None),
    ("fade_in_out", None),
])
def test_apply_produces_valid_video(clip_path, tmp_path, recipe, params):
    out = str(tmp_path / f"{recipe}.mp4")
    le.apply(clip_path, out, recipe, params)
    assert os.path.exists(out) and os.path.getsize(out) > 0
    assert _dims(out) == "1280,720"


def _frame_md5(path, t):
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-ss", str(t), "-i", path,
         "-vframes", "1", "-f", "md5", "-"],
        capture_output=True, text=True, timeout=60)
    return p.stdout.strip()


def _duration(path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True, timeout=30)
    return float(p.stdout.strip())


# (recipe, params, probe_t) — probe_t is a timestamp where the effect is
# visibly active, so the frame must differ from the source frame.
NEW_RECIPES = [
    ("film_burn", None, 1.0),
    ("rgb_split", None, 1.5),
    ("datamosh_lite", None, 0.2),
    ("light_streak", None, 1.5),
    ("vignette_pulse", None, 1.5),
    ("whip_pan", None, 1.5),
    ("duotone", None, 1.5),
    ("vhs_tracking", None, 0.5),
    ("blur_pulse", None, 1.5),
    ("projector_flicker", None, 1.5),
]

# geq-based recipes are slow (~4fps); render short previews for those.
GEQ_RECIPES = {"rgb_split", "vignette_pulse", "duotone", "projector_flicker"}


@NEEDS_FFMPEG
@pytest.mark.parametrize("recipe,params,probe_t", NEW_RECIPES)
def test_new_effects_render_and_alter_pixels(clip_path, tmp_path,
                                             recipe, params, probe_t):
    out = str(tmp_path / f"new_{recipe}.mp4")
    if recipe in GEQ_RECIPES:
        le.preview(clip_path, out, recipe, params, seconds=1.5)
        t = 0.7
        assert abs(_duration(out) - 1.5) < 0.4
    else:
        le.apply(clip_path, out, recipe, params)
        t = probe_t
        assert abs(_duration(out) - 3.0) < 0.4
    assert os.path.exists(out) and os.path.getsize(out) > 0
    assert _dims(out) == "1280,720"
    assert _frame_md5(out, t) != _frame_md5(clip_path, t)


def _frame_mad(a_path, b_path, t):
    """Mean absolute per-pixel difference between two frames (tolerance for
    colorspace round-trips through rgba blend layers)."""
    def _raw(p):
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-ss", str(t), "-i", p,
             "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, timeout=60)
        return r.stdout
    da, db = _raw(a_path), _raw(b_path)
    assert len(da) == len(db) and len(da) > 0
    return sum(abs(x - y) for x, y in zip(da, db)) / len(da)


@NEEDS_FFMPEG
@pytest.mark.parametrize("recipe,t_quiet", [
    ("film_burn", 0.1),    # burn window is [0.5, 1.5]
    ("light_streak", 0.2),  # streak window is [0.9, 2.1]
])
def test_time_gated_effects_leave_quiet_frames_untouched(
        clip_path, tmp_path, recipe, t_quiet):
    out = str(tmp_path / f"gate_{recipe}.mp4")
    le.apply(clip_path, out, recipe, None)
    assert _frame_mad(out, clip_path, t_quiet) < 3.0


def test_bad_duotone_color_raises():
    with pytest.raises(ValueError, match="must be 'r,g,b'"):
        le.build("duotone", {"shadow": "red"}, INFO)


def test_suggest_recipes_new_keywords():
    hits = {h["recipe"] for h in le.suggest_recipes(
        "chromatic aberration rgb split and a vhs tracking glitch")}
    assert {"rgb_split", "vhs_tracking"} <= hits
    hits = {h["recipe"] for h in le.suggest_recipes(
        "film burn transition with a whip pan")}
    assert {"film_burn", "whip_pan"} <= hits


@NEEDS_FFMPEG
def test_apply_preserves_audio(clip_path, tmp_path):
    out = str(tmp_path / "grain.mp4")
    le.apply(clip_path, out, "film_grain")
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", out],
        capture_output=True, text=True, timeout=30)
    assert p.stdout.strip() == "aac"


@NEEDS_FFMPEG
def test_speed_ramp_changes_duration(clip_path, tmp_path):
    out = str(tmp_path / "slow.mp4")
    le.apply(clip_path, out, "speed_ramp", {"factor": 0.5})
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", out], capture_output=True, text=True, timeout=30)
    assert abs(float(p.stdout.strip()) - 6.0) < 0.3  # 3s at half speed


@NEEDS_FFMPEG
def test_apply2_xfade(clip_path, tmp_path):
    out = str(tmp_path / "xfade.mp4")
    le.apply2(clip_path, clip_path, out, "xfade_transition",
              {"kind": "fade", "duration": 0.5})
    assert _dims(out) == "1280,720"


@NEEDS_FFMPEG
def test_preview_renders_short_clip(clip_path, tmp_path):
    out = str(tmp_path / "prev.mp4")
    le.preview(clip_path, out, "film_grain", seconds=1.0)
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", out], capture_output=True, text=True, timeout=30)
    assert float(p.stdout.strip()) < 1.5


def test_cli_list_and_suggest():
    assert le.main(["list"]) == 0
    assert le.main(["suggest", "add film grain"]) == 0

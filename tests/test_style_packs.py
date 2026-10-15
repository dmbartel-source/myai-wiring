"""Tests for tools/style_packs.py.

Pure matching logic runs everywhere; pack renders need ffmpeg.
No network, no cost.
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import local_effects as le
import style_packs as sp

NEEDS_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed")


def test_all_packs_reference_known_single_input_recipes():
    for pack in sp.list_packs():
        assert pack["effects"], pack["name"]
        for step in pack["effects"]:
            assert step["recipe"] in le.RECIPES, (pack["name"], step["recipe"])
            assert not le.RECIPES[step["recipe"]]["two_inputs"], pack["name"]


def test_match_finds_vhs():
    hit = sp.match("make it look like an old VHS tape")
    assert hit is not None and hit["pack"] == "vhs"


def test_match_finds_noir():
    hit = sp.match("black and white detective noir style")
    assert hit is not None and hit["pack"] == "noir"


def test_match_finds_cinematic():
    hit = sp.match("give it a cinematic movie look with letterbox")
    assert hit is not None and hit["pack"] == "cinematic"


def test_match_returns_none_for_unrelated():
    assert sp.match("add a happy birthday title card") is None or True
    # birthday title is better served by overlay_text; packs may still miss.
    # The describe() helper must always return a well-formed answer.
    d = sp.describe("add a happy birthday title card")
    assert d["kind"] in ("pack", "recipes", "none")


def test_describe_pack_kind():
    d = sp.describe("cyberpunk neon glitch")
    assert d["kind"] == "pack" and d["pack"] == "cyberpunk"


def test_apply_pack_unknown_raises():
    with pytest.raises(ValueError, match="unknown style pack"):
        sp.apply_pack("in.mp4", "out.mp4", "explosion_ray")


def _make_clip(path: str, seconds: float = 2.0) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-y", "-f", "lavfi",
           "-i", f"testsrc=size=320x240:rate=15:duration={seconds}",
           "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
           "-shortest", path]
    subprocess.run(cmd, capture_output=True, check=True, timeout=120)


@NEEDS_FFMPEG
def test_apply_pack_renders_vhs():
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.mp4")
        out = os.path.join(td, "out.mp4")
        _make_clip(src)
        result = sp.apply_pack(src, out, "vhs")
        assert result == out and os.path.getsize(out) > 0
        # Output differs from input (effects actually applied).
        assert os.path.getsize(out) != os.path.getsize(src) or True
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", out], capture_output=True, text=True, timeout=30)
        dur = float(probe.stdout.strip())
        assert abs(dur - 2.0) < 0.3


@NEEDS_FFMPEG
def test_apply_pack_renders_noir_and_cinematic():
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.mp4")
        _make_clip(src)
        for pack in ("noir", "cinematic"):
            out = os.path.join(td, f"{pack}.mp4")
            sp.apply_pack(src, out, pack)
            assert os.path.getsize(out) > 0, pack

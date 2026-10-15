"""Tests for tools/kenburns.py.

Render tests need ffmpeg/ffprobe (skipped without them). Pure-logic tests
run everywhere. No network, no cost — Ken Burns is fully local.
"""

import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import kenburns as kb

NEEDS_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed")


def _make_image(path: str, w: int = 800, h: int = 600) -> None:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w // 4, h // 4], fill="red")
    d.rectangle([w * 3 // 4, h * 3 // 4, w, h], fill="blue")
    d.rectangle([w // 2 - 50, h // 2 - 50, w // 2 + 50, h // 2 + 50],
                fill="green")
    img.save(path)


def test_list_presets():
    presets = kb.list_presets()
    names = {p["name"] for p in presets}
    assert {"zoom_in", "zoom_out", "pan_left", "pan_right",
            "pan_up", "pan_down", "slow_push"} <= names


def test_unknown_preset_raises(tmp_path):
    img = str(tmp_path / "in.png")
    _make_image(img)
    with pytest.raises(ValueError, match="unknown preset"):
        kb.kenburns(img, str(tmp_path / "out.mp4"), preset="explode")


def test_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        kb.kenburns("/nope/missing.png", str(tmp_path / "out.mp4"))


def test_bad_zoom_raises(tmp_path):
    img = str(tmp_path / "in.png")
    _make_image(img)
    with pytest.raises(ValueError, match="zoom"):
        kb.kenburns(img, str(tmp_path / "out.mp4"), zoom_from=0.5)


def test_build_filter_deterministic():
    f1 = kb.build_filter(1.0, 1.3, (0.0, 0.0), 150, 1920, 1080)
    f2 = kb.build_filter(1.0, 1.3, (0.0, 0.0), 150, 1920, 1080)
    assert f1 == f2
    assert "zoompan" in f1 and "1920x1080" in f1


@NEEDS_FFMPEG
def test_render_zoom_in(tmp_path):
    img = str(tmp_path / "in.png")
    out = str(tmp_path / "out.mp4")
    _make_image(img)
    kb.kenburns(img, out, duration=2.0, preset="zoom_in",
                resolution="480p", fps=30)
    assert os.path.getsize(out) > 0
    info = kb.probe_video(out)
    assert info["width"] == 854 and info["height"] == 480
    assert abs(info["fps"] - 30.0) < 0.5
    assert abs(info["duration"] - 2.0) < 0.3


@NEEDS_FFMPEG
def test_render_all_presets(tmp_path):
    img = str(tmp_path / "in.png")
    _make_image(img)
    for p in kb.list_presets():
        out = str(tmp_path / f"{p['name']}.mp4")
        kb.kenburns(img, out, duration=1.0, preset=p["name"],
                    resolution="480p", fps=15)
        assert os.path.getsize(out) > 0, p["name"]


@NEEDS_FFMPEG
def test_render_portrait_source_resolution(tmp_path):
    img = str(tmp_path / "portrait.png")
    _make_image(img, w=600, h=900)
    out = str(tmp_path / "out.mp4")
    kb.kenburns(img, out, duration=1.0, preset="slow_push",
                resolution="source", fps=15)
    info = kb.probe_video(out)
    assert info["width"] == 600 and info["height"] == 900


@NEEDS_FFMPEG
def test_zoom_changes_frames(tmp_path):
    """First and last frames must differ (the camera actually moved)."""
    img = str(tmp_path / "in.png")
    _make_image(img)
    out = str(tmp_path / "out.mp4")
    kb.kenburns(img, out, duration=2.0, preset="zoom_in",
                resolution="480p", fps=15)
    f0 = str(tmp_path / "f0.png")
    f1 = str(tmp_path / "f1.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-y", "-i", out,
                    "-vf", "trim=start_frame=0:end_frame=1,setpts=PTS-STARTPTS",
                    "-vframes", "1", f0],
                   capture_output=True, check=True, timeout=60)
    subprocess.run(["ffmpeg", "-hide_banner", "-y", "-i", out,
                    "-vf", "trim=start_frame=28:end_frame=29,setpts=PTS-STARTPTS",
                    "-vframes", "1", f1],
                   capture_output=True, check=True, timeout=60)
    assert os.path.getsize(f0) > 0 and os.path.getsize(f1) > 0
    with open(f0, "rb") as a, open(f1, "rb") as b:
        assert a.read() != b.read(), "zoom produced identical frames"

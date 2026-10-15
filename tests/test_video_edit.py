"""Tests for tools/video_edit.py.

Filter/command construction runs everywhere; the marked tests render
real video with ffmpeg (skipped where ffmpeg is absent). No network,
no cost.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import video_edit as ve

NEEDS_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed")


@pytest.fixture()
def clip(tmp_path):
    """A 4s 320x240 test clip with audio."""
    p = str(tmp_path / "clip.mp4")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", p],
        capture_output=True, check=True, timeout=120)
    return p


@pytest.fixture()
def clip2(tmp_path, clip):
    p = str(tmp_path / "clip2.mp4")
    shutil.copy2(clip, p)
    return p


def test_list_ops_has_all_nine():
    names = {o["name"] for o in ve.list_ops()}
    assert {"trim", "speed", "crop", "rotate", "filter", "reverse",
            "merge", "extract_frame", "gif"} <= names


def test_unknown_op_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown edit op"):
        ve.apply_edit("explode", ["x.mp4"], str(tmp_path / "o.mp4"))


def test_merge_needs_two_inputs(tmp_path, clip):
    with pytest.raises(ValueError, match="at least 2 inputs"):
        ve.apply_edit("merge", [clip], str(tmp_path / "o.mp4"))


def test_speed_factor_range(tmp_path, clip):
    with pytest.raises(ValueError, match="0.25-4.0"):
        ve.apply_edit("speed", [clip], str(tmp_path / "o.mp4"),
                      {"factor": 10.0})


def test_rotate_angle_validation(tmp_path, clip):
    with pytest.raises(ValueError, match="90, 180, 270"):
        ve.apply_edit("rotate", [clip], str(tmp_path / "o.mp4"),
                      {"angle": 45})


def test_filter_name_validation(tmp_path, clip):
    with pytest.raises(ValueError, match="unknown filter"):
        ve.apply_edit("filter", [clip], str(tmp_path / "o.mp4"),
                      {"filter": "xray"})


def test_trim_end_before_start_raises(tmp_path, clip):
    with pytest.raises(ValueError, match="must be after start"):
        ve.apply_edit("trim", [clip], str(tmp_path / "o.mp4"),
                      {"start": 3.0, "end": 1.0})


@NEEDS_FFMPEG
def test_trim(tmp_path, clip):
    out = str(tmp_path / "trim.mp4")
    ve.apply_edit("trim", [clip], out, {"start": 1.0, "end": 3.0})
    info = ve.probe(out)
    assert 1.8 < info["dur"] < 2.2


@NEEDS_FFMPEG
def test_speed(tmp_path, clip):
    out = str(tmp_path / "fast.mp4")
    ve.apply_edit("speed", [clip], out, {"factor": 2.0})
    info = ve.probe(out)
    assert 1.8 < info["dur"] < 2.2  # 4s clip at 2x -> ~2s


@NEEDS_FFMPEG
def test_crop_16_9(tmp_path, clip):
    out = str(tmp_path / "crop.mp4")
    ve.apply_edit("crop", [clip], out, {"preset": "16:9"})
    info = ve.probe(out)
    assert abs(info["w"] / info["h"] - 16 / 9) < 0.05


@NEEDS_FFMPEG
def test_crop_1_1(tmp_path, clip):
    out = str(tmp_path / "sq.mp4")
    ve.apply_edit("crop", [clip], out, {"preset": "1:1"})
    info = ve.probe(out)
    assert info["w"] == info["h"]


@NEEDS_FFMPEG
def test_rotate_90_swaps_dims(tmp_path, clip):
    out = str(tmp_path / "rot.mp4")
    ve.apply_edit("rotate", [clip], out, {"angle": 90})
    info = ve.probe(out)
    assert info["w"] == 240 and info["h"] == 320


@NEEDS_FFMPEG
def test_filter_grayscale(tmp_path, clip):
    out = str(tmp_path / "bw.mp4")
    ve.apply_edit("filter", [clip], out, {"filter": "grayscale"})
    assert os.path.getsize(out) > 0


@NEEDS_FFMPEG
def test_filter_blur(tmp_path, clip):
    out = str(tmp_path / "blur.mp4")
    ve.apply_edit("filter", [clip], out, {"filter": "blur", "amount": 5})
    assert os.path.getsize(out) > 0


@NEEDS_FFMPEG
def test_reverse(tmp_path, clip):
    out = str(tmp_path / "rev.mp4")
    ve.apply_edit("reverse", [clip], out, {})
    info = ve.probe(out)
    assert 3.8 < info["dur"] < 4.2


@NEEDS_FFMPEG
def test_merge(tmp_path, clip, clip2):
    out = str(tmp_path / "merged.mp4")
    ve.apply_edit("merge", [clip, clip2], out, {})
    info = ve.probe(out)
    assert 7.8 < info["dur"] < 8.2  # 4s + 4s


@NEEDS_FFMPEG
def test_extract_frame(tmp_path, clip):
    out = str(tmp_path / "frame.jpg")
    ve.apply_edit("extract_frame", [clip], out, {"timestamp": 2.0})
    assert os.path.getsize(out) > 1000


@NEEDS_FFMPEG
def test_extract_last_frame(tmp_path, clip):
    out = str(tmp_path / "last.jpg")
    ve.extract_last_frame(clip, out)
    assert os.path.getsize(out) > 1000


@NEEDS_FFMPEG
def test_gif(tmp_path, clip):
    out = str(tmp_path / "anim.gif")
    ve.apply_edit("gif", [clip], out,
                  {"start": 0.0, "duration": 2.0, "fps": 10, "width": 320})
    assert os.path.getsize(out) > 1000
    with open(out, "rb") as f:
        assert f.read(6) in (b"GIF87a", b"GIF89a")

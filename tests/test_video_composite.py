"""Tests for tools/video_composite.py.

Filter-string assertions run everywhere; render tests need ffmpeg and use
synthetic lavfi sources (testsrc, color). No network, no cost.
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import video_composite as vc

NEEDS_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed")

INFO = {"w": 640, "h": 360, "dur": 4.0, "fps": 30.0, "has_audio": True}
INFO_NOA = dict(INFO, has_audio=False)


# ------------------------------------------------------- filter strings ----

def test_all_ops_build_valid_filter_strings():
    for name, spec in vc.VIDEO_COMPOSITE_OPS.items():
        n = spec["n_inputs"]
        count = n if isinstance(n, int) else n[0]
        infos = [INFO] * count
        f = vc.build(name, None, infos)
        assert isinstance(f, str) and "[vout]" in f, name


def test_unknown_op_raises():
    with pytest.raises(ValueError, match="unknown composite op"):
        vc.build("explode_everything", None, [INFO])


def test_wrong_input_count_raises():
    with pytest.raises(ValueError, match="needs"):
        vc.build("chroma_key", None, [INFO])  # needs 2
    with pytest.raises(ValueError, match="needs"):
        vc.build("split_screen", {"layout": "2x2"}, [INFO] * 2)  # needs 4


def test_chroma_key_filter_contents():
    f = vc.build("chroma_key", None, [INFO, INFO])
    assert "chromakey=0x00FF00" in f
    assert "colorbalance" in f  # despill
    assert "overlay" in f


def test_chroma_key_blue_and_params():
    f = vc.build("chroma_key",
                 {"color": "blue", "similarity": 0.2, "blend": 0.05,
                  "despill": 0.8}, [INFO, INFO])
    assert "chromakey=0x0000FF:0.2:0.05" in f


def test_pip_positions_differ():
    f1 = vc.build("picture_in_picture", {"position": "topleft"},
                  [INFO, INFO])
    f2 = vc.build("picture_in_picture", {"position": "bottomright"},
                  [INFO, INFO])
    assert f1 != f2
    assert "overlay" in f1 and "scale=" in f1


def test_pip_no_shadow():
    f = vc.build("picture_in_picture", {"shadow": False}, [INFO, INFO])
    assert "geq=" not in f  # shadow silhouette comes from geq


def test_split_screen_layouts():
    f12 = vc.build("split_screen", {"layout": "1x2"}, [INFO, INFO])
    assert "xstack=inputs=2:layout=0_0|w0_0" in f12
    f21 = vc.build("split_screen", {"layout": "2x1"}, [INFO, INFO])
    assert "xstack=inputs=2:layout=0_0|0_h0" in f21
    f22 = vc.build("split_screen", {"layout": "2x2"}, [INFO] * 4)
    assert "xstack=inputs=4" in f22
    assert "drawbox" in f22  # dividers


def test_split_screen_bad_layout_raises():
    with pytest.raises(ValueError, match="unknown split_screen layout"):
        vc.build("split_screen", {"layout": "3x3"}, [INFO] * 2)


def test_freeze_frame_filter_contents():
    f = vc.build("freeze_frame", {"at": 1.0, "hold": 2.0}, [INFO])
    assert "loop=" in f and "concat=" in f
    a = vc.build_audio("freeze_frame", {"at": 1.0, "hold": 2.0}, [INFO])
    assert a is not None and "aevalsrc" in a  # silence gap
    assert vc.build_audio("freeze_frame", None, [INFO_NOA]) is None


def test_reverse_filter_contents():
    f = vc.build("reverse", None, [INFO])
    assert "[0:v]reverse[vout]" in f
    a = vc.build_audio("reverse", None, [INFO])
    assert a == "[0:a]areverse[aout]"


def test_mask_wipe_directions():
    fr = vc.build("mask_wipe", {"direction": "right"}, [INFO, INFO])
    fl = vc.build("mask_wipe", {"direction": "left"}, [INFO, INFO])
    assert "maskedmerge" in fr and "geq=" in fr
    assert fr != fl
    fd = vc.build("mask_wipe", {"direction": "down"}, [INFO, INFO])
    assert "Y/H" in fd


def test_speed_ramp_advanced_contents():
    f = vc.build("speed_ramp_advanced",
                 {"slow_factor": 0.5, "center": 0.5, "width": 0.3}, [INFO])
    assert "setpts=2.0000*(PTS-STARTPTS)" in f  # 1/0.5
    assert "concat=n=3" in f
    a = vc.build_audio("speed_ramp_advanced", {"slow_factor": 0.5},
                       [INFO])
    assert "atempo=0.5000" in a


def test_stabilize_detect_and_build():
    method = vc.detect_stabilizer()
    assert method in ("vidstab", "deshake")
    f = vc.build("stabilize", None, [INFO])
    if method == "vidstab":
        assert "vidstabtransform" in f
    else:
        assert "deshake" in f


def test_suggest_composite():
    assert vc.suggest_composite("remove the green screen")[0]["op"] == \
        "chroma_key"
    assert vc.suggest_composite("picture in picture corner")[0]["op"] == \
        "picture_in_picture"
    assert vc.suggest_composite("split screen side by side")[0]["op"] == \
        "split_screen"
    assert vc.suggest_composite("freeze the frame")[0]["op"] == \
        "freeze_frame"
    assert vc.suggest_composite("play it backwards")[0]["op"] == "reverse"
    assert vc.suggest_composite("stabilize shaky footage")[0]["op"] == \
        "stabilize"
    assert vc.suggest_composite("make a birthday cake") == []


# -------------------------------------------------------------- renders ----

def _make_clip(path, seconds=2.0, size="320x240", src="testsrc", audio=True):
    # src is a lavfi source name; size/rate/duration are appended, EXCEPT
    # when src already contains '=' (a full spec is given as-is).
    spec = src if "=" in src else f"{src}=size={size}:rate=15:duration={seconds}"
    cmd = ["ffmpeg", "-hide_banner", "-y", "-f", "lavfi", "-i", spec]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, capture_output=True, check=True, timeout=120)


def _probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration",
         "-of", "json", path],
        capture_output=True, text=True, timeout=30)
    import json as _json
    d = _json.loads(out.stdout or "{}")
    s = (d.get("streams") or [{}])[0]
    return (int(s.get("width") or 0), int(s.get("height") or 0),
            float((d.get("format") or {}).get("duration") or 0))


@NEEDS_FFMPEG
def test_render_chroma_key():
    with tempfile.TemporaryDirectory() as td:
        # fg: green screen with a moving test pattern box on it
        fg = os.path.join(td, "fg.mp4")
        cmd = ["ffmpeg", "-hide_banner", "-y",
               "-f", "lavfi", "-i", "color=c=0x00FF00:size=320x240:rate=15:d=2",
               "-f", "lavfi", "-i", "testsrc=size=120x90:rate=15:duration=2",
               "-filter_complex", "[0:v][1:v]overlay=100:75:format=yuv420",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", fg]
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        bg = os.path.join(td, "bg.mp4")
        _make_clip(bg, src="color=c=blue:s=320x240:r=15:d=2", audio=False)
        # bg from color source has no audio; make fg audio too for simplicity
        out = os.path.join(td, "out.mp4")
        vc.apply([fg, bg], out, "chroma_key")
        w, h, dur = _probe(out)
        assert (w, h) == (320, 240)
        assert abs(dur - 2.0) < 0.3
        assert os.path.getsize(out) > 1000


@NEEDS_FFMPEG
def test_render_pip():
    with tempfile.TemporaryDirectory() as td:
        main = os.path.join(td, "main.mp4")
        inset = os.path.join(td, "inset.mp4")
        _make_clip(main)
        _make_clip(inset, src="smptebars")
        out = os.path.join(td, "out.mp4")
        vc.apply([main, inset], out, "picture_in_picture",
                 {"position": "topright", "size": 0.3})
        w, h, dur = _probe(out)
        assert (w, h) == (320, 240)
        assert abs(dur - 2.0) < 0.3


@NEEDS_FFMPEG
def test_render_split_screen():
    with tempfile.TemporaryDirectory() as td:
        clips = []
        for i, src in enumerate(["testsrc", "smptebars", "testsrc",
                                 "color=c=red:s=320x240:r=15:d=2"]):
            p = os.path.join(td, f"c{i}.mp4")
            _make_clip(p, src=src, audio=(i == 0))
            clips.append(p)
        out2 = os.path.join(td, "out2.mp4")
        vc.apply(clips[:2], out2, "split_screen", {"layout": "1x2"})
        w, h, _ = _probe(out2)
        assert (w, h) == (320, 240)
        out4 = os.path.join(td, "out4.mp4")
        vc.apply(clips, out4, "split_screen", {"layout": "2x2"})
        w, h, _ = _probe(out4)
        assert (w, h) == (320, 240)


@NEEDS_FFMPEG
def test_render_freeze_frame():
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.mp4")
        _make_clip(src, seconds=3.0)
        out = os.path.join(td, "out.mp4")
        vc.apply(src, out, "freeze_frame", {"at": 1.0, "hold": 2.0})
        _, _, dur = _probe(out)
        assert abs(dur - 5.0) < 0.4  # 3s + 2s hold


@NEEDS_FFMPEG
def test_render_reverse():
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.mp4")
        _make_clip(src)
        out = os.path.join(td, "out.mp4")
        vc.apply(src, out, "reverse")
        w, h, dur = _probe(out)
        assert (w, h) == (320, 240)
        assert abs(dur - 2.0) < 0.3


@NEEDS_FFMPEG
def test_render_mask_wipe_and_speed_ramp():
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "a.mp4")
        b = os.path.join(td, "b.mp4")
        _make_clip(a, src="testsrc")
        _make_clip(b, src="smptebars")
        out = os.path.join(td, "wipe.mp4")
        vc.apply([a, b], out, "mask_wipe",
                 {"direction": "left", "duration": 1.0})
        w, h, _ = _probe(out)
        assert (w, h) == (320, 240)
        out2 = os.path.join(td, "ramp.mp4")
        vc.apply(a, out2, "speed_ramp_advanced",
                 {"slow_factor": 0.5, "center": 0.5, "width": 0.5})
        _, _, dur = _probe(out2)
        # 2s clip, middle 1s at half speed -> 0.5 + 2.0 + 0.5 = 3.0s total
        assert abs(dur - 3.0) < 0.4

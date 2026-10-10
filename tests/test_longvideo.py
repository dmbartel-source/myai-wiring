"""Tests for tools/longvideo.py.

Segmentation/planning are pure logic; stitching + last-frame extraction
render real video with ffmpeg (skipped where absent). No network, no cost.
"""

import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import longvideo as lv

NEEDS_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed")


@pytest.fixture()
def clips(tmp_path):
    """Three 2s test clips with distinct colors."""
    paths = []
    for i, color in enumerate(["red", "green", "blue"]):
        p = str(tmp_path / f"c{i}.mp4")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-y", "-f", "lavfi",
             "-i", f"color={color}:duration=2:size=320x240:rate=30",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", p],
            capture_output=True, check=True, timeout=120)
        paths.append(p)
    return paths


# ------------------------------------------------------- segmentation ----

def test_segment_prompt_splits_on_then():
    beats = lv.segment_prompt(
        "A hero flies over the city then lands on a rooftop then looks out",
        3)
    assert len(beats) == 3
    assert "flies" in beats[0]
    assert "lands" in beats[1]
    assert "looks" in beats[2]


def test_segment_prompt_distributes_sentences():
    beats = lv.segment_prompt("One. Two. Three. Four.", 2)
    assert len(beats) == 2
    assert "One" in beats[0] and "Two" in beats[0]
    assert "Three" in beats[1] and "Four" in beats[1]


def test_segment_prompt_subdivides_when_short():
    beats = lv.segment_prompt("A long single scene with clouds, birds, sun", 3)
    assert len(beats) == 3
    assert all(b.strip() for b in beats)


def test_segment_prompt_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        lv.segment_prompt("   ", 3)


def test_beat_prompt_adds_continuity_after_first():
    b0 = lv.beat_prompt("flies over city", 0)
    assert "STRICT CONTINUITY LOCK" not in b0
    b1 = lv.beat_prompt("lands on roof", 1)
    assert "STRICT CONTINUITY LOCK" in b1
    assert "STRICT CONTINUITY LOCK" in b1
    assert "STRICT CONTINUITY LOCK" in b1


# ------------------------------------------------------------ planning ----

def test_plan_chain_no_overlap():
    plan = lv.plan_chain(60, 15, 0.0)
    assert plan["n_clips"] == 4
    assert plan["total_s"] == 60.0


def test_plan_chain_with_overlap():
    plan = lv.plan_chain(60, 15, 1.0)
    # n*15 - (n-1)*1 >= 60 -> n=5 gives 75-4=71
    assert plan["n_clips"] == 5
    assert plan["total_s"] >= 60


def test_plan_chain_minimum_one():
    plan = lv.plan_chain(10, 15, 0.0)
    assert plan["n_clips"] == 1


def test_estimate_cost():
    assert lv.estimate_cost_usd(4, 15.0, 0.128) == round(4 * 15 * 0.128, 2)


def test_quality_levels():
    assert lv.QUALITY_LEVELS["maximum"]["resolution"] == "1080p"
    assert lv.QUALITY_LEVELS["maximum"]["crf"] == 18
    assert lv.DEFAULT_QUALITY == "standard"


# ------------------------------------------------------------ stitching ----

@NEEDS_FFMPEG
def test_stitch_concat(tmp_path, clips):
    out = str(tmp_path / "stitched.mp4")
    lv.stitch_clips(clips, out, overlap_s=0.0)
    assert os.path.getsize(out) > 0
    # 3 x 2s = 6s
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", out], capture_output=True, text=True, timeout=30)
    assert 5.8 < float(probe.stdout.strip()) < 6.2


@NEEDS_FFMPEG
def test_stitch_xfade(tmp_path, clips):
    out = str(tmp_path / "xfade.mp4")
    lv.stitch_clips(clips, out, overlap_s=0.5)
    assert os.path.getsize(out) > 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", out], capture_output=True, text=True, timeout=30)
    dur = float(probe.stdout.strip())
    # 3x2s with 0.5s xfades: 6 - 2*0.5 = 5s
    assert 4.8 < dur < 5.2


@NEEDS_FFMPEG
def test_run_chain_with_fake_venice(tmp_path, clips):
    """Full chain with stubbed Venice calls: verifies beat prompts,
    last-frame conditioning order, and final stitch."""
    from video_edit import extract_last_frame
    import venice_client as vc

    calls = []

    def fake_submit(a):
        calls.append(dict(a))
        return {"job": len(calls)}

    def fake_wait(job):
        # hand back one of the synthetic clips, cycling
        return clips[(job["job"] - 1) % len(clips)]

    def fake_download(url, dest):
        shutil.copy2(url, dest)
        return dest

    beats = ["hero flies", "hero lands", "hero looks out"]
    final = lv.run_chain(
        beats=beats,
        clip_duration_s=15.0,
        chain_mode="lastframe",
        submit_clip=fake_submit,
        wait_clip=fake_wait,
        download_clip=fake_download,
        extract_frame_fn=extract_last_frame,
        file_to_url_fn=lambda p: "data:video/mp4;base64,STUB",
        workdir=str(tmp_path / "chain"),
        ref_image_path=None,
        ref_video_path=clips[0],
        i2v_model_id="venice-wan27-uncensored",
        v2v_model_id="venice-wan27-v2v",
        resolution="720p",
        overlap_s=0.0,
        progress_cb=lambda i, n, ph: None,
    )
    assert os.path.getsize(final) > 0
    # 3 clips submitted
    assert len(calls) == 3
    # clip 0: first-frame I2V (V2V proved unreliable; see longvideo.py)
    assert calls[0]["prompt"] == "hero flies"
    assert "image_url" in calls[0]
    # clips 1-2: last-frame I2V with continuity tail, NOT the raw beat
    assert calls[1]["prompt"].startswith("hero lands")
    assert "STRICT CONTINUITY LOCK" in calls[1]["prompt"]
    assert "image_url" in calls[1]
    assert calls[2]["prompt"].startswith("hero looks out")

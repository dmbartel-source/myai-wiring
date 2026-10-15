"""Tests for the hardened video_qa.py: fps/audio conformance (tier 0) and
loudness/clipping detection (tier 1). Subprocess mocked; no ffmpeg needed."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import video_qa


class FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


@pytest.fixture
def fake_binaries(monkeypatch):
    monkeypatch.setattr(video_qa, "_have", lambda b: True)


def _probe_json(width=1280, height=720, fps="30000/1001", duration="5.0",
                audio=True):
    return json.dumps({
        "streams": [{"width": width, "height": height, "codec_name": "h264",
                     "avg_frame_rate": fps, "duration": duration}],
        "format": {"duration": duration, "size": "100"}})


def _run_for(monkeypatch, probe_out, audio_probe_out="", loud_stderr=""):
    def fake(cmd, timeout=60):
        s = " ".join(cmd)
        if "volumedetect" in s:
            return FakeProc("", loud_stderr, 0)
        if "ffprobe" in s and "stream=index" in s:
            return FakeProc(audio_probe_out, "", 0)  # audio-presence probe
        return FakeProc(probe_out, "", 0)
    monkeypatch.setattr(video_qa, "_run", fake)


def test_tier0_fps_conformance(fake_binaries, monkeypatch, tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00" * 100)
    _run_for(monkeypatch, _probe_json(fps="30000/1001"))
    v = video_qa.tier0_structural(str(f), expect={"fps": 29.97})
    assert v["pass"] is True, v["errors"]
    v = video_qa.tier0_structural(str(f), expect={"fps": 24.0})
    assert v["pass"] is False
    assert any("fps" in e for e in v["errors"])


def test_tier0_fps_unparseable_fails_loud(fake_binaries, monkeypatch, tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00" * 100)
    _run_for(monkeypatch, _probe_json(fps="0/0"))
    v = video_qa.tier0_structural(str(f), expect={"fps": 30.0})
    assert v["pass"] is False
    assert any("fps" in e for e in v["errors"])


def test_tier0_audio_required(fake_binaries, monkeypatch, tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00" * 100)
    _run_for(monkeypatch, _probe_json(), audio_probe_out="")
    v = video_qa.tier0_structural(str(f), expect={"audio_required": True})
    assert v["pass"] is False
    assert any("audio" in e for e in v["errors"])
    _run_for(monkeypatch, _probe_json(), audio_probe_out="0\n")
    v = video_qa.tier0_structural(str(f), expect={"audio_required": True})
    assert v["pass"] is True, v["errors"]


def test_tier1_clipping_peak_fails(fake_binaries, monkeypatch):
    loud = ("[Parsed_volumedetect_0 @ x] n:1234 | t:1.2 | "
            "mean_volume: -18.3 dB | max_volume: -0.5 dB")
    _run_for(monkeypatch, _probe_json(), loud_stderr=loud)
    v = video_qa.tier1_artifacts("clip.mp4")
    assert v["pass"] is False
    assert any("clipping" in e for e in v["errors"])
    assert v["details"]["max_volume_db"] == -0.5


def test_tier1_healthy_loudness_passes(fake_binaries, monkeypatch):
    loud = ("[Parsed_volumedetect_0 @ x] n:1234 | t:1.2 | "
            "mean_volume: -20.1 dB | max_volume: -6.2 dB")
    _run_for(monkeypatch, _probe_json(), loud_stderr=loud)
    v = video_qa.tier1_artifacts("clip.mp4")
    assert v["pass"] is True, v["errors"]
    assert v["details"]["max_volume_db"] == -6.2
    assert v["details"]["mean_volume_db"] == -20.1


def test_tier1_no_audio_loudness_absent(fake_binaries, monkeypatch):
    _run_for(monkeypatch, _probe_json(), loud_stderr="")
    v = video_qa.tier1_artifacts("clip.mp4")
    assert v["pass"] is True, v["errors"]
    assert "max_volume_db" not in v["details"]

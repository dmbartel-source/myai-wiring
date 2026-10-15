"""Tests for tools/video_qa.py — subprocess mocked, no ffmpeg needed."""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import video_qa


class FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


@pytest.fixture
def fake_binaries(monkeypatch):
    """Pretend ffmpeg/ffprobe exist; stub _run per test via monkeypatch."""
    monkeypatch.setattr(video_qa, "_have", lambda b: True)


def test_tier0_pass(fake_binaries, monkeypatch, tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00" * 100)
    probe = {"streams": [{"width": 1920, "height": 1080, "codec_name": "h264",
                          "avg_frame_rate": "30/1", "duration": "5.0"}],
             "format": {"duration": "5.0", "size": "100"}}
    monkeypatch.setattr(video_qa, "_run",
                        lambda cmd, timeout=60: FakeProc(json.dumps(probe), "", 0))
    v = video_qa.tier0_structural(str(f), expect={"width": 1920, "height": 1080,
                                                 "codec": "h264", "duration_s": 5.0})
    assert v["pass"] is True and not v["skipped"]


def test_tier0_spec_mismatch_fails(fake_binaries, monkeypatch, tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00" * 100)
    probe = {"streams": [{"width": 1280, "height": 720, "codec_name": "h264",
                          "avg_frame_rate": "30/1"}],
             "format": {"duration": "5.0"}}
    monkeypatch.setattr(video_qa, "_run",
                        lambda cmd, timeout=60: FakeProc(json.dumps(probe), "", 0))
    v = video_qa.tier0_structural(str(f), expect={"width": 1920})
    assert v["pass"] is False
    assert any("width" in e for e in v["errors"])


def test_tier0_missing_file():
    v = video_qa.tier0_structural("/nonexistent/clip.mp4")
    assert v["pass"] is False


def test_tier0_skipped_without_ffprobe(monkeypatch, tmp_path):
    monkeypatch.setattr(video_qa, "_have", lambda b: False)
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00" * 10)
    v = video_qa.tier0_structural(str(f))
    assert v["skipped"] is True and v["pass"] is False  # loud, never silent-pass


def test_tier1_detects_black_frames(fake_binaries, monkeypatch):
    stderr = ("[blackdetect @ 0x] black_start:0 black_end:1.2 black_duration:1.2\n"
              "[silencedetect @ 0x] silence_start: 2.0\n")
    monkeypatch.setattr(video_qa, "_run",
                        lambda cmd, timeout=600: FakeProc("", stderr, 0))
    v = video_qa.tier1_artifacts("clip.mp4")
    assert v["pass"] is False
    assert v["details"]["black_segments"] == 1
    assert "silence_note" in v["details"]  # silence warns, doesn't fail


def test_tier1_clean_pass(fake_binaries, monkeypatch):
    monkeypatch.setattr(video_qa, "_run",
                        lambda cmd, timeout=600: FakeProc("", "", 0))
    v = video_qa.tier1_artifacts("clip.mp4")
    assert v["pass"] is True


def test_tier2_ssim_psnr_thresholds(fake_binaries, monkeypatch):
    stderr = ("[Parsed_ssim_0 @ 0x] SSIM Y:0.962341 (26.5 dB) U:0.98 V:0.98 All:0.97\n"
              "[Parsed_psnr_1 @ 0x] PSNR y:34.21 average:33.0\n")
    monkeypatch.setattr(video_qa, "_run",
                        lambda cmd, timeout=600: FakeProc("", stderr, 0))
    v = video_qa.tier2_metrics("a.mp4", "ref.mp4")
    assert v["pass"] is True
    assert v["details"]["ssim_y"] > 0.95


def test_tier2_low_ssim_fails(fake_binaries, monkeypatch):
    stderr = ("[Parsed_ssim_0 @ 0x] SSIM Y:0.901234 (20.1 dB) U:0.95 V:0.95 All:0.93\n"
              "[Parsed_psnr_1 @ 0x] PSNR y:28.50 average:27.0\n")
    monkeypatch.setattr(video_qa, "_run",
                        lambda cmd, timeout=600: FakeProc("", stderr, 0))
    v = video_qa.tier2_metrics("a.mp4", "ref.mp4")
    assert v["pass"] is False
    assert any("SSIM" in e for e in v["errors"])


def test_tier3_extracts_at_decision_points(fake_binaries, monkeypatch, tmp_path):
    frames_dir = tmp_path / "frames"

    def fake_run(cmd, timeout=60):
        out = cmd[-1]
        open(out, "wb").write(b"png")
        return FakeProc("", "", 0)

    monkeypatch.setattr(video_qa, "_run", fake_run)
    v = video_qa.tier3_check_frames("clip.mp4", str(frames_dir),
                                    decision_points=[0.5, 4.8])
    assert v["pass"] is True
    assert len(v["details"]["frames"]) == 2
    assert "decision" in v["details"]["note"].lower() or "decision points" in v["details"]["note"]


def test_cli_exit_codes(fake_binaries, monkeypatch, tmp_path, capsys):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00" * 100)
    probe = {"streams": [{"width": 1920, "height": 1080, "codec_name": "h264",
                          "avg_frame_rate": "30/1"}],
             "format": {"duration": "5.0"}}
    monkeypatch.setattr(video_qa, "_run",
                        lambda cmd, timeout=60: FakeProc(json.dumps(probe), "", 0))
    # tier3 will fail (no frames written) -> overall fail -> exit 1
    rc = video_qa.main([str(f), "--frames-dir", str(tmp_path / "fr")])
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["overall_pass"] is False

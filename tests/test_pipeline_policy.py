"""Tests for the pipeline's policy gates: ToS refusals, privacy blocks,
egress blocks, and the local tools. Policy refusals must fail FAST (no
retry sleeps) and LOUD (StageFailed with a trail)."""

import json
import os
import shutil
import subprocess
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import pipeline as pipeline_mod
from pipeline import Pipeline, StageFailed
import fal_client

NEEDS_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def edl_with(job_id, stage_id, tool, args):
    return {
        "edl_version": 1,
        "job_id": job_id,
        "stages": [{"id": stage_id, "tool": tool, "args": args,
                    "idempotency_key": "k-" + stage_id}],
    }


def test_fal_submit_rejected_route_fails_fast_no_retry(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(pipeline_mod.time, "sleep", lambda s: sleeps.append(s))
    p = Pipeline(str(tmp_path / "jobs"))
    edl = edl_with("job-r", "gen", "fal_submit",
                   {"route": "kling-3", "duration_s": 5,
                    "arguments": {"prompt": "x"},
                    "allow_third_party": True})
    with pytest.raises(StageFailed, match="policy refusal"):
        p.run(edl)
    assert sleeps == [], "rejected ToS route must not burn retry sleeps"


def test_fal_submit_needs_explicit_third_party_consent(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(pipeline_mod.time, "sleep", lambda s: sleeps.append(s))
    p = Pipeline(str(tmp_path / "jobs"))
    edl = edl_with("job-p", "gen", "fal_submit",
                   {"route": "wan-25", "duration_s": 5,
                    "arguments": {"prompt": "x"}})  # no allow_third_party
    with pytest.raises(StageFailed, match="policy refusal"):
        p.run(edl)
    assert sleeps == []


def test_fal_submit_deepseek_brain_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod.time, "sleep", lambda s: None)
    p = Pipeline(str(tmp_path / "jobs"))
    edl = edl_with("job-d", "gen", "fal_submit",
                   {"route": "wan-25", "duration_s": 5,
                    "arguments": {"prompt": "x"},
                    "allow_third_party": True,
                    "brain": "deepseek/deepseek-v4.1-flash"})
    with pytest.raises(StageFailed, match="[Dd]eepseek"):
        p.run(edl)


def test_fal_submit_happy_path_with_consent(tmp_path, monkeypatch):
    monkeypatch.setenv("MYAI_EGRESS_LOG", str(tmp_path / "egress.jsonl"))

    def fake_submit(route, arguments, webhook_url=None, idempotency_key=None,
                    opener=None, tos_ack=False):
        assert route == "wan-25"
        return types.SimpleNamespace(request_id="req-mock")

    monkeypatch.setattr(fal_client, "submit", fake_submit)
    p = Pipeline(str(tmp_path / "jobs"))
    edl = edl_with("job-ok", "gen", "fal_submit",
                   {"route": "wan-25", "duration_s": 2,
                    "arguments": {"prompt": "x"},
                    "allow_third_party": True,
                    "sensitivity": "private"})
    manifest = p.run(edl)
    assert manifest["status"] == "done"
    assert manifest["stages"]["gen"]["result"]["request_id"] == "req-mock"
    assert manifest["stages"]["gen"]["result"]["estimate_usd"] == 0.20


def test_fal_submit_conditional_needs_tos_ack(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod.time, "sleep", lambda s: None)
    p = Pipeline(str(tmp_path / "jobs"))
    edl = edl_with("job-c", "gen", "fal_submit",
                   {"route": "veo3-fast", "duration_s": 5,
                    "arguments": {"prompt": "x"},
                    "allow_third_party": True})  # no tos_ack
    with pytest.raises(StageFailed, match="CONDITIONAL"):
        p.run(edl)


@NEEDS_FFMPEG
def test_local_fx_stage_runs_offline(tmp_path):
    src = str(tmp_path / "src.mp4")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", "-f", "lavfi",
         "-i", "testsrc=duration=2:size=640x360:rate=24",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", src],
        capture_output=True, check=True, timeout=120)
    p = Pipeline(str(tmp_path / "jobs"))
    edl = edl_with("job-fx", "fx", "local_fx",
                   {"recipe": "film_grain", "params": {"strength": 6},
                    "input": src, "filename": "grain.mp4"})
    manifest = p.run(edl)
    out = manifest["stages"]["fx"]["result"]["file"]
    assert os.path.exists(out)
    assert manifest["stages"]["fx"]["result"]["cost_usd"] == 0.0
    assert manifest["stages"]["fx"]["result"]["private"] is True


def test_local_gen_stage_refuses_loud_until_wired(tmp_path, monkeypatch):
    monkeypatch.delenv("MYAI_LOCAL_GEN", raising=False)
    monkeypatch.setattr(pipeline_mod.time, "sleep", lambda s: None)
    p = Pipeline(str(tmp_path / "jobs"))
    edl = edl_with("job-lg", "gen", "local_gen", {"prompt": "a red cube"})
    with pytest.raises(StageFailed, match="[Nn]ot [Ww]ired|[Nn]ot enabled"):
        p.run(edl)

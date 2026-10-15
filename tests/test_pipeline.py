"""Tests for tools/pipeline.py — EDL validation, resume, failure ladder."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from pipeline import Pipeline, validate_edl, EDLValidationError, StageFailed


def good_edl(job_id="job-1"):
    return {
        "edl_version": 1,
        "job_id": job_id,
        "stages": [
            {"id": "s1", "tool": "checkpoint", "args": {},
             "idempotency_key": "k1", "checkpoint": True},
            {"id": "s2", "tool": "checkpoint", "args": {},
             "idempotency_key": "k2"},
        ],
    }


def test_validate_edl_ok():
    assert validate_edl(good_edl())["job_id"] == "job-1"


def test_validate_edl_rejects():
    bad = good_edl()
    bad["edl_version"] = 99
    with pytest.raises(EDLValidationError, match="edl_version"):
        validate_edl(bad)
    bad2 = good_edl()
    bad2["stages"][1]["idempotency_key"] = "k1"  # duplicate
    with pytest.raises(EDLValidationError, match="Duplicate idempotency"):
        validate_edl(bad2)
    bad3 = good_edl()
    del bad3["stages"][0]["tool"]
    with pytest.raises(EDLValidationError, match="missing required field"):
        validate_edl(bad3)
    with pytest.raises(EDLValidationError):
        validate_edl({"edl_version": 1, "job_id": "x", "stages": []})


def test_run_completes_and_checkpoints(tmp_path):
    p = Pipeline(str(tmp_path / "jobs"))
    manifest = p.run(good_edl("job-a"))
    assert manifest["status"] == "done"
    assert os.path.exists(str(tmp_path / "jobs" / "job-a" / "checkpoints" / "s1.json"))
    for sub in ("inputs", "takes", "work", "qa", "final"):
        assert os.path.isdir(str(tmp_path / "jobs" / "job-a" / sub))


def test_resume_skips_completed_stages(tmp_path):
    p = Pipeline(str(tmp_path / "jobs"))
    calls = []

    def counting(args, ctx):
        calls.append(ctx["stage_id"])
        return {"n": len(calls)}

    p.register_tool("counting", counting)
    edl = good_edl("job-r")
    edl["stages"][0]["tool"] = "counting"
    edl["stages"][1]["tool"] = "counting"
    p.run(edl)
    assert calls == ["s1", "s2"]
    # Re-run: both stages already done -> nothing re-executes (idempotent).
    p2 = Pipeline(str(tmp_path / "jobs"))
    p2.register_tool("counting", counting)
    p2.run(edl)
    assert calls == ["s1", "s2"]


def test_transient_failure_retried_then_succeeds(tmp_path, monkeypatch):
    p = Pipeline(str(tmp_path / "jobs"), max_retries=3)
    monkeypatch.setattr("pipeline.time.sleep", lambda s: None)
    attempts = []

    def flaky(args, ctx):
        attempts.append(1)
        if len(attempts) < 3:
            raise StageFailed(ctx["stage_id"], "transient network blip")
        return {"ok": True}

    p.register_tool("flaky", flaky)
    edl = good_edl("job-f")
    edl["stages"] = [{"id": "s1", "tool": "flaky", "args": {},
                      "idempotency_key": "kf"}]
    manifest = p.run(edl)
    assert manifest["status"] == "done"
    assert len(attempts) == 3


def test_validation_failure_fails_fast_no_retry(tmp_path, monkeypatch):
    p = Pipeline(str(tmp_path / "jobs"), max_retries=5)
    monkeypatch.setattr("pipeline.time.sleep", lambda s: None)
    attempts = []

    def bad(args, ctx):
        attempts.append(1)
        raise StageFailed(ctx["stage_id"], "validation error: bad argument")

    p.register_tool("bad", bad)
    edl = good_edl("job-v")
    edl["stages"] = [{"id": "s1", "tool": "bad", "args": {},
                      "idempotency_key": "kv"}]
    with pytest.raises(StageFailed, match="validation error"):
        p.run(edl)
    assert len(attempts) == 1  # zero retries on validation failures


def test_unknown_tool_loud_failure(tmp_path):
    p = Pipeline(str(tmp_path / "jobs"))
    edl = good_edl("job-u")
    edl["stages"] = [{"id": "s1", "tool": "nope", "args": {},
                      "idempotency_key": "ku"}]
    with pytest.raises(StageFailed, match="unknown tool"):
        p.run(edl)


def test_fallback_degrade_path(tmp_path, monkeypatch):
    p = Pipeline(str(tmp_path / "jobs"), max_retries=0)
    monkeypatch.setattr("pipeline.time.sleep", lambda s: None)

    def primary(args, ctx):
        raise StageFailed(ctx["stage_id"], "transient outage")

    def fallback(args, ctx):
        return {"degraded": True}

    p.register_tool("primary", primary)
    p.register_tool("fallback", fallback)
    edl = good_edl("job-d")
    edl["stages"] = [{"id": "s1", "tool": "primary", "args": {},
                      "idempotency_key": "kd",
                      "fallback": {"tool": "fallback", "args": {}}}]
    manifest = p.run(edl)
    assert manifest["stages"]["s1"]["result"] == {"degraded": True}

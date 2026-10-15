"""Tests for tools/venice_client.py V2V extensions.

Payload-shape assertions use a mock opener — no network, no cost, no key
needed (VENICE_API_KEY is stubbed via env).
"""

import base64
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import venice_client as vc


@pytest.fixture(autouse=True)
def _stub_venice_key(monkeypatch):
    # Stub via monkeypatch (auto-undone per test) so the real env var
    # doesn't leak into other test modules running in the same process.
    monkeypatch.setenv("VENICE_API_KEY", "test-key-do-not-use")


class FakeResp:
    def __init__(self, body: dict, ctype="application/json"):
        self.status = 200
        self.headers = {"Content-Type": ctype}
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_opener(captured: dict, response: dict):
    def opener(req, timeout=None):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode())
        captured["auth"] = req.headers.get("Authorization")
        return FakeResp(response)
    return opener


def test_v2v_models_registered():
    models = vc.v2v_models()
    assert "venice-wan27-v2v" in models
    assert "venice-runway-v2v" in models
    assert "venice-seedance-r2v" in models
    # uncensored flag present on the Wan 2.7 line
    assert models["venice-wan27-v2v"]["uncensored"] is True


def test_quality_levels():
    assert set(vc.QUALITY_LEVELS) == {"draft", "standard", "maximum"}
    assert vc.DEFAULT_QUALITY == "standard"
    assert vc.QUALITY_LEVELS["maximum"]["resolution"] == "1080p"
    assert vc.QUALITY_LEVELS["draft"]["resolution"] == "480p"


def test_submit_v2v_sends_video_url(tmp_path):
    captured = {}
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp4 fake")
    job = vc.submit_v2v(
        "venice-wan27-v2v", prompt="restyle test",
        video_path=str(vid), duration_s=5.0,
        opener=make_opener(captured, {"request_id": "r1"}))
    assert job.request_id == "r1"
    p = captured["payload"]
    assert p["model"] == "wan-2-7-video-to-video"
    assert p["video_url"].startswith("data:video/mp4;base64,")
    # User text passes through, with David's standing consent
    # attestation appended (his explicit direction, Oct 14 2026).
    assert p["prompt"].startswith("restyle test")
    assert "Consent attestation:" in p["prompt"]
    assert "AI-generated or a consenting adult model" in p["prompt"]
    assert captured["auth"] == "Bearer test-key-do-not-use"


def test_submit_v2v_r2v_sends_reference_urls_with_extend(tmp_path):
    captured = {}
    vid = tmp_path / "prev.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp4 fake")
    vc.submit_v2v(
        "venice-seedance-r2v", prompt="continue the scene",
        video_path=str(vid), task_type="extend", duration_s=10.0,
        opener=make_opener(captured, {"request_id": "r2"}))
    p = captured["payload"]
    assert p["model"] == "seedance-2-0-reference-to-video"
    assert len(p["reference_video_urls"]) == 1
    assert p["reference_video_urls"][0].startswith("data:video/mp4;base64,")
    # Venice rejects omni_reference_task_type on Seedance 2.0 (HTTP 400);
    # submit_v2v must not send it even when a task_type is requested.
    assert "omni_reference_task_type" not in p
    assert "video_url" not in p


def test_submit_v2v_r2v_edit_task_type(tmp_path):
    captured = {}
    vid = tmp_path / "src.mp4"
    vid.write_bytes(b"fake")
    vc.submit_v2v(
        "venice-runway-v2v", prompt="anime style",
        video_path=str(vid),
        opener=make_opener(captured, {"request_id": "r3"}))
    p = captured["payload"]
    assert p["model"] == "runway-gen4-turbo-v2v"
    assert p["video_url"].startswith("data:video/mp4;base64,")


def test_submit_v2v_rejects_i2v_model():
    with pytest.raises(vc.VeniceValidationError, match="not a video-to-video"):
        vc.submit_v2v("venice-wan3", prompt="x", video_path="/tmp/x.mp4")


def test_submit_v2v_needs_video():
    with pytest.raises(vc.VeniceValidationError, match="needs a source video"):
        vc.submit_v2v("venice-wan27-v2v", prompt="x")


def test_file_to_data_url_size_cap(tmp_path):
    big = tmp_path / "big.mp4"
    big.write_bytes(b"x" * (vc.MAX_DATA_URL_BYTES + 1))
    with pytest.raises(ValueError, match="50MB"):
        vc.file_to_data_url(str(big))


def test_submit_passes_aspect_ratio():
    captured = {}
    vc.submit("venice-seedance-r2v",
              {"prompt": "x", "duration": "10s",
               "reference_video_urls": ["data:video/mp4;base64,AAA"],
               "omni_reference_task_type": "extend",
               "aspect_ratio": "adaptive"},
              opener=make_opener(captured, {"request_id": "r4"}))
    p = captured["payload"]
    assert p["aspect_ratio"] == "adaptive"
    assert p["omni_reference_task_type"] == "extend"


def test_estimate_cost():
    assert vc.estimate_cost("venice-wan27-v2v", 10.0) == round(0.13 * 10, 4)


def test_submit_answers_consent_challenge(tmp_path):
    import urllib.error
    calls = {"n": 0}
    captured = {}

    def opener(req, timeout=None):
        calls["n"] += 1
        captured["payload"] = json.loads(req.data.decode())
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                req.full_url, 409, "conflict", {},
                io.BytesIO(json.dumps(
                    {"policy_text": "attest likeness consent"}).encode()))
        return FakeResp({"request_id": "after-consent"})

    job = vc.submit("venice-seedance-r2v",
                    {"prompt": "replace the model",
                     "reference_video_urls": ["data:video/mp4;base64,AAA"],
                     "aspect_ratio": "adaptive"},
                    opener=opener)
    assert job.request_id == "after-consent"
    assert calls["n"] == 2
    assert "Consent attestation:" in captured["payload"]["prompt"]

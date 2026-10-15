"""Tests for animate_client (WaveSpeed Wan 2.2 Animate).

All HTTP is stubbed through the opener hook — no live calls, no key.
"""

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import animate_client as ac


class FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self, n=-1):
        if n is None or n < 0:
            out, self._b = self._b, b""
            return out
        out, self._b = self._b[:n], self._b[n:]
        return out

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("WAVESPEED_API_KEY", "test-key")


def _media(tmp_path):
    img = tmp_path / "model.jpg"
    img.write_bytes(b"\xff\xd8fake-jpeg")
    vid = tmp_path / "base.mp4"
    vid.write_bytes(b"fake-mp4")
    return str(img), str(vid)


def test_estimate_cost():
    assert ac.estimate_cost(5.0, "720p") == 0.40
    assert ac.estimate_cost(5.0, "480p") == 0.20
    assert ac.estimate_cost(1.0, "720p") == 0.24  # 3s minimum billed
    assert ac.estimate_cost(7.2, "720p") == round(8 * 0.08, 2)  # rounds up


def test_submit_replace_payload(tmp_path):
    img, vid = _media(tmp_path)
    seen = {}

    def opener(req):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return FakeResp({"data": {"id": "job-123"}})

    job = ac.submit_replace(img, vid, opener=opener)
    assert job.job_id == "job-123"
    assert seen["url"].endswith("/wavespeed-ai/wan-2.2/animate")
    body = seen["body"]
    assert body["mode"] == "replace"
    assert body["enable_safety_checker"] is False
    assert body["image"].startswith("data:image/jpeg;base64,")
    assert body["video"].startswith("data:video/mp4;base64,")


def test_poll_completed_extracts_url():
    def opener(req):
        return FakeResp({"data": {"status": "completed",
                                  "outputs": ["https://x/out.mp4"]}})

    st = ac.poll(ac.AnimateJob("j1"), opener=opener)
    assert st["status"] == "completed"
    assert st["url"] == "https://x/out.mp4"


def test_wait_returns_url(monkeypatch):
    monkeypatch.setattr(ac.time, "sleep", lambda s: None)
    seq = iter([
        {"data": {"status": "processing"}},
        {"data": {"status": "completed", "outputs": ["https://x/o.mp4"]}},
    ])

    def opener(req):
        return FakeResp(next(seq))

    url = ac.wait(ac.AnimateJob("j2"), opener=opener)
    assert url == "https://x/o.mp4"


def test_wait_failed_raises(monkeypatch):
    monkeypatch.setattr(ac.time, "sleep", lambda s: None)

    def opener(req):
        return FakeResp({"data": {"status": "failed", "error": "boom"}})

    with pytest.raises(ac.AnimateError, match="failed"):
        ac.wait(ac.AnimateJob("j3"), opener=opener)


def test_missing_key_raises_setup_message(monkeypatch, tmp_path):
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ac.AnimateAuthError, match="wavespeed.ai"):
        ac.get_api_key()


def test_key_from_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    d = tmp_path / ".nanobot"
    d.mkdir()
    (d / "wavespeed.env").write_text("WAVESPEED_API_KEY=file-key-9\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert ac.get_api_key() == "file-key-9"

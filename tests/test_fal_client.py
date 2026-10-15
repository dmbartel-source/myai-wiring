"""Tests for tools/fal_client.py — mocked fal.ai, no network."""

import io
import json
import os
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import fal_client
from fal_client import (
    FalAuthError, FalTransientError, FalValidationError, FalTosError,
    estimate_cost_usd, make_idempotency_key, submit, poll_status,
    webhook_payload_ok, verify_webhook_signature, FalJob, JobStatus,
    require_route, MODEL_REGISTRY, MODEL_ROUTES,
)
from net import EgressBlocked
import net as netmod


class FakeResp:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status = status

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_opener(responses):
    """responses: list of FakeResp or Exception, consumed in order."""
    calls = {"n": 0}

    def opener(req, timeout=None):
        calls["n"] += 1
        r = responses[min(calls["n"] - 1, len(responses) - 1)]
        if isinstance(r, Exception):
            raise r
        return r

    opener.calls = calls
    return opener


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("FAL_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def isolated_egress_log(tmp_path, monkeypatch):
    monkeypatch.setenv("MYAI_EGRESS_LOG", str(tmp_path / "egress.jsonl"))
    netmod.set_egress_log(None)
    yield
    netmod.set_egress_log(None)


def test_missing_key_fails_cleanly():
    with pytest.raises(FalAuthError, match="FAL_API_KEY is not set"):
        fal_client.get_api_key()


def test_estimate_math():
    assert estimate_cost_usd("wan-25", 5.0) == 0.5
    assert estimate_cost_usd("seedance-mini", 10.0) == 3.0
    with pytest.raises(FalValidationError):
        estimate_cost_usd("nope", 5.0)


def test_rejected_routes_refused_outright():
    # Kling / Pika / MiniMax-Hailuo: ToS review FAILED. No estimate, no
    # submit, no override — the refusal must name the reason.
    for route in ("kling-3", "pika-2", "minimax-hailuo"):
        assert MODEL_REGISTRY[route]["tos"] == "rejected"
        assert route not in MODEL_ROUTES  # not an active route
        with pytest.raises(FalTosError, match="REJECTED"):
            estimate_cost_usd(route, 5.0)
        with pytest.raises(FalTosError, match="REJECTED"):
            require_route(route)
        with pytest.raises(FalTosError, match="REJECTED"):
            require_route(route, tos_ack=True)  # ack changes nothing


def test_conditional_routes_need_tos_ack(monkeypatch):
    with pytest.raises(FalTosError, match="CONDITIONAL"):
        estimate_cost_usd("veo3-fast", 5.0)
    assert estimate_cost_usd("veo3-fast", 5.0, tos_ack=True) == 0.75
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    opener = make_opener([FakeResp({"request_id": "req-c"})])
    with pytest.raises(FalTosError, match="CONDITIONAL"):
        submit("veo3-fast", {"prompt": "x"}, opener=opener)
    job = submit("veo3-fast", {"prompt": "x"}, opener=opener, tos_ack=True)
    assert job.request_id == "req-c"


def test_submit_rejected_route_never_reaches_network(monkeypatch):
    monkeypatch.setenv("FAL_API_KEY", "test-key")

    def opener(req, timeout=None):  # pragma: no cover - must never run
        raise AssertionError("rejected route must not touch the network")

    with pytest.raises(FalTosError):
        submit("kling-3", {"prompt": "x"}, opener=opener)


def test_webhook_url_must_be_ours_and_https(monkeypatch):
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    opener = make_opener([FakeResp({"request_id": "req-w"})])
    with pytest.raises(FalValidationError, match="https"):
        submit("wan-25", {"prompt": "x"}, opener=opener,
               webhook_url="http://myserver.example/hook")
    with pytest.raises(FalValidationError, match="credentials"):
        submit("wan-25", {"prompt": "x"}, opener=opener,
               webhook_url="https://user:pass@myserver.example/hook")
    job = submit("wan-25", {"prompt": "x"}, opener=opener,
                 webhook_url="https://myserver.example/hook")
    assert job.request_id == "req-w"


def test_egress_gate_enforced_even_with_injected_opener(monkeypatch):
    # _http_json with a non-allowlisted URL must raise EgressBlocked even
    # though the transport is mocked.
    def opener(req, timeout=None):  # pragma: no cover - must never run
        raise AssertionError("blocked host must not reach the transport")

    with pytest.raises(EgressBlocked):
        fal_client._http_json("GET", "https://evil.example/x", "k",
                              opener=opener)


def test_idempotency_key_deterministic():
    assert make_idempotency_key("a") == make_idempotency_key("a")
    assert make_idempotency_key("a") != make_idempotency_key("b")


def test_submit_success(monkeypatch):
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    opener = make_opener([FakeResp({"request_id": "req-123"})])
    job = submit("wan-25", {"prompt": "x", "duration": 5}, opener=opener)
    assert job.request_id == "req-123"
    assert job.status == JobStatus.IN_QUEUE
    assert opener.calls["n"] == 1


def test_submit_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    err = urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b"busy"))
    opener = make_opener([err, FakeResp({"request_id": "req-9"})])
    job = submit("wan-25", {"prompt": "x"}, opener=opener)
    assert job.request_id == "req-9"
    assert opener.calls["n"] == 2  # one retry happened


def test_submit_validation_error_never_retried(monkeypatch):
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    err = urllib.error.HTTPError("u", 422, "bad", {}, io.BytesIO(b"bad args"))
    opener = make_opener([err])
    with pytest.raises(FalValidationError):
        submit("wan-25", {"prompt": "x"}, opener=opener)
    assert opener.calls["n"] == 1  # zero retries


def test_submit_auth_error(monkeypatch):
    monkeypatch.setenv("FAL_API_KEY", "bad-key")
    err = urllib.error.HTTPError("u", 401, "nope", {}, io.BytesIO(b"nope"))
    opener = make_opener([err])
    with pytest.raises(FalAuthError):
        submit("wan-25", {"prompt": "x"}, opener=opener)


def test_poll_status_transitions(monkeypatch):
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    job = FalJob(request_id="r1", endpoint="fal-ai/wan-25/text-to-video",
                 idempotency_key="k")
    opener = make_opener([
        FakeResp({"status": "IN_PROGRESS"}),
        FakeResp({"status": "COMPLETED"}),
    ])
    assert poll_status(job, opener) == JobStatus.IN_PROGRESS
    assert poll_status(job, opener) == JobStatus.COMPLETED


def test_poll_failed_validation(monkeypatch):
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    job = FalJob(request_id="r1", endpoint="e", idempotency_key="k")
    opener = make_opener([FakeResp({"status": "FAILED",
                                    "error": {"code": 422, "message": "validation"}})])
    with pytest.raises(FalValidationError):
        poll_status(job, opener)


def test_wait_timeout(monkeypatch):
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    import fal_client as fc
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    job = FalJob(request_id="r1", endpoint="e", idempotency_key="k")
    opener = make_opener([FakeResp({"status": "IN_PROGRESS"})] * 50)
    # force immediate deadline by monkeypatching monotonic
    t = {"now": 0.0}
    monkeypatch.setattr(fc.time, "monotonic", lambda: t.__setitem__("now", t["now"] + 100.0) or t["now"])
    with pytest.raises(fc.FalTimeoutError):
        fc.wait(job, timeout_s=10, poll_interval_s=1, opener=opener)


def test_webhook_payload_ok():
    assert webhook_payload_ok({"status": "OK", "request_id": "r"}) is True
    assert webhook_payload_ok({"status": "ERROR", "request_id": "r"}) is False
    assert webhook_payload_ok({"status": "OK"}) is False


def test_webhook_verify_fails_closed_without_pynacl():
    # PyNaCl is not installed in this env: verification must fail closed.
    with pytest.raises(fal_client.FalError, match="PyNaCl"):
        verify_webhook_signature(b"{}", "sig", {"kty": "OKP", "crv": "Ed25519", "x": "abc"})

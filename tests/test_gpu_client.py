"""Tests for tools/gpu_client.py — ComfyUI client for David's RunPod pod.

Mock-opener style like test_venice_client.py: no network, no pod. The
egress gate still validates every URL (the policy is not mocked), so
tests use a proxy.runpod.net base URL — the allowlisted pod host.
"""

import json
import os
import sys
import urllib.error
import urllib.parse

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import net
import gpu_client as gc


BASE = "https://testpod-8188.proxy.runpod.net"


@pytest.fixture(autouse=True)
def _gpu_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_BASE_URL", BASE)
    monkeypatch.delenv("GPU_API_KEY", raising=False)
    monkeypatch.setenv("MYAI_EGRESS_LOG", str(tmp_path / "egress.jsonl"))
    net.set_egress_log(None)
    yield
    net.set_egress_log(None)


class FakeResp:
    def __init__(self, body, ctype="application/json", status=200):
        self.status = status
        self.headers = {"Content-Type": ctype}
        self._body = body if isinstance(body, bytes) else json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


HISTORY_OK = {
    "pid-1": {
        "status": {"status_str": "success", "completed": True,
                   "messages": []},
        "outputs": {
            "11": {"images": [{"filename": "edit_00001_.png",
                               "subfolder": "myai", "type": "output"}]},
            "14": {"gifs": [{"filename": "i2v_00001_.mp4",
                             "subfolder": "myai", "type": "output"}]},
        },
    }
}


def make_router(captured: dict, history=HISTORY_OK):
    """Opener routing by path, recording every request in `captured`."""
    def opener(req, timeout=None):
        url = req.full_url
        path = urllib.parse.urlsplit(url).path
        captured.setdefault("requests", []).append((req.method, path))
        captured["last_req"] = req
        if path == "/upload/image":
            assert req.headers.get("Content-type", "").startswith(
                "multipart/form-data")
            return FakeResp({"name": "up_00001.png", "subfolder": "",
                             "type": "input"})
        if path == "/prompt":
            payload = json.loads(req.data.decode())
            captured["prompt_payload"] = payload
            return FakeResp({"prompt_id": "pid-1", "node_errors": {}})
        if path == "/history/pid-1":
            return FakeResp(history)
        if path == "/queue":
            return FakeResp({"queue_running": [], "queue_pending": []})
        if path == "/view":
            return FakeResp(b"FAKE-MEDIA-BYTES", ctype="video/mp4")
        if path == "/interrupt":
            return FakeResp({})
        if path == "/system_stats":
            return FakeResp({"system": {"os": "linux"}, "devices": []})
        raise AssertionError(f"unexpected request: {req.method} {url}")
    return opener


# ------------------------------------------------------------------ config --

def test_base_url_from_env_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("GPU_BASE_URL", BASE + "/")
    assert gc.get_base_url() == BASE


def test_base_url_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / "gpu.env"
    env_file.write_text(f"GPU_BASE_URL={BASE}/\nGPU_API_KEY=file-key\n")
    monkeypatch.delenv("GPU_BASE_URL")
    monkeypatch.delenv("GPU_API_KEY", raising=False)
    monkeypatch.setattr(gc, "ENV_PATH", str(env_file))
    assert gc.get_base_url() == BASE
    assert gc.get_api_key() == "file-key"


def test_not_configured_raises_typed_error(tmp_path, monkeypatch):
    monkeypatch.delenv("GPU_BASE_URL")
    monkeypatch.setattr(gc, "ENV_PATH", str(tmp_path / "missing.env"))
    with pytest.raises(gc.GpuNotConfigured):
        gc.get_base_url()
    assert gc.is_configured() is False


def test_estimate_cost_uses_hourly_rate():
    # Default H100 Secure mid-range $2.89/hr; 20 min render ~ $0.96.
    assert gc.estimate_cost_usd(1200.0) == pytest.approx(0.9633, abs=1e-3)
    assert gc.estimate_cost_usd(0.0) == 0.0


# ------------------------------------------------------------------ upload --

def test_upload_image_returns_server_filename(tmp_path):
    img = tmp_path / "photo.png"
    img.write_bytes(b"\x89PNG fake")
    captured = {}
    name = gc.upload_image(str(img), opener=make_router(captured))
    assert name == "up_00001.png"
    assert ("POST", "/upload/image") in captured["requests"]


def test_upload_video_returns_server_filename(tmp_path):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp4 fake")
    captured = {}
    name = gc.upload_video(str(vid), opener=make_router(captured))
    assert name == "up_00001.png"  # router's canned name


def test_upload_rejects_wrong_kind(tmp_path):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp4 fake")
    with pytest.raises(gc.GpuValidationError):
        gc.upload_image(str(vid), opener=make_router({}))


def test_upload_missing_file():
    with pytest.raises(gc.GpuValidationError):
        gc.upload_image("/nonexistent/photo.png", opener=make_router({}))


# ------------------------------------------------------------------- queue --

def test_queue_workflow_returns_job():
    captured = {}
    job = gc.queue_workflow({"1": {"class_type": "SaveImage", "inputs": {}}},
                            opener=make_router(captured))
    assert job.prompt_id == "pid-1"
    assert job.status == gc.JobStatus.IN_QUEUE
    # client_id is attached for ComfyUI's websocket bookkeeping
    assert captured["prompt_payload"]["client_id"]
    assert "1" in captured["prompt_payload"]["prompt"]


def test_queue_workflow_node_errors_raise():
    def opener(req, timeout=None):
        return FakeResp({"prompt_id": "pid-x",
                         "node_errors": {"9": {"errors": ["bad"]}}})
    with pytest.raises(gc.GpuValidationError):
        gc.queue_workflow({"9": {"class_type": "KSampler", "inputs": {}}},
                          opener=opener)


def test_queue_workflow_rejects_empty():
    with pytest.raises(gc.GpuValidationError):
        gc.queue_workflow({}, opener=make_router({}))


def test_api_key_header_sent_when_configured(monkeypatch):
    monkeypatch.setenv("GPU_API_KEY", "pod-key")
    captured = {}
    gc.queue_workflow({"1": {"class_type": "SaveImage", "inputs": {}}},
                      opener=make_router(captured))
    assert captured["last_req"].headers.get("Authorization") == \
        "Bearer pod-key"


# -------------------------------------------------------------------- poll --

def test_poll_complete_sets_outputs():
    job = gc.GpuJob(prompt_id="pid-1")
    status = gc.poll_status(job, opener=make_router({}))
    assert status == gc.JobStatus.COMPLETED
    assert job.outputs["11"]["images"][0]["filename"] == "edit_00001_.png"
    files = list(gc.iter_output_files(job.outputs))
    assert ("i2v_00001_.mp4", "myai", "output") in files


def test_poll_error_raises_job_failed():
    history = {"pid-1": {
        "status": {"status_str": "error", "completed": True,
                   "messages": [["execution_start", {}],
                                ["execution_error",
                                 {"node_id": "9", "node_type": "KSampler",
                                  "exception_type": "RuntimeError",
                                  "exception_message": "CUDA out of memory"}]]},
        "outputs": {}}}
    job = gc.GpuJob(prompt_id="pid-1")
    with pytest.raises(gc.GpuJobFailedError, match="out of memory"):
        gc.poll_status(job, opener=make_router({}, history=history))
    assert job.status == gc.JobStatus.FAILED


def test_poll_still_queued():
    def opener(req, timeout=None):
        path = urllib.parse.urlsplit(req.full_url).path
        if path == "/history/pid-1":
            return FakeResp({})
        if path == "/queue":
            return FakeResp({
                "queue_running": [],
                "queue_pending": [[3, "pid-1", {}, {}, []]]})
        raise AssertionError(path)
    job = gc.GpuJob(prompt_id="pid-1")
    assert gc.poll_status(job, opener=opener) == gc.JobStatus.IN_QUEUE


def test_wait_returns_outputs():
    job = gc.GpuJob(prompt_id="pid-1")
    outputs = gc.wait(job, timeout_s=30, poll_interval_s=0.01,
                      opener=make_router({}))
    assert outputs is job.outputs
    assert job.status == gc.JobStatus.COMPLETED


# -------------------------------------------------------------- unreachable --

def _dead_opener(req, timeout=None):
    raise urllib.error.URLError("Connection refused")


def test_unreachable_pod_raises_typed_error():
    with pytest.raises(gc.GpuUnreachableError):
        gc.health(opener=_dead_opener)
    with pytest.raises(gc.GpuUnreachableError):
        gc.queue_workflow({"1": {"class_type": "SaveImage", "inputs": {}}},
                          opener=_dead_opener)


def test_http_error_mapping():
    def opener_500(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 503, "down", {},
                                     __import__("io").BytesIO(b"busy"))
    with pytest.raises(gc.GpuTransientError):
        gc.health(opener=opener_500)

    def opener_401(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "no", {},
                                     __import__("io").BytesIO(b"denied"))
    with pytest.raises(gc.GpuAuthError):
        gc.health(opener=opener_401)


# ---------------------------------------------------------------- end-to-end --

def test_run_workflow_downloads_results(tmp_path):
    dest = tmp_path / "out"
    saved = gc.run_workflow(
        {"1": {"class_type": "SaveImage", "inputs": {}}}, str(dest),
        timeout_s=30, poll_interval_s=0.01, opener=make_router({}))
    names = sorted(os.path.basename(p) for p in saved)
    assert names == ["edit_00001_.png", "i2v_00001_.mp4"]
    for p in saved:
        with open(p, "rb") as f:
            assert f.read() == b"FAKE-MEDIA-BYTES"


def test_interrupt_posts_to_pod():
    captured = {}
    gc.interrupt(opener=make_router(captured))
    assert ("POST", "/interrupt") in captured["requests"]

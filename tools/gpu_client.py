#!/usr/bin/env python3
"""
Self-hosted GPU client for MYAI Studio — talks to David's own ComfyUI
instance running on a rented RunPod pod (see runpod/README.md).

Mirrors the venice_client.py interface shape (submit -> poll -> result,
typed errors, `opener` test seam) so it drops into the studio the same
way. The differences that matter:

  - No third-party API and no per-generation charge. The "provider" is
    David's own pod; media flows ONLY between the Studio server and that
    pod, nowhere else.
  - The pod is rented by the hour and is OFF most of the time. An
    unreachable pod is a normal state, not a bug: every call raises
    GpuUnreachableError (or GpuNotConfigured when gpu.env is missing)
    and the studio reports "pod is off" instead of crashing.
  - All HTTP goes through tools/net.py's egress gate. The RunPod proxy
    host (proxy.runpod.net) is on the allowlist for exactly this client.

ComfyUI API used:
  POST /upload/image   multipart file -> {name, subfolder, type}
  POST /prompt         {prompt, client_id} -> {prompt_id, node_errors}
  GET  /history/<id>   execution result + output file references
  GET  /queue          pending/running prompts (status before history)
  GET  /view           ?filename&subfolder&type -> result bytes
  POST /interrupt      cancel the currently running prompt
  GET  /system_stats   liveness / device info

Config: ~/.nanobot/gpu.env (or environment):
  GPU_BASE_URL   e.g. https://<pod-id>-8188.proxy.runpod.net
  GPU_API_KEY    optional; sent as Authorization: Bearer when set
                 (only needed if the pod is put behind an auth proxy)
  GPU_HOURLY_USD optional; pod rental rate for cost estimates
                 (default: H100 Secure Cloud mid-range, $2.89/hr)
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

try:
    from . import net
except ImportError:  # tools/ used as plain scripts dir
    import net


ENV_PATH = os.path.expanduser("~/.nanobot/gpu.env")

# H100 80GB SXM, RunPod Secure Cloud on-demand (Oct 2026 range
# $2.89-$3.49/hr). Used only for studio cost *estimates* — the real
# charge is pod rental time, billed by RunPod, not per generation.
DEFAULT_HOURLY_USD = 2.89

# Rough H100 wall-clock expectations per studio option, for estimates.
EST_RENDER_S = {
    "gpu_image_edit": 90.0,     # Qwen-Image-Edit-2509, ~1-2 min
    "gpu_i2v": 1200.0,          # Wan 2.2 14B dual-pass, ~10-20 min
    "gpu_t2v": 1200.0,
    "gpu_replace": 1200.0,      # VACE 14B
    "raw": 600.0,
}


class GpuError(Exception):
    """Base GPU/ComfyUI error."""


class GpuNotConfigured(GpuError):
    """gpu.env / GPU_BASE_URL missing — pod was never set up."""


class GpuUnreachableError(GpuError):
    """Pod did not answer (off, expired, or wrong URL). Normal state."""


class GpuAuthError(GpuError):
    """Pod rejected the API key (401/403)."""


class GpuValidationError(GpuError):
    """Bad request (4xx) — workflow or upload rejected by ComfyUI."""


class GpuTransientError(GpuError):
    """Retryable (429, 5xx)."""


class GpuJobFailedError(GpuError):
    """ComfyUI executed the workflow and a node errored."""


class JobStatus(Enum):
    IN_QUEUE = "IN_QUEUE"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class GpuJob:
    prompt_id: str
    status: JobStatus = JobStatus.IN_QUEUE
    outputs: dict = field(default_factory=dict)  # node_id -> output dict
    attempts: int = 0


# ------------------------------------------------------------------ config --

def _load_env_file() -> dict:
    """Parse ~/.nanobot/gpu.env (KEY=VALUE lines) -> dict. {} if absent."""
    out: dict = {}
    try:
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return out


def get_base_url() -> str:
    """ComfyUI base URL for David's pod. Raises GpuNotConfigured."""
    url = os.environ.get("GPU_BASE_URL", "").strip()
    if not url:
        url = _load_env_file().get("GPU_BASE_URL", "").strip()
    if not url:
        raise GpuNotConfigured(
            "GPU_BASE_URL is not set. Create ~/.nanobot/gpu.env with the "
            "pod's ComfyUI proxy URL (see runpod/README.md), or set the "
            "GPU_BASE_URL environment variable.")
    return url.rstrip("/")


def get_api_key() -> str:
    """Optional pod API key ('' when unset — plain ComfyUI has no auth)."""
    key = os.environ.get("GPU_API_KEY", "").strip()
    if not key:
        key = _load_env_file().get("GPU_API_KEY", "").strip()
    return key


def hourly_rate_usd() -> float:
    raw = os.environ.get("GPU_HOURLY_USD", "").strip()
    if not raw:
        raw = _load_env_file().get("GPU_HOURLY_USD", "").strip()
    try:
        return float(raw) if raw else DEFAULT_HOURLY_USD
    except ValueError:
        return DEFAULT_HOURLY_USD


def estimate_cost_usd(render_seconds: float) -> float:
    """Pod-rental share of a render: rate x wall-clock. Estimate only."""
    return round(hourly_rate_usd() * float(render_seconds) / 3600.0, 4)


def is_configured() -> bool:
    try:
        get_base_url()
        return True
    except GpuNotConfigured:
        return False


# -------------------------------------------------------------------- http --

def _request(
    method: str,
    path: str,
    data: Optional[bytes] = None,
    content_type: Optional[str] = None,
    timeout: float = 60.0,
    opener: Optional[Callable] = None,
) -> tuple[int, bytes, str]:
    """One HTTP call to the pod. Returns (status, raw_body, content_type).

    Raises the typed Gpu* errors; connection failures become
    GpuUnreachableError so callers can say "pod is off".
    """
    url = f"{get_base_url()}{path}"
    req = urllib.request.Request(url, data=data, method=method)
    key = get_api_key()
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with net.gated_urlopen(req, timeout=timeout, opener=opener) as resp:
            return resp.status, resp.read(), resp.headers.get(
                "Content-Type", "")
    except net.EgressBlocked:
        raise
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        if e.code in (401, 403):
            raise GpuAuthError(f"pod auth failed (HTTP {e.code}): {body}")
        if e.code == 429 or 500 <= e.code < 600:
            raise GpuTransientError(
                f"pod transient error (HTTP {e.code}): {body}")
        raise GpuValidationError(
            f"pod rejected request (HTTP {e.code}): {body}")
    except (urllib.error.URLError, ConnectionError, TimeoutError,
            OSError) as e:
        raise GpuUnreachableError(
            f"GPU pod unreachable at {get_base_url()} ({e}). "
            "The pod is probably stopped — start it in the RunPod "
            "console, or check GPU_BASE_URL in ~/.nanobot/gpu.env.")


def _request_json(method: str, path: str, payload: Optional[dict] = None,
                  timeout: float = 60.0,
                  opener: Optional[Callable] = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    _, raw, _ = _request(method, path, data=data,
                         content_type="application/json" if data else None,
                         timeout=timeout, opener=opener)
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise GpuValidationError(
            f"pod returned non-JSON for {method} {path}: {raw[:200]!r}")
    if not isinstance(body, dict):
        raise GpuValidationError(
            f"pod returned unexpected JSON for {method} {path}: {body!r}"[:300])
    return body


def health(opener: Optional[Callable] = None) -> dict:
    """GET /system_stats. Raises GpuUnreachableError when the pod is off."""
    return _request_json("GET", "/system_stats", timeout=15.0, opener=opener)


# ------------------------------------------------------------------ upload --

def _multipart_encode(fields: dict, file_field: str, filename: str,
                      data: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = f"----myai-gpu-{uuid.uuid4().hex[:12]}"
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'
                   .encode())
        body.write(f"{v}\r\n".encode())
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{filename}"\r\n'.encode())
    body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.write(data)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    return body.getvalue(), boundary


def _upload_media(path: str, kind: str,
                  opener: Optional[Callable] = None) -> str:
    """Upload one media file to the pod's ComfyUI input dir.

    ComfyUI's /upload/image endpoint is its generic file intake (it
    stores any file under input/; VHS and the core loaders read video
    files from there too). Returns the server-side filename to use in
    LoadImage / VHS_LoadVideo nodes.
    """
    if not os.path.isfile(path):
        raise GpuValidationError(f"media not found: {path}")
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    if not ctype.startswith(kind + "/"):
        article = "an" if kind == "image" else "a"
        raise GpuValidationError(
            f"not {article} {kind} (content-type {ctype}): {path}")
    with open(path, "rb") as f:
        data = f.read()
    filename = os.path.basename(path)
    payload, boundary = _multipart_encode(
        {"type": "input", "overwrite": "true"},
        "image", filename, data, ctype)
    # /upload/image is multipart, not JSON — go through _request directly.
    _, raw, _ = _request(
        "POST", "/upload/image", data=payload,
        content_type=f"multipart/form-data; boundary={boundary}",
        timeout=600.0, opener=opener)
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise GpuValidationError(f"upload returned non-JSON: {raw[:200]!r}")
    name = body.get("name")
    if not name:
        raise GpuValidationError(f"upload returned no filename: {body}")
    return name


def upload_image(path: str, opener: Optional[Callable] = None) -> str:
    """Upload an image to the pod. Returns the ComfyUI input filename."""
    return _upload_media(path, "image", opener=opener)


def upload_video(path: str, opener: Optional[Callable] = None) -> str:
    """Upload a video to the pod. Returns the ComfyUI input filename."""
    return _upload_media(path, "video", opener=opener)


# ------------------------------------------------------------------- queue --

def queue_workflow(workflow: dict,
                   opener: Optional[Callable] = None) -> GpuJob:
    """Queue an arbitrary ComfyUI API-format workflow dict (POST /prompt).

    This is the raw escape hatch: any workflow ComfyUI can run, this
    runs — full creative control lives here. Typed builders in
    gpu_workflows.py cover the common jobs.
    """
    if not isinstance(workflow, dict) or not workflow:
        raise GpuValidationError("workflow must be a non-empty dict")
    payload = {"prompt": workflow, "client_id": uuid.uuid4().hex}
    last_err: Optional[GpuError] = None
    for attempt in range(3):
        try:
            body = _request_json("POST", "/prompt", payload,
                                 timeout=60.0, opener=opener)
            node_errors = body.get("node_errors") or {}
            if node_errors:
                raise GpuValidationError(
                    f"ComfyUI rejected the workflow: {node_errors}"[:500])
            prompt_id = body.get("prompt_id")
            if not prompt_id:
                raise GpuValidationError(
                    f"ComfyUI returned no prompt_id: {body}"[:300])
            return GpuJob(prompt_id=prompt_id, attempts=attempt + 1)
        except GpuTransientError as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise last_err or GpuError("ComfyUI queue failed")


def _history_entry(job: GpuJob,
                   opener: Optional[Callable] = None) -> Optional[dict]:
    body = _request_json("GET", f"/history/{job.prompt_id}",
                         timeout=30.0, opener=opener)
    return body.get(job.prompt_id)


def poll_status(job: GpuJob,
                opener: Optional[Callable] = None) -> JobStatus:
    """Check a queued workflow. Sets job.outputs when complete.

    Raises GpuJobFailedError when ComfyUI reports a node error.
    """
    entry = _history_entry(job, opener)
    if entry is None:
        # Not in history yet: distinguish queued vs running via /queue.
        q = _request_json("GET", "/queue", timeout=30.0, opener=opener)
        running = [item[1] for item in q.get("queue_running", [])
                   if isinstance(item, (list, tuple)) and len(item) > 1]
        job.status = (JobStatus.PROCESSING if job.prompt_id in running
                      else JobStatus.IN_QUEUE)
        return job.status
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        detail = ""
        for kind, payload in status.get("messages", []):
            if kind == "execution_error" and isinstance(payload, dict):
                detail = (f"{payload.get('node_type', '?')} node "
                          f"{payload.get('node_id', '?')}: "
                          f"{payload.get('exception_type', '')} "
                          f"{payload.get('exception_message', '')}")
                break
        job.status = JobStatus.FAILED
        raise GpuJobFailedError(
            f"ComfyUI workflow failed ({detail or status})"[:500])
    if status.get("completed"):
        job.status = JobStatus.COMPLETED
        job.outputs = entry.get("outputs") or {}
        return job.status
    job.status = JobStatus.PROCESSING
    return job.status


def wait(job: GpuJob, timeout_s: float = 1800.0, poll_interval_s: float = 5.0,
         opener: Optional[Callable] = None,
         progress_cb: Optional[Callable] = None) -> dict:
    """Poll until completion. Returns the outputs dict (node_id -> files).

    Raises GpuJobFailedError on node errors, GpuTransientError on
    timeout. progress_cb(job) is invoked after every poll — the studio
    uses it for phase updates and cancellation checks.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status = poll_status(job, opener)
        except GpuTransientError:
            time.sleep(poll_interval_s)
            continue
        if progress_cb is not None:
            progress_cb(job)
        if status == JobStatus.COMPLETED:
            return job.outputs
        time.sleep(poll_interval_s)
    raise GpuTransientError(
        f"GPU job {job.prompt_id} timed out after {timeout_s}s "
        "(it may still finish on the pod — check the ComfyUI UI)")


def interrupt(opener: Optional[Callable] = None) -> None:
    """Cancel the currently running workflow on the pod (POST /interrupt)."""
    _request("POST", "/interrupt", data=b"{}", 
             content_type="application/json", timeout=30.0, opener=opener)


# ----------------------------------------------------------------- results --

_OUTPUT_KEYS = ("images", "gifs", "videos", "video", "audio")


def iter_output_files(outputs: dict):
    """Yield (filename, subfolder, type) for every file in an outputs dict."""
    for node_out in (outputs or {}).values():
        if not isinstance(node_out, dict):
            continue
        for key in _OUTPUT_KEYS:
            entries = node_out.get(key)
            if isinstance(entries, dict):
                entries = [entries]
            if not isinstance(entries, list):
                continue
            for e in entries:
                if isinstance(e, dict) and e.get("filename"):
                    yield (e["filename"], e.get("subfolder", ""),
                           e.get("type", "output"))


def download_output(filename: str, subfolder: str, otype: str,
                    dest_path: str,
                    opener: Optional[Callable] = None) -> str:
    """GET /view one result file -> dest_path (atomic: temp + rename)."""
    import tempfile
    qs = urllib.parse.urlencode({"filename": filename,
                                 "subfolder": subfolder, "type": otype})
    _, raw, _ = _request("GET", f"/view?{qs}", timeout=600.0, opener=opener)
    fd, tmp = tempfile.mkstemp(prefix=".dl-",
                               dir=os.path.dirname(dest_path) or ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, dest_path)
    return dest_path


def download_outputs(job: GpuJob, dest_dir: str,
                     opener: Optional[Callable] = None) -> list:
    """Download every output file of a completed job into dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    saved = []
    for filename, subfolder, otype in iter_output_files(job.outputs):
        dest = os.path.join(dest_dir, os.path.basename(filename))
        saved.append(download_output(filename, subfolder, otype, dest,
                                     opener=opener))
    return saved


def run_workflow(workflow: dict, dest_dir: str, timeout_s: float = 1800.0,
                 poll_interval_s: float = 5.0,
                 opener: Optional[Callable] = None,
                 progress_cb: Optional[Callable] = None) -> list:
    """Queue -> wait -> download. Returns local paths of result files."""
    job = queue_workflow(workflow, opener=opener)
    wait(job, timeout_s=timeout_s, poll_interval_s=poll_interval_s,
         opener=opener, progress_cb=progress_cb)
    return download_outputs(job, dest_dir, opener=opener)

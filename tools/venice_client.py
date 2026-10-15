#!/usr/bin/env python3
"""
Venice AI video client for MYAI Studio.

Mirrors the fal_client.py interface (submit -> poll -> result) so it drops
into the existing i2v_client.py provider abstraction.

Venice flow:
  POST /video/quote    (optional, exact pre-charge pricing)
  POST /video/queue    (submit, returns request_id)
  POST /video/retrieve (poll until video/mp4 or COMPLETED + download_url)

Model: wan-3-0-image-to-video (Wan 3.0 I2V)
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


VENICE_API_BASE = "https://api.venice.ai/api/v1"

# Model registry: model_id -> spec
# "workflow": i2v | v2v | r2v  (drives which media fields submit() sends)
# "uncensored": True marks models in the uncensored family (Wan 2.7 line).
#   David's standing permission covers fictional adult creative work; the
#   studio never filters prompts — it passes them to Venice exactly as written.
VENICE_MODELS = {
    "venice-wan3": {
        "model": "wan-3-0-image-to-video",
        "label": "Wan 3.0 (Venice)",
        "workflow": "i2v",
        "uncensored": False,
        # ~$0.65 per 5s 720p clip (use /video/quote for exact)
        "cost_per_s": 0.13,
        "default_duration_s": 5.0,
        "max_duration_s": 30.0,
        "resolutions": ["720p", "1080p"],
    },
    "venice-wan27-uncensored": {
        "model": "wan-2-7-enhanced-image-to-video",
        "label": "Wan 2.7 Enhanced (Venice, uncensored)",
        "workflow": "i2v",
        "uncensored": True,
        "cost_per_s": 0.128,
        "default_duration_s": 5.0,
        "max_duration_s": 15.0,
        "resolutions": ["720p", "1080p"],
    },
    # --- video-to-video / restyle ---
    "venice-wan27-v2v": {
        "model": "wan-2-7-video-to-video",
        "label": "Wan 2.7 Video-to-Video (uncensored)",
        "workflow": "v2v",
        "uncensored": True,
        "cost_per_s": 0.13,
        "default_duration_s": 5.0,
        "max_duration_s": 5.0,
        "resolutions": ["720p", "1080p"],
        "description": ("Restyle or transform an existing clip. Wan 2.7 is "
                        "Venice's uncensored-creative-freedom line."),
    },
    "venice-runway-v2v": {
        "model": "runway-gen4-turbo-v2v",
        "label": "Runway Gen-4 Turbo V2V",
        "workflow": "v2v",
        "uncensored": False,
        "cost_per_s": 0.15,
        "default_duration_s": 5.0,
        "max_duration_s": 10.0,
        "resolutions": ["720p", "1080p"],
        "description": ("Cinematic restyle with strong motion physics. "
                        "Silent output (no audio)."),
    },
    "venice-seedance-r2v": {
        "model": "seedance-2-0-reference-to-video",
        "label": "Seedance 2.0 Reference-to-Video",
        "workflow": "r2v",
        "uncensored": False,
        "supports_extend": True,
        "cost_per_s": 0.30,
        "default_duration_s": 10.0,
        "max_duration_s": 15.0,
        "resolutions": ["720p", "1080p"],
        "description": ("Edit or Extend workflows via reference_video_urls. "
                        "Native Extend is built for seamless continuation. "
                        "Note: public Seedance may reject clips with "
                        "detectable persons — use the Venice app for the "
                        "full feature set."),
    },
}

# Quality levels for generation UI. David picks; default is standard.
# Drives resolution + local encode settings. Model-variant overrides only
# where the variant ID is verified (never invented).
QUALITY_LEVELS = {
    "draft": {
        "label": "Draft",
        "resolution": "480p",
        "crf": 23,
        "preset": "veryfast",
        "description": "Fast + cheap. For iterating on prompts.",
    },
    "standard": {
        "label": "Standard",
        "resolution": "720p",
        "crf": 20,
        "preset": "medium",
        "description": "Balanced quality and cost.",
    },
    "maximum": {
        "label": "Maximum",
        "resolution": "1080p",
        "crf": 18,
        "preset": "slow",
        "description": "Best quality. Pro output.",
    },
}
DEFAULT_QUALITY = "standard"


class VeniceError(Exception):
    """Base Venice API error."""


class VeniceAuthError(VeniceError):
    """Missing/invalid API key."""


class VeniceValidationError(VeniceError):
    """Bad request (4xx)."""


class VeniceTransientError(VeniceError):
    """Retryable (429, 5xx)."""


class VeniceConsentRequired(VeniceError):
    """409 needs_consent — face detected, attestation required (no charge)."""

    def __init__(self, message: str, policy_text: str = ""):
        super().__init__(message)
        self.policy_text = policy_text


class JobStatus(Enum):
    IN_QUEUE = "IN_QUEUE"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class VeniceJob:
    request_id: str
    model_id: str
    status: JobStatus = JobStatus.IN_QUEUE
    result_url: Optional[str] = None
    attempts: int = 0


def get_api_key() -> str:
    key = os.environ.get("VENICE_API_KEY", "").strip()
    if not key:
        # Fall back to ~/.nanobot/venice.env (same pattern as fal.env)
        env_path = os.path.expanduser("~/.nanobot/venice.env")
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("VENICE_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
                    elif line and not line.startswith("#") and "=" not in line:
                        key = line  # bare key
                        break
        except FileNotFoundError:
            pass
    if not key:
        raise VeniceAuthError(
            "VENICE_API_KEY is not set. Add it to ~/.nanobot/venice.env "
            "or set the VENICE_API_KEY environment variable. "
            "Get a key at https://venice.ai/settings/api"
        )
    return key


def _http_json(
    method: str,
    path: str,
    api_key: str,
    payload: Optional[dict] = None,
    opener: Optional[Callable] = None,
) -> tuple[int, dict | bytes, str]:
    """Returns (status, parsed_body_or_raw_bytes, content_type)."""
    url = f"{VENICE_API_BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")

    _open = opener or urllib.request.urlopen
    try:
        with _open(req, timeout=120) as resp:
            status = resp.status
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read()
    except urllib.error.HTTPError as e:
        status = e.code
        raw = e.read()
        ctype = e.headers.get("Content-Type", "") if e.headers else ""
        body_text = raw.decode("utf-8", errors="replace")[:500]
        if status in (401, 403):
            raise VeniceAuthError(f"Venice auth failed (HTTP {status}): {body_text}")
        if status == 409:
            # Consent attestation required — parse policy text
            policy = ""
            try:
                policy = json.loads(body_text).get("policy_text", body_text)
            except Exception:
                policy = body_text
            raise VeniceConsentRequired(
                f"Venice requires likeness consent attestation (HTTP 409, no charge).",
                policy_text=policy,
            )
        if status == 429 or 500 <= status < 600:
            raise VeniceTransientError(f"Venice transient error (HTTP {status}): {body_text}")
        raise VeniceValidationError(f"Venice request invalid (HTTP {status}): {body_text}")

    if "application/json" in ctype:
        return status, json.loads(raw.decode("utf-8")), ctype
    return status, raw, ctype


def quote(model_id: str, args: dict, opener: Optional[Callable] = None) -> dict:
    """Get exact pre-charge pricing. Free to call."""
    api_key = get_api_key()
    spec = VENICE_MODELS[model_id]
    resolution = args.get("resolution", "720p")
    supported = spec.get("resolutions", ["720p", "1080p"])
    if resolution not in supported:
        order = ["480p", "720p", "1080p"]
        resolution = min(supported, key=lambda r: order.index(r) if r in order else 99)
    payload = {
        "model": spec["model"],
        "prompt": args.get("prompt", ""),
        "duration": args.get("duration", "5s"),
        "resolution": resolution,
    }
    for key in ("image_url", "video_url", "reference_image_urls",
                "reference_video_urls", "aspect_ratio"):
        if args.get(key) is not None:
            payload[key] = args[key]
    status, body, _ = _http_json("POST", "/video/quote", api_key, payload, opener)
    return body if isinstance(body, dict) else {}


def estimate_cost(model_id: str, duration_s: float) -> float:
    """Local cost estimate (no API call)."""
    spec = VENICE_MODELS[model_id]
    return round(spec["cost_per_s"] * duration_s, 4)


def submit(
    model_id: str,
    args: dict,
    opener: Optional[Callable] = None,
) -> VeniceJob:
    """Submit a video generation job. Returns VeniceJob with request_id."""
    api_key = get_api_key()
    spec = VENICE_MODELS[model_id]

    # Normalize duration to "Ns" format
    duration = args.get("duration", "5s")
    if isinstance(duration, (int, float)):
        duration = f"{int(duration)}s"
    elif isinstance(duration, str) and not duration.endswith("s"):
        duration = f"{duration}s"

    # Clamp resolution to what this model supports (e.g. V2V models reject 480p)
    resolution = args.get("resolution", "720p")
    supported = spec.get("resolutions", ["720p", "1080p"])
    if resolution not in supported:
        # Pick the lowest supported resolution as fallback
        order = ["480p", "720p", "1080p"]
        resolution = min(supported, key=lambda r: order.index(r) if r in order else 99)

    payload = {
        "model": spec["model"],
        "prompt": args.get("prompt", ""),
        "duration": duration,
        "resolution": resolution,
    }
    # Media passthrough per Venice /video/queue spec (URL or data URL):
    #   image_url              I2V first frame
    #   video_url              V2V source clip (mp4/mov/webm)
    #   reference_image_urls   R2V character/style consistency (<=30)
    #   reference_video_urls   R2V edit/extend source clips (<=10)
    #   omni_reference_task_type  Seedance 2.x: auto|reference|edit|extend
    for key in ("image_url", "end_image_url", "audio_url", "video_url",
                "reference_image_urls", "reference_video_urls",
                "reference_audio_urls", "omni_reference_task_type",
                "aspect_ratio"):
        if args.get(key) is not None:
            payload[key] = args[key]
    if args.get("negative_prompt"):
        payload["negative_prompt"] = args["negative_prompt"]
    if args.get("seed") is not None:
        payload["seed"] = args["seed"]

    last_err: Optional[VeniceError] = None
    for attempt in range(3):
        try:
            status, body, _ = _http_json("POST", "/video/queue", api_key, payload, opener)
            if not isinstance(body, dict):
                raise VeniceValidationError(f"Venice returned non-JSON: {body!r}"[:200])
            request_id = body.get("request_id") or body.get("id") or body.get("queue_id")
            if not request_id:
                raise VeniceValidationError(f"Venice returned no request_id: {body}")
            return VeniceJob(
                request_id=request_id,
                model_id=model_id,
                attempts=attempt + 1,
            )
        except VeniceTransientError as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise last_err or VeniceError("Venice submit failed")


def poll_status(
    job: VeniceJob,
    opener: Optional[Callable] = None,
) -> JobStatus:
    """Check job status. Returns JobStatus. Sets job.result_url when done."""
    api_key = get_api_key()
    # Venice /video/retrieve requires the model alongside the queue_id
    model = VENICE_MODELS.get(job.model_id, {}).get("model", "")
    payload = {"model": model, "queue_id": job.request_id}
    status, body, ctype = _http_json("POST", "/video/retrieve", api_key, payload, opener)

    # Venice returns video/mp4 binary when done — save it and use file:// URL
    if "video/" in ctype:
        import tempfile
        fd, tmp = tempfile.mkstemp(prefix="venice-vid-", suffix=".mp4")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(body if isinstance(body, bytes) else b"")
        except BaseException:
            try: os.remove(tmp)
            except OSError: pass
            raise
        job.status = JobStatus.COMPLETED
        job.result_url = f"file://{tmp}"
        return job.status

    if not isinstance(body, dict):
        raise VeniceValidationError(f"Venice poll returned unexpected: {body!r}"[:200])

    state = body.get("status", "").upper()
    if state in ("COMPLETED", "SUCCEEDED", "SUCCESS"):
        job.status = JobStatus.COMPLETED
        # Prefer download_url (24h pre-signed); check multiple possible keys
        job.result_url = (body.get("download_url") or body.get("video_url")
                          or body.get("url") or body.get("result_url")
                          or body.get("output_url"))
        if not job.result_url:
            # Log the full response keys for debugging
            import sys
            print(f"[venice-debug] COMPLETED but no URL. Keys: {list(body.keys())}",
                  file=sys.stderr, flush=True)
    elif state in ("FAILED", "ERROR", "CANCELLED", "CANCELED"):
        job.status = JobStatus.FAILED
        raise VeniceValidationError(f"Venice job failed: {body.get('error', body)}"[:300])
    elif state in ("PROCESSING", "IN_PROGRESS", "QUEUED", "PENDING"):
        job.status = JobStatus.PROCESSING if "PROCESS" in state or "PROGRESS" in state else JobStatus.IN_QUEUE
    else:
        # Unknown state — treat as still processing to avoid false failures
        job.status = JobStatus.PROCESSING
    return job.status


def wait(
    job: VeniceJob,
    timeout_s: float = 900.0,
    poll_interval_s: float = 10.0,
    opener: Optional[Callable] = None,
) -> str:
    """Poll until completion. Returns result video URL. Raises on failure/timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status = poll_status(job, opener)
        except VeniceTransientError:
            time.sleep(poll_interval_s)
            continue
        if status == JobStatus.COMPLETED:
            if not job.result_url:
                raise VeniceError("Venice job completed but returned no video URL")
            return job.result_url
        time.sleep(poll_interval_s)
    raise VeniceTransientError(f"Venice job timed out after {timeout_s}s")


# ------------------------------------------------------- video-to-video ----

_MIME_BY_EXT = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".m4v": "video/mp4", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp",
}

# Keep data-URL payloads sane: Venice caps reference clips at 50MB each.
MAX_DATA_URL_BYTES = 50 * 1024 * 1024


def file_to_data_url(path: str, mime: Optional[str] = None) -> str:
    """Read a local media file -> data: URL for /video/queue.

    Raises ValueError if the file exceeds Venice's per-clip media cap.
    """
    import base64
    size = os.path.getsize(path)
    if size > MAX_DATA_URL_BYTES:
        raise ValueError(
            f"{path} is {size / 1e6:.1f}MB; Venice caps API media at 50MB. "
            "Trim/compress the clip first (Edit tab).")
    if mime is None:
        mime = _MIME_BY_EXT.get(os.path.splitext(path)[1].lower(),
                                "application/octet-stream")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def download_to_file(url: str, dest_path: str,
                     opener: Optional[Callable] = None) -> str:
    """Download a result URL to dest_path (atomic: temp + rename)."""
    import tempfile
    # Handle file:// URLs (binary video saved by poll_status) via copy
    if url.startswith("file://"):
        import shutil
        src = url[7:]
        shutil.copy2(src, dest_path)
        try: os.remove(src)
        except OSError: pass
        return dest_path
    req = urllib.request.Request(url,
                                 headers={"User-Agent": "myai-wiring/1.0"})
    _open = opener or urllib.request.urlopen
    fd, tmp = tempfile.mkstemp(prefix=".dl-",
                               dir=os.path.dirname(dest_path) or ".")
    try:
        with _open(req, timeout=300) as resp, os.fdopen(fd, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, dest_path)
    return dest_path


def v2v_models() -> dict:
    """V2V/R2V-capable models for the studio model picker."""
    return {mid: s for mid, s in VENICE_MODELS.items()
            if s.get("workflow") in ("v2v", "r2v")}


def submit_v2v(
    model_id: str,
    prompt: str,
    video_path: Optional[str] = None,
    video_data_url: Optional[str] = None,
    reference_video_paths: Optional[list] = None,
    task_type: Optional[str] = None,
    duration_s: float = 5.0,
    resolution: str = "720p",
    aspect_ratio: Optional[str] = None,
    opener: Optional[Callable] = None,
) -> VeniceJob:
    """Submit a video-to-video / edit / extend job.

    Workflow routing by model spec:
      v2v models (wan-2-7-video-to-video, runway-gen4-turbo-v2v):
        source clip -> video_url
      r2v models (seedance-2-0-reference-to-video):
        source clip(s) -> reference_video_urls (+ omni_reference_task_type
        "edit" for restyle, "extend" for seamless continuation)

    Prompts pass through exactly as given — no filtering, ever.
    """
    spec = VENICE_MODELS.get(model_id)
    if spec is None:
        raise VeniceValidationError(f"unknown Venice model {model_id!r}")
    workflow = spec.get("workflow")
    if workflow not in ("v2v", "r2v"):
        raise VeniceValidationError(
            f"model {model_id!r} is not a video-to-video model")

    # Clamp duration to model max (Venice rejects over-limit with 422)
    max_d = float(spec.get("max_duration_s", 10.0))
    duration_s = min(float(duration_s), max_d)

    args: dict = {"prompt": prompt, "duration": duration_s,
                  "resolution": resolution}
    if aspect_ratio:
        args["aspect_ratio"] = aspect_ratio

    if workflow == "v2v":
        if video_data_url:
            args["video_url"] = video_data_url
        elif video_path:
            args["video_url"] = file_to_data_url(video_path)
        else:
            raise VeniceValidationError("v2v needs a source video")
    else:  # r2v
        refs = []
        if reference_video_paths:
            refs = [file_to_data_url(p) for p in reference_video_paths]
        elif video_data_url:
            refs = [video_data_url]
        elif video_path:
            refs = [file_to_data_url(video_path)]
        if not refs:
            raise VeniceValidationError("r2v needs at least one reference video")
        args["reference_video_urls"] = refs
        # NOTE: no omni_reference_task_type — Venice rejects it on
        # Seedance 2.0 R2V models (HTTP 400, verified Oct 14 2026:
        # "only supported on Seedance 2.5 reference-to-video models").
        # The task is conveyed by the prompt + reference media instead.

    return submit(model_id, args, opener=opener)

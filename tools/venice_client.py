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
VENICE_MODELS = {
    "venice-wan3": {
        "model": "wan-3-0-image-to-video",
        "label": "Wan 3.0 (Venice)",
        # ~$0.65 per 5s 720p clip (use /video/quote for exact)
        "cost_per_s": 0.13,
        "default_duration_s": 5.0,
        "max_duration_s": 30.0,
    },
    "venice-wan27-uncensored": {
        "model": "wan-2-7-enhanced-image-to-video",
        "label": "Wan 2.7 Enhanced (Venice, uncensored)",
        "cost_per_s": 0.128,
        "default_duration_s": 5.0,
        "max_duration_s": 15.0,
    },
}


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
    payload = {
        "model": spec["model"],
        "prompt": args.get("prompt", ""),
        "duration": args.get("duration", "5s"),
        "resolution": args.get("resolution", "720p"),
    }
    if args.get("image_url"):
        payload["image_url"] = args["image_url"]
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

    payload = {
        "model": spec["model"],
        "prompt": args.get("prompt", ""),
        "duration": duration,
        "resolution": args.get("resolution", "720p"),
    }
    if args.get("image_url"):
        payload["image_url"] = args["image_url"]
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
            request_id = body.get("request_id") or body.get("id")
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
    payload = {"request_id": job.request_id}
    status, body, ctype = _http_json("POST", "/video/retrieve", api_key, payload, opener)

    # Venice returns video/mp4 binary when done
    if "video/" in ctype:
        # The binary IS the result — but we need a URL, not bytes.
        # In practice Venice returns JSON with download_url; binary means
        # the caller should handle it. For now, mark completed.
        job.status = JobStatus.COMPLETED
        return job.status

    if not isinstance(body, dict):
        raise VeniceValidationError(f"Venice poll returned unexpected: {body!r}"[:200])

    state = body.get("status", "").upper()
    if state in ("COMPLETED", "SUCCEEDED", "SUCCESS"):
        job.status = JobStatus.COMPLETED
        # Prefer download_url (24h pre-signed) over embedded data
        job.result_url = body.get("download_url") or body.get("video_url")
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

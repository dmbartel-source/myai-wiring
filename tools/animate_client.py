"""WaveSpeed AI client — Wan 2.2 Animate (full-body character replacement).

Wan-Animate is the model built for exactly one job: take a driving
video + a character photo and REPLACE the performer in the video with
the person from the photo, preserving the original motion, expressions,
scene and camera. (Its other mode, "animate", puppeteers a still image
with a video's motion; the Studio uses "replace".)

Hosted on WaveSpeed (not Venice — Venice's Wan lineup has no Animate):
  POST https://api.wavespeed.ai/api/v3/wavespeed-ai/wan-2.2/animate
  GET  https://api.wavespeed.ai/api/v3/predictions/{id}

Pricing (Oct 2026): 720p $0.40 per 5s, 480p $0.20 per 5s; billed on
input duration rounded up to whole seconds, 3s minimum, max 120s.

The Studio sends media as data URLs (David's server is private, so
nothing he owns is publicly reachable) and always submits with the
host's safety checker disabled — his material is pre-vetted
consenting-adult / AI-generated work (see MEMORY.md standing rule).

API key: WAVESPEED_API_KEY env var, or ~/.nanobot/wavespeed.env
(same file pattern as venice.env / fal.env). Key values are NEVER
printed, logged, or stored anywhere else.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from typing import Callable, Optional

API_BASE = "https://api.wavespeed.ai/api/v3"
MODEL_PATH = "wavespeed-ai/wan-2.2/animate"

# Keep request bodies sane; WaveSpeed media inputs are generous but a
# phone-shot 4K clip as base64 is still a bad idea.
MAX_DATA_URL_BYTES = 100 * 1024 * 1024

_MIME_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".mp4": "video/mp4", ".mov": "video/quicktime",
    ".webm": "video/webm",
}

# USD per second of input video, by output resolution.
COST_PER_S = {"720p": 0.08, "480p": 0.04}
MAX_DURATION_S = 120.0


class AnimateError(RuntimeError):
    """Base class for WaveSpeed/Wan-Animate failures."""


class AnimateAuthError(AnimateError):
    """No API key configured, or the key was rejected."""


def get_api_key() -> str:
    key = os.environ.get("WAVESPEED_API_KEY", "").strip()
    if not key:
        env_path = os.path.expanduser("~/.nanobot/wavespeed.env")
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("WAVESPEED_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
                    elif line and not line.startswith("#") and "=" not in line:
                        key = line  # bare key
                        break
        except FileNotFoundError:
            pass
    if not key:
        raise AnimateAuthError(
            "WaveSpeed is not set up yet. Create an account at "
            "wavespeed.ai, add funds, copy your API key, then save it on "
            "the server in ~/.nanobot/wavespeed.env as one line: "
            "WAVESPEED_API_KEY=<your key>"
        )
    return key


def file_to_data_url(path: str, mime: Optional[str] = None) -> str:
    import base64
    size = os.path.getsize(path)
    if size > MAX_DATA_URL_BYTES:
        raise ValueError(
            f"{path} is {size / 1e6:.1f}MB; too large to send. "
            "Trim/compress the clip first (Edit tab).")
    if mime is None:
        mime = _MIME_BY_EXT.get(os.path.splitext(path)[1].lower(),
                                "application/octet-stream")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def estimate_cost(duration_s: float, resolution: str = "720p") -> float:
    """Billed on input duration rounded up, 3s minimum."""
    import math
    billed = max(3.0, math.ceil(float(duration_s)))
    return round(billed * COST_PER_S.get(resolution, 0.08), 2)


def _request(method: str, url: str, payload: Optional[dict] = None,
             opener: Optional[Callable] = None) -> dict:
    data = None
    headers = {"Authorization": f"Bearer {get_api_key()}"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=method)
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        if e.code in (401, 403):
            raise AnimateAuthError(
                f"WaveSpeed rejected the API key (HTTP {e.code}). "
                "Check ~/.nanobot/wavespeed.env on the server.") from e
        raise AnimateError(f"WaveSpeed HTTP {e.code}: {body}") from e


class AnimateJob:
    def __init__(self, job_id: str):
        self.job_id = job_id

    def __repr__(self):
        return f"AnimateJob({self.job_id!r})"


def submit_replace(image_path: str, video_path: str,
                   resolution: str = "720p",
                   seed: Optional[int] = None,
                   opener: Optional[Callable] = None) -> AnimateJob:
    """Replace the performer in video_path with the person in image_path.

    mode="replace" keeps the source video's motion, scene and camera.
    The safety checker is disabled: David's inputs are pre-vetted.
    """
    if resolution not in COST_PER_S:
        raise AnimateError(f"unknown resolution {resolution!r}")
    payload = {
        "image": file_to_data_url(image_path),
        "video": file_to_data_url(video_path),
        "mode": "replace",
        "resolution": resolution,
        "enable_safety_checker": False,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    resp = _request("POST", f"{API_BASE}/{MODEL_PATH}", payload,
                    opener=opener)
    data = resp.get("data") or resp
    job_id = data.get("id") or data.get("task_id")
    if not job_id:
        raise AnimateError(f"WaveSpeed submit returned no job id: "
                           f"{str(resp)[:300]}")
    return AnimateJob(str(job_id))


def poll(job: AnimateJob, opener: Optional[Callable] = None) -> dict:
    """One status read. Returns {status, url|None, raw}."""
    resp = _request("GET", f"{API_BASE}/predictions/{job.job_id}",
                    opener=opener)
    data = resp.get("data") or resp
    status = str(data.get("status") or "").lower()
    url = None
    if status in ("completed", "succeeded", "success"):
        outputs = data.get("outputs") or []
        if isinstance(outputs, list) and outputs:
            url = outputs[0]
        elif isinstance(outputs, str):
            url = outputs
        elif data.get("output"):
            out = data["output"]
            url = out[0] if isinstance(out, list) else out
    return {"status": status, "url": url, "raw": data}


def wait(job: AnimateJob, timeout_s: float = 1800.0,
         poll_interval_s: float = 10.0,
         opener: Optional[Callable] = None) -> str:
    """Poll until the job completes; return the output video URL."""
    deadline = time.time() + timeout_s
    while True:
        st = poll(job, opener=opener)
        if st["url"]:
            return st["url"]
        if st["status"] in ("failed", "error", "cancelled", "canceled"):
            raise AnimateError(
                f"WaveSpeed job {job.job_id} failed: "
                f"{str(st['raw'])[:300]}")
        if time.time() >= deadline:
            raise AnimateError(
                f"WaveSpeed job {job.job_id} timed out after "
                f"{timeout_s:.0f}s (status {st['status']!r})")
        time.sleep(poll_interval_s)


def download_to_file(url: str, dest_path: str,
                     opener: Optional[Callable] = None) -> str:
    req = urllib.request.Request(url, method="GET")
    open_fn = opener or urllib.request.urlopen
    with open_fn(req) as resp, open(dest_path, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest_path

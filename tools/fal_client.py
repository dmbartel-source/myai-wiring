"""fal.ai client for the MYAI media pipeline.

Patterns: submit -> poll with backoff (webhook preferred for long jobs),
immediate download, ED25519 webhook verification, async job state machine
(IN_QUEUE -> IN_PROGRESS -> COMPLETED), retry budgets by failure class,
idempotency keys.

Credentials: reads FAL_API_KEY from the environment. Fails cleanly with a
clear error when the key is absent — no network calls are attempted.

No real API calls are made unless submit()/poll() are invoked with a key.
All network paths are fully mockable (see tests/).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import random
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

try:
    from . import net
except ImportError:  # tools/ used as plain scripts dir
    import net


FAL_QUEUE_BASE = "https://queue.fal.run"

# Model registry with ToS status from the Oct 2026 terms review.
#   approved    — fal.ai / Replicate platform terms accepted with caveats.
#   conditional — usable only after written clearance / paid tier /
#                 commercial authorization (pass tos_ack=True per call).
#   rejected    — ToS review FAILED (IP/training terms). require_route()
#                 REFUSES these outright; there is no override.
# Planning rates ($/s) are Oct 2026 research figures — re-verify live
# before wiring billing.
MODEL_REGISTRY = {
    # Draft tier (approved)
    "wan-25": {
        "endpoint": "fal-ai/wan-25/text-to-video",
        "usd_per_sec": 0.05,
        "tier": "draft",
        "supports": ["t2v", "i2v"],
        "tos": "approved",
        "tos_note": "fal.ai platform: approved with caveats (Oct 2026 review).",
    },
    "seedance-mini": {
        "endpoint": "fal-ai/seedance/v2-0-mini",
        "usd_per_sec": 0.0113,
        "tier": "draft",
        "supports": ["t2v", "i2v"],
        "tos": "approved",
        "tos_note": "fal.ai platform: approved with caveats (Oct 2026 review).",
    },
    # Seedance 2.0 reference-to-video: up to 9 image refs, 3 video refs,
    # 3 audio refs, cited in the prompt as @Image1/@Video1/etc.
    # Oct 2026 research rate ~$0.30/s at 720p; video refs get ~40%
    # discount per the same research. Re-verify live before billing.
    "seedance-2.0": {
        "endpoint": "bytedance/seedance-2.0/reference-to-video",
        "usd_per_sec": 0.30,
        "tier": "final",  # ~$1.50/5s clip: above the $0.30 draft cap
        "supports": ["r2v"],
        "tos": "approved",
        "tos_note": "fal.ai platform: approved with caveats (Oct 2026 review).",
    },
    # Final tier (conditional — Google Veo paid tier; verify data-use terms)
    "veo3-fast": {
        "endpoint": "fal-ai/veo3/fast",
        "usd_per_sec": 0.15,
        "tier": "final",
        "supports": ["t2v", "i2v"],
        "tos": "conditional",
        "tos_note": ("Google Veo via fal: paid tier only; confirm training/data "
                     "terms in writing before enabling (Oct 2026 review)."),
    },
    # Hero tier (conditional; explicit ask only)
    "veo3": {
        "endpoint": "fal-ai/veo3",
        "usd_per_sec": 0.40,  # with audio; 0.20 without
        "tier": "hero",
        "supports": ["t2v", "i2v"],
        "tos": "conditional",
        "tos_note": ("Google Veo via fal: paid tier only; confirm training/data "
                     "terms in writing before enabling (Oct 2026 review)."),
    },
    "runway-gen45": {
        "endpoint": "fal-ai/runway-gen4-5/text-to-video",
        "usd_per_sec": 0.12,
        "tier": "hero",
        "supports": ["t2v", "i2v"],
        "tos": "conditional",
        "tos_note": ("Runway: pending written clarification of IP/training "
                     "terms (Oct 2026 review). Do not enable until cleared."),
    },
    # --- REJECTED. Listed so attempts fail LOUDLY, not as "unknown". ---
    "kling-3": {
        "endpoint": "fal-ai/kling-video/v3/standard/text-to-video",
        "usd_per_sec": 0.112,
        "tier": "final",
        "supports": ["t2v", "i2v"],
        "tos": "rejected",
        "tos_note": ("Kling ToS REJECTED in Oct 2026 review (IP/training terms "
                     "unacceptable). Never route here."),
    },
    "pika-2": {
        "endpoint": "fal-ai/pika/v2/text-to-video",
        "usd_per_sec": 0.0,
        "tier": "final",
        "supports": ["t2v", "i2v"],
        "tos": "rejected",
        "tos_note": "Pika ToS REJECTED in Oct 2026 review. Never route here.",
    },
    "minimax-hailuo": {
        "endpoint": "fal-ai/minimax/hailuo-02/text-to-video",
        "usd_per_sec": 0.0,
        "tier": "final",
        "supports": ["t2v", "i2v"],
        "tos": "rejected",
        "tos_note": "MiniMax/Hailuo ToS REJECTED in Oct 2026 review. Never route here.",
    },
}

# Active routes: everything not rejected. Kept under the old name for
# backward compatibility (pipeline.py reads MODEL_ROUTES[route]["tier"]).
MODEL_ROUTES = {
    name: spec for name, spec in MODEL_REGISTRY.items()
    if spec["tos"] != "rejected"
}


class JobStatus(str, Enum):
    IN_QUEUE = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FalError(Exception):
    """Base error for fal client failures."""


class FalTosError(FalError):
    """Model route refused on terms-of-service grounds. Fail closed."""


class FalAuthError(FalError):
    """Missing key or 401/403 from the API."""


class FalValidationError(FalError):
    """4xx other than auth/rate-limit: bad args. NEVER retried."""


class FalTransientError(FalError):
    """Network errors, 5xx, 429: safe to retry with backoff."""


class FalTimeoutError(FalError):
    """Job did not complete within the poll budget."""


@dataclass
class FalJob:
    request_id: str
    endpoint: str
    idempotency_key: str
    status: JobStatus = JobStatus.IN_QUEUE
    result: Optional[dict] = None
    attempts: int = 0


def get_api_key() -> str:
    """Read FAL_API_KEY from env. Raise a clear error if absent."""
    key = os.environ.get("FAL_API_KEY", "").strip()
    if not key:
        raise FalAuthError(
            "FAL_API_KEY is not set. Export it (e.g. from a mode-600 "
            "EnvironmentFile) before submitting generation jobs. "
            "No network call was attempted."
        )
    return key


def make_idempotency_key(job_label: str) -> str:
    """Deterministic idempotency key from a job label (effect id + take)."""
    digest = hashlib.sha256(job_label.encode("utf-8")).hexdigest()[:32]
    return f"myai-{digest}"


def require_route(route: str, tos_ack: bool = False) -> dict:
    """ToS gate for model routes. Returns the route spec or raises.

    - Unknown route -> FalValidationError.
    - tos="rejected" -> FalTosError, outright, no override.
    - tos="conditional" -> FalTosError unless tos_ack=True (explicit
      per-call acknowledgement that David cleared the terms).
    """
    spec = MODEL_REGISTRY.get(route)
    if spec is None:
        raise FalValidationError(f"Unknown model route: {route!r}")
    tos = spec["tos"]
    if tos == "rejected":
        raise FalTosError(
            f"Model route {route!r} is REJECTED by the Oct 2026 ToS review: "
            f"{spec['tos_note']} Refused outright."
        )
    if tos == "conditional" and not tos_ack:
        raise FalTosError(
            f"Model route {route!r} is CONDITIONAL: {spec['tos_note']} "
            f"Pass tos_ack=True only after David clears the terms in writing."
        )
    return spec


def estimate_cost_usd(route: str, duration_s: float, tos_ack: bool = False) -> float:
    """Pre-generation cost estimate. Called BEFORE every paid call.

    Enforces the ToS gate too: rejected models get no estimate, only a
    refusal; conditional models need tos_ack=True.
    """
    spec = require_route(route, tos_ack=tos_ack)
    return round(spec["usd_per_sec"] * duration_s, 4)


def _classify_http_error(status: int, body: str) -> FalError:
    if status in (401, 403):
        return FalAuthError(f"fal.ai auth failed (HTTP {status}): {body[:200]}")
    if status == 429 or 500 <= status < 600:
        return FalTransientError(f"fal.ai transient failure (HTTP {status}): {body[:200]}")
    return FalValidationError(f"fal.ai request invalid (HTTP {status}): {body[:200]}")


def _http_json(
    method: str,
    url: str,
    api_key: str,
    payload: Optional[dict] = None,
    extra_headers: Optional[dict] = None,
    opener: Optional[Callable] = None,
) -> dict:
    """Single HTTP round trip, behind the egress gate (tools/net.py).

    `opener` is injectable for tests — the transport is mocked, but the
    gate still validates the URL, so tests exercise the policy too.
    """
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Key {api_key}")
    req.add_header("Content-Type", "application/json")
    for k, v in (extra_headers or {}).items():
        req.add_header(k, v)
    try:
        with net.gated_urlopen(req, timeout=30, opener=opener) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise _classify_http_error(e.code, body)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise FalTransientError(f"Network failure contacting fal.ai: {e}")


def _validate_webhook_url(webhook_url: str) -> None:
    """Webhooks make fal.ai POST job metadata to us — the URL must be our
    own https endpoint, never a third party, and never carry credentials."""
    parts = urllib.parse.urlsplit(webhook_url)
    if parts.scheme.lower() != "https":
        raise FalValidationError(
            f"webhook_url must be https (got {parts.scheme!r}) — "
            "job metadata must not travel in cleartext."
        )
    if parts.username or parts.password:
        raise FalValidationError("webhook_url must not embed credentials.")
    if not parts.hostname:
        raise FalValidationError("webhook_url has no host.")


def submit(
    route: str,
    arguments: dict,
    webhook_url: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    opener: Optional[Callable] = None,
    tos_ack: bool = False,
) -> FalJob:
    """Submit a generation job. Returns a FalJob in IN_QUEUE state.

    ToS gate: rejected routes are refused outright (FalTosError);
    conditional routes require tos_ack=True.

    Retry policy: transient errors retried with jittered backoff (max 3
    attempts); validation errors raised immediately (0 retries).
    """
    api_key = get_api_key()
    spec = require_route(route, tos_ack=tos_ack)
    if webhook_url:
        _validate_webhook_url(webhook_url)
    key = idempotency_key or make_idempotency_key(
        f"{route}:{json.dumps(arguments, sort_keys=True)}"
    )
    url = f"{FAL_QUEUE_BASE}/{spec['endpoint']}"
    headers = {"X-Fal-Idempotency-Key": key}
    if webhook_url:
        # Webhook preferred for long jobs; poll is the fallback.
        payload = {"webhook_url": webhook_url, **arguments}
    else:
        payload = dict(arguments)

    last_err: Optional[FalError] = None
    for attempt in range(3):
        try:
            resp = _http_json("POST", url, api_key, payload, headers, opener)
            request_id = resp.get("request_id")
            if not request_id:
                raise FalValidationError(f"fal.ai returned no request_id: {resp}")
            return FalJob(
                request_id=request_id,
                endpoint=spec["endpoint"],
                idempotency_key=key,
                attempts=attempt + 1,
            )
        except FalTransientError as e:
            last_err = e
            time.sleep(_backoff_s(attempt))
        except FalError:
            raise  # validation/auth: never retried
    raise last_err or FalTransientError("submit failed after retries")


def poll_status(job: FalJob, opener: Optional[Callable] = None) -> JobStatus:
    """One status check. Raises on terminal failure states."""
    api_key = get_api_key()
    url = f"{FAL_QUEUE_BASE}/{job.endpoint}/requests/{job.request_id}/status"
    resp = _http_json("GET", url, api_key, opener=opener)
    status = resp.get("status", "")
    try:
        job.status = JobStatus(status)
    except ValueError:
        raise FalError(f"Unknown fal.ai job status: {status!r}")
    if job.status == JobStatus.FAILED:
        # Classify: fal includes error detail; treat 4xx-class as validation.
        detail = json.dumps(resp.get("error", {}))
        if "422" in detail or "validation" in detail.lower():
            raise FalValidationError(f"Job failed (validation): {detail[:300]}")
        raise FalTransientError(f"Job failed (transient): {detail[:300]}")
    return job.status


def fetch_result(job: FalJob, opener: Optional[Callable] = None) -> dict:
    """Fetch the completed result payload (contains output URLs)."""
    api_key = get_api_key()
    url = f"{FAL_QUEUE_BASE}/{job.endpoint}/requests/{job.request_id}"
    resp = _http_json("GET", url, api_key, opener=opener)
    job.result = resp
    job.status = JobStatus.COMPLETED
    return resp


def wait(
    job: FalJob,
    timeout_s: float = 600.0,
    poll_interval_s: float = 5.0,
    opener: Optional[Callable] = None,
    on_status: Optional[Callable[[JobStatus], None]] = None,
) -> dict:
    """Poll until COMPLETED or timeout. Returns the result payload."""
    deadline = time.monotonic() + timeout_s
    interval = poll_interval_s
    while True:
        try:
            status = poll_status(job, opener)
        except FalTransientError:
            # Transient poll failure: back off and keep trying within budget.
            status = job.status
        if on_status:
            on_status(status)
        if status == JobStatus.COMPLETED:
            return fetch_result(job, opener)
        if time.monotonic() >= deadline:
            raise FalTimeoutError(
                f"Job {job.request_id} did not complete within {timeout_s}s"
            )
        time.sleep(interval)
        interval = min(interval * 1.5, 30.0)  # capped exponential backoff


def download_to_file(
    url: str, dest_path: str, opener: Optional[Callable] = None
) -> str:
    """Download an artifact URL to dest_path (atomic: temp + rename).

    Provider URLs expire — call immediately after the job completes.
    The URL goes through the egress gate: artifact CDNs (fal.media etc.)
    are allowlisted; anything else is refused. Error messages redact the
    URL (pre-signed tokens) instead of truncating it.
    """
    import tempfile

    req = urllib.request.Request(url, headers={"User-Agent": "myai-wiring/1.0"})
    try:
        with net.gated_urlopen(req, timeout=120, opener=opener) as resp:
            fd, tmp = tempfile.mkstemp(
                prefix=".dl-", dir=os.path.dirname(dest_path) or "."
            )
            with os.fdopen(fd, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
    except (urllib.error.URLError, OSError) as e:
        raise FalTransientError(
            f"Download failed for {net.redact_url(url)}: {e}"
        )
    os.replace(tmp, dest_path)  # atomic; never a partial file at dest
    return dest_path


def _backoff_s(attempt: int) -> float:
    """Jittered exponential backoff: 1s, 2s, 4s base + jitter."""
    return (2.0 ** attempt) + random.uniform(0, 0.5)


# ---------------------------------------------------------------------------
# Webhook verification (Ed25519). fal.ai signs webhooks; the public key is
# published via JWKS. Verification requires a JWK — fetched once and cached
# by the caller, or passed directly (tests inject a fixed key).
# ---------------------------------------------------------------------------

def verify_webhook_signature(
    raw_body: bytes,
    signature_header: str,
    public_key_jwk: dict,
) -> bool:
    """Verify fal.ai's Ed25519 webhook signature.

    signature_header: value of the X-Fal-Webhook-Signature header
        (base64url-encoded signature over the raw body).
    public_key_jwk: JWK dict with kty="OKP", crv="Ed25519", x=base64url pubkey.

    Returns True on valid signature, False otherwise. Raises FalError if
    PyNaCl is unavailable (fail closed: unverifiable = rejected by caller).
    """
    try:
        from nacl.signing import VerifyKey
        from nacl.exceptions import BadSignature
    except ImportError:
        raise FalError(
            "PyNaCl is required for webhook verification (pip install pynacl). "
            "Failing closed: treat the webhook as unverified."
        )
    try:
        if public_key_jwk.get("kty") != "OKP" or public_key_jwk.get("crv") != "Ed25519":
            return False
        x = public_key_jwk["x"]
        # base64url without padding
        x_padded = x + "=" * (-len(x) % 4)
        pubkey_bytes = base64.urlsafe_b64decode(x_padded)
        sig_padded = signature_header + "=" * (-len(signature_header) % 4)
        sig_bytes = base64.urlsafe_b64decode(sig_padded)
        VerifyKey(pubkey_bytes).verify(raw_body, sig_bytes)
        return True
    except Exception:
        return False


def webhook_payload_ok(payload: dict) -> bool:
    """fal.ai webhook payloads carry status OK/ERROR (distinct from queue
    statuses). Returns True only for successful completions."""
    return payload.get("status") == "OK" and bool(payload.get("request_id"))

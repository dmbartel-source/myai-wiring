"""Egress gate: the single choke point for ALL outbound HTTP in this package.

Default-deny. Every HTTP(S) request made by tools/ must pass through
gated_urlopen() (or call check_url() first). The gate:

  - allows only an explicit domain allowlist (fal.ai platform hosts,
    fal artifact CDN, openrouter.ai) plus loopback http for the
    self-hosted local generative endpoint;
  - blocks everything else by raising EgressBlocked;
  - logs EVERY attempt — allowed and blocked — as JSONL, with query
    strings redacted (pre-signed URLs carry tokens).

There is no silent transmission: if a request leaves this machine, there
is a log line for it. If the log write itself fails, the request still
goes through (logging must never break the pipeline) — but the failure
is swallowed only inside the logger, never the gate decision.

The `opener` parameter is a test-only seam: injected transports still go
through check_url(), so tests exercise the gate without touching the
network.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Optional, Union


class EgressBlocked(Exception):
    """Raised when an outbound request is not on the egress allowlist."""


# --- Allowlist -------------------------------------------------------------
# Suffix match: "queue.fal.run" also covers "x.queue.fal.run" but NOT
# "evilqueue.fal.run" (the leading-dot rule below prevents that).
ALLOWED_SUFFIXES = (
    "queue.fal.run",   # fal.ai queue API
    "fal.run",         # fal.ai platform
    "fal.ai",          # fal.ai platform / docs
    "rest.fal.ai",     # fal.ai REST API (storage upload initiate)
    "fal.media",       # fal artifact CDN (v3.fal.media result URLs)
    "cdn.fal.ai",      # fal CDN edge
    "openrouter.ai",   # model brain API (nanobot presets)
)
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

_EGRESS_LOG_OVERRIDE: Optional[str] = None


def set_egress_log(path: Optional[str]) -> None:
    """Override the egress log path (tests use a tmp dir). None = default."""
    global _EGRESS_LOG_OVERRIDE
    _EGRESS_LOG_OVERRIDE = path


def egress_log_path() -> str:
    if _EGRESS_LOG_OVERRIDE:
        return _EGRESS_LOG_OVERRIDE
    return os.environ.get(
        "MYAI_EGRESS_LOG", os.path.expanduser("~/.myai/egress.jsonl")
    )


def redact_url(url: str) -> str:
    """Strip query string and fragment (pre-signed URLs carry tokens)."""
    try:
        p = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, "", ""))
    except Exception:
        return "<unparseable-url>"


def _host_allowed(host: str) -> bool:
    return any(host == d or host.endswith("." + d) for d in ALLOWED_SUFFIXES)


def check_url(url: str) -> str:
    """Validate an outbound URL. Returns the host if allowed.

    Raises EgressBlocked for: non-http(s) schemes, embedded credentials,
    plain http to non-loopback hosts, and any host not on the allowlist.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except Exception as e:
        raise EgressBlocked(f"unparseable URL: {e}")
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise EgressBlocked(f"scheme {scheme!r} not permitted (http/https only)")
    if parts.username or parts.password:
        raise EgressBlocked("URL contains embedded credentials — refused")
    host = (parts.hostname or "").lower()
    if not host:
        raise EgressBlocked("URL has no host — refused")
    if host in LOOPBACK_HOSTS:
        return host  # loopback: self-hosted endpoints only (e.g. local ComfyUI)
    if scheme == "http":
        raise EgressBlocked("plain http refused for non-loopback hosts")
    if not _host_allowed(host):
        raise EgressBlocked(
            f"host {host!r} is not on the egress allowlist "
            f"{sorted(ALLOWED_SUFFIXES)} — refusing to transmit"
        )
    return host


def _write_log(record: dict) -> None:
    try:
        path = egress_log_path()
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # logging must never break the gate or the pipeline


def log_attempt(method: str, url: str, allowed: bool, reason: str = "") -> None:
    _write_log({
        "ts": time.time(),
        "method": method,
        "url": redact_url(url),
        "allowed": bool(allowed),
        "reason": reason,
    })


def _method_of(req: Union[str, urllib.request.Request]) -> str:
    if isinstance(req, urllib.request.Request):
        return req.get_method()
    return "GET"


def _url_of(req: Union[str, urllib.request.Request]) -> str:
    if isinstance(req, urllib.request.Request):
        return req.full_url
    return req


def gated_urlopen(
    req: Union[str, urllib.request.Request],
    timeout: float = 30.0,
    opener=None,
):
    """urlopen() behind the egress gate. Always validates the URL first —
    even when a test `opener` is injected (the transport is mocked, the
    policy is not). Logs allowed and blocked attempts. Raises EgressBlocked
    on policy violation (logged), or the transport's own errors otherwise.
    """
    url = _url_of(req)
    method = _method_of(req)
    try:
        host = check_url(url)
    except EgressBlocked as e:
        log_attempt(method, url, False, str(e))
        raise
    log_attempt(method, url, True, f"allowlisted host {host}")
    open_fn = opener or urllib.request.urlopen
    return open_fn(req, timeout=timeout)

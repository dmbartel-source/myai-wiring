"""Local generative path — self-hosted diffusion endpoint (FUTURE WIRING).

This is the copyright-privacy endgame for generative work: a diffusion
server (e.g. ComfyUI, or an SD WebUI / `sd --api` instance) running on
David's own VPS, reached only over loopback. Zero third-party contact —
prompts and media never leave the machine.

STATUS: stub. The config, health check, and call shape are defined so the
pipeline can reserve the hook (`local_gen` stage tool), but generate()
refuses until an operator explicitly enables it AND the endpoint answers
a health check. No live calls are made by this package today.

To wire it later (on the VPS):
  1. Run ComfyUI (or equivalent) on 127.0.0.1:8188 with --api-only.
  2. Export MYAI_LOCAL_GEN=1 and MYAI_LOCAL_GEN_URL=http://127.0.0.1:8188.
  3. Implement the workflow POST in generate() for the chosen server's API
     (ComfyUI: POST /prompt with a workflow JSON; poll /history/<id>).
  4. The egress gate (tools/net.py) already permits loopback http and
     blocks everything else — no policy change needed.

Cost: $0 marginal (VPS CPU; GPU instance if David adds one later).
Privacy: total — nothing leaves localhost.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

try:
    from . import net
except ImportError:  # tools/ used as plain scripts dir
    import net


class LocalGenNotWired(Exception):
    """The local generative endpoint is not configured/enabled yet."""


@dataclass
class LocalGenConfig:
    """Config for the self-hosted diffusion endpoint."""
    url: str = "http://127.0.0.1:8188"  # loopback only, always
    kind: str = "comfyui"               # "comfyui" | "sd-webui" | "custom"
    workflow: str = "txt2img"           # workflow/pipeline selector
    timeout_s: float = 600.0
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LocalGenConfig":
        return cls(
            url=os.environ.get("MYAI_LOCAL_GEN_URL", "http://127.0.0.1:8188"),
            kind=os.environ.get("MYAI_LOCAL_GEN_KIND", "comfyui"),
        )

    def validate(self) -> None:
        """Only loopback URLs are ever permitted — enforced, not advised."""
        host = net.check_url(self.url)  # raises EgressBlocked otherwise
        if self.kind not in ("comfyui", "sd-webui", "custom"):
            raise LocalGenNotWired(f"unknown local gen kind {self.kind!r}")


def enabled() -> bool:
    """Explicit operator opt-in. Default off."""
    return os.environ.get("MYAI_LOCAL_GEN", "0") == "1"


def health_check(config: Optional[LocalGenConfig] = None) -> bool:
    """Ask the local endpoint if it is alive. Loopback http only (the
    egress gate enforces this). Returns False on any failure — never raises
    for an unhealthy endpoint (raises only on policy violation)."""
    cfg = config or LocalGenConfig.from_env()
    cfg.validate()
    # ComfyUI exposes /system_stats; treat any 2xx as alive. This is a
    # best-effort probe — generate() re-checks before doing work.
    req = urllib.request.Request(cfg.url.rstrip("/") + "/system_stats",
                                 headers={"User-Agent": "myai-wiring/1.0"})
    try:
        with net.gated_urlopen(req, timeout=10) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def generate(prompt: str,
             config: Optional[LocalGenConfig] = None,
             **kwargs) -> dict:
    """Generate media on the local endpoint.

    Today this ALWAYS raises LocalGenNotWired: either the operator has not
    opted in (MYAI_LOCAL_GEN=1), the endpoint fails its health check, or
    the workflow POST for the configured server kind is not implemented
    yet. When it is implemented, it returns
    {"file": <path>, "prompt": ..., "seed": ..., "ms": ...}.
    """
    cfg = config or LocalGenConfig.from_env()
    if not enabled():
        raise LocalGenNotWired(
            "Local generative path is not enabled. To wire it: run a "
            "diffusion server on 127.0.0.1:8188, export MYAI_LOCAL_GEN=1, "
            "and implement the workflow POST for kind "
            f"{cfg.kind!r} in tools/local_gen.py. Nothing was sent anywhere."
        )
    cfg.validate()
    if not health_check(cfg):
        raise LocalGenNotWired(
            f"Local endpoint {cfg.url} did not answer its health check. "
            "Is the diffusion server running? Nothing was sent anywhere."
        )
    # --- implement the per-server workflow POST here (ComfyUI /prompt etc.)
    raise LocalGenNotWired(
        f"Workflow POST for kind {cfg.kind!r} is not implemented yet. "
        "See the module docstring for the wiring steps."
    )

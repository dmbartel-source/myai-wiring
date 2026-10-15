# MYAI Media Wiring — Security Policy

How this package keeps David's media private, and how to verify it.
Written for a reviewer with the source tree and a terminal.

## 1. Egress policy (no silent transmission)

**Rule:** every outbound HTTP(S) request in `tools/` goes through
`tools/net.py::gated_urlopen()`. Default-deny: the request is refused
with `EgressBlocked` unless the host is on the explicit allowlist.

**Allowlist** (`net.ALLOWED_SUFFIXES`, suffix match with leading-dot rule
so `evilfal.ai` does NOT match `fal.ai`):

| Host suffix | Why |
|---|---|
| `queue.fal.run` | fal.ai queue API (submit/poll/result) |
| `fal.run`, `fal.ai`, `cdn.fal.ai` | fal.ai platform |
| `fal.media` | fal artifact CDN (`v3.fal.media` result download URLs) |
| `openrouter.ai` | model brain API (preset config; reserved) |

Plus loopback (`localhost`, `127.0.0.1`, `::1`) over **http only** — for the
self-hosted diffusion endpoint (`tools/local_gen.py`). Plain http to any
non-loopback host is refused. Non-http(s) schemes are refused. URLs with
embedded credentials (`user:pass@host`) are refused.

**Logging:** every attempt — allowed AND blocked — is appended as JSONL to
`~/.myai/egress.jsonl` (override with `MYAI_EGRESS_LOG` env var or
`net.set_egress_log()`). Query strings are redacted before logging
(pre-signed URLs carry tokens). If the log write fails, the request still
proceeds — logging must never break the pipeline — but the gate decision
itself never fails open.

**Test seam:** injected `opener` callables (used by the test suite) mock the
*transport* but still pass through `check_url()` — tests exercise the
policy without touching the network.

**Out of scope:** `pipeline`'s `shell`/`ffmpeg` tools run local subprocesses
from EDL plans the agent wrote. They cannot be egress-gated at this layer
(that needs OS-level controls on the VPS); the EDL is the audit trail.

## 2. Privacy guarantee (copyright)

David's rule: *the products of MYAI are completely private for copyright
reasons.* Encoded in `tools/privacy.py`:

- Every media asset carries a **Sensitivity** (`sensitive` / `private` /
  `public`). **Default is `sensitive`** — David's footage, descriptions,
  and outputs are his IP unless he says otherwise.
- **Third-party processing always requires explicit per-call
  `allow_third_party=True`.** No standing permission, no config toggle,
  no "the skill said it was fine". The flag is per call, per asset.
- **Default route is local-only**: `tools/local_effects.py` (12
  deterministic ffmpeg recipes, $0) and the `local_gen` hook for a
  self-hosted diffusion endpoint. Generative APIs are for what local
  processing cannot do (new pixels).
- **DeepSeek ban enforced in code**: `privacy.check_brain_for_media()`
  refuses any brain/provider/model id containing `deepseek`
  (case-insensitive) for media work. Called by the pipeline's `fal_submit`
  stage and available to any caller. A config comment is not enforcement;
  this function is.
- Model routes carry a **ToS status** (`fal_client.MODEL_REGISTRY`):
  `approved` / `conditional` / `rejected`. `require_route()` refuses
  `rejected` models outright (no override exists); `conditional` models
  need explicit per-call `tos_ack=True`.

## 3. Model allowlist (Oct 2026 ToS review)

| Route | Endpoint | Tier | ToS | Notes |
|---|---|---|---|---|
| `wan-25` | `fal-ai/wan-25/text-to-video` | draft | **approved** | fal.ai platform terms accepted with caveats |
| `seedance-mini` | `fal-ai/seedance/v2-0-mini` | draft | **approved** | same |
| `veo3-fast` | `fal-ai/veo3/fast` | final | **conditional** | Google Veo: paid tier only; verify data-use terms in writing, then `tos_ack=True` |
| `veo3` | `fal-ai/veo3` | hero | **conditional** | same as veo3-fast; explicit ask only |
| `runway-gen45` | `fal-ai/runway-gen4-5/text-to-video` | hero | **conditional** | Runway: pending written clarification; do not enable until cleared |
| `kling-3` | `fal-ai/kling-video/v3/...` | — | **REJECTED** | Kling ToS failed review (IP/training terms). Client refuses. |
| `pika-2` | `fal-ai/pika/v2/...` | — | **REJECTED** | Pika ToS failed review. Client refuses. |
| `minimax-hailuo` | `fal-ai/minimax/hailuo-02/...` | — | **REJECTED** | MiniMax/Hailuo ToS failed review. Client refuses. |

Rejected routes are listed in the registry *so attempts fail loudly*
(`FalTosError`) instead of dying as "unknown route". Planning rates
($/s) are Oct 2026 research figures — re-verify on fal.ai before wiring
billing.

## 4. Credentials

- `FAL_API_KEY` read from the environment only (`fal_client.get_api_key()`).
  Absent key → clean `FalAuthError`, no network attempted.
- `config/presets.json` contains only `${ENV_VAR}` placeholders. Real keys
  live in a mode-600 EnvironmentFile on the VPS, never in chat, config, or
  the repo.
- The key is sent only as an `Authorization: Key ...` header to
  allowlisted fal.ai hosts. It is never logged (error paths use
  `net.redact_url()`).

## 5. Reviewer checklist

To re-verify this package after any change:

1. `grep -rn "urlopen\|urllib.request\|http.client\|socket\." tools/ --include="*.py"` — every hit must be inside `net.py` (the gate) or go through `net.gated_urlopen()`. Any new direct network call is a finding.
2. `grep -rniE "kling|pika|minimax|hailuo" tools/ skills/ --include="*.py" --include="*.md"` — hits must ONLY be in `rejected` registry entries, refusal tests, and this document. Any recommended/default usage is a finding.
3. `grep -rni "deepseek" tools/ --include="*.py"` — must only appear in `privacy.BANNED_MEDIA_BRAINS` and its tests.
4. Run the suite: `.venv/bin/python -m pytest tests/ -q` — must be green, zero network (unplug to be sure; the suite is fully mocked except local ffmpeg renders).
5. `python tools/net.py` sanity: `check_url("https://evil.example/x")` must raise `EgressBlocked`.
6. Inspect `~/.myai/egress.jsonl` after any live run — every outbound request must have a log line. A request without a log line means it bypassed the gate: treat as a security bug.
7. Confirm `MODEL_REGISTRY` ToS notes still match the latest terms review before enabling any `conditional` route.

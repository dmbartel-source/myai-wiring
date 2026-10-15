# MYAI Media Wiring Package

The complete integration code for David's AI video-effects system — **ready to deploy the moment the VPS exists**. All reversible prep work: no real API calls were made, nothing was spent, everything is tested against mocks.

**Status:** built and tested 2026-10-08; security & privacy hardened 2026-10-08 (v1.1.0). Test suite: **139/139 passing** (`pytest tests/`).

## What each piece is

| Path | What it is |
|---|---|
| `skills/effect/SKILL.md` | The `$effect` skill: brief → prompt engineering → **LOCAL-FIRST routing** (deterministic recipes before any API) → **binding** keyframe/contact-sheet approval gate → cost estimate + hard cap → ToS-gated fal.ai generate → download → ffmpeg finish → 4-tier QC → deliver. ToS-gated model table (approved / conditional / rejected); Kling/Pika/MiniMax/Hailuo are refused by the client. |
| `skills/reference/SKILL.md` | The `$reference` skill: photo editing for VFX reference material. Routing rule (PIL vs OpenCV vs vision model vs generative API), search-first, deterministic-local-before-generative, thumbnail spec, JSON sidecar on every output. |
| `tools/net.py` | **Egress gate (new):** the single choke point for ALL outbound HTTP. Default-deny domain allowlist (fal.ai hosts, fal artifact CDN, openrouter.ai; loopback http only for the local endpoint); blocks everything else with `EgressBlocked`; logs every allowed AND blocked attempt as JSONL (query strings redacted). |
| `tools/privacy.py` | **Copyright-privacy policy in code (new):** asset sensitivity (default `sensitive`), per-call `allow_third_party=True` required for ANY third-party processing, ToS-status enforcement, and the **DeepSeek ban enforced in code** (`check_brain_for_media`). |
| `tools/fal_client.py` | fal.ai client: `submit` → poll with capped backoff (webhook preferred for long jobs), immediate atomic download, ED25519 webhook verification, job state machine (IN_QUEUE→IN_PROGRESS→COMPLETED), retry budgets by failure class (**0 retries on validation errors**, jittered backoff on transient), idempotency keys. **ToS-gated model registry** (`MODEL_REGISTRY` with approved/conditional/rejected; `require_route()` refuses rejected models outright). All HTTP via the egress gate; webhook URLs must be our own https. Reads `FAL_API_KEY` from env; fails cleanly without it. |
| `tools/local_effects.py` | **Deterministic local effect recipes (new):** 12 parameterized ffmpeg recipes (film grain, vignette, light leaks, glitch cuts, zoom punches, speed ramps, color grades, text overlays, shake, letterbox, fades, xfade transitions). **$0, no API, fully private** — the default route. `suggest()` maps descriptions to recipes; `preview()` renders short checks. |
| `tools/local_gen.py` | **Self-hosted diffusion hook (new, stub):** config + health check + call shape for a loopback diffusion endpoint (e.g. ComfyUI on 127.0.0.1:8188). Refuses loudly until explicitly enabled (`MYAI_LOCAL_GEN=1`) and healthy. The copyright-privacy endgame for generative work; see module docstring for wiring steps. |
| `tools/cost_ledger.py` | Spend ledger enforcing the cost policy **in code**: pre-generation estimate logged before every paid call; per-effect caps (draft $0.30 / final $2.00 / hero $10.00 + explicit approval); daily $5 soft / $10 hard; monthly $100 hard with 50/80/100% alerts. **New:** dry-run mode (plans spend without consequences) and per-project monthly budgets. Over-cap calls raise `BudgetBlocked`. JSONL log. |
| `tools/video_qa.py` | 4-tier QC gates returning JSON, non-zero exit on failure: T0 ffprobe structural (spec match ±3s, **fps conformance, audio_required**), T1 black/freeze/silence detect in one decode pass **+ volumedetect loudness (near-clipping peaks ≥ −1 dBFS fail)**, T2 SSIM ≥ 0.95 / PSNR ≥ 30 dB vs reference (VMAF deliberately skipped), T3 check-frame extraction **at decision points** for vision-model review. Missing ffmpeg → tiers marked `skipped`, never silently passed. |
| `tools/pipeline.py` | Plan-on-disk EDL executor: validated EDL JSON contract, deterministic stages, checkpoints between phases, **resumable** (completed stages never re-run — idempotent), SQLite job store (no broker), failure ladder (transient retry → fallback → loud failure with trail, never silent). **Policy refusals (ToS/privacy/egress/unwired) fail fast with zero retries.** Built-in tools: `fal_submit` (ToS + privacy gated), `fal_wait`, `download` (egress-gated), `ffmpeg`, `shell`, `qa`, `checkpoint`, **`local_fx`** (deterministic local recipes), **`local_gen`** (self-hosted hook); register custom tools via `register_tool()`. |
| `tools/prompt_director.py` | Effect description → engineered shot prompt. Pure functions, no network: 6-part formula, camera vocabulary (15 moves, one per clip), negative-prompt builder, deterministic seeds + house seed, `motion_only()` I2V variant. |
| `SECURITY.md` | **Security policy (new):** egress policy, privacy guarantee, model allowlist with ToS statuses, credential handling, and a reviewer checklist for re-auditing after any change. |
| `config/presets.json` | nanobot model preset templates: `triage` (DeepSeek V4.1 Flash, ZDR — **never for media/sensitive work**), `daily` (Sonnet 5.5), `deep` (Opus 5.5), `western` (GPT-6 Luna trial); fallback chain; OpenRouter extras (`zdr:true`, `data_collection:"deny"`, `require_parameters:true`); media spend caps. **Only `${ENV_VAR}` placeholders — no real keys.** |
| `tests/` | 139 pytest tests: the original 50 (all still passing) plus 89 new — egress allow/block + logging, privacy refusals + DeepSeek ban, ToS-gate refusals, webhook validation, 12 local recipes (filter strings + real ffmpeg renders), ledger dry-run + project budgets, QA loudness/fps/audio additions, pipeline policy fail-fast. All mocked except local ffmpeg renders; zero network, zero spend. |

## Deploy order on the real VPS

1. **Server ready** (Contabo Core 4, Ubuntu 24.04, hardened per runbook Phase 2) with nanobot installed (Phase 3).
2. **Copy this package** to the VPS: `tools/` → `~/.nanobot/workspace/media/tools/` (or `nanobot/agent/tools/` if wiring as native tools); `skills/` → `~/.nanobot/workspace/skills/`; `config/presets.json` → merge into `~/.nanobot/config.json`; `SECURITY.md` → keep next to the code for reviewers.
3. **Python venv** on the VPS: `python3 -m venv ~/.nanobot/workspace/media/.venv && .venv/bin/pip install pytest pynacl` (pynacl enables real ED25519 webhook verification — currently fail-closed without it).
4. **System packages**: `apt install ffmpeg` (QC tiers + local_effects recipes + finishing need real ffmpeg/ffprobe).
5. **Secrets** (mode-600 EnvironmentFile, never in chat/config): `FAL_API_KEY`, `OPENROUTER_API_KEY`.
6. **Verify**: `pytest tests/` → 139/139; `python tools/video_qa.py --help`; `python tools/local_effects.py list`; dry-run an EDL with the `checkpoint` tool only. Confirm the egress log appears at `~/.myai/egress.jsonl` after any networked test (or set `MYAI_EGRESS_LOG`).
7. **First live test**: fund fal.ai with a small amount, run one **draft-tier** `$effect` end-to-end on a non-sensitive asset, confirm the ledger AND the egress log, then enable finals.

## Local generative path (future wiring)

`tools/local_gen.py` is a stub defining the endgame for copyright-sensitive generative work: a diffusion server (ComfyUI or equivalent) on the VPS itself, reached only over loopback. The egress gate already permits loopback http, so no policy change is needed. Wiring steps are in the module docstring. Until it is wired, `local_gen` pipeline stages fail loudly with `LocalGenNotWired` — never silently, never falling back to a third party.

## What still needs keys / the live server

- **`FAL_API_KEY`** — fal.ai account + small prepaid balance. Nothing here has ever touched the real API.
- **`OPENROUTER_API_KEY`** — for the `daily`/`deep`/`triage`/`western` presets (separate from media spend).
- **Live price re-verification** — model $/s rates in `fal_client.py` are Oct 2026 research figures; re-check fal.ai model pages before wiring billing.
- **ToS vetting** — commercial video APIs may only be enabled after the terms review passes (copyright-privacy hard constraint). Fully-private paths (CPU render, self-hosted open models) need no vetting.
- **Real-box benchmarks** — ffmpeg throughput, VLM check-frame review costs, and the `reasoning_content` echo behavior for DeepSeek thinking mode all need live verification.
- **Webhook endpoint** — fal.ai webhooks need an inbound HTTPS receiver; until then the poll fallback works (it is the default in `fal_client.wait`).

## Design decisions worth knowing

- **Cost policy lives in code** (`cost_ledger.py`), not in prompts — agents ignore prompt budgets.
- **Copyright privacy is structural**: media work is routed away from DeepSeek at the preset level (`triage` preset is banned from media work in config), and the `$effect` skill carries the hard rule.
- **Never overwrite**: versioned takes, atomic renames, content-hash-friendly layout.
- **Loud failures only**: every failure path raises with a trail; nothing is silently dropped or retried blindly.

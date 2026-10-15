# CHANGELOG — MYAI media wiring package

## RunPod GPU (ComfyUI) backend (2026-10-14)

- **GPU client** (`tools/gpu_client.py`, new): ComfyUI API client for
  David's own RunPod pod. Config from `~/.nanobot/gpu.env`
  (`GPU_BASE_URL`, optional `GPU_API_KEY`). Upload image/video, queue
  workflows (`POST /prompt`), poll history, download results
  (`GET /view`), interrupt. Typed errors throughout — an offline pod
  raises `GpuUnreachableError` and Studio degrades gracefully (the pod
  is off most of the time; that's normal). All traffic goes through the
  `net.py` egress gate; `proxy.runpod.net` added to the allowlist for
  exactly this destination. Rental-share cost estimates via
  `estimate_cost_usd()` (H100 Secure rate, `GPU_HOURLY_USD` override).
- **Workflow builders** (`tools/gpu_workflows.py`, new): programmatic
  ComfyUI API-format workflows with every knob parameterized (prompt,
  negative, seed, steps, cfg, resolution, frames, fps, denoise,
  reference strength). `image_edit` (Qwen-Image-Edit-2509,
  identity-preserving photo edits), `image_to_video` / `text_to_video`
  (Wan 2.2 14B FP8, dual high/low-noise pass, native nodes),
  `video_replace_model` (Wan VACE 14B reference-to-video), plus a raw
  `queue_workflow()` escape hatch for arbitrary workflows. Open models
  only; prompts pass through exactly as written.
- **Pod bootstrap** (`runpod/`, new): `setup_pod.sh` (idempotent —
  installs ComfyUI + Manager/VideoHelperSuite/KJNodes onto the network
  volume and downloads the model set with hf_transfer), `models.txt`
  (the exact download list; kept in sync with `gpu_workflows.MODELS` by
  test), `README.md` (pod spec: H100 80GB SXM, Secure Cloud,
  on-demand only, 150GB volume — the model set is ~128GB, so the
  originally planned 75GB could not hold it).
- **Studio wiring** (`tools/studio_server.py`, `tools/studio_ui.html`):
  four new Create options — Edit Photo (Your GPU), Animate Photo
  (Your GPU), Text to Video (Your GPU), Replace Model (Your GPU) —
  dispatched in `/api/generate` ahead of the fal.ai gate (no third
  party involved), run as `gpu_generate` jobs in the existing job
  system with progress phases and pod interrupt on cancel. The UI adds
  a parameter block (seed/steps/cfg/size/frames/denoise) to the
  generate confirmation for GPU options and a base-video +
  reference-photo picker for Replace Model. No new media-serving
  endpoints; results flow through the existing job result route.
- Tests: `tests/test_gpu_client.py` (20) + `tests/test_gpu_workflows.py`
  (8), mock-transport only — no pod, no network, no spend.

## v1.3.0 — Face/body replacement VFX (2026-10-08, unreleased)

- **Face/body replacement pipeline** (`tools/face_vfx.py`, new): professional
  head-replacement VFX, fully local (OpenCV + NumPy, $0, private).
  `detect_faces` (DNN when models present, Haar fallback),
  `estimate_landmarks` (dlib 68-pt if available, else proportion heuristic),
  `align_face` (similarity warp), `composite_face` (feathered face-oval mask
  + LAB color-match + Poisson blend), `track_face` (CSRT/KCF/MIL with
  deterministic template fallback), `replace_in_video` (detect -> track ->
  composite per frame — "replace over a real movie"), `replace_region` +
  `ellipse_mask`/`rect_mask` for body parts and objects. Consent notice in
  module docstring.
- **Model downloader** (`tools/download_models.py`, new, not auto-run):
  fetches the OpenCV DNN face model and dlib 68-landmark predictor with
  documented URLs and sizes. The pipeline works without them (Haar fallback).
- **Skill** (`skills/face_swap/SKILL.md`, new): consent framing and the
  working procedure for face/body replacement jobs.
- Tests: 20 new (synthetic cartoon-face fixtures, no real faces), all passing.
- **Automatic metadata stripping** (`tools/privacy.py`): EVERY upload
  to MYAI (studio, upload page, fal.ai) now has ALL metadata stripped
  before storage or transmission — EXIF, GPS, device info, container tags.
  Pixels untouched. Enforced in code, not policy.

## v1.3.0 — Photo manipulation toolkit (2026-10-08, unreleased)

- **Photo toolkit** (`tools/photo_tools.py`, new): Photoshop-style local
  image manipulation. Pure PIL + OpenCV + NumPy, no network, deterministic.
  30 registered ops: adjustments (brightness/contrast/saturation/sharpness,
  levels, curves, white balance, HSL, vibrance, clarity), filters (gaussian/
  motion/lens blur, sharpen, unsharp mask, edge enhance, oil paint,
  watercolor, pencil sketch, posterize, cross-process, bleach bypass,
  faded film), geometry (resize/crop/rotate/straighten/perspective),
  thumbnail, text overlay. 11 exact-math blend modes (multiply, screen,
  overlay, soft/hard light, darken/lighten, difference, color dodge/burn)
  plus `composite_layers()` with feathered masks. Utilities: smart-crop
  thumbnails (saliency, face-agnostic), contact sheets, EXIF-preserving
  save. `suggest_ops()` keyword router + `apply_op`/`apply_chain`.
  31 tests, all passing.

## v1.3.0 — Media Studio + manipulation suite (2026-10-08, unreleased)

- **Media Studio** (`tools/studio_server.py` + `tools/studio_ui.html`, new):
  mobile-first web UI for David's phone. Big buttons, no chat. Screens:
  Upload (progress bars), Gallery (thumbnail grid), Effects (Style Packs /
  Photo / Video / Animate cards, all FREE), Generate (paid fal.ai options
  with tap-to-confirm cost estimates + free Ken Burns alternative),
  Result (live progress, preview, download). Backend: stdlib HTTP on
  127.0.0.1, thread-pool jobs (max 2), JSONL job log, 320px thumbnails
  (PIL/ffmpeg). Paid endpoints stubbed: `POST /api/generate` returns
  `needs_setup` without FAL_API_KEY, `coming_soon` once configured.
  21 tests, all passing.
- **$animate skill** (`tools/kenburns.py` + `tools/i2v_client.py` +
  `skills/animate/SKILL.md`, new): picture-to-video. Local-first: Ken Burns
  camera moves (zoom_in/out, pan x4, slow_push) via deterministic ffmpeg
  zoompan — free, instant, private, any input aspect, 1080p/720p/480p/source.
  Generative path: fal.ai I2V (`seedance-mini` default, `wan-25`) with
  ToS gate, privacy gate (`allow_third_party=True` required), cost ledger
  authorize/record, motion-only prompts via `prompt_director.motion_only`,
  image upload through the egress gate, download to `jobs/<id>/takes/`.
  21 tests (9 Ken Burns renders + 12 mocked I2V), all passing.
- **Video compositing** (`tools/video_composite.py`, new): 8 ffmpeg-based
  compositing ops — `chroma_key` (green/blue screen + despill, composites
  over a background), `picture_in_picture` (position presets, border, drop
  shadow), `split_screen` (1x2 / 2x1 / 2x2 grids with dividers),
  `freeze_frame` (frame hold with silence-filled audio gap), `reverse`
  (video + audio), `mask_wipe` (directional gradient-mask reveal between
  two clips), `speed_ramp_advanced` (slow-mo zone with pitch-corrected
  audio), `stabilize` (2-pass vidstab when available, deshake fallback,
  auto-detected). Registry `VIDEO_COMPOSITE_OPS` + `suggest_composite()`
  keyword matcher. 21 tests, all passing (incl. real renders).

## v1.2.0 — Style packs + private upload server (2026-10-08, unreleased)

- **Style packs** (`tools/style_packs.py`, new): 8 curated multi-effect chains
  (vhs, vintage_film, noir, cyberpunk, cinematic, dreamy, action,
  music_video) with natural-language matching. `match()` maps "make it look
  like an old VHS tape" to the right chain; `apply_pack()` renders it.
  9 tests, all passing.
- **Upload server** (`tools/upload_server.py`, new): stdlib-only private
  upload page for large phone videos (the WebUI caps video at 20 MiB).
  Binds 127.0.0.1, meant to sit behind `tailscale serve`; filename
  sanitization, 2 GiB/file cap. 2 tests, all passing.
- **Local effects expansion** (`tools/local_effects.py`): 10 new
  deterministic ffmpeg recipes (22 total): `film_burn`, `rgb_split`,
  `datamosh_lite`, `light_streak`, `vignette_pulse`, `whip_pan`,
  `duotone`, `vhs_tracking`, `blur_pulse`, `projector_flicker`. All
  local, $0, fully private. Keyword routing extended; `rgb split` and
  `datamosh` keywords moved from `glitch_cut` to their dedicated
  recipes. Tests: render + pixel-differs + duration/dims for all ten;
  time-gating verified (quiet frames untouched within colorspace
  tolerance).

## v1.1.0 — Security & privacy hardening (2026-10-08)

Security audit + hardening + capability expansion. Test suite: 139/139 passing
(50 original, all still green; 89 new). Zero network, zero spend in tests.

### Security & privacy (the point of this release)
- **Egress gate** (`tools/net.py`, new): single choke point for ALL outbound
  HTTP. Default-deny domain allowlist (fal.ai hosts, fal artifact CDN,
  openrouter.ai; loopback http only for the self-hosted endpoint). Blocks
  everything else with `EgressBlocked`; logs every allowed AND blocked
  attempt as JSONL with query strings redacted. Injected test transports
  still pass through the policy check.
- **Copyright-privacy policy in code** (`tools/privacy.py`, new): asset
  sensitivity (default `sensitive`), explicit per-call
  `allow_third_party=True` required for ANY third-party processing, ToS
  status enforcement, and the **DeepSeek ban enforced in code**
  (`check_brain_for_media`) instead of a config comment.
- **ToS-gated model registry** (`tools/fal_client.py`): `MODEL_REGISTRY`
  records approved/conditional/rejected per the Oct 2026 terms review.
  `require_route()` refuses Kling/Pika/MiniMax-Hailuo outright (no
  override); conditional models (Veo, Runway) need per-call `tos_ack=True`.
  Removed the dead `FAL_JWKS_URL` constant; webhook URLs must be our own
  https endpoint; error messages redact URLs instead of truncating them.
- **Pipeline policy fail-fast**: ToS/privacy/egress/unwired refusals raise
  `StageFailed` immediately — zero retries burned, loud trail preserved.
- **Docs**: `SECURITY.md` (egress policy, privacy guarantee, model
  allowlist, reviewer checklist), `$effect` skill rewritten local-first
  with the ToS-gated route table, `$reference` skill privacy note
  strengthened.

### Capability maximization
- **`tools/local_effects.py`** (new): 12 deterministic ffmpeg recipes
  (film grain, vignette, light leaks, glitch cuts, zoom punches, speed
  ramps with pitch-corrected audio, 5 color grades, text overlays,
  shake, letterbox, fades, xfade transitions). $0, no API, fully
  private — the default route. `suggest()` maps descriptions to recipes;
  `preview()` renders short checks. All 12 verified with real ffmpeg.
- **`tools/local_gen.py`** (new, stub): config + health check + call shape
  for a self-hosted diffusion endpoint on loopback (the copyright-privacy
  endgame). Refuses loudly until explicitly enabled and healthy.
- **`tools/video_qa.py`**: tier 1 gains a volumedetect loudness pass
  (near-clipping peaks ≥ −1 dBFS fail); tier 0 `expect` gains fps
  conformance and `audio_required`.
- **`tools/cost_ledger.py`**: dry-run mode (plans spend, never authorizes,
  never counts) and per-project monthly budgets (backward compatible —
  legacy entries without a project field still count globally).
- **Pipeline**: new `local_fx` (deterministic local recipes) and
  `local_gen` (self-hosted hook) stage tools; `fal_submit` now gates on
  ToS + privacy before estimating; `download` is egress-gated.

### Changed behavior to know about
- `fal_client.submit()` / `estimate_cost_usd()` take a new `tos_ack`
  keyword (default False). Rejected routes raise `FalTosError`.
- `CostLedger()` takes `dry_run` and `project_caps` keywords (defaults
  preserve old behavior). `authorize()` takes `project`; `record_actual()`
  takes `project`; `summary()` takes `project`.
- `pipeline` EDL `fal_submit` stages now need `allow_third_party: true`
  (and `tos_ack: true` for conditional models) or they fail fast.
- The `$effect` skill's "Standard final" route changed: Kling removed
  (ToS-rejected); finals route to `veo3-fast` (conditional, needs David's
  written clearance) or stay on approved draft models / local recipes.

## v1.0.0 — Initial wiring package (2026-10-08)
- `$effect` + `$reference` skills, fal.ai client, cost ledger, 4-tier QC,
  EDL pipeline, prompt director, nanobot presets, README. 50/50 mocked
  tests. No live API calls, no spend.

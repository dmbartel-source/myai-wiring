"""Image-to-video via fal.ai, built on tools/fal_client.py.

$animate's generative path (when Ken Burns camera moves aren't enough).
Pipeline: upload image -> submit I2V -> poll -> download to jobs/<id>/takes/.

All gates enforced, in order:
  1. ToS gate   — fal_client.require_route() (rejected models refused).
  2. Privacy    — privacy.authorize_third_party() needs allow_third_party=True.
  3. Cost       — cost_ledger.authorize() before the paid call; record_actual
                  after. BudgetBlocked stops the job before any spend.

No real API calls unless animate_image()/upload_image() are invoked with a
key. Everything is mockable via the `opener` injectable (see tests/).
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# API key loading: env var first, then ~/.nanobot/fal.env (mode 600).
# ---------------------------------------------------------------------------

def load_fal_key() -> str:
    """Return the fal.ai API key, or '' if not configured."""
    key = os.environ.get("FAL_API_KEY", "").strip()
    if key:
        return key
    # Fall back to the key file David installed on the server.
    for candidate in (
        os.path.expanduser("~/.nanobot/fal.env"),
        "/home/nanobot/.nanobot/fal.env",
    ):
        try:
            if os.path.isfile(candidate):
                with open(candidate) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("FAL_API_KEY="):
                            key = line.split("=", 1)[1].strip().strip("'\"")
                            if key:
                                return key
        except OSError:
            continue
    return ""

try:
    from . import fal_client
    from . import net
    from . import privacy
    from . import cost_ledger
    from . import prompt_director
except ImportError:  # tools/ used as plain scripts dir
    import fal_client
    import net
    import privacy
    import cost_ledger
    import prompt_director


# Draft-tier, ToS-approved, i2v-capable. wan-25 and seedance-mini only —
# conditional/hero models (veo3 etc.) need David's written clearance and
# are intentionally NOT offered here; use tools/fal_client.py directly.
I2V_MODELS = ("wan-25", "seedance-mini")

# fal.ai file storage upload (two-step: initiate -> PUT bytes).
# As of 2026 the old one-step https://fal.run/storage/upload multipart endpoint
# returns 404. The current flow (used by the official fal clients) is:
#   1. POST https://rest.fal.ai/storage/upload/initiate
#        {"file_name": ..., "content_type": ...}  -> {upload_url, file_url}
#   2. PUT raw bytes to upload_url (no Authorization header; signed URL)
#   3. file_url becomes the hosted URL usable as image_url / video_url.
# rest.fal.ai must be on the egress allowlist in tools/net.py.
FAL_UPLOAD_INITIATE_URL = (
    "https://rest.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3"
)

# Per-model I2V argument defaults. duration is a STRING on fal.ai
# ("4".."15" or "auto").
I2V_ARGUMENT_DEFAULTS = {
    "wan-25": {"duration": "5"},
    "seedance-mini": {"duration": "5", "resolution": "720p",
                      "aspect_ratio": "16:9", "generate_audio": False},
}


class I2VError(Exception):
    """Image-to-video pipeline failure (upload/submit/result)."""


def _multipart_encode(field_name: str, filename: str,
                      data: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = f"----myai-i2v-{os.urandom(8).hex()}"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        f'Content-Disposition: form-data; name="{field_name}"; '
        f'filename="{filename}"\r\n'.encode())
    body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.write(data)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    return body.getvalue(), boundary


def _upload_media(path: str, kind: str,
                  opener: Optional[Callable] = None) -> str:
    """Upload one media file (kind "image" or "video") to fal.ai storage.

    Returns the hosted file URL. Metadata is stripped from a temp copy
    first; the user's original is never modified. Raises I2VError.
    """
    api_key = load_fal_key()
    if not api_key:
        raise I2VError("FAL_API_KEY is not set — cannot upload to fal.ai")
    if not os.path.isfile(path):
        raise I2VError(f"media not found: {path}")
    # Privacy: strip ALL metadata before anything leaves the server.
    # Work on a sanitized temp copy; never modify the user's original.
    import tempfile
    from privacy import strip_metadata
    _tmp = tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(path)[1], delete=False)
    _tmp.close()
    try:
        import shutil
        shutil.copy2(path, _tmp.name)
        strip_metadata(_tmp.name)
        with open(_tmp.name, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(_tmp.name)
        except OSError:
            pass
    if len(data) > 100 * 1024 * 1024:
        raise I2VError(f"media exceeds fal.ai 100 MB upload limit: {path}")
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    if not ctype.startswith(kind + "/"):
        article = "an" if kind == "image" else "a"
        raise I2VError(f"not {article} {kind} (content-type {ctype}): {path}")

    # Two-step fal.ai storage upload (2026 flow):
    # 1. POST /storage/upload/initiate -> {upload_url, file_url}
    # 2. PUT raw bytes to upload_url (signed URL, no auth header)
    init_body = json.dumps({
        "file_name": os.path.basename(path),
        "content_type": ctype,
    }).encode("utf-8")
    init_req = urllib.request.Request(
        FAL_UPLOAD_INITIATE_URL, data=init_body, method="POST")
    init_req.add_header("Authorization", f"Key {api_key}")
    init_req.add_header("Content-Type", "application/json")
    init_req.add_header("Content-Length", str(len(init_body)))
    try:
        with net.gated_urlopen(init_req, timeout=60, opener=opener) as resp:
            init_payload = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise I2VError(f"media upload failed: {e}")
    upload_url = init_payload.get("upload_url", "")
    file_url = init_payload.get("file_url", "")
    if not upload_url or not file_url:
        raise I2VError(f"upload initiate returned no URLs: {init_payload}")
    put_req = urllib.request.Request(upload_url, data=data, method="PUT")
    put_req.add_header("Content-Type", ctype)
    put_req.add_header("Content-Length", str(len(data)))
    try:
        with net.gated_urlopen(put_req, timeout=120, opener=opener) as resp:
            resp.read()
    except Exception as e:
        raise I2VError(f"media upload failed: {e}")
    return file_url


def upload_image(image_path: str,
                 opener: Optional[Callable] = None) -> str:
    """Upload an image to fal.ai storage via the egress gate.

    Returns the hosted file URL. Raises I2VError on failure.
    """
    return _upload_media(image_path, "image", opener=opener)


def upload_video(video_path: str,
                 opener: Optional[Callable] = None) -> str:
    """Upload a video to fal.ai storage via the egress gate.

    Used for reference-to-video video refs. Returns the hosted file URL.
    Raises I2VError on failure.
    """
    return _upload_media(video_path, "video", opener=opener)


def build_i2v_arguments(
    image_url: str,
    shot: "prompt_director.ShotPrompt",
    model: str,
    duration_s: float = 5.0,
    **overrides: Any,
) -> dict:
    """Build the fal.ai I2V arguments payload for a model.

    shot: a prompt_director.ShotPrompt (use prompt_director.motion_only()
    for the recommended motion-only I2V prompt).
    """
    if model not in I2V_ARGUMENT_DEFAULTS:
        raise I2VError(
            f"no I2V argument profile for {model!r}; "
            f"supported: {list(I2V_ARGUMENT_DEFAULTS)}")
    args = dict(I2V_ARGUMENT_DEFAULTS[model])
    args["image_url"] = image_url
    args["prompt"] = shot.positive
    args["duration"] = str(int(duration_s))
    args.update(overrides)
    return args


def _extract_video_url(result: dict) -> str:
    v = result.get("video")
    if isinstance(v, dict) and v.get("url"):
        return v["url"]
    if isinstance(result.get("video_url"), str):
        return result["video_url"]
    # Last resort: first .mp4-ish URL anywhere in the payload.
    for key in ("url", "file_url", "access_url"):
        if isinstance(result.get(key), str):
            return result[key]
    raise I2VError(f"no video URL in result payload: {str(result)[:300]}")


def default_ledger_path() -> str:
    return os.environ.get(
        "MYAI_SPEND_LEDGER", os.path.expanduser("~/.myai/spend.jsonl"))


def default_jobs_root() -> str:
    return os.environ.get("MYAI_JOBS_ROOT",
                          os.path.join(os.getcwd(), "jobs"))


def animate_image(
    image_path: str,
    prompt: str,
    model: str = "seedance-mini",
    duration_s: float = 5.0,
    *,
    allow_third_party: bool = False,
    sensitivity: str = "sensitive",
    effect_id: Optional[str] = None,
    ledger: Optional["cost_ledger.CostLedger"] = None,
    jobs_root: Optional[str] = None,
    camera: str = "static",
    aspect: str = "16:9",
    tos_ack: bool = False,
    opener: Optional[Callable] = None,
    # --- Research-backed precision controls (Oct 2026 fal.ai research) ---
    enable_prompt_expansion: bool = False,
    end_image_url: Optional[str] = None,
    seed: Optional[int] = None,
    **i2v_overrides: Any,
) -> dict:
    """Full I2V pipeline. Returns a manifest dict.

    Gates (fail closed, in order): ToS -> privacy -> cost. The paid call
    happens only after all three pass. Spend is recorded afterwards.

    prompt: plain motion description, e.g. "slow dolly in, hair stirring
    in the wind". A motion-only engineered prompt is built via
    prompt_director.motion_only(). The prompt is passed to fal.ai exactly
    as engineered — this code never filters, sanitizes, or alters prompt
    content for any reason.

    Research-backed precision controls:
      enable_prompt_expansion: Wan 2.5 only. fal.ai runs an LLM over your
        prompt by default (enable_prompt_expansion=true) and it can inject
        unprompted content (their own docs show it adding choir sounds and
        armor details nobody asked for). Defaults to False here because
        David wants precise control — what he writes is what gets sent.
        Ignored for non-Wan models.
      end_image_url: Seedance only. URL of an end-frame image; pins BOTH
        the start and end frames so the model interpolates between them
        instead of drifting. Maximum source fidelity for locked shots.
        Ignored for non-Seedance models.
      seed: all models. Integer seed for reproducibility — lock it once a
        composition reads, reuse byte-identical prompts across clips for
        character consistency. None (default) lets fal.ai randomize.

    The returned manifest includes "actual_prompt" when the API reports
    what prompt was actually used (Wan 2.5 with expansion enabled) —
    compare it against your input to see what the rewriter changed.
    """
    # 1. ToS gate + model support check.
    spec = fal_client.require_route(model, tos_ack=tos_ack)
    if model not in I2V_MODELS or "i2v" not in spec.get("supports", []):
        raise I2VError(
            f"model {model!r} is not offered for I2V here; "
            f"supported: {list(I2V_MODELS)}")

    # 2. Privacy gate.
    asset = privacy.Asset(path=image_path, sensitivity=sensitivity)
    route_tos = spec["tos"]
    if route_tos == "conditional" and tos_ack:
        route_tos = "conditional+ack"
    privacy.authorize_third_party(
        asset, allow_third_party=allow_third_party,
        route_tos=route_tos, model=model)

    # 3. Cost gate.
    estimate = fal_client.estimate_cost_usd(model, duration_s, tos_ack=tos_ack)
    tier = spec["tier"]
    effect_id = effect_id or f"i2v-{int(time.time())}"
    ledger = ledger or cost_ledger.CostLedger(default_ledger_path())
    auth = ledger.authorize(effect_id, tier, estimate)

    # 4. Upload the image (egress-gated).
    image_url = upload_image(image_path, opener=opener)

    # 5. Engineer the motion-only prompt, submit, wait.
    shot = prompt_director.motion_only(
        image_path, prompt, camera=camera, duration_s=duration_s,
        aspect=aspect)
    arguments = build_i2v_arguments(
        image_url, shot, model, duration_s, **i2v_overrides)
    # --- Precision controls (model-aware passthrough) ---
    if seed is not None:
        arguments["seed"] = seed
    if model == "wan-25":
        # Wan 2.5 prompt-expansion LLM: default OFF for precise control.
        arguments["enable_prompt_expansion"] = enable_prompt_expansion
    if model.startswith("seedance") and end_image_url:
        # Pin start + end frames for maximum fidelity.
        arguments["end_image_url"] = end_image_url
    job = fal_client.submit(model, arguments, opener=opener, tos_ack=tos_ack)
    result = fal_client.wait(job, opener=opener)

    # 6. Download immediately (provider URLs expire).
    video_url = _extract_video_url(result)
    jobs_root = jobs_root or default_jobs_root()
    take_dir = os.path.join(jobs_root, effect_id, "takes")
    os.makedirs(take_dir, exist_ok=True)
    dest = os.path.join(take_dir, "take01.mp4")
    fal_client.download_to_file(video_url, dest, opener=opener)

    # 7. Record actual spend.
    ledger.record_actual(effect_id, tier, estimate,
                         detail=f"i2v {model} {duration_s}s")

    manifest = {
        "effect_id": effect_id,
        "model": model,
        "tier": tier,
        "estimate_usd": estimate,
        "authorization": auth,
        "image_url": image_url,
        "prompt": shot.to_dict(),
        "request_id": job.request_id,
        "output_path": dest,
        "video_url": video_url,
    }
    # Wan 2.5 reports the prompt actually used (after expansion rewriting).
    if isinstance(result.get("actual_prompt"), str):
        manifest["actual_prompt"] = result["actual_prompt"]
    return manifest


# ------------------------------------------------- reference-to-video ----

# Seedance 2.0 reference-to-video caps (Oct 2026 research).
REFERENCE_LIMITS = {"image": 9, "video": 3, "audio": 3}
R2V_MODEL = "seedance-2.0"

# Research pricing figures (Oct 2026): ~$0.30/s at 720p for image refs;
# video refs get a 40% discount on the request. Re-verify live before
# quoting billing.
R2V_USD_PER_SEC = 0.30
R2V_VIDEO_REF_MULTIPLIER = 0.6


def _validate_reference_lists(image_paths: list,
                              video_paths: list) -> None:
    image_paths = image_paths or []
    video_paths = video_paths or []
    if not image_paths and not video_paths:
        raise I2VError("reference_to_video needs at least one image or "
                       "video reference")
    if len(image_paths) > REFERENCE_LIMITS["image"]:
        raise I2VError(
            f"too many image references ({len(image_paths)}); "
            f"Seedance 2.0 allows up to {REFERENCE_LIMITS['image']}")
    if len(video_paths) > REFERENCE_LIMITS["video"]:
        raise I2VError(
            f"too many video references ({len(video_paths)}); "
            f"Seedance 2.0 allows up to {REFERENCE_LIMITS['video']}")


def build_reference_arguments(
    image_urls: list[str],
    shot: "prompt_director.ShotPrompt",
    video_urls: Optional[list[str]] = None,
    duration_s: float = 5.0,
    resolution: str = "720p",
    **overrides: Any,
) -> dict:
    """Build the Seedance 2.0 reference-to-video arguments payload.

    image_urls: 1-9 hosted image URLs (cited in the prompt as @Image1..).
    video_urls: 0-3 hosted video URLs (cited as @Video1..).
    shot: a prompt_director.ShotPrompt — use
    prompt_director.build_reference_prompt() so the prompt references
    the refs by their @ImageN/@VideoN tags.
    """
    image_urls = list(image_urls or [])
    video_urls = list(video_urls or [])
    _validate_reference_lists(image_urls, video_urls)
    args: dict[str, Any] = {
        "image_urls": image_urls,
        "prompt": shot.positive,
        "duration": str(int(duration_s)),
        "resolution": resolution,
    }
    if video_urls:
        args["video_urls"] = video_urls
    args.update(overrides)
    return args


def estimate_reference_cost_usd(duration_s: float,
                                has_video_refs: bool) -> float:
    """Research-based R2V estimate. Re-verify live before billing."""
    estimate = R2V_USD_PER_SEC * duration_s
    if has_video_refs:
        estimate *= R2V_VIDEO_REF_MULTIPLIER  # 40% video-ref discount
    return round(estimate, 4)


def reference_to_video(
    image_paths: Optional[list[str]] = None,
    video_paths: Optional[list[str]] = None,
    prompt: str = "",
    duration_s: float = 5.0,
    resolution: str = "720p",
    *,
    image_roles: Optional[list[str]] = None,
    video_roles: Optional[list[str]] = None,
    allow_third_party: bool = False,
    sensitivity: str = "sensitive",
    effect_id: Optional[str] = None,
    ledger: Optional["cost_ledger.CostLedger"] = None,
    jobs_root: Optional[str] = None,
    camera: str = "static",
    aspect: str = "16:9",
    tos_ack: bool = False,
    seed: Optional[int] = None,
    opener: Optional[Callable] = None,
    **r2v_overrides: Any,
) -> dict:
    """Reference-to-video pipeline (Seedance 2.0). Returns a manifest dict.

    Uploads up to 9 image refs + up to 3 video refs and generates video
    guided by them — the multi-reference path for stronger character /
    model consistency. Single-image calls should keep using
    image_to_video() via animate_image(); this is the 2+ refs route.

    Gates (fail closed, in order): ToS -> privacy (every ref asset) ->
    cost. The paid call happens only after all three pass.

    The prompt is engineered by prompt_director.build_reference_prompt()
    so refs are cited as @Image1..@ImageN / @Video1..@VideoM by role.
    image_roles / video_roles align positionally with image_paths /
    video_paths; omitted roles get sensible defaults. The prompt is
    passed to fal.ai exactly as engineered — this code never filters,
    sanitizes, or alters prompt content for any reason.
    """
    image_paths = list(image_paths or [])
    video_paths = list(video_paths or [])
    _validate_reference_lists(image_paths, video_paths)

    # 1. ToS gate.
    spec = fal_client.require_route(R2V_MODEL, tos_ack=tos_ack)
    tier = spec["tier"]
    route_tos = spec["tos"]

    # 2. Privacy gate: every reference asset is authorized individually.
    assets = ([privacy.Asset(path=p, sensitivity=sensitivity)
               for p in image_paths]
              + [privacy.Asset(path=p, sensitivity=sensitivity)
                 for p in video_paths])
    for asset in assets:
        privacy.authorize_third_party(
            asset, allow_third_party=allow_third_party,
            route_tos=route_tos, model=R2V_MODEL)

    # 3. Cost gate.
    estimate = estimate_reference_cost_usd(duration_s,
                                           has_video_refs=bool(video_paths))
    effect_id = effect_id or f"r2v-{int(time.time())}"
    ledger = ledger or cost_ledger.CostLedger(default_ledger_path())
    auth = ledger.authorize(effect_id, tier, estimate)

    # 4. Upload all references (egress-gated, metadata stripped).
    image_urls = [upload_image(p, opener=opener) for p in image_paths]
    video_urls = [upload_video(p, opener=opener) for p in video_paths]

    # 5. Engineer the reference prompt, submit, wait.
    shot = prompt_director.build_reference_prompt(
        image_roles=image_roles, video_roles=video_roles,
        n_images=len(image_urls), n_videos=len(video_urls),
        motion=prompt, camera=camera, duration_s=duration_s, aspect=aspect)
    arguments = build_reference_arguments(
        image_urls, shot, video_urls=video_urls, duration_s=duration_s,
        resolution=resolution, **r2v_overrides)
    if seed is not None:
        arguments["seed"] = seed
    job = fal_client.submit(R2V_MODEL, arguments, opener=opener,
                            tos_ack=tos_ack)
    result = fal_client.wait(job, opener=opener)

    # 6. Download immediately (provider URLs expire).
    video_url = _extract_video_url(result)
    jobs_root = jobs_root or default_jobs_root()
    take_dir = os.path.join(jobs_root, effect_id, "takes")
    os.makedirs(take_dir, exist_ok=True)
    dest = os.path.join(take_dir, "take01.mp4")
    fal_client.download_to_file(video_url, dest, opener=opener)

    # 7. Record actual spend.
    ledger.record_actual(effect_id, tier, estimate,
                         detail=f"r2v {R2V_MODEL} {duration_s}s "
                                f"{len(image_urls)}img/{len(video_urls)}vid")

    manifest = {
        "effect_id": effect_id,
        "model": R2V_MODEL,
        "tier": tier,
        "estimate_usd": estimate,
        "authorization": auth,
        "image_urls": image_urls,
        "video_urls": video_urls,
        "prompt": shot.to_dict(),
        "request_id": job.request_id,
        "output_path": dest,
        "video_url": video_url,
    }
    if isinstance(result.get("actual_prompt"), str):
        manifest["actual_prompt"] = result["actual_prompt"]
    return manifest


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="fal.ai image-to-video")
    ap.add_argument("image", nargs="?", default=None,
                    help="input image path (single-image mode)")
    ap.add_argument("prompt", nargs="?", default=None,
                    help="motion description")
    ap.add_argument("--reference-images", nargs="+", default=None,
                    help="2-9 reference images for better character "
                         "consistency (reference-to-video mode)")
    ap.add_argument("--reference-videos", nargs="+", default=None,
                    help="up to 3 reference video clips "
                         "(reference-to-video mode)")
    ap.add_argument("--image-role", nargs="+", default=None,
                    help="role labels for --reference-images, in order "
                         '(e.g. "character front view" "outfit detail")')
    ap.add_argument("--video-role", nargs="+", default=None,
                    help="role labels for --reference-videos, in order")
    ap.add_argument("--model", default="seedance-mini", choices=list(I2V_MODELS))
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--resolution", default="720p",
                    help="reference-to-video resolution (e.g. 720p, 1080p)")
    ap.add_argument("--allow-third-party", action="store_true")
    ap.add_argument("--effect-id", default=None)
    ap.add_argument("--enable-prompt-expansion", action="store_true",
                    help="Wan 2.5 only: let fal.ai's LLM rewrite the prompt "
                         "(default off for precise control)")
    ap.add_argument("--end-image-url", default=None,
                    help="Seedance only: URL of an end-frame image to pin "
                         "both start and end frames")
    ap.add_argument("--seed", type=int, default=None,
                    help="integer seed for reproducibility")
    a = ap.parse_args(argv)
    if a.reference_images or a.reference_videos:
        if not a.prompt:
            ap.error("a prompt is required with --reference-images/"
                     "--reference-videos")
        manifest = reference_to_video(
            image_paths=a.reference_images, video_paths=a.reference_videos,
            prompt=a.prompt, duration_s=a.duration,
            resolution=a.resolution, image_roles=a.image_role,
            video_roles=a.video_role,
            allow_third_party=a.allow_third_party, effect_id=a.effect_id,
            seed=a.seed)
    else:
        if not a.image or not a.prompt:
            ap.error("image and prompt are required (or use "
                     "--reference-images/--reference-videos)")
        manifest = animate_image(
            a.image, a.prompt, model=a.model, duration_s=a.duration,
            allow_third_party=a.allow_third_party, effect_id=a.effect_id,
            enable_prompt_expansion=a.enable_prompt_expansion,
            end_image_url=a.end_image_url, seed=a.seed)
    print(json.dumps({k: v for k, v in manifest.items()
                      if k != "prompt"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

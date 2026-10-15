"""Copyright-privacy policy enforced in code, not in prompts.

David's hard rule: the products of MYAI are completely private for
copyright reasons. This module encodes that rule so a misconfigured
prompt or a careless agent cannot silently route his IP through a
third party.

Model:

  - Every media asset carries a Sensitivity. The DEFAULT is SENSITIVE:
    David's footage, descriptions, and outputs are his IP unless he
    says otherwise.
  - Third-party processing ALWAYS requires an explicit per-call flag
    (allow_third_party=True). There is no standing permission, no
    "the skill said it was fine", no config toggle that quietly opts
    everything in. The flag is per call, per asset, on purpose.
  - Model routes carry a ToS status (see fal_client.MODEL_REGISTRY):
    approved / conditional / rejected. Rejected models are refused
    outright; conditional models additionally require tos_ack=True.
  - The DeepSeek ban is enforced here, in code: any brain/provider id
    containing "deepseek" is refused for media work. A config comment
    is not enforcement; this function is.

Nothing here touches the network. These are pure policy checks that
raise PrivacyBlocked (fail closed) before any bytes move.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PrivacyBlocked(Exception):
    """Raised when a media operation would violate the privacy policy."""


class Sensitivity(str, Enum):
    SENSITIVE = "sensitive"  # David's IP. Local-only unless he explicitly says otherwise.
    PRIVATE = "private"      # David's, but cleared for ToS-vetted third parties per call.
    PUBLIC = "public"        # No IP concern (e.g. stock/test media).


# Substring match, case-insensitive, against brain/provider/model ids.
BANNED_MEDIA_BRAINS = ("deepseek",)


@dataclass
class Asset:
    """A media asset under the privacy policy."""
    path: str = ""
    sensitivity: Sensitivity = Sensitivity.SENSITIVE
    note: str = ""

    def __post_init__(self):
        if isinstance(self.sensitivity, str):
            try:
                self.sensitivity = Sensitivity(self.sensitivity.lower())
            except ValueError:
                raise PrivacyBlocked(
                    f"Unknown sensitivity {self.sensitivity!r}; expected one of "
                    f"{[s.value for s in Sensitivity]}"
                )


def check_brain_for_media(brain_id: Optional[str]) -> None:
    """Enforce the DeepSeek ban on media work. Raises PrivacyBlocked.

    `brain_id` may be a preset name, provider id, or model slug — anything
    containing a banned substring (case-insensitive) is refused. Empty/None
    means "no brain specified": nothing to check, passes.
    """
    if not brain_id:
        return
    lowered = str(brain_id).lower()
    for banned in BANNED_MEDIA_BRAINS:
        if banned in lowered:
            raise PrivacyBlocked(
                f"Media work may never use {brain_id!r}: it matches the banned "
                f"brain {banned!r} (trains on inputs by default; copyright-privacy "
                f"hard rule). Route media planning/review through the "
                f"'daily' (Sonnet) or 'deep' (Opus) presets instead."
            )


def authorize_third_party(
    asset: Optional[Asset] = None,
    *,
    allow_third_party: bool = False,
    route_tos: str = "approved",
    brain: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Gate for any third-party media processing. Fail closed.

    Checks, in order:
      1. Brain ban (DeepSeek etc. never touch media work).
      2. Route ToS status: "rejected" -> refused outright; "conditional"
         without an explicit ack -> refused (pass route_tos="conditional+ack"
         once David has cleared that model in writing).
      3. Per-call consent: allow_third_party=True is required for ANY
         third-party contact, regardless of sensitivity. SENSITIVE assets
         additionally require David's explicit say-so per the skill's
         approval gate — this function enforces the flag, the skill
         enforces whose finger is on it.

    Returns an authorization record on success. Raises PrivacyBlocked.
    """
    check_brain_for_media(brain)
    check_brain_for_media(model)

    tos = route_tos.lower()
    if tos == "rejected":
        raise PrivacyBlocked(
            "Route ToS status is 'rejected' (Oct 2026 ToS review: IP/training "
            "terms unacceptable). Refused outright — no override exists."
        )
    if tos == "conditional":
        raise PrivacyBlocked(
            "Route ToS status is 'conditional' (needs written clearance, paid "
            "tier, or commercial authorization per the Oct 2026 review). "
            "Re-check with route_tos='conditional+ack' after David clears it."
        )
    if tos not in ("approved", "conditional+ack"):
        raise PrivacyBlocked(f"Unknown route ToS status {route_tos!r} — failing closed.")

    asset = asset or Asset()
    if not allow_third_party:
        raise PrivacyBlocked(
            f"Third-party processing of {asset.path or 'this asset'} "
            f"(sensitivity={asset.sensitivity.value}) requires explicit "
            f"per-call allow_third_party=True. Default is local-only: use "
            f"tools/local_effects.py or the local generative endpoint. "
            f"No silent transmission is permitted."
        )
    return {
        "authorized": True,
        "asset_path": asset.path,
        "sensitivity": asset.sensitivity.value,
        "route_tos": tos,
        "allow_third_party": True,
    }


def default_brain_check() -> None:
    """Convenience: check the MYAI_MEDIA_BRAIN env var if the deploy sets it.

    Deployments may export MYAI_MEDIA_BRAIN=<preset-or-model> so every
    media stage validates the brain it runs under. Unset = skipped.
    """
    check_brain_for_media(os.environ.get("MYAI_MEDIA_BRAIN", ""))


# ------------------------------------------------- metadata stripping ----

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def strip_image_metadata(src: str, dest: Optional[str] = None) -> str:
    """Rewrite an image with ALL metadata removed (EXIF, GPS, IPTC, XMP).

    Pixels are preserved exactly; only metadata is dropped. If dest is None,
    the source file is replaced in place (via a temp file + atomic rename).
    Returns the path written.
    """
    from PIL import Image

    dest = dest or src
    tmp = dest + ".strip_tmp"
    try:
        with Image.open(src) as im:
            # Fresh image from pixel data: no EXIF, no ICC surprises carried
            # over except color profile (kept for correct rendering).
            icc = im.info.get("icc_profile")
            clean = Image.new(im.mode, im.size)
            clean.putdata(list(im.getdata()))
            save_kw: dict = {"format": im.format or "PNG"}
            if icc:
                save_kw["icc_profile"] = icc
            # PNG text chunks / JPEG APP markers are dropped by not copying
            # im.info / im.text.
            clean.save(tmp, **save_kw)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp) and tmp != dest:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return dest


def strip_video_metadata(src: str, dest: Optional[str] = None) -> str:
    """Rewrite a video with ALL container/stream metadata removed.

    Uses ffmpeg `-map_metadata -1` with stream copy (no re-encode, pixels
    untouched). Falls back to a clear error if ffmpeg is missing.
    Returns the path written.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not installed — cannot strip video metadata")
    dest = dest or src
    tmp = dest + ".strip_tmp.mp4"
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-y", "-i", src,
             "-map_metadata", "-1", "-c", "copy", tmp],
            capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg metadata strip failed: "
                               f"{proc.stderr[-500:]}")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp) and tmp != dest:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return dest


def strip_metadata(path: str, dest: Optional[str] = None) -> str:
    """Strip ALL metadata from an image or video. Dispatches by extension.

    This is the single entry point every MYAI upload path must call before
    the file is stored or sent anywhere. Unknown extensions are left alone
    (returned unchanged) — callers handling sensitive types should validate
    extensions first.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return strip_image_metadata(path, dest)
    if ext in VIDEO_EXTS:
        return strip_video_metadata(path, dest)
    return path

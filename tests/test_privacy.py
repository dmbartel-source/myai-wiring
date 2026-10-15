"""Tests for tools/privacy.py — the copyright-privacy policy in code."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import privacy
from privacy import (
    Asset, PrivacyBlocked, Sensitivity,
    authorize_third_party, check_brain_for_media,
)


def test_default_sensitivity_is_sensitive():
    a = Asset(path="clip.mp4")
    assert a.sensitivity == Sensitivity.SENSITIVE


def test_unknown_sensitivity_fails_closed():
    with pytest.raises(PrivacyBlocked):
        Asset(path="x", sensitivity="topsecret")


def test_third_party_requires_explicit_per_call_flag():
    with pytest.raises(PrivacyBlocked, match="allow_third_party=True"):
        authorize_third_party(Asset("clip.mp4"), route_tos="approved")


def test_third_party_ok_with_flag_and_approved_tos():
    rec = authorize_third_party(
        Asset("clip.mp4", Sensitivity.PRIVATE),
        allow_third_party=True, route_tos="approved")
    assert rec["authorized"] is True
    assert rec["sensitivity"] == "private"


def test_rejected_tos_refused_even_with_flag():
    with pytest.raises(PrivacyBlocked, match="[Rr]ejected"):
        authorize_third_party(Asset("clip.mp4"), allow_third_party=True,
                              route_tos="rejected")


def test_conditional_tos_needs_ack():
    with pytest.raises(PrivacyBlocked, match="[Cc]onditional"):
        authorize_third_party(Asset("clip.mp4"), allow_third_party=True,
                              route_tos="conditional")
    rec = authorize_third_party(Asset("clip.mp4"), allow_third_party=True,
                                route_tos="conditional+ack")
    assert rec["authorized"] is True


def test_unknown_tos_fails_closed():
    with pytest.raises(PrivacyBlocked):
        authorize_third_party(Asset("clip.mp4"), allow_third_party=True,
                              route_tos="whatever")


@pytest.mark.parametrize("brain", [
    "deepseek/deepseek-v4.1-flash",
    "DeepSeek-R1",
    "triage",  # preset name alone is fine...
])
def test_deepseek_brain_banned_for_media(brain):
    if brain == "triage":
        check_brain_for_media(brain)  # preset names are not model ids
    else:
        with pytest.raises(PrivacyBlocked, match="[Dd]eepseek"):
            check_brain_for_media(brain)


@pytest.mark.parametrize("brain", [
    "anthropic/claude-sonnet-5.5",
    "anthropic/claude-opus-5.5",
    "openai/gpt-6-luna",
    "daily",
    None,
    "",
])
def test_allowed_brains_pass(brain):
    check_brain_for_media(brain)  # no raise


def test_brain_ban_applies_inside_authorize():
    with pytest.raises(PrivacyBlocked, match="[Dd]eepseek"):
        authorize_third_party(Asset("clip.mp4"), allow_third_party=True,
                              route_tos="approved",
                              brain="deepseek/deepseek-v4.1-flash")


def test_default_brain_check_uses_env(monkeypatch):
    monkeypatch.setenv("MYAI_MEDIA_BRAIN", "deepseek/deepseek-v4.1-flash")
    with pytest.raises(PrivacyBlocked):
        privacy.default_brain_check()
    monkeypatch.setenv("MYAI_MEDIA_BRAIN", "daily")
    privacy.default_brain_check()  # no raise
    monkeypatch.delenv("MYAI_MEDIA_BRAIN")
    privacy.default_brain_check()  # unset: skipped, no raise


# ------------------------------------------------- metadata stripping ----

def test_strip_image_metadata_removes_exif(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    import privacy

    src = str(tmp_path / "photo.jpg")
    img = Image.new("RGB", (64, 64), color="blue")
    exif = img.getexif()
    exif[271] = "SpyCamera"  # Make
    exif[272] = "SpyModel"   # Model
    exif[306] = "2026:01:01 00:00:00"  # DateTime
    img.save(src, exif=exif)
    assert len(Image.open(src).getexif()) > 0

    privacy.strip_metadata(src)
    assert len(Image.open(src).getexif()) == 0
    # Pixels survive.
    assert Image.open(src).getpixel((32, 32))[2] > 200


def test_strip_metadata_passthrough_non_media(tmp_path):
    import privacy
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    assert privacy.strip_metadata(str(f)) == str(f)
    assert f.read_text() == "hello"


def test_strip_video_metadata(tmp_path):
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    import privacy

    src = str(tmp_path / "clip.mp4")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", "-f", "lavfi",
         "-i", "testsrc=size=160x120:rate=10:duration=1",
         "-metadata", "title=SecretTitle", "-metadata", "comment=GPS:here",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", src],
        capture_output=True, check=True, timeout=60)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags",
         "-of", "csv=p=0", src], capture_output=True, text=True, timeout=30)
    assert "SecretTitle" in probe.stdout

    privacy.strip_metadata(src)
    probe2 = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags",
         "-of", "csv=p=0", src], capture_output=True, text=True, timeout=30)
    assert "SecretTitle" not in probe2.stdout

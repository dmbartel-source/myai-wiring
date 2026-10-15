"""Tests for tools/i2v_client.py — everything mocked, no network, no spend."""

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import i2v_client
from i2v_client import (
    I2V_MODELS, build_i2v_arguments, upload_image, upload_video,
    animate_image, I2VError, _extract_video_url,
    build_reference_arguments, reference_to_video,
    estimate_reference_cost_usd,
)
import fal_client
import cost_ledger
import privacy
from privacy import PrivacyBlocked
from cost_ledger import BudgetBlocked


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("FAL_API_KEY", raising=False)
    monkeypatch.setenv("FAL_API_KEY", "test-key")


def _make_image(path):
    from PIL import Image
    Image.new("RGB", (64, 64), "red").save(path)


class FakeUploadResp:
    def __init__(self, payload):
        self._p = payload

    def read(self):
        return json.dumps(self._p).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _upload_opener(url):
    def opener(req, timeout=None):
        if "storage/upload/initiate" in req.full_url:
            assert req.get_header("Authorization") == "Key test-key"
            return FakeUploadResp({"upload_url": "https://fal.media/signed-put",
                                   "file_url": url})
        # Signed PUT to the upload URL carries no auth header.
        return FakeUploadResp({})
    return opener


def test_upload_image_returns_url(tmp_path):
    img = str(tmp_path / "pic.png")
    _make_image(img)
    url = upload_image(img, opener=_upload_opener("https://v3.fal.media/x.png"))
    assert url == "https://v3.fal.media/x.png"


def test_upload_image_missing_file():
    with pytest.raises(I2VError, match="not found"):
        upload_image("/nope/missing.png")


def test_upload_image_rejects_non_image(tmp_path):
    f = str(tmp_path / "notes.txt")
    open(f, "w").write("hello")
    with pytest.raises(I2VError, match="not an image"):
        upload_image(f, opener=_upload_opener("https://x"))


def test_build_i2v_arguments_seedance():
    import prompt_director as pd
    shot = pd.motion_only("img.png", "slow push in")
    args = build_i2v_arguments("https://img", shot, "seedance-mini", 5.0)
    assert args["image_url"] == "https://img"
    assert args["duration"] == "5"
    assert "push" in args["prompt"].lower() or "animate" in args["prompt"].lower()
    assert args["resolution"] == "720p"


def test_build_i2v_arguments_wan():
    import prompt_director as pd
    shot = pd.motion_only("img.png", "pan left")
    args = build_i2v_arguments("https://img", shot, "wan-25", 5.0)
    assert args["image_url"] == "https://img"


def test_build_i2v_arguments_unknown_model():
    import prompt_director as pd
    shot = pd.motion_only("img.png", "x")
    with pytest.raises(I2VError, match="no I2V argument profile"):
        build_i2v_arguments("https://img", shot, "veo3")


def test_extract_video_url():
    assert _extract_video_url({"video": {"url": "https://v.mp4"}}) == "https://v.mp4"
    assert _extract_video_url({"video_url": "https://w.mp4"}) == "https://w.mp4"
    with pytest.raises(I2VError):
        _extract_video_url({"nothing": 1})


def _mock_fal(monkeypatch, video_url="https://cdn/video.mp4"):
    job = fal_client.FalJob(request_id="req-123", endpoint="e",
                            idempotency_key="k")

    def fake_submit(route, arguments, opener=None, tos_ack=False, **kw):
        assert route in I2V_MODELS
        assert "image_url" in arguments and "prompt" in arguments
        return job

    def fake_wait(j, opener=None, **kw):
        assert j.request_id == "req-123"
        return {"video": {"url": video_url}}

    downloaded = {}

    def fake_download(url, dest, opener=None):
        assert url == video_url
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"FAKEMP4")
        downloaded["dest"] = dest
        return dest

    monkeypatch.setattr(fal_client, "submit", fake_submit)
    monkeypatch.setattr(fal_client, "wait", fake_wait)
    monkeypatch.setattr(fal_client, "download_to_file", fake_download)
    return downloaded


def test_animate_image_full_flow(tmp_path, monkeypatch):
    img = str(tmp_path / "pic.png")
    _make_image(img)
    _mock_fal(monkeypatch)
    ledger = cost_ledger.CostLedger(str(tmp_path / "spend.jsonl"))
    manifest = animate_image(
        img, "slow push in", model="seedance-mini", duration_s=5.0,
        allow_third_party=True, ledger=ledger,
        jobs_root=str(tmp_path / "jobs"),
        opener=_upload_opener("https://v3.fal.media/pic.png"))
    assert manifest["model"] == "seedance-mini"
    assert manifest["request_id"] == "req-123"
    assert os.path.isfile(manifest["output_path"])
    assert manifest["output_path"].startswith(str(tmp_path / "jobs"))
    assert "/takes/take01.mp4" in manifest["output_path"]
    assert manifest["estimate_usd"] > 0
    # spend was recorded
    summary = ledger.summary()
    assert summary["day_spend_usd"] > 0


def test_animate_image_privacy_blocks(tmp_path, monkeypatch):
    img = str(tmp_path / "pic.png")
    _make_image(img)
    _mock_fal(monkeypatch)
    ledger = cost_ledger.CostLedger(str(tmp_path / "spend.jsonl"))
    with pytest.raises(PrivacyBlocked):
        animate_image(img, "push in", allow_third_party=False,
                      ledger=ledger, jobs_root=str(tmp_path / "jobs"),
                      opener=_upload_opener("https://x"))


def test_animate_image_rejects_bad_model(tmp_path, monkeypatch):
    img = str(tmp_path / "pic.png")
    _make_image(img)
    ledger = cost_ledger.CostLedger(str(tmp_path / "spend.jsonl"))
    with pytest.raises(Exception):  # FalTosError or I2VError
        animate_image(img, "push in", model="kling-3",
                      allow_third_party=True, ledger=ledger,
                      jobs_root=str(tmp_path / "jobs"),
                      opener=_upload_opener("https://x"))


def test_animate_image_cost_blocks(tmp_path, monkeypatch):
    img = str(tmp_path / "pic.png")
    _make_image(img)
    _mock_fal(monkeypatch)
    # tiny monthly cap forces a block
    ledger = cost_ledger.CostLedger(str(tmp_path / "spend.jsonl"))
    monkeypatch.setattr(cost_ledger, "MONTHLY_HARD_CAP", 0.001)
    with pytest.raises(BudgetBlocked):
        animate_image(img, "push in", model="seedance-mini", duration_s=5.0,
                      allow_third_party=True, ledger=ledger,
                      jobs_root=str(tmp_path / "jobs"),
                      opener=_upload_opener("https://x"))


def test_animate_image_uses_motion_only_prompt(tmp_path, monkeypatch):
    img = str(tmp_path / "pic.png")
    _make_image(img)
    captured = {}

    def fake_submit(route, arguments, opener=None, tos_ack=False, **kw):
        captured["prompt"] = arguments["prompt"]
        job = fal_client.FalJob(request_id="r", endpoint="e",
                                idempotency_key="k")
        return job

    monkeypatch.setattr(fal_client, "submit", fake_submit)
    monkeypatch.setattr(fal_client, "wait",
                        lambda j, opener=None, **kw: {"video": {"url": "https://v.mp4"}})
    monkeypatch.setattr(fal_client, "download_to_file",
                        lambda url, dest, opener=None: dest)
    ledger = cost_ledger.CostLedger(str(tmp_path / "spend.jsonl"))
    # pre-create the take dir so download mock (no-op) doesn't fail on makedirs
    animate_image(img, "hair stirring in the wind", model="wan-25",
                  allow_third_party=True, ledger=ledger,
                  jobs_root=str(tmp_path / "jobs"), duration_s=2.0,
                  opener=_upload_opener("https://v3.fal.media/pic.png"))
    assert "hair stirring" in captured["prompt"]
    assert "Animate this still image" in captured["prompt"]


# ---------------------------------------------------------------------------
# Research-backed precision controls (Oct 2026): prompt expansion, end-frame
# pinning, and seed passthrough. All mocked — no network, no spend.
# ---------------------------------------------------------------------------

def _capture_submit(monkeypatch, wait_result=None):
    """Replace fal_client.submit/wait/download; return captured arguments."""
    captured = {}

    def fake_submit(route, arguments, opener=None, tos_ack=False, **kw):
        captured["route"] = route
        captured["arguments"] = dict(arguments)
        return fal_client.FalJob(request_id="req-1", endpoint="e",
                                 idempotency_key="k")

    def fake_wait(j, opener=None, **kw):
        return wait_result or {"video": {"url": "https://v.mp4"}}

    monkeypatch.setattr(fal_client, "submit", fake_submit)
    monkeypatch.setattr(fal_client, "wait", fake_wait)
    monkeypatch.setattr(fal_client, "download_to_file",
                        lambda url, dest, opener=None: dest)
    return captured


def _run_i2v(tmp_path, monkeypatch, holder, **kwargs):
    img = str(tmp_path / "pic.png")
    _make_image(img)
    wait_result = holder.pop("wait", None)
    captured = _capture_submit(monkeypatch, wait_result=wait_result)
    ledger = cost_ledger.CostLedger(str(tmp_path / "spend.jsonl"))
    base = dict(model="seedance-mini", allow_third_party=True, ledger=ledger,
                jobs_root=str(tmp_path / "jobs"),
                opener=_upload_opener("https://v3.fal.media/pic.png"))
    # wan-25 at $0.10/s: use 2s to stay under the $0.30 draft cap in tests.
    if kwargs.get("model") == "wan-25" and "duration_s" not in kwargs:
        kwargs["duration_s"] = 2.0
    base.update(kwargs)
    manifest = animate_image(img, "slow push in", **base)
    # fake_submit populated `captured` by reference during the run.
    holder.update(captured)
    return manifest


def test_prompt_expansion_defaults_off_for_wan(tmp_path, monkeypatch):
    holder = {}
    _run_i2v(tmp_path, monkeypatch, holder, model="wan-25")
    args = holder["arguments"]
    assert args["enable_prompt_expansion"] is False


def test_prompt_expansion_can_be_enabled_for_wan(tmp_path, monkeypatch):
    holder = {}
    _run_i2v(tmp_path, monkeypatch, holder, model="wan-25",
             enable_prompt_expansion=True)
    assert holder["arguments"]["enable_prompt_expansion"] is True


def test_prompt_expansion_not_sent_for_seedance(tmp_path, monkeypatch):
    holder = {}
    _run_i2v(tmp_path, monkeypatch, holder, model="seedance-mini",
             enable_prompt_expansion=True)
    assert "enable_prompt_expansion" not in holder["arguments"]


def test_end_image_url_passed_for_seedance(tmp_path, monkeypatch):
    holder = {}
    _run_i2v(tmp_path, monkeypatch, holder, model="seedance-mini",
             end_image_url="https://v3.fal.media/end.png")
    assert holder["arguments"]["end_image_url"] == "https://v3.fal.media/end.png"


def test_end_image_url_not_sent_for_wan(tmp_path, monkeypatch):
    holder = {}
    _run_i2v(tmp_path, monkeypatch, holder, model="wan-25",
             end_image_url="https://v3.fal.media/end.png")
    assert "end_image_url" not in holder["arguments"]


def test_seed_passed_through(tmp_path, monkeypatch):
    for model in ("wan-25", "seedance-mini"):
        holder = {}
        _run_i2v(tmp_path, monkeypatch, holder, model=model, seed=42)
        assert holder["arguments"]["seed"] == 42


def test_seed_absent_when_not_given(tmp_path, monkeypatch):
    holder = {}
    _run_i2v(tmp_path, monkeypatch, holder, model="wan-25")
    assert "seed" not in holder["arguments"]


def test_actual_prompt_in_manifest_when_returned(tmp_path, monkeypatch):
    holder = {"wait": {"video": {"url": "https://v.mp4"},
                       "actual_prompt": "rewritten by fal"}}
    manifest = _run_i2v(tmp_path, monkeypatch, holder, model="wan-25",
                        enable_prompt_expansion=True)
    assert manifest["actual_prompt"] == "rewritten by fal"


def test_actual_prompt_absent_when_not_returned(tmp_path, monkeypatch):
    holder = {}
    manifest = _run_i2v(tmp_path, monkeypatch, holder, model="wan-25")
    assert "actual_prompt" not in manifest


# ---------------------------------------------------------------------------
# Reference-to-video (Seedance 2.0): multi-image + video refs.
# All network mocked — no spend.
# ---------------------------------------------------------------------------

def _make_video(path):
    # Real 1-frame mp4 via ffmpeg so metadata-stripping sees valid video.
    import subprocess
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=0.5",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
        capture_output=True)
    if r.returncode != 0 or not os.path.isfile(path):
        raise RuntimeError(f"test video fixture failed: {r.stderr[:200]}")


def _mock_uploads(monkeypatch):
    monkeypatch.setattr(
        i2v_client, "upload_image",
        lambda path, opener=None: f"https://up/{os.path.basename(path)}")
    monkeypatch.setattr(
        i2v_client, "upload_video",
        lambda path, opener=None: f"https://up/{os.path.basename(path)}")


def test_upload_video_returns_url(tmp_path):
    v = str(tmp_path / "clip.mp4")
    _make_video(v)
    url = upload_video(v, opener=_upload_opener("https://v3.fal.media/c.mp4"))
    assert url == "https://v3.fal.media/c.mp4"


def test_upload_video_rejects_non_video(tmp_path):
    f = str(tmp_path / "pic.png")
    _make_image(f)
    with pytest.raises(I2VError, match="not a video"):
        upload_video(f, opener=_upload_opener("https://x"))


def test_build_reference_arguments_images_only():
    import prompt_director as pd
    shot = pd.build_reference_prompt(
        image_roles=["front view", "side view"], motion="she walks forward")
    args = build_reference_arguments(
        ["https://i1", "https://i2"], shot, duration_s=5.0)
    assert args["image_urls"] == ["https://i1", "https://i2"]
    assert "video_urls" not in args
    assert args["duration"] == "5"
    assert args["resolution"] == "720p"
    assert "@Image1 (front view)" in args["prompt"]
    assert "@Image2 (side view)" in args["prompt"]


def test_build_reference_arguments_with_videos():
    import prompt_director as pd
    shot = pd.build_reference_prompt(
        image_roles=["front view"], video_roles=["walk cycle"],
        motion="she walks forward")
    args = build_reference_arguments(
        ["https://i1"], shot, video_urls=["https://v1"], duration_s=5.0)
    assert args["image_urls"] == ["https://i1"]
    assert args["video_urls"] == ["https://v1"]
    assert "@Video1 (walk cycle)" in args["prompt"]


def test_build_reference_arguments_validation():
    import prompt_director as pd
    shot = pd.build_reference_prompt(image_roles=["a"], motion="x")
    with pytest.raises(I2VError, match="at least one"):
        build_reference_arguments([], shot)
    with pytest.raises(I2VError, match="too many image"):
        build_reference_arguments([f"https://i{i}" for i in range(10)], shot)
    with pytest.raises(I2VError, match="too many video"):
        build_reference_arguments(
            ["https://i"], shot,
            video_urls=[f"https://v{i}" for i in range(4)])


def test_estimate_reference_cost_usd():
    assert estimate_reference_cost_usd(5.0, False) == round(0.30 * 5.0, 4)
    # video refs get the 40% discount
    assert estimate_reference_cost_usd(5.0, True) == round(0.30 * 5.0 * 0.6, 4)


def _run_r2v(tmp_path, monkeypatch, **kwargs):
    imgs = []
    for i in range(2):
        p = str(tmp_path / f"pic{i}.png")
        _make_image(p)
        imgs.append(p)
    _mock_uploads(monkeypatch)
    captured = {}

    def fake_submit(route, arguments, opener=None, tos_ack=False, **kw):
        captured["route"] = route
        captured["arguments"] = dict(arguments)
        return fal_client.FalJob(request_id="req-r2v", endpoint="e",
                                 idempotency_key="k")

    monkeypatch.setattr(fal_client, "submit", fake_submit)
    monkeypatch.setattr(fal_client, "wait",
                        lambda j, opener=None, **kw: {"video": {"url": "https://v.mp4"}})
    monkeypatch.setattr(fal_client, "download_to_file",
                        lambda url, dest, opener=None: dest)
    ledger = cost_ledger.CostLedger(str(tmp_path / "spend.jsonl"))
    base = dict(prompt="she walks forward",
                image_roles=["character front view", "character side view"],
                allow_third_party=True, ledger=ledger,
                jobs_root=str(tmp_path / "jobs"))
    base.update(kwargs)
    manifest = reference_to_video(image_paths=imgs, **base)
    return manifest, captured, ledger


def test_reference_to_video_images_only(tmp_path, monkeypatch):
    manifest, captured, ledger = _run_r2v(tmp_path, monkeypatch)
    assert captured["route"] == "seedance-2.0"
    args = captured["arguments"]
    assert len(args["image_urls"]) == 2
    assert "video_urls" not in args
    assert "@Image1 (character front view)" in args["prompt"]
    assert "@Image2 (character side view)" in args["prompt"]
    assert manifest["model"] == "seedance-2.0"
    assert manifest["request_id"] == "req-r2v"
    assert manifest["image_urls"] == args["image_urls"]
    assert manifest["video_urls"] == []
    # no video refs: full research rate, no discount
    assert manifest["estimate_usd"] == round(0.30 * 5.0, 4)
    summary = ledger.summary()
    assert summary["day_spend_usd"] > 0


def test_reference_to_video_with_video_ref(tmp_path, monkeypatch):
    v = str(tmp_path / "clip.mp4")
    _make_video(v)
    manifest, captured, _ = _run_r2v(
        tmp_path, monkeypatch, video_paths=[v],
        video_roles=["walk cycle sample"])
    args = captured["arguments"]
    assert len(args["video_urls"]) == 1
    assert "@Video1 (walk cycle sample)" in args["prompt"]
    assert len(manifest["video_urls"]) == 1
    # 40% video-ref discount applied
    assert manifest["estimate_usd"] == round(0.30 * 5.0 * 0.6, 4)


def test_reference_to_video_seed_passthrough(tmp_path, monkeypatch):
    _, captured, _ = _run_r2v(tmp_path, monkeypatch, seed=7)
    assert captured["arguments"]["seed"] == 7


def test_reference_to_video_privacy_blocks(tmp_path, monkeypatch):
    imgs = [str(tmp_path / f"p{i}.png") for i in range(2)]
    for p in imgs:
        _make_image(p)
    _mock_uploads(monkeypatch)
    ledger = cost_ledger.CostLedger(str(tmp_path / "spend.jsonl"))
    with pytest.raises(PrivacyBlocked):
        reference_to_video(image_paths=imgs, prompt="x",
                           allow_third_party=False, ledger=ledger,
                           jobs_root=str(tmp_path / "jobs"))


def test_reference_to_video_validation(tmp_path):
    with pytest.raises(I2VError, match="at least one"):
        reference_to_video(prompt="x")
    with pytest.raises(I2VError, match="too many image"):
        reference_to_video(image_paths=[f"i{i}" for i in range(10)],
                           prompt="x")

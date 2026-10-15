"""Tests for tools/studio_server.py.

Runs the real server in a thread (127.0.0.1, ephemeral port). Exercises
upload, media listing, thumbnails, effects catalog, generate stubs, job
lifecycle, and path-traversal defenses. No network beyond localhost.
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

os.environ.pop("FAL_API_KEY", None)  # generate tests need "not configured"

import studio_server as ss
from http.server import ThreadingHTTPServer

NEEDS_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed")
NEEDS_PIL = pytest.mark.skipif(ss.Image is None, reason="PIL not installed")


@pytest.fixture(scope="module")
def server():
    tmp = tempfile.mkdtemp(prefix="studio_test_")
    up = os.path.join(tmp, "uploads")
    res = os.path.join(tmp, "jobs")
    os.makedirs(up)
    ss.UPLOAD_DIR = up
    ss.RESULTS_DIR = res
    ss.JOBS_LOG = os.path.join(res, "jobs.jsonl")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ss.StudioHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)
    yield f"http://127.0.0.1:{port}", up, res
    srv.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


def _get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return r.status, r.read(), r.headers.get("Content-Type", "")


def _post_json(base, path, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _multipart(fields):
    boundary = "STUDIOBOUNDARY"
    body = io.BytesIO()
    for name, filename, data, ctype in fields:
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"; '
                   f'filename="{filename}"\r\n'.encode())
        body.write(f"Content-Type: {ctype}\r\n\r\n".encode())
        body.write(data + b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    return body.getvalue(), boundary


def _upload(base, filename, data, ctype):
    body, boundary = _multipart([("files", filename, data, ctype)])
    req = urllib.request.Request(
        base + "/api/upload", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _make_png():
    from PIL import Image
    img = Image.new("RGB", (160, 120), (200, 40, 40))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _make_mp4(path):
    cmd = ["ffmpeg", "-hide_banner", "-y", "-f", "lavfi",
           "-i", "testsrc=size=320x240:rate=15:duration=2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    with open(path, "rb") as f:
        return f.read()


def _wait_job(base, job_id, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        with urllib.request.urlopen(base + "/api/job/" + job_id) as r:
            job = json.loads(r.read())
        if job["status"] in ("done", "error"):
            return job
        time.sleep(2)
    raise TimeoutError(f"job {job_id} did not finish")


# ---------------------------------------------------------------- tests ----

def test_serves_ui(server):
    base, _, _ = server
    status, body, ctype = _get(base, "/")
    assert status == 200
    assert b"MYAI Studio" in body
    assert "text/html" in ctype


@NEEDS_PIL
def test_upload_image(server):
    base, up, _ = server
    status, resp = _upload(base, "shot.png", _make_png(), "image/png")
    assert status == 200
    assert resp["saved"] and resp["saved"][0]["type"] == "image"
    assert os.path.isfile(os.path.join(up, resp["saved"][0]["name"]))


@NEEDS_FFMPEG
def test_upload_video(server):
    base, up, _ = server
    with tempfile.TemporaryDirectory() as td:
        data = _make_mp4(os.path.join(td, "clip.mp4"))
    status, resp = _upload(base, "clip.mp4", data, "video/mp4")
    assert status == 200
    assert resp["saved"] and resp["saved"][0]["type"] == "video"


def test_upload_rejects_non_media(server):
    base, _, _ = server
    status, resp = _upload(base, "evil.exe", b"MZ...", "application/x-ms")
    # An upload where EVERYTHING is refused now says so (400 + reason),
    # instead of the old silent-skip 200 that made files "vanish".
    assert status == 400
    assert "evil.exe" in resp["error"]
    assert resp["rejected"]


def test_upload_accepts_download_formats(server):
    # Videos pulled from a phone's Downloads folder are often .avi/.ts —
    # they must be accepted as videos, not silently skipped.
    base, up, _ = server
    status, resp = _upload(base, "movie.avi", b"RIFF....", "video/x-msvideo")
    assert status == 200
    assert resp["saved"] and resp["saved"][0]["type"] == "video"
    assert resp["saved"][0]["name"] == "movie.avi"


def test_media_listing(server):
    base, _, _ = server
    status, body, _ = _get(base, "/api/media")
    assert status == 200
    data = json.loads(body)
    names = [m["name"] for m in data["media"]]
    assert "shot.png" in names
    entry = next(m for m in data["media"] if m["name"] == "shot.png")
    assert entry["type"] == "image" and entry["size"] > 0
    assert entry["thumb"] == "/api/thumb/shot.png"


@NEEDS_PIL
def test_thumbnail_image(server):
    base, _, _ = server
    status, body, ctype = _get(base, "/api/thumb/shot.png")
    assert status == 200
    assert "image/jpeg" in ctype
    assert body[:2] == b"\xff\xd8"  # JPEG magic


def test_thumbnail_missing_is_404(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as ei:
        _get(base, "/api/thumb/nope.png")
    assert ei.value.code == 404


def test_thumbnail_traversal_blocked(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as ei:
        _get(base, "/api/thumb/..%2F..%2Fetc%2Fpasswd")
    assert ei.value.code == 404


def test_effects_catalog(server):
    base, _, _ = server
    status, body, _ = _get(base, "/api/effects")
    data = json.loads(body)
    assert len(data["style_pack"]) == 8
    assert len(data["video_effect"]) >= 12
    assert all(e["applies_to"] in ("image", "video") for e in data["video_effect"])
    # video_edit kind: 9 ops, all apply to video
    assert len(data["video_edit"]) == 9
    assert all(e["applies_to"] == "video" for e in data["video_edit"])
    names = {e["name"] for e in data["video_edit"]}
    assert {"trim", "speed", "crop", "rotate", "filter", "reverse",
            "merge", "extract_frame", "gif"} == names


def test_generate_options(server):
    base, _, _ = server
    status, body, _ = _get(base, "/api/generate_options")
    data = json.loads(body)
    assert data["configured"] is False
    ids = [o["id"] for o in data["options"]]
    assert ids == ["i2v", "i2v_venice", "i2v_venice_uncensored", "kenburns_free", "t2v_draft", "t2v_hero",
                   "v2v_restyle", "model_replace",
                   "gpu_image_edit", "gpu_i2v", "gpu_t2v", "gpu_replace",
                   "long_video"]
    free = next(o for o in data["options"] if o["id"] == "kenburns_free")
    paid = next(o for o in data["options"] if o["id"] == "t2v_hero")
    assert free["badge"] == "FREE" and not free["paid"]
    assert paid["badge"] == "$" and paid["paid"]


def test_generate_needs_setup_without_key(server):
    base, _, _ = server
    status, resp = _post_json(base, "/api/generate",
                              {"id": "t2v_draft", "prompt": "a sunset"})
    assert status == 200
    assert resp["status"] == "needs_setup"


def test_generate_unknown_id_400(server):
    base, _, _ = server
    status, resp = _post_json(base, "/api/generate", {"id": "nope"})
    assert status == 400


def test_generate_free_option_rejected(server):
    base, _, _ = server
    status, resp = _post_json(base, "/api/generate", {"id": "kenburns_free"})
    assert status == 400


def test_apply_bad_kind_400(server):
    base, _, _ = server
    status, resp = _post_json(base, "/api/apply",
                              {"media": "shot.png", "kind": "nope",
                               "name": "vhs", "params": {}})
    assert status == 400


def test_apply_unknown_name_400(server):
    base, _, _ = server
    status, resp = _post_json(base, "/api/apply",
                              {"media": "shot.png", "kind": "style_pack",
                               "name": "nope", "params": {}})
    assert status == 400


def test_apply_missing_media_404(server):
    base, _, _ = server
    status, resp = _post_json(base, "/api/apply",
                              {"media": "ghost.mp4", "kind": "style_pack",
                               "name": "vhs", "params": {}})
    assert status == 404


def test_apply_wrong_media_type_400(server):
    base, _, _ = server
    # noir style pack applies to video; shot.png is an image
    status, resp = _post_json(base, "/api/apply",
                              {"media": "shot.png", "kind": "style_pack",
                               "name": "noir", "params": {}})
    assert status == 400


def test_apply_traversal_media_404(server):
    base, _, _ = server
    status, resp = _post_json(base, "/api/apply",
                              {"media": "../../etc/passwd",
                               "kind": "style_pack", "name": "noir",
                               "params": {}})
    assert status == 404


@NEEDS_FFMPEG
def test_style_pack_job_lifecycle(server):
    base, _, _ = server
    status, resp = _post_json(base, "/api/apply",
                              {"media": "clip.mp4", "kind": "style_pack",
                               "name": "noir", "params": {}})
    assert status == 200
    job_id = resp["job_id"]
    assert job_id.startswith("job-")
    job = _wait_job(base, job_id)
    assert job["status"] == "done", job.get("error")
    assert job["progress"] == 100
    # download the result
    st, body, ctype = _get(base, job["result_url"])
    assert st == 200 and len(body) > 0
    assert "video" in ctype


def test_job_unknown_404(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as ei:
        _get(base, "/api/job/job-9999")
    assert ei.value.code == 404


def test_result_traversal_blocked(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as ei:
        _get(base, "/api/result/job-0001/..%2F..%2Fsecret.txt")
    assert ei.value.code == 404


def _upload_pngs(base, colors):
    from PIL import Image as PILImage
    fields = []
    for i, color in enumerate(colors):
        buf = io.BytesIO()
        PILImage.new("RGB", (32, 32), color).save(buf, "PNG")
        fields.append(("files", f"ref{i}.png", buf.getvalue(), "image/png"))
    body, boundary = _multipart(fields)
    req = urllib.request.Request(
        base + "/api/upload", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as r:
        saved = json.loads(r.read())["saved"]
    return [s["name"] for s in saved]


def test_generate_reference_to_video(server, monkeypatch):
    """POST /api/generate with reference_images routes to r2v (mocked)."""
    base, up, res = server
    names = _upload_pngs(base, ["red", "blue"])
    assert len(names) == 2

    monkeypatch.setenv("FAL_API_KEY", "test-key")
    import i2v_client

    seen = {}

    def fake_r2v(**kwargs):
        seen.update(kwargs)
        d = os.path.join(kwargs["jobs_root"], "fake-r2v", "takes")
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, "take01.mp4")
        with open(out, "wb") as f:
            f.write(b"FAKEMP4")
        return {"output_path": out, "estimate_usd": 1.5}

    monkeypatch.setattr(i2v_client, "reference_to_video", fake_r2v)

    status, resp = _post_json(
        base, "/api/generate",
        {"id": "i2v", "prompt": "she walks forward",
         "reference_images": names})
    assert status == 200
    assert resp["status"] == "started"
    assert "job_id" in resp

    # background job resolves both refs and calls reference_to_video
    job_id = resp["job_id"]
    job = None
    for _ in range(100):
        _, body, _ = _get(base, "/api/job/" + job_id)
        job = json.loads(body)
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert job["status"] == "done", job
    assert len(seen["image_paths"]) == 2
    assert seen["video_paths"] in (None, [])
    assert seen["prompt"] == "she walks forward"
    assert job["result_name"].endswith("_r2v.mp4")


def test_generate_reference_too_many_400(server, monkeypatch):
    base, up, res = server
    monkeypatch.setenv("FAL_API_KEY", "test-key")
    names = _upload_pngs(base, ["red", "green"])
    status, resp = _post_json(
        base, "/api/generate",
        {"id": "i2v", "prompt": "x",
         "reference_images": names * 5})  # 10 refs > cap of 9
    assert status == 400
    assert "max 9" in resp["error"]


# ------------------------------------------------- v2v / longvideo ----

def test_v2v_models_lists_uncensored_first(server):
    base, _, _ = server
    status, body, _ = _get(base, "/api/v2v_models")
    data = json.loads(body)
    assert status == 200
    ids = [m["id"] for m in data["models"]]
    assert "venice-wan27-v2v" in ids
    assert "venice-runway-v2v" in ids
    assert "venice-seedance-r2v" in ids
    # uncensored models sort first
    first_uncensored = [m for m in data["models"] if m["uncensored"]]
    assert first_uncensored and first_uncensored[0]["id"] == "venice-wan27-v2v"
    assert set(data["quality_levels"]) == {"draft", "standard", "maximum"}
    assert data["default_quality"] == "standard"
    assert any("lastframe" == c["id"] for c in data["chain_modes"])


def test_v2v_needs_setup_without_key(server):
    base, up, _ = server
    # upload a tiny mp4 so media validation passes before key check
    mp4 = _make_test_mp4(up, "v.mp4")
    status, resp = _post_json(base, "/api/v2v",
                              {"media": "v.mp4", "model": "venice-wan27-v2v",
                               "prompt": "noir restyle"})
    assert status == 200
    assert resp["status"] == "needs_setup"


def test_v2v_rejects_unknown_model(server):
    base, _, _ = server
    import os as _os
    _os.environ["VENICE_API_KEY"] = "test-key"
    try:
        status, resp = _post_json(base, "/api/v2v",
                                  {"media": "x.mp4", "model": "nope",
                                   "prompt": "y"})
        assert status == 400
        assert "unknown V2V model" in resp["error"]
    finally:
        del _os.environ["VENICE_API_KEY"]


def test_v2v_rejects_image_media(server, monkeypatch):
    base, up, _ = server
    monkeypatch.setenv("VENICE_API_KEY", "test-key")
    names = _upload_pngs(base, ["red"])
    status, resp = _post_json(base, "/api/v2v",
                              {"media": names[0], "model": "venice-wan27-v2v",
                               "prompt": "y"})
    assert status == 400
    assert "needs a video" in resp["error"]


def test_segment_prompt_endpoint(server):
    base, _, _ = server
    status, resp = _post_json(base, "/api/segment_prompt",
                              {"prompt": "A flies then B lands then C looks",
                               "n_beats": 3})
    assert status == 200
    assert len(resp["beats"]) == 3
    assert "flies" in resp["beats"][0]


def test_segment_prompt_empty_400(server):
    base, _, _ = server
    status, resp = _post_json(base, "/api/segment_prompt",
                              {"prompt": "   ", "n_beats": 2})
    assert status == 400


def test_longvideo_needs_reference(server, monkeypatch):
    base, _, _ = server
    monkeypatch.setenv("VENICE_API_KEY", "test-key")
    status, resp = _post_json(base, "/api/longvideo",
                              {"prompt": "a story", "target_duration_s": 60})
    assert status == 400
    assert "reference" in resp["error"]


def test_longvideo_duration_bounds(server, monkeypatch):
    base, up, _ = server
    monkeypatch.setenv("VENICE_API_KEY", "test-key")
    names = _upload_pngs(base, ["red"])
    status, resp = _post_json(base, "/api/longvideo",
                              {"media": names[0], "prompt": "a story",
                               "target_duration_s": 5})
    assert status == 400
    assert "15-600" in resp["error"]


def test_longvideo_beat_count_mismatch_400(server, monkeypatch):
    base, up, _ = server
    monkeypatch.setenv("VENICE_API_KEY", "test-key")
    names = _upload_pngs(base, ["red"])
    status, resp = _post_json(base, "/api/longvideo",
                              {"media": names[0], "prompt": "a story",
                               "target_duration_s": 60,
                               "beats": ["only one"]})
    assert status == 400
    assert "needs 4 clips" in resp["error"]


def _make_test_mp4(updir, name):
    import subprocess as _sp
    p = os.path.join(updir, name)
    _sp.run(["ffmpeg", "-hide_banner", "-y", "-f", "lavfi",
             "-i", "testsrc=duration=1:size=160x120:rate=15",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", p],
            capture_output=True, check=True, timeout=60)
    return p


@NEEDS_FFMPEG
def test_apply_video_edit_trim(server):
    base, up, _ = server
    _make_test_mp4(up, "trimme.mp4")
    status, resp = _post_json(base, "/api/apply",
                              {"media": "trimme.mp4", "kind": "video_edit",
                               "name": "trim", "params": {}})
    assert status == 200
    job = _wait_job_done(base, resp["job_id"])
    assert job["status"] == "done"
    assert job["result_name"].endswith("_trim.mp4")


@NEEDS_FFMPEG
def test_apply_video_edit_reverse(server):
    base, up, _ = server
    _make_test_mp4(up, "revme.mp4")
    status, resp = _post_json(base, "/api/apply",
                              {"media": "revme.mp4", "kind": "video_edit",
                               "name": "reverse", "params": {}})
    assert status == 200
    job = _wait_job_done(base, resp["job_id"])
    assert job["status"] == "done"


def _wait_job_done(base, job_id, timeout=120):
    job = None
    for _ in range(int(timeout / 0.5)):
        _, body, _ = _get(base, "/api/job/" + job_id)
        job = json.loads(body)
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.5)
    return job


# --- long-video chunking for model replace (any length) ---

def test_chunk_bounds_cover_full_duration():
    b = ss._chunk_bounds(219.0, 110.0)
    assert b == [(0.0, 110.0), (110.0, 109.0)]
    assert ss._chunk_bounds(60.0, 110.0) == [(0.0, 60.0)]
    assert ss._chunk_bounds(110.0, 110.0) == [(0.0, 110.0)]
    total = sum(length for _, length in ss._chunk_bounds(1000.0, 110.0))
    assert abs(total - 1000.0) < 0.01


def test_even_bounds_no_runt_tail():
    b = ss._even_bounds(219.0, 14.0)
    assert len(b) == 16
    assert all(length <= 14.0 and length >= 2.0 for _, length in b)
    assert abs(sum(length for _, length in b) - 219.0) < 0.1
    # The case greedy chunking ruins: 15.5s must be 2 x 7.75, not 14+1.5
    b = ss._even_bounds(15.5, 14.0)
    assert len(b) == 2 and all(length >= 2.0 for _, length in b)
    assert ss._even_bounds(10.0, 14.0) == [(0.0, 10.0)]

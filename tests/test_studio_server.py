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


def test_largest_face_crop_box():
    box = ss._largest_face_crop_box(1000, 800, [(100, 100, 50, 50),
                                                (300, 200, 200, 200)])
    assert box == (210, 110, 590, 490)  # largest face + 45% margin
    assert ss._largest_face_crop_box(100, 100, [(0, 0, 40, 40)]) == \
        (0, 0, 58, 58)  # clamped to the image
    assert ss._largest_face_crop_box(100, 100, []) is None


def test_run_scail_chunks_chains_and_stitches(tmp_path, monkeypatch):
    """Long-video SCAIL: 81-frame chunks on a 76-frame step, each
    continuation anchored on the previous chunk's tail, heads
    trimmed by the overlap, stitched in order."""
    calls = {"build": [], "run": []}
    counts = iter([81, 81, 48])

    class FakeGC:
        def upload_video(self, path):
            return "up_" + os.path.basename(path)

        def run_workflow(self, wf, jobdir, **kw):
            calls["run"].append(jobdir)
            os.makedirs(jobdir, exist_ok=True)
            out = os.path.join(jobdir, "out.mp4")
            open(out, "wb").write(b"x")
            return [out]

    class FakeGW:
        def video_replace_model_scail(self, *a, **kw):
            calls["build"].append(kw)
            return {"wf": True}

    def fake_count(path):
        return next(counts)

    monkeypatch.setattr(ss, "_count_frames", fake_count)
    monkeypatch.setattr(ss, "_scail_segment",
                        lambda s, a, b, c, out: open(out, "wb").write(b"x"))
    monkeypatch.setattr(ss, "_scail_trim_head",
                        lambda s, h, out: open(out, "wb").write(b"x"))
    monkeypatch.setattr(ss, "_concat_normalized",
                        lambda parts, out, like: open(out, "wb").write(b"x"))
    jobdir = str(tmp_path)
    src = os.path.join(jobdir, "src.mp4")
    open(src, "wb").write(b"x")
    out = ss._run_scail_chunks(
        FakeGC(), FakeGW(), "job-x", jobdir, src, "ref_up", "prompt",
        "person", 0.4, {"width": 704, "height": 512, "steps": 24},
        {"duration": 200 / 16.0})
    assert os.path.basename(out) == "_scail_stitched.mp4"
    builds = calls["build"]
    assert [b["frames"] for b in builds] == [81, 81, 48]
    assert [b["frame_offset"] for b in builds] == [0, 76, 152]
    assert builds[0]["previous_frames_video"] is None
    assert builds[1]["previous_frames_video"] == "up_anchor_000.mp4"
    assert builds[2]["previous_frames_video"] == "up_anchor_001.mp4"
    assert len(calls["run"]) == 3


def test_run_seedvr2_finish_roundtrip(tmp_path):
    """SeedVR2 finish: uploads the stitched video, runs the
    standalone upscale graph at 1024, returns the pod output."""
    seen = {}

    class FakeGC:
        def upload_video(self, path):
            seen["up"] = path
            return "up_stitched.mp4"

        def run_workflow(self, wf, jobdir, **kw):
            seen["wf"] = wf
            seen["jobdir"] = jobdir
            os.makedirs(jobdir, exist_ok=True)
            out = os.path.join(jobdir, "finished.mp4")
            open(out, "wb").write(b"x")
            return [out]

    class FakeGW:
        def video_seedvr2_upscale(self, name, **kw):
            seen["name"] = name
            seen["kw"] = kw
            return {"wf": True}

    jobdir = str(tmp_path)
    src = os.path.join(jobdir, "stitched.mp4")
    open(src, "wb").write(b"x")
    out = ss._run_seedvr2_finish(FakeGC(), FakeGW(), jobdir, src)
    assert os.path.basename(out) == "finished.mp4"
    assert seen["up"] == src
    assert seen["name"] == "up_stitched.mp4"
    assert seen["kw"]["resolution"] == 1024


# ------------------------------------------------- face lock reporting ----

def _longvideo_job_env(tmp_path, monkeypatch):
    """Point the server's dirs at tmp and stub out Venice."""
    up = tmp_path / "uploads"
    res = tmp_path / "jobs"
    up.mkdir()
    res.mkdir()
    monkeypatch.setattr(ss, "UPLOAD_DIR", str(up))
    monkeypatch.setattr(ss, "RESULTS_DIR", str(res))
    monkeypatch.setattr(ss, "JOBS_LOG", str(res / "jobs.jsonl"))
    (up / "ref.jpg").write_bytes(b"\xff\xd8\xff" + b"0" * 64)
    (up / "base.jpg").write_bytes(b"\xff\xd8\xff" + b"0" * 64)

    class FakeVC:
        DEFAULT_QUALITY = "standard"
        QUALITY_LEVELS = {"standard": {"resolution": "720p", "crf": 20,
                                       "preset": "medium"}}

        @staticmethod
        def file_to_data_url(path):
            return "data:image/jpeg;base64,STUB"

    monkeypatch.setattr(ss, "vc", FakeVC)


def _fake_chain_returning(kwargs_seen):
    def fake_run_chain(**kwargs):
        kwargs_seen.update(kwargs)
        out = os.path.join(kwargs["workdir"], "longvideo_final.mp4")
        os.makedirs(kwargs["workdir"], exist_ok=True)
        _make_mp4(out)
        return out
    return fake_run_chain


def _longvideo_params():
    return {"beats": ["she walks in", "she sits down"],
            "clip_duration_s": 15.0,
            "quality": "standard",
            "chain_mode": "lastframe",
            "face_lock": True,
            "ref_image": "ref.jpg"}


@NEEDS_FFMPEG
def test_longvideo_face_lock_failure_is_surfaced(tmp_path, monkeypatch):
    """A failed face lock must not pass silently: the finished job
    carries the failure in its phase and a warning field, and the
    unlocked result is what ships."""
    _longvideo_job_env(tmp_path, monkeypatch)
    seen = {}
    monkeypatch.setattr(ss.lv, "run_chain", _fake_chain_returning(seen))

    def bad_lock(*a, **k):
        raise RuntimeError("no face detected in reference photo")

    monkeypatch.setattr(ss, "_face_lock", bad_lock)
    job_id = ss.JOBS.new("base.jpg", "longvideo", "longvideo",
                         _longvideo_params())
    ss._run_job(job_id)
    job = ss.JOBS.get(job_id)
    assert job["status"] == "done", job
    assert "face lock FAILED" in (job.get("phase") or "")
    assert "face lock FAILED" in (job.get("warning") or "")
    # The reason itself must survive (it was swallowed before Oct 17).
    assert "no face detected in reference photo" in job["warning"]
    assert job["result_name"].endswith("_longvideo.mp4")
    # The per-clip lock was wired into the chain.
    assert seen.get("clip_lock_fn") is not None


@NEEDS_FFMPEG
def test_longvideo_face_lock_success_reports_plain_done(tmp_path,
                                                        monkeypatch):
    """A successful lock keeps the plain 'done' phase, no warning, and
    ships the locked result file."""
    _longvideo_job_env(tmp_path, monkeypatch)
    seen = {}
    monkeypatch.setattr(ss.lv, "run_chain", _fake_chain_returning(seen))

    def ok_lock(video, ref, out, workdir):
        shutil.copy2(video, out)

    monkeypatch.setattr(ss, "_face_lock", ok_lock)
    job_id = ss.JOBS.new("base.jpg", "longvideo", "longvideo",
                         _longvideo_params())
    ss._run_job(job_id)
    job = ss.JOBS.get(job_id)
    assert job["status"] == "done", job
    assert job.get("phase") == "done"
    assert not job.get("warning")
    assert job["result_name"].endswith("_longvideo_facelock.mp4")


# ------------------------------------------------------- Stage 1 GUI ----

def _seed_job_with_files(server, files, result_name):
    """Register a done job and plant files in its results folder."""
    _, _, res = server
    job_id = ss.JOBS.new("base.mp4", "gpu_replace", "Test Job", {})
    jobdir = os.path.join(res, job_id)
    os.makedirs(jobdir, exist_ok=True)
    for name in files:
        with open(os.path.join(jobdir, name), "wb") as f:
            f.write(b"x" * 64)
    ss.JOBS.update(job_id, status="done", progress=100,
                   result_name=result_name,
                   result_url=f"/api/result/{job_id}/{result_name}")
    return job_id


def test_job_deliverables_locked_is_primary(server):
    job_id = _seed_job_with_files(server, [
        "clip_gpu_replace.mp4", "clip_locked_neutral.mp4",
        "scail_replace_00001.mp4", "_facelock_silent.mp4",
    ], "clip_gpu_replace.mp4")
    with urllib.request.urlopen(server[0] + "/api/job/" + job_id) as r:
        job = json.loads(r.read())
    dels = job["deliverables"]
    names = [d["name"] for d in dels]
    # Locked final first; engine intermediate + scratch files hidden.
    assert names == ["clip_locked_neutral.mp4", "clip_gpu_replace.mp4"]
    assert dels[0]["primary"] is True
    assert dels[0]["label"] == "Face-locked final"
    assert dels[1]["label"] == "Raw render"
    assert job["primary_name"] == "clip_locked_neutral.mp4"
    assert job["primary_url"].endswith("/clip_locked_neutral.mp4")


def test_jobs_list_primary_prefers_locked(server):
    job_id = _seed_job_with_files(server, ["out.mp4", "out_locked.mp4"],
                                  "out.mp4")
    with urllib.request.urlopen(server[0] + "/api/jobs") as r:
        jobs = json.loads(r.read())["jobs"]
    entry = next(j for j in jobs if j["job_id"] == job_id)
    assert entry["primary_name"] == "out_locked.mp4"
    assert entry["has_clips"] is False


def test_job_clips_route_not_shadowed(server):
    # The /clips route must return JSON (it used to be shadowed by the
    # generic /api/job/ prefix route and 404 for every job).
    job_id = _seed_job_with_files(server, ["out.mp4"], "out.mp4")
    chain = os.path.join(server[2], job_id, "chain")
    os.makedirs(chain, exist_ok=True)
    with open(os.path.join(chain, "clip_01.mp4"), "wb") as f:
        f.write(b"x")
    status, body, _ = _get(server[0], f"/api/job/{job_id}/clips")
    assert status == 200
    data = json.loads(body)
    assert [c["name"] for c in data["clips"]] == ["clip_01.mp4"]
    with urllib.request.urlopen(server[0] + "/api/job/" + job_id) as r:
        job = json.loads(r.read())
    assert job["has_clips"] is True


@NEEDS_PIL
def test_media_listing_roles_and_file_route(server):
    base, up, _ = server
    png = _make_png()
    with open(os.path.join(up, "ref1.jpg"), "wb") as f:
        f.write(png)
    with open(os.path.join(up, ".media_roles.json"), "w") as f:
        json.dump({"ref1.jpg": {"role": "face_ref",
                                "label": "Face reference"}}, f)
    with urllib.request.urlopen(base + "/api/media") as r:
        media = json.loads(r.read())["media"]
    item = next(m for m in media if m["name"] == "ref1.jpg")
    assert item["role"] == "face_ref"
    assert item["role_label"] == "Face reference"
    assert item["url"] == "/api/media/file/ref1.jpg"
    assert item["mtime"] > 0
    status, body, _ = _get(base, "/api/media/file/ref1.jpg")
    assert status == 200 and body == png
    # Traversal must not escape the uploads dir.
    try:
        _get(base, "/api/media/file/..%2Fstudio_server.py")
        raise AssertionError("expected 404 for traversal")
    except urllib.error.HTTPError as e:
        assert e.code == 404

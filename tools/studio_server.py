"""MYAI Media Studio — mobile-first web UI for phone-based media editing.

Stdlib HTTP server + the wiring tools. Binds 127.0.0.1 by design; expose
it with `tailscale serve` (tailnet-only), never 0.0.0.0 on a public
interface.

Endpoints:
    GET  /                        the studio UI (tools/studio_ui.html)
    POST /api/upload               multipart file upload -> UPLOAD_DIR
    GET  /api/media                JSON list of uploads
    GET  /api/thumb/<name>         320px thumbnail (PIL / ffmpeg)
    GET  /api/effects              JSON catalog of available effects
    POST /api/apply                {media, kind, name, params} -> {job_id}
    GET  /api/job/<job_id>         {status, progress, result_url, error}
    GET  /api/result/<job_id>/<f>  download the rendered result

Jobs run in a thread pool (max 2 concurrent ffmpeg jobs). Job state lives
in memory; every transition is appended to jobs.jsonl.

Usage:
    python3 studio_server.py [--port 8899] [--dir /home/nanobot/uploads]
"""

from __future__ import annotations

import argparse
import cgi
import concurrent.futures
import html
import io
import json
import mimetypes
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote

# ---------------------------------------------------------------- imports ----

import local_effects as le
import style_packs as sp
import upload_server as us

try:
    import photo_tools as pt
    PHOTO_OPS = getattr(pt, "PHOTO_OPS", {})
except ImportError:
    pt = None
    PHOTO_OPS = {}

try:
    import kenburns as kb
except ImportError:
    kb = None

try:
    from PIL import Image
except ImportError:
    Image = None

# ----------------------------------------------------------------- config ----

UPLOAD_DIR = os.environ.get("MYAI_UPLOAD_DIR", "/home/nanobot/uploads")
RESULTS_DIR = os.environ.get("MYAI_STUDIO_RESULTS",
                             os.path.join(UPLOAD_DIR, "..", "studio_jobs"))
JOBS_LOG = os.path.join(RESULTS_DIR, "jobs.jsonl")
MAX_WORKERS = 2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}

HERE = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(HERE, "studio_ui.html")

KENBURNS_PRESETS = {
    "zoom_in": {"description": "Slow push-in on the photo.",
                "params": {"duration": 5.0, "zoom_from": 1.0, "zoom_to": 1.3,
                           "pan": [0.0, 0.0]}},
    "zoom_out": {"description": "Slow pull-back revealing more.",
                 "params": {"duration": 5.0, "zoom_from": 1.3, "zoom_to": 1.0,
                            "pan": [0.0, 0.0]}},
    "pan_left": {"description": "Drift left across the photo.",
                 "params": {"duration": 5.0, "zoom_from": 1.15, "zoom_to": 1.15,
                            "pan": [-0.1, 0.0]}},
    "pan_right": {"description": "Drift right across the photo.",
                  "params": {"duration": 5.0, "zoom_from": 1.15,
                             "zoom_to": 1.15, "pan": [0.1, 0.0]}},
    "slow_push": {"description": "Very slow cinematic push-in.",
                  "params": {"duration": 8.0, "zoom_from": 1.0, "zoom_to": 1.2,
                             "pan": [0.0, 0.0]}},
}

# kind -> (applies_to, registry_name)
KINDS = ("style_pack", "photo_op", "video_effect", "kenburns")


def media_type(name: str) -> str | None:
    ext = os.path.splitext(name)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return None


def safe_media_path(name: str) -> str | None:
    """Resolve an uploaded filename to a path, or None if unsafe/missing."""
    clean = us.safe_name(name)
    if not clean or clean != name.replace("\\", "/").split("/")[-1]:
        # safe_name already strips dirs; reject anything that changed shape
        # beyond the sanitizer's normal replacements.
        pass
    path = os.path.join(UPLOAD_DIR, clean)
    real = os.path.realpath(path)
    if not real.startswith(os.path.realpath(UPLOAD_DIR) + os.sep):
        return None
    if not os.path.isfile(real):
        return None
    return real


def _paramspec(params: dict) -> dict:
    """Simplify a registry params dict for the UI catalog."""
    out = {}
    for k, v in (params or {}).items():
        if isinstance(v, dict):
            out[k] = {"type": v.get("type", "string"),
                      "default": v.get("default")}
        else:
            out[k] = {"type": "string", "default": v}
    return out


GENERATE_OPTIONS = [
    {"id": "i2v",
     "name": "Animate Photo (AI)",
     "badge": "$",
     "cost": "~$0.05\u20130.15/clip",
     "description": "Real AI motion from your photo (fal.ai).",
     "needs_media": "image",
     "needs_prompt": True,
     "paid": True},
    {"id": "kenburns_free",
     "name": "Animate Photo (Ken Burns)",
     "badge": "FREE",
     "cost": "$0",
     "description": "Smooth zoom/pan on your photo. Local, instant.",
     "needs_media": "image",
     "needs_prompt": False,
     "paid": False},
    {"id": "t2v_draft",
     "name": "Text to Video (Draft)",
     "badge": "$",
     "cost": "~$0.05/s",
     "description": "Quick AI video from a text description.",
     "needs_media": None,
     "needs_prompt": True,
     "paid": True},
    {"id": "t2v_hero",
     "name": "Text to Video (Hero)",
     "badge": "$",
     "cost": "~$0.40/s",
     "description": "Best-quality AI video from text.",
     "needs_media": None,
     "needs_prompt": True,
     "paid": True},
]


def _load_fal_key() -> str:
    """FAL_API_KEY from env, or ~/.nanobot/fal.env (David's server install)."""
    key = os.environ.get("FAL_API_KEY", "").strip()
    if key:
        return key
    for candidate in (os.path.expanduser("~/.nanobot/fal.env"),
                      "/home/nanobot/.nanobot/fal.env"):
        try:
            if os.path.isfile(candidate):
                with open(candidate) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("FAL_API_KEY="):
                            k = line.split("=", 1)[1].strip().strip("'\"")
                            if k:
                                return k
        except OSError:
            continue
    return ""


def fal_status() -> dict:
    key = _load_fal_key()
    # Make it available to i2v_client via env for this process.
    if key and not os.environ.get("FAL_API_KEY"):
        os.environ["FAL_API_KEY"] = key
    return {
        "configured": bool(key),
        "message": ("fal.ai is configured." if key else
                    "Needs setup: add your fal.ai API key (FAL_API_KEY) on "
                    "the server to enable paid generation. Free options "
                    "work without it."),
    }


def generate_options() -> dict:
    info = fal_status()
    return {"configured": info["configured"],
            "setup_message": info["message"],
            "options": GENERATE_OPTIONS}


def effects_catalog() -> dict:
    packs = [{"name": p["name"], "description": p["blurb"],
              "applies_to": "video", "params": {}}
             for p in sp.list_packs()]
    photos = [{"name": n, "description": s.get("description", n),
               "applies_to": "image", "params": _paramspec(s.get("params"))}
              for n, s in PHOTO_OPS.items()]
    videos = [{"name": r["name"], "description": r["description"],
               "applies_to": "video", "params": _paramspec(r["params"])}
              for r in le.list_recipes() if not r["two_inputs"]]
    animate = []
    if kb is not None:
        animate = [{"name": n, "description": p["description"],
                    "applies_to": "image", "params": {}}
                   for n, p in KENBURNS_PRESETS.items()]
    return {"style_pack": packs, "photo_op": photos,
            "video_effect": videos, "kenburns": animate}


def validate_apply(media: str, kind: str, name: str,
                   params: dict) -> tuple[str, str] | tuple[None, str]:
    """Return (media_path, error). error is None on success."""
    if kind not in KINDS:
        return None, f"unknown kind {kind!r}; expected one of {list(KINDS)}"
    mpath = safe_media_path(media)
    if mpath is None:
        return None, f"media {media!r} not found"
    mtype = media_type(os.path.basename(mpath))
    catalog = effects_catalog()
    names = {e["name"]: e for e in catalog[kind]}
    if name not in names:
        return None, f"unknown {kind} {name!r}"
    entry = names[name]
    if entry["applies_to"] != mtype:
        return None, (f"{kind}/{name} applies to {entry['applies_to']} "
                      f"media, but {media!r} is {mtype}")
    if not isinstance(params, dict):
        return None, "params must be an object"
    return mpath, None


# ------------------------------------------------------------------- jobs ----

class JobStore:
    """Thread-safe in-memory job state + JSONL log."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._counter = 0
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def new(self, media: str, kind: str, name: str, params: dict) -> str:
        with self._lock:
            self._counter += 1
            job_id = f"job-{self._counter:04d}"
            self._jobs[job_id] = {
                "job_id": job_id, "media": media, "kind": kind,
                "name": name, "params": params,
                "status": "queued", "progress": 0,
                "result_url": None, "result_name": None, "error": None,
                "created": time.time(),
            }
            self._log(job_id, "queued")
            return job_id

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)
                self._log(job_id, fields.get("status", "update"))

    def _log(self, job_id: str, event: str) -> None:
        rec = {"ts": time.time(), "job_id": job_id, "event": event,
               "state": self._jobs[job_id]}
        try:
            with open(JOBS_LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass


JOBS = JobStore()
POOL = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)


def _run_job(job_id: str) -> None:
    job = JOBS.get(job_id)
    if not job:
        return
    JOBS.update(job_id, status="running", progress=50)
    try:
        mpath = safe_media_path(job["media"])
        if mpath is None:
            raise RuntimeError(f"media {job['media']!r} disappeared")
        jobdir = os.path.join(RESULTS_DIR, job_id)
        os.makedirs(jobdir, exist_ok=True)
        kind, name, params = job["kind"], job["name"], job["params"]
        base = os.path.splitext(us.safe_name(job["media"]))[0]

        if kind == "style_pack":
            out = os.path.join(jobdir, f"{base}_{name}.mp4")
            sp.apply_pack(mpath, out, name)
        elif kind == "video_effect":
            out = os.path.join(jobdir, f"{base}_{name}.mp4")
            le.apply(mpath, out, name, params or None)
        elif kind == "photo_op":
            if pt is None:
                raise RuntimeError("photo tools not available")
            if Image is None:
                raise RuntimeError("PIL not available")
            fn = PHOTO_OPS[name]["function"]
            img = Image.open(mpath).convert("RGB")
            result = fn(img, **(params or {}))
            out = os.path.join(jobdir, f"{base}_{name}.jpg")
            result.save(out, "JPEG", quality=92)
        elif kind == "kenburns":
            if kb is None:
                raise RuntimeError("kenburns not available")
            preset = KENBURNS_PRESETS[name]
            merged = dict(preset["params"])
            merged.update(params or {})
            pan = merged.get("pan", [0.0, 0.0])
            out = os.path.join(jobdir, f"{base}_{name}.mp4")
            kb.kenburns(mpath, out,
                        duration=float(merged.get("duration", 5.0)),
                        zoom_from=float(merged.get("zoom_from", 1.0)),
                        zoom_to=float(merged.get("zoom_to", 1.3)),
                        pan=(float(pan[0]), float(pan[1])),
                        fps=int(merged.get("fps", 30)))
        elif kind == "paid_generate":
            # Paid fal.ai image-to-video. David approved fal.ai + funding.
            try:
                from i2v_client import animate_image
            except ImportError:
                import sys
                sys.path.insert(0, os.path.dirname(__file__))
                from i2v_client import animate_image
            manifest = animate_image(
                mpath,
                job["params"].get("prompt", ""),
                model=job["params"].get("model", "seedance-mini"),
                duration_s=float(job["params"].get("duration_s", 5.0)),
                allow_third_party=True,  # David approved fal.ai
                tos_ack=True,            # David approved the terms
                jobs_root=RESULTS_DIR,
            )
            out = manifest["output_path"]
            # Copy to jobdir with a clean name.
            import shutil
            result_name = f"{base}_i2v.mp4"
            dest = os.path.join(jobdir, result_name)
            shutil.copy2(out, dest)
            out = dest
            result_name = os.path.basename(out)
            JOBS.update(job_id, status="done", progress=100,
                        result_url=f"/api/result/{job_id}/{result_name}",
                        result_name=result_name)
            return
        else:
            raise RuntimeError(f"unknown kind {kind!r}")

        result_name = os.path.basename(out)
        JOBS.update(job_id, status="done", progress=100,
                    result_url=f"/api/result/{job_id}/{result_name}",
                    result_name=result_name)
    except Exception as e:  # noqa: BLE001 - job errors are reported, not raised
        JOBS.update(job_id, status="error", error=str(e)[:500])


# -------------------------------------------------------------- thumbnails ----

def thumbnail_for(name: str) -> bytes | None:
    """Return JPEG thumbnail bytes for an uploaded file, or None."""
    mpath = safe_media_path(name)
    if mpath is None:
        return None
    mtype = media_type(name)
    thumbdir = os.path.join(RESULTS_DIR, "_thumbs")
    os.makedirs(thumbdir, exist_ok=True)
    tpath = os.path.join(thumbdir, us.safe_name(name) + ".jpg")
    if os.path.isfile(tpath):
        with open(tpath, "rb") as f:
            return f.read()
    try:
        if mtype == "image" and Image is not None:
            img = Image.open(mpath).convert("RGB")
            img.thumbnail((320, 320))
            img.save(tpath, "JPEG", quality=80)
        elif mtype == "video":
            cmd = ["ffmpeg", "-hide_banner", "-y", "-ss", "0.5", "-i", mpath,
                   "-vframes", "1", "-vf", "scale=320:-1",
                   "-q:v", "4", tpath]
            proc = subprocess.run(cmd, capture_output=True, timeout=60)
            if proc.returncode != 0 or not os.path.isfile(tpath):
                return None
        else:
            return None
        with open(tpath, "rb") as f:
            return f.read()
    except Exception:  # noqa: BLE001 - thumbnail failure just means no thumb
        return None


# ---------------------------------------------------------------- handler ----

class StudioHandler(BaseHTTPRequestHandler):
    server_version = "MYAIStudio/1.0"

    def log_message(self, *args):
        pass

    # -- helpers --

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, ctype: str, filename: str | None = None):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if filename:
            self.send_header("Content-Disposition",
                             f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            return None
        if length <= 0 or length > 4 * 1024 * 1024:
            return None
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, OSError):
            return None

    # -- routes --

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._serve_ui()
        elif path == "/api/media":
            self._api_media()
        elif path == "/api/effects":
            self._send_json(effects_catalog())
        elif path == "/api/generate_options":
            self._send_json(generate_options())
        elif path == "/api/fal_status":
            self._send_json(fal_status())
        elif path.startswith("/api/thumb/"):
            self._api_thumb(unquote(path[len("/api/thumb/"):]))
        elif path.startswith("/api/job/"):
            self._api_job(path[len("/api/job/"):])
        elif path.startswith("/api/result/"):
            rest = path[len("/api/result/"):]
            parts = rest.split("/", 1)
            if len(parts) == 2:
                self._api_result(unquote(parts[0]), unquote(parts[1]))
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self._api_upload()
        elif parsed.path == "/api/apply":
            self._api_apply()
        elif parsed.path == "/api/generate":
            self._api_generate()
        else:
            self.send_error(404)

    # -- handlers --

    def _serve_ui(self):
        try:
            with open(UI_PATH, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(500, "studio_ui.html missing")
            return
        self._send_bytes(body, "text/html; charset=utf-8")

    def _api_media(self):
        items = []
        try:
            names = sorted(os.listdir(UPLOAD_DIR))
        except OSError:
            names = []
        for n in names:
            mtype = media_type(n)
            if not mtype:
                continue
            full = os.path.join(UPLOAD_DIR, n)
            if not os.path.isfile(full):
                continue
            items.append({
                "name": n,
                "type": mtype,
                "size": os.path.getsize(full),
                "thumb": f"/api/thumb/{n}",
            })
        self._send_json({"media": items})

    def _api_thumb(self, name: str):
        data = thumbnail_for(name)
        if data is None:
            self.send_error(404)
            return
        self._send_bytes(data, "image/jpeg")

    def _api_job(self, job_id: str):
        if not re.fullmatch(r"job-\d{4}", job_id or ""):
            self.send_error(404)
            return
        job = JOBS.get(job_id)
        if job is None:
            self._send_json({"error": "unknown job"}, 404)
            return
        self._send_json(job)

    def _api_result(self, job_id: str, filename: str):
        if not re.fullmatch(r"job-\d{4}", job_id or ""):
            self.send_error(404)
            return
        clean = us.safe_name(filename)
        jobdir = os.path.realpath(os.path.join(RESULTS_DIR, job_id))
        if not jobdir.startswith(os.path.realpath(RESULTS_DIR) + os.sep):
            self.send_error(404)
            return
        fpath = os.path.join(jobdir, clean)
        if not os.path.isfile(fpath):
            self.send_error(404)
            return
        ctype, _ = mimetypes.guess_type(fpath)
        with open(fpath, "rb") as f:
            self._send_bytes(f.read(), ctype or "application/octet-stream",
                             filename=clean)

    def _api_upload(self):
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > us.MAX_BYTES * 4:
            self._send_json({"error": "empty or too large"}, 400)
            return
        form = cgi.FieldStorage(
            fp=self.rfile, headers=self.headers,
            environ={"REQUEST_METHOD": "POST",
                     "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                     "CONTENT_LENGTH": str(length)})
        saved = []
        items = form["files"] if "files" in form else []
        if not isinstance(items, list):
            items = [items]
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        for item in items:
            if not getattr(item, "filename", None):
                continue
            name = us.safe_name(item.filename)
            if media_type(name) is None:
                continue  # only images and videos
            dest = os.path.join(UPLOAD_DIR, name)
            stem, ext = os.path.splitext(name)
            n = 1
            while os.path.exists(dest):
                dest = os.path.join(UPLOAD_DIR, f"{stem}-{n}{ext}")
                n += 1
            size = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = item.file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > us.MAX_BYTES:
                        f.close()
                        os.remove(dest)
                        self._send_json({"error": "file too large"}, 413)
                        return
                    f.write(chunk)
            # Privacy: strip ALL metadata from every upload, before storage.
            try:
                from privacy import strip_metadata
                strip_metadata(dest)
            except Exception:
                pass  # never fail an upload on metadata stripping
            saved.append({"name": os.path.basename(dest),
                          "type": media_type(dest), "size": size})
        self._send_json({"saved": saved})

    def _api_apply(self):
        body = self._read_json()
        if not body:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        media = body.get("media", "")
        kind = body.get("kind", "")
        name = body.get("name", "")
        params = body.get("params") or {}
        _, err = validate_apply(media, kind, name, params)
        if err:
            # Distinguish "media not found" (404) from bad requests (400).
            self._send_json({"error": err},
                            404 if "not found" in err else 400)
            return
        job_id = JOBS.new(media, kind, name, params)
        POOL.submit(_run_job, job_id)
        self._send_json({"job_id": job_id})

    def _api_generate(self):
        """Paid generation endpoint. Stubbed until i2v_client lands."""
        body = self._read_json()
        if not body:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        gen_id = body.get("id", "")
        prompt = (body.get("prompt") or "").strip()
        media = body.get("media") or ""
        opt = next((o for o in GENERATE_OPTIONS if o["id"] == gen_id), None)
        if opt is None:
            self._send_json({"error": f"unknown generate option {gen_id!r}"},
                            400)
            return
        if not opt["paid"]:
            self._send_json({"error": "use /api/apply for free options"},
                            400)
            return
        if opt["needs_prompt"] and not prompt:
            self._send_json({"error": "a prompt is required"}, 400)
            return
        if opt["needs_media"]:
            mpath = safe_media_path(media)
            if mpath is None:
                self._send_json({"error": f"media {media!r} not found"}, 404)
                return
            if media_type(media) != opt["needs_media"]:
                self._send_json(
                    {"error": f"this needs an {opt['needs_media']}"}, 400)
                return
        info = fal_status()
        if not info["configured"]:
            self._send_json({"status": "needs_setup",
                             "message": info["message"]})
            return
        # Only image-to-video is wired up. Text-to-video needs a backend.
        if gen_id != "i2v":
            self._send_json({
                "status": "not_available",
                "message": ("Text-to-video isn't wired up yet — "
                            "image-to-video is ready now."),
            })
            return
        # Launch the paid I2V job in the background.
        mpath = safe_media_path(media)
        job_id = JOBS.new(media, "paid_generate", gen_id, {
            "prompt": prompt,
            "model": "seedance-mini",  # approved route
            "duration_s": 5.0,
        })
        POOL.submit(_run_job, job_id)
        self._send_json({"job_id": job_id, "status": "started",
                         "estimate": opt["cost"]})


def main(argv=None) -> int:
    global UPLOAD_DIR, RESULTS_DIR, JOBS_LOG
    ap = argparse.ArgumentParser(description="MYAI Media Studio server")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--dir", default=UPLOAD_DIR,
                    help="upload directory")
    ap.add_argument("--results", default=RESULTS_DIR,
                    help="job results directory")
    args = ap.parse_args(argv)

    UPLOAD_DIR = args.dir
    RESULTS_DIR = args.results
    JOBS_LOG = os.path.join(RESULTS_DIR, "jobs.jsonl")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), StudioHandler)
    print(f"MYAI Media Studio on 127.0.0.1:{args.port} "
          f"(uploads: {UPLOAD_DIR})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

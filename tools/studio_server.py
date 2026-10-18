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
    import video_edit as ve
    VIDEO_EDIT_OPS = ve.EDIT_OPS
except ImportError:
    ve = None
    VIDEO_EDIT_OPS = {}

try:
    import longvideo as lv
except ImportError:
    lv = None

try:
    import face_vfx as fv
except ImportError:
    fv = None

try:
    import animate_client as ac
except ImportError:
    ac = None

try:
    import venice_client as vc
except ImportError:
    vc = None

try:
    import gpu_client as gc
    import gpu_workflows as gw
except ImportError:
    gc = None
    gw = None

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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi", ".flv",
              ".wmv", ".ts", ".m2ts", ".mpg", ".mpeg", ".3gp", ".ogv"}

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
KINDS = ("style_pack", "photo_op", "video_effect", "kenburns", "video_edit")


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


_MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS
# Engine scratch outputs inside a job folder: real files, but not
# deliverables a user should ever be shown as a "result".
_INTERMEDIATE_PREFIXES = ("scail_replace", "gap_", "wan_", "vace_")


def job_deliverables(job_id: str, result_name: str | None) -> list[dict]:
    """Viewable deliverables in a job folder, best first.

    Post-step finals (face-locked) outrank the recorded render, so the
    file a job presents as its result is the finished one even when a
    later step added a better version after the job record was written.
    Scratch files (leading underscore/dot) and engine intermediates are
    hidden.
    """
    jobdir = os.path.join(RESULTS_DIR, job_id)
    try:
        names = os.listdir(jobdir)
    except OSError:
        return []
    found = []
    for n in names:
        if n.startswith(("_", ".")):
            continue
        if os.path.splitext(n)[1].lower() not in _MEDIA_EXTS:
            continue
        full = os.path.join(jobdir, n)
        if not os.path.isfile(full):
            continue
        low = n.lower()
        if "locked" in low or "facefusion" in low:
            rank, label = 0, "Face-locked final"
        elif result_name and n == result_name:
            rank, label = 1, "Result"
        elif low.startswith(_INTERMEDIATE_PREFIXES):
            continue
        else:
            rank = 2
            label = os.path.splitext(n)[0].replace("_", " ").strip().title()
        try:
            st = os.stat(full)
            size, mtime = st.st_size, st.st_mtime
        except OSError:
            size, mtime = 0, 0
        found.append({
            "name": n, "label": label, "kind": media_type(n),
            "url": f"/api/result/{job_id}/{n}",
            "size": size, "mtime": mtime, "_rank": rank,
        })
    found.sort(key=lambda d: (d["_rank"], -d["mtime"]))
    for i, d in enumerate(found):
        d["primary"] = (i == 0)
        del d["_rank"]
    if len(found) > 1 and found[0]["label"] == "Result":
        found[0]["label"] = "Final render"
    if any(d["label"] == "Face-locked final" for d in found):
        for d in found:
            if d["label"] == "Result":
                d["label"] = "Raw render"
    return found


def job_primary(job_id: str, result_name: str | None,
                result_url: str | None) -> dict:
    """The one file a job should present as its result."""
    dels = job_deliverables(job_id, result_name)
    prim = next((d for d in dels if d["primary"]), None)
    if prim:
        return {"primary_url": prim["url"], "primary_name": prim["name"],
                "primary_kind": prim["kind"]}
    return {"primary_url": result_url, "primary_name": result_name,
            "primary_kind": media_type(result_name) if result_name else None}


def job_has_clips(job_id: str) -> bool:
    chain = os.path.join(RESULTS_DIR, job_id, "chain")
    try:
        return any(n.startswith("clip_") and n.endswith(".mp4")
                   for n in os.listdir(chain))
    except OSError:
        return False


def _media_roles() -> dict:
    """Optional filename -> {role, label} map for the Library.

    Lives at <UPLOAD_DIR>/.media_roles.json; absent file = no roles.
    """
    try:
        with open(os.path.join(UPLOAD_DIR, ".media_roles.json"),
                  encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


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
     "cost": "~$1.50/clip (5s)",
     "description": ("Real AI motion from your photo (fal.ai). Select 1 photo "
                     "to animate, or 2-9 photos (and up to 3 short clips) "
                     "for better character consistency."),
     "needs_media": "image",
     "multi_ref": True,
     "needs_prompt": True,
     "paid": True},
    {"id": "i2v_venice",
     "name": "Animate Photo (Venice AI)",
     "badge": "$",
     "cost": "~$0.65/clip (5s)",
     "description": ("AI motion via Venice (Wan 3.0). No platform content filter; "
                     "privacy-friendly. Select 1 photo to animate."),
     "needs_media": "image",
     "multi_ref": False,
     "needs_prompt": True,
     "paid": True},
    {"id": "i2v_venice_uncensored",
     "name": "Animate Photo (Venice Uncensored)",
     "badge": "$",
     "cost": "~$0.64/clip (5s)",
     "description": ("AI motion via Venice enhanced uncensored model (Wan 2.7). "
                     "Maximum creative freedom. Select 1 photo to animate."),
     "needs_media": "image",
     "multi_ref": False,
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
    {"id": "v2v_restyle",
     "name": "AI Restyle (Video-to-Video)",
     "badge": "$",
     "cost": "~$0.65-1.50/clip",
     "description": ("Transform an existing clip with AI — restyle, change "
                     "mood, reimagine. Uncensored Wan 2.7 available."),
     "needs_media": "video",
     "needs_prompt": True,
     "paid": True},
    {"id": "model_replace",
     "name": "Replace Model in Video",
     "badge": "$",
     "cost": "AI ~$3.00/clip · Face FREE",
     "description": ("Put a different model into an existing video. Pick "
                     "the base video, then a reference photo or reference "
                     "video of the new model. AI does full-body swaps; "
                     "Face mode is free/local and swaps just the face."),
     "needs_media": "video",
     "needs_prompt": False,
     "paid": True},
    {"id": "gpu_image_edit",
     "name": "Edit Photo (Your GPU)",
     "badge": "$",
     "cost": "~$0.10/edit (pod time)",
     "description": ("Edit a photo by instruction on your own RunPod GPU "
                     "(Qwen-Image-Edit) — the subject stays true to the "
                     "reference. Open model, no filter. Needs the pod "
                     "running (see runpod/README.md)."),
     "needs_media": "image",
     "needs_prompt": True,
     "paid": True},
    {"id": "gpu_i2v",
     "name": "Animate Photo (Your GPU)",
     "badge": "$",
     "cost": "~$1.00/clip (pod time)",
     "description": ("AI motion via Wan 2.2 on your own RunPod GPU. "
                     "No per-clip API charge and no platform filter — "
                     "just pod time. Needs the pod running."),
     "needs_media": "image",
     "needs_prompt": True,
     "paid": True},
    {"id": "gpu_t2v",
     "name": "Text to Video (Your GPU)",
     "badge": "$",
     "cost": "~$1.00/clip (pod time)",
     "description": ("AI video from text via Wan 2.2 on your own RunPod "
                     "GPU. Open model, full parameter control. Needs "
                     "the pod running."),
     "needs_media": None,
     "needs_prompt": True,
     "paid": True},
    {"id": "gpu_replace",
     "name": "Replace Model (Your GPU)",
     "badge": "$",
     "cost": "~$1.00/clip (pod time)",
     "description": ("Put a different model into an existing video via "
                     "Wan VACE on your own RunPod GPU. Pick the base "
                     "video AND a reference photo or video of the new model. "
                     "Needs the pod running."),
     "needs_media": "video",
     "needs_prompt": False,
     "paid": True},
    {"id": "long_video",
     "name": "Long Video (Chained, minutes)",
     "badge": "$",
     "cost": "per-clip × N clips",
     "description": ("Minutes-long continuous video: your prompt becomes a "
                     "narrative arc, generated clip-by-clip with last-frame "
                     "continuity and stitched seamlessly."),
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
    edits = [{"name": n, "description": s["description"],
              "applies_to": "video", "params": _paramspec(s["params"]),
              "output": s.get("output", "video"), "multi": s.get("multi", False)}
             for n, s in VIDEO_EDIT_OPS.items()]
    return {"style_pack": packs, "photo_op": photos,
            "video_effect": videos, "kenburns": animate,
            "video_edit": edits}


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
        self._lock = threading.RLock()
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
        self._snapshot()

    def _snapshot(self) -> None:
        """Persist full job state to disk so jobs survive restarts.
        Called with the lock already held (from new/update via _log)."""
        snap_path = os.path.join(RESULTS_DIR, "jobs_snapshot.json")
        try:
            jobs_copy = dict(self._jobs)
            counter = self._counter
            with open(snap_path + ".tmp", "w") as f:
                json.dump({"counter": counter, "jobs": jobs_copy}, f)
            os.replace(snap_path + ".tmp", snap_path)
        except OSError:
            pass

    def restore(self) -> None:
        """Load jobs from disk snapshot. Running jobs -> interrupted."""
        snap_path = os.path.join(RESULTS_DIR, "jobs_snapshot.json")
        try:
            with open(snap_path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        with self._lock:
            self._counter = data.get("counter", 0)
            for job_id, job in data.get("jobs", {}).items():
                # Jobs that were running when the server stopped
                # didn't finish — mark them interrupted
                if job.get("status") in ("queued", "running"):
                    job["status"] = "interrupted"
                    job["error"] = "Server restarted during this job"
                    job["phase"] = "interrupted"
                self._jobs[job_id] = job


JOBS = JobStore()
JOBS.restore()  # Recover jobs from previous run, if any
POOL = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)


def _probe_video_info(path: str) -> dict:
    """ffprobe facts about a video file ({} when unreadable)."""
    import subprocess as _sp
    try:
        pr = _sp.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration,format_name,size:stream=codec_type,codec_name,width,height",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30)
        data = json.loads(pr.stdout or "{}")
    except Exception:
        return {}
    fmt = data.get("format", {})
    vcodec = ""
    vwidth = vheight = 0
    for st in data.get("streams", []):
        if st.get("codec_type") == "video":
            vcodec = st.get("codec_name", "")
            try:
                vwidth = int(st.get("width") or 0)
                vheight = int(st.get("height") or 0)
            except (TypeError, ValueError):
                vwidth = vheight = 0
            break
    try:
        duration = float(fmt.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    try:
        size = int(fmt.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    return {"duration": duration, "format": fmt.get("format_name", ""),
            "vcodec": vcodec, "size": size,
            "width": vwidth, "height": vheight}


def _venice_ready_video(path: str, workdir: str, tag: str) -> str:
    """Return a Seedance-spec copy of a video (MP4/MOV + H.264/H.265).

    Venice's Seedance reference inputs accept only mp4/mov containers
    with H.264/HEVC video; anything else (webm, mkv, VP9, AV1...) makes
    their endpoint fail with a generic 500. Transcode off-spec files
    locally with ffmpeg; on-spec files pass through untouched.
    """
    info = _probe_video_info(path)
    fmt = (info.get("format") or "").lower()
    vcodec = (info.get("vcodec") or "").lower()
    if ("mp4" in fmt or "mov" in fmt) and vcodec in ("h264", "hevc", "h265"):
        return path
    out = os.path.join(workdir, f"{tag}_venice.mp4")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", path,
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-movflags", "+faststart", out]
    import subprocess as _sp
    r = _sp.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(
            "could not convert the video to MP4/H.264 for Venice: "
            + (r.stderr or "")[:200])
    return out


# Wan Animate accepts at most 120s per job; long base videos are
# processed in chunks under that cap and stitched back together.
ANIMATE_CHUNK_S = 110.0


def _chunk_bounds(duration: float, chunk_s: float) -> list:
    """[(start, length), ...] covering `duration` in <= chunk_s parts."""
    bounds = []
    start = 0.0
    while start < duration - 0.05:
        length = min(chunk_s, duration - start)
        bounds.append((start, length))
        start += length
    return bounds or [(0.0, duration)]


def _even_bounds(duration: float, max_chunk: float) -> list:
    """Even [(start, length), ...] parts, each <= max_chunk and >= 2s
    (Seedance reference clips need 2-15s; the tail must not strand a
    sub-2s runt)."""
    import math
    n = max(1, int(math.ceil(duration / max_chunk)))
    step = duration / n
    return [(round(i * step, 3), round(step, 3)) for i in range(n)]


def _split_video_chunks(path: str, workdir: str, chunk_s: float,
                        even: bool = False) -> list:
    """Split a video into re-encoded MP4 chunks (no audio needed:
    the original audio is muxed back onto the final stitch).

    even=True spreads the duration into equal parts <= chunk_s so no
    runt tail is left (used for Seedance's 2-15s clip window).
    """
    import subprocess as _sp
    info = _probe_video_info(path)
    duration = info.get("duration") or 0.0
    if duration <= chunk_s + 0.05:
        return [path]
    chunks = []
    bounds = (_even_bounds(duration, chunk_s) if even
              else _chunk_bounds(duration, chunk_s))
    for i, (start, length) in enumerate(bounds):
        out = os.path.join(workdir, f"chunk_{i:03d}.mp4")
        r = _sp.run(["ffmpeg", "-y", "-v", "error",
                     "-ss", f"{start:.3f}", "-i", path,
                     "-t", f"{length:.3f}", "-an",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                    capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError("splitting the video failed: "
                               + (r.stderr or "")[:200])
        chunks.append(out)
    return chunks


def _concat_normalized(parts: list, out_path: str, like_path: str) -> None:
    """Concatenate videos, normalizing each to like_path's size/fps
    first so provider outputs with slightly different specs still join
    cleanly. Falls back to a plain concat when probing fails."""
    import subprocess as _sp
    info = _probe_video_info(like_path)
    w = h = 0
    try:
        pr = _sp.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                      "-show_entries", "stream=width,height,avg_frame_rate",
                      "-of", "json", like_path],
                     capture_output=True, text=True, timeout=30)
        st = (json.loads(pr.stdout or "{}").get("streams") or [{}])[0]
        w, h = int(st.get("width") or 0), int(st.get("height") or 0)
    except Exception:
        pass
    workdir = os.path.dirname(out_path)
    norm = []
    for i, part in enumerate(parts):
        n = os.path.join(workdir, f"norm_{i:03d}.mp4")
        vf = []
        if w and h:
            vf.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                      f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2")
        vf.append("fps=30")
        vf.append("format=yuv420p")
        r = _sp.run(["ffmpeg", "-y", "-v", "error", "-i", part,
                     "-vf", ",".join(vf), "-an",
                     "-c:v", "libx264", n],
                    capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not os.path.exists(n):
            raise RuntimeError("normalizing a chunk failed: "
                               + (r.stderr or "")[:200])
        norm.append(n)
    listfile = os.path.join(workdir, "concat_list.txt")
    with open(listfile, "w") as f:
        for n in norm:
            f.write(f"file '{n}'\n")
    r = _sp.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                 "-i", listfile, "-c", "copy", out_path],
                capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError("stitching chunks failed: "
                           + (r.stderr or "")[:200])


def _ensure_photo_ref(ref_path: str, jobdir: str) -> str:
    """Reference image path for a photo or video reference.

    Face Replace and Wan Animate take a still image of the model.
    When the reference is a VIDEO, grab a frame from early in the
    clip (face usually visible near the start) and use that — so a
    video of the model works as a reference everywhere.
    """
    if media_type(os.path.basename(ref_path)) == "image":
        return ref_path
    info = _probe_video_info(ref_path)
    dur = info.get("duration") or 0.0
    at = min(2.0, dur / 4.0) if dur else 0.0
    out = os.path.join(jobdir, "ref_frame.jpg")
    import subprocess as _sp
    r = _sp.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.2f}",
                 "-i", ref_path, "-frames:v", "1", "-q:v", "2", out],
                capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("could not pull a still frame from the "
                           "reference video")
    return out



def _ref_sheet(paths: list, out_path: str, cell: int = 896) -> str:
    """Tile 2-4 reference photos into ONE composite reference image.

    VACE conditions on a single reference image. A sheet of several
    consistent photos (face close-up, full body, side view) gives the
    model more identity evidence than any one shot and measurably
    reduces drift when the source video changes angle. Each photo is
    letterboxed into its cell (never cropped or stretched).
    """
    if Image is None:
        raise RuntimeError("PIL not available for the reference sheet")
    import math as _math
    n = len(paths)
    cols = 2 if n <= 4 else 3
    rows = _math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * cell, rows * cell), (24, 24, 24))
    for i, pth in enumerate(paths):
        im = Image.open(pth).convert("RGB")
        im.thumbnail((cell, cell))
        x = (i % cols) * cell + (cell - im.width) // 2
        y = (i // cols) * cell + (cell - im.height) // 2
        sheet.paste(im, (x, y))
    sheet.save(out_path, "JPEG", quality=92)
    return out_path

def _mux_audio(silent_video: str, audio_source: str, out_path: str) -> str:
    """Copy silent_video's picture + audio_source's audio into out_path.

    Falls back to a plain copy of silent_video when the source has no
    audio or ffmpeg fails. Returns out_path.
    """
    import subprocess as _sp
    import shutil as _sh
    r = _sp.run(
        ["ffmpeg", "-y", "-i", silent_video, "-i", audio_source,
         "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
         "-c:a", "aac", "-shortest", out_path],
        capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not os.path.isfile(out_path):
        _sh.copy2(silent_video, out_path)
    return out_path


def _largest_face_crop_box(img_w: int, img_h: int, boxes):
    """Crop box (x0, y0, x1, y1) around the largest detected face,
    with ~45% margin, clamped to the image. None when no boxes."""
    if not boxes:
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    mx, my = int(w * 0.45), int(h * 0.45)
    return (max(0, x - mx), max(0, y - my),
            min(img_w, x + w + mx), min(img_h, y + h + my))


def _ref_face_crop(ref_photo_path: str):
    """The reference's FACE CROP (BGR array).

    face_vfx.composite_face is contracted for a face crop; feeding it
    a full photo warps the ENTIRE picture into the target face oval --
    the floating 'bubble with the whole figure' artifact. Raises when
    no face can be found (callers fall back to the un-locked video).
    """
    if fv is None:
        raise RuntimeError("face_vfx module not available")
    import cv2 as _cv2
    img = _cv2.imread(ref_photo_path)
    if img is None:
        raise RuntimeError("could not read reference photo")
    box = _largest_face_crop_box(img.shape[1], img.shape[0],
                                 fv.detect_faces(img))
    if box is None:
        raise RuntimeError("no face detected in reference photo")
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


# Shown on the finished job when the true-to-reference face lock could
# not be applied, so an unlocked result is never passed off as locked.
FACE_LOCK_FAILED_NOTE = "face lock FAILED — showing unlocked version"


def _lock_failure_note(e: BaseException) -> str:
    """The Face Lock failure note WITH its reason. Until Oct 17 the
    call sites swallowed the exception into a bare boolean, so the
    failure repeated for days with no diagnosable record anywhere.
    Capture it in the job's warning AND print it (the user service's
    stdout lands in the journal)."""
    note = f"{FACE_LOCK_FAILED_NOTE}: {type(e).__name__}: {e}"
    print(f"[face_lock] {note}", flush=True)
    return note


def _face_lock(video_path: str, ref_photo_path: str, out_path: str,
               workdir: str) -> str:
    """Composite the reference photo's face onto every frame of a video.

    This is the true-to-reference step: generative models redraw the
    person and drift; this puts the reference's actual face back in,
    tracked per frame, then muxes the video's own audio back.
    Returns out_path. Raises on failure (callers decide the fallback).
    """
    if fv is None:
        raise RuntimeError("face_vfx module not available")
    src_face = _ref_face_crop(ref_photo_path)
    silent = os.path.join(workdir, "_facelock_silent.mp4")
    fv.replace_in_video(video_path, src_face, silent)
    # OpenCV writes MPEG-4 Part 2 ("mp4v"), which phone and browser
    # players render as a BLACK SCREEN (audio plays, picture does
    # not). Re-encode the composited video to H.264 and attach the
    # source video's audio in the same pass.
    import subprocess as _sp
    r = _sp.run(["ffmpeg", "-y", "-v", "error",
                 "-i", silent, "-i", video_path,
                 "-map", "0:v:0", "-map", "1:a:0?",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-shortest", out_path],
                capture_output=True, text=True, timeout=3600)
    if r.returncode != 0 or not os.path.exists(out_path):
        # Last resort: mux the un-re-encoded composite (may not
        # play everywhere, but keeps the job's work).
        _mux_audio(silent, video_path, out_path)
    try:
        os.remove(silent)
    except OSError:
        pass
    return out_path


def _count_frames(path: str) -> int:
    """Exact frame count via ffprobe; 0 when probing fails."""
    import subprocess as _sp
    try:
        r = _sp.run(["ffprobe", "-v", "error", "-count_frames",
                     "-select_streams", "v:0", "-show_entries",
                     "stream=nb_read_frames", "-of", "csv=p=0", path],
                    capture_output=True, text=True, timeout=120)
        return int((r.stdout or "0").strip() or 0)
    except Exception:
        return 0


def _scail_segment(src: str, start_f: int, count: int, fps: float,
                   out: str) -> None:
    """Cut an exact frame window of the 16fps timeline from src."""
    import subprocess as _sp
    r = _sp.run(["ffmpeg", "-y", "-v", "error",
                 "-ss", f"{start_f / fps:.3f}", "-i", src,
                 "-vf", f"fps={fps:g}", "-frames:v", str(count),
                 "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"segment extract failed: {r.stderr[-300:]}")


def _scail_trim_head(src: str, head: int, out: str) -> None:
    """Drop the first `head` frames (the re-rendered anchor overlap)."""
    import subprocess as _sp
    r = _sp.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                 "-vf", f"select='gte(n\\,{head})',setpts=N/FRAME_RATE/TB",
                 "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"chunk trim failed: {r.stderr[-300:]}")


def _run_scail_chunks(gc_, gw_, job_id: str, jobdir: str, vpath: str,
                      ref_up: str, prompt: str, mask_prompt: str,
                      mask_threshold: float, sk: dict, info: dict,
                      progress_cb=None) -> str:
    """Render a long SCAIL-2 replace in 81-frame chunks chained by
    5-frame anchors (the model's trained extension recipe: 81-frame
    chunks, 76-frame step), then stitch. Each chunk after the first
    re-renders the previous chunk's last 5 frames as its head; those
    are trimmed at the stitch so the timeline stays exact. Returns
    the stitched silent video path (the caller's audio mux + Face
    Lock post-passes run on it unchanged)."""
    import subprocess as _sp
    fps = 16.0
    total = int(float(info.get("duration") or 0.0) * fps)
    width, height = int(sk["width"]), int(sk["height"])
    base = {k: v for k, v in sk.items()
            if k not in ("width", "height", "frames")}
    step, chunk_len = 76, 81
    starts = []
    s = 0
    while s < total:
        starts.append(s)
        s += step
    if len(starts) > 1 and total - starts[-1] < 17:
        # A runt tail chunk borrows frames from the previous window.
        starts[-1] = max(starts[-2] + 17, total - 33)
    workdir = os.path.join(jobdir, "_scail_chunks")
    os.makedirs(workdir, exist_ok=True)
    kept, anchor_up, covered_end = [], None, 0
    for k, start in enumerate(starts):
        seg_len = min(chunk_len, total - start)
        seg = os.path.join(workdir, f"seg_{k:03d}.mp4")
        _scail_segment(vpath, start, seg_len, fps, seg)
        seg_up = gc_.upload_video(seg)
        wf = gw_.video_replace_model_scail(
            seg_up, ref_up, prompt, mask_prompt=mask_prompt,
            mask_threshold=mask_threshold, width=width, height=height,
            frames=seg_len, previous_frames_video=anchor_up,
            frame_offset=start, **base)
        JOBS.update(job_id, status="running",
                    progress=25 + int(60 * k / max(1, len(starts))),
                    phase=f"rendering chunk {k + 1} of {len(starts)} "
                          "on your GPU pod")
        outs = gc_.run_workflow(wf, os.path.join(workdir, f"out_{k:03d}"),
                                timeout_s=3600.0, poll_interval_s=10.0,
                                progress_cb=progress_cb)
        if not outs:
            raise RuntimeError(
                f"GPU pod produced no output for chunk {k + 1}")
        out = outs[0]
        produced = _count_frames(out) or seg_len
        head = max(0, covered_end - start)
        if 0 < head < produced:
            piece = os.path.join(workdir, f"piece_{k:03d}.mp4")
            _scail_trim_head(out, head, piece)
        else:
            piece = out
        kept.append(piece)
        covered_end = start + produced - head
        if k < len(starts) - 1:
            anchor = os.path.join(workdir, f"anchor_{k:03d}.mp4")
            _scail_trim_head(out, max(0, produced - 5), anchor)
            anchor_up = gc_.upload_video(anchor)
    if len(kept) == 1:
        return kept[0]
    stitched = os.path.join(jobdir, "_scail_stitched.mp4")
    listfile = os.path.join(workdir, "concat.txt")
    with open(listfile, "w") as fh:
        for piece in kept:
            fh.write(f"file '{os.path.abspath(piece)}'\n")
    r = _sp.run(["ffmpeg", "-y", "-v", "error", "-f", "concat",
                 "-safe", "0", "-i", listfile, "-c", "copy", stitched],
                capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not os.path.exists(stitched):
        _concat_normalized(kept, stitched, kept[0])
    return stitched


def _run_seedvr2_finish(gc_, gw_, jobdir: str, video_path: str,
                        resolution: int = 1024) -> str:
    """SeedVR2 detail finish over a finished silent video -- the
    long-video counterpart of the in-graph SCAIL pass: upload the
    stitched video, run the standalone upscale graph, download.
    Returns the finished path; raises on pod failure (the caller
    falls back to the un-finished video)."""
    up = gc_.upload_video(video_path)
    wf = gw_.video_seedvr2_upscale(up, resolution=resolution, fps=16.0)
    outs = gc_.run_workflow(wf, os.path.join(jobdir, "_seedvr2"),
                            timeout_s=3600.0, poll_interval_s=10.0)
    if not outs:
        raise RuntimeError("GPU pod produced no SeedVR2 output")
    return outs[0]


def _run_gpu_job(job_id: str, job: dict) -> None:
    """Self-hosted GPU generation on David's RunPod ComfyUI pod.

    kind "gpu_generate"; name is the generate-option id (gpu_image_edit /
    gpu_i2v / gpu_t2v / gpu_replace). Media flows only between this
    server and David's own pod — no third-party API is involved. An
    offline pod raises GpuUnreachableError and the job ends in a clear
    error state; Studio itself is unaffected. Raises on failure —
    _run_job's handler records the error.
    """
    if gc is None or gw is None:
        raise RuntimeError("gpu modules not available")
    p = job["params"]
    gen_id = job["name"]
    jobdir = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(jobdir, exist_ok=True)
    media_name = p.get("media_name") or ""
    base = (os.path.splitext(us.safe_name(media_name))[0]
            if media_name else gen_id)

    def _f(key, default):
        try:
            return float(p.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _i(key, default):
        return int(_f(key, default))

    def _progress(_job):
        # Cancellation: the cancel endpoint flips status; interrupt the
        # pod render and bail with InterruptedError (handled upstream).
        if (JOBS.get(job_id) or {}).get("status") == "cancelled":
            try:
                gc.interrupt()
            except Exception:
                pass
            raise InterruptedError("Job cancelled by user")
        JOBS.update(job_id, status="running", progress=40,
                    phase="rendering on your GPU pod")

    video_knobs = dict(
        negative_prompt=p.get("negative_prompt") or gw.DEFAULT_NEGATIVE,
        seed=_i("seed", 0), steps=_i("steps", 20), cfg=_f("cfg", 5.0),
        width=_i("width", 832), height=_i("height", 480),
        frames=_i("frames", 81), fps=_f("fps", 16.0))

    JOBS.update(job_id, status="running", progress=10,
                phase="uploading to your GPU pod")
    scail_chunks = None   # gpu_replace SCAIL: long-video chunk plan
    scail_seedvr2 = False  # gpu_replace SCAIL: SeedVR2 detail finish
    use_gap = False        # gpu_replace: GAP orchestrator engine
    gpu_face_ref = None   # gpu_replace: face-lock source image
    gpu_src_video = None  # gpu_replace: original video (audio source)
    if gen_id == "gpu_image_edit":
        mpath = safe_media_path(media_name)
        if mpath is None:
            raise RuntimeError(f"media {media_name!r} disappeared")
        up = gc.upload_image(mpath)
        wf = gw.image_edit(
            up, p.get("prompt", ""),
            negative_prompt=p.get("negative_prompt", ""),
            seed=_i("seed", 0), steps=_i("steps", 20), cfg=_f("cfg", 4.0),
            denoise=_f("denoise", 1.0),
            width=(int(p["width"]) if p.get("width") else None),
            height=(int(p["height"]) if p.get("height") else None))
        suffix, timeout_s = "gpuedit", 1200.0
    elif gen_id == "gpu_i2v":
        mpath = safe_media_path(media_name)
        if mpath is None:
            raise RuntimeError(f"media {media_name!r} disappeared")
        up = gc.upload_image(mpath)
        wf = gw.image_to_video(up, p.get("prompt", ""), **video_knobs)
        suffix, timeout_s = "gpu_i2v", 3600.0
    elif gen_id == "gpu_t2v":
        wf = gw.text_to_video(p.get("prompt", ""), **video_knobs)
        suffix, timeout_s = "gpu_t2v", 3600.0
    elif gen_id == "gpu_replace":
        vpath = safe_media_path(media_name)
        if vpath is None:
            raise RuntimeError("base video disappeared")
        ref_names = list(p.get("ref_media_names")
                         or [p.get("ref_media_name") or ""])
        ref_paths = []
        for rn in ref_names:
            rp = safe_media_path(rn)
            if rp is None:
                raise RuntimeError("a reference file disappeared")
            if media_type(rn) == "video":
                # VACE conditions on a still image; take a frame of
                # the reference video (same helper as Model Replace).
                rp = _ensure_photo_ref(rp, jobdir)
            ref_paths.append(rp)
        if len(ref_paths) > 1:
            rpath = _ref_sheet(ref_paths,
                               os.path.join(jobdir, "_ref_sheet.jpg"))
        else:
            rpath = ref_paths[0]
        knobs = dict(video_knobs)
        info = _probe_video_info(vpath)
        if "frames" not in p:
            # Cover the WHOLE clip, not the 81-frame (~5s) workflow
            # default: derive the frame count from the real duration
            # (4n+1 grid), capped per single VACE pass.
            dur = float(info.get("duration") or 0.0)
            if dur > 0:
                raw = int(dur * float(knobs.get("fps", 16.0)))
                knobs["frames"] = max(17, min(321, 4 * round((raw - 1) / 4) + 1))
        if not p.get("width") and info.get("width"):
            # Render at the source's own resolution (16-aligned):
            # faster than upscaling and keeps its aspect ratio true.
            knobs["width"] = max(16, (int(info["width"]) // 16) * 16)
            knobs["height"] = max(16, (int(info["height"]) // 16) * 16)
        prompt = (p.get("prompt") or "").strip() or (
            "a photorealistic woman, identical to the woman in the "
            "reference image, with the same face, hair, body and skin "
            "tone, performing exactly the same actions and poses as "
            "the woman in the source video, natural skin texture, "
            "consistent lighting")
        gpu_face_ref = ref_paths[0]
        gpu_src_video = vpath
        vup = gc.upload_video(vpath)
        # SCAIL-2 is the primary engine: it is also the DEFAULT when
        # the request carries no engine (a cached pre-selector Studio
        # page sends none, and must not silently land on the VACE
        # fallback -- pod-verified Oct 15, two VACE runs killed).
        engine = (p.get("engine") or "scail").strip().lower()
        if engine in ("scail", "gap"):
            use_gap = engine == "gap"
            # SCAIL-2 end-to-end replacement: ONE reference image (the
            # first), not the tiled sheet -- its coloured-mask identity
            # binding wants a single clean subject. Works larger than
            # the source: 32-aligned, ~704 wide, aspect preserved.
            iup = gc.upload_image(ref_paths[0])
            sk = dict(knobs)
            # SCAIL-2 finish: ZONED (preservation) composite.
            # David's current spec (Oct 15, supersedes the earlier
            # act-performed-by-the-models clarification): EVERYTHING
            # in the original video is preserved; only the female
            # model is swapped for the reference. So her tracked
            # region minus the contact patch is the only generated
            # area; the partner, the room and the contact itself
            # stay source pixels. (The union finish -- both bodies
            # and their contact jointly generated -- let the base
            # model redraw the contact zone and censor it with an
            # invented towel; it stays available via zoned=false.)
            # Seam softening: grow 16, feather 8.
            sk.setdefault("zoned", True)
            sk.setdefault("composite_expand", 16)
            sk.setdefault("composite_feather", 8)
            if info.get("width"):
                sw, sh = int(info["width"]), int(info["height"])
                tw = 704 if sw <= 1408 else (sw // 32) * 32
                sk["width"] = tw
                sk["height"] = max(32, (round(sh * tw / sw) // 32) * 32)
            # Sampling profile (research sweep, Oct 15): the
            # distilled 6-step / cfg-1 path matches neither official
            # SCAIL-2 recipe and distilled sampling is reported to
            # collapse multi-identity binding. The measured
            # identity-holding compromise is 24 steps / cfg 5 /
            # shift 3 with the LightX2V distill LoRA OFF.
            if "steps" not in p:
                sk["steps"] = 24
            if "cfg" not in p:
                sk["cfg"] = 5.0
            if "shift" not in p:
                sk["shift"] = 3.0
            if "lightx2v_strength" not in p:
                sk["lightx2v_strength"] = 0.0
            # SeedVR2 detail finish (optional, off by default):
            # single-pass jobs take it in-graph; long jobs get it
            # as a post-pass over the stitched video instead (chunk
            # anchors are render-size, so it cannot run in a chunk).
            scail_seedvr2 = (bool(p.get("seedvr2"))
                             and p.get("seedvr2")
                             not in (False, 0, "false"))
            # Person LoRA (skswoman, trained on her own frames):
            # rides in sk so the single-pass AND chunked paths both
            # pick it up; the GAP call passes it by name below.
            person_on = (bool(p.get("person_lora"))
                         and p.get("person_lora")
                         not in (False, 0, "false"))
            scail_person = 0.0
            if person_on:
                # Ladder-2 probe (Oct 18): an explicit strength
                # overrides the 1.0 default -- the trained LoRAs
                # carry alpha 1 at dim 32, so a 3.0 run tests
                # whether the alpha gate is what mutes them.
                try:
                    scail_person = float(
                        p.get("person_lora_strength", 1.0))
                except (TypeError, ValueError):
                    scail_person = 1.0
                scail_person = min(5.0, max(0.0, scail_person))
            sk["person_lora_strength"] = scail_person
            # Pivot wiring (Oct 18): the single-target SCAIL path
            # picks HER track by SAM3 text prompts; the generic
            # defaults ("woman"/"person") were part of the old
            # fragility in a two-person frame. Specific per-job
            # terms now reach the builder via sk, and composite
            # joins sk so engine=scail can run the canonical
            # no-composite finish (the GAP call reads it from p
            # directly; the SCAIL calls take **sk).
            if p.get("hero_prompt"):
                sk["hero_prompt"] = str(p["hero_prompt"])
            if p.get("ref_prompt"):
                sk["ref_prompt"] = str(p["ref_prompt"])
            sk["composite"] = (p.get("composite", True)
                                not in (False, 0, "false"))
            # Which trained LoRA the selector loads: "v1"/"v2" (or a
            # known filename); unknown/missing falls back to v1.
            sk["person_lora_file"] = gw.person_lora_filename(
                p.get("person_lora_version"))
            # The prompt must describe the post-replacement scene
            # with BOTH people and their interaction (official
            # SCAIL-2 prompt semantics); the shared default only
            # describes her.
            scail_prompt = (p.get("prompt") or "").strip() or (
                "the woman from the reference image together with "
                "the man from the source video, in close physical "
                "contact, both fully visible throughout, "
                "photorealistic, natural skin texture, consistent "
                "lighting")
            total_frames = int(float(info.get("duration") or 0.0)
                               * 16.0)
            if use_gap:
                # GAP orchestrator: chunks internally over the
                # full video in ONE workflow (no server chunk
                # plan); the run timeout scales with length.
                # Ladder-2 levers (Oct 18): the face close-up extra
                # view (a crop of the SAME reference the Face Lock
                # uses) and the fp16 DiT the person LoRAs were
                # trained against -- both opt-in per job.
                gap_closeup = ""
                if (p.get("closeup")
                        and p.get("closeup")
                        not in (False, 0, "false")):
                    try:
                        import cv2 as _cv2
                        import tempfile
                        crop = _ref_face_crop(gpu_face_ref)
                        fd, cpath = tempfile.mkstemp(suffix=".png")
                        os.close(fd)
                        try:
                            _cv2.imwrite(cpath, crop)
                            gap_closeup = gc.upload_image(cpath)
                        finally:
                            os.unlink(cpath)
                    except Exception:
                        gap_closeup = ""
                gap_dit = ("wan2.1_14B_SCAIL_2_fp16.safetensors"
                           if p.get("dit_fp16")
                           not in (None, False, 0, "false")
                           else None)
                wf = gw.video_replace_model_scail_gap(
                    vup, iup, scail_prompt,
                    mask_prompt=(p.get("mask_prompt") or "person"),
                    mask_threshold=_f("mask_threshold", 0.4),
                    width=int(sk["width"]), height=int(sk["height"]),
                    frames=total_frames,
                    seed=int(sk.get("seed", 0)),
                    steps=int(sk.get("steps", 24)),
                    cfg=float(sk.get("cfg", 5.0)),
                    shift=float(sk.get("shift", 3.0)),
                    lightx2v_strength=float(
                        sk.get("lightx2v_strength", 0.0)),
                    composite=(p.get("composite", True)
                               not in (False, 0, "false")),
                    composite_expand=int(sk.get("composite_expand", 16)),
                    composite_feather=int(sk.get("composite_feather", 8)),
                    gap_swap=bool(p.get("gap_swap")),
                    zoned=p.get("zoned", True)
                        not in (False, 0, "false"),
                    closeup_image=gap_closeup,
                    dit_file=gap_dit,
                    person_lora_strength=scail_person,
                    person_lora_file=sk["person_lora_file"],
                    cache_id=f"studio-{job_id}",
                    seedvr2_resolution=(1024 if scail_seedvr2 else 0))
                gap_timeout = max(3600.0, total_frames * 4.0)
            elif total_frames > 81:
                # Long video: 81-frame chunks chained by 5-frame
                # anchors (see _run_scail_chunks); the single-pass
                # workflow is built per chunk there instead.
                scail_chunks = dict(
                    ref_up=iup, prompt=scail_prompt,
                    mask_prompt=(p.get("mask_prompt") or "person"),
                    mask_threshold=_f("mask_threshold", 0.4),
                    sk=sk, info=info)
                wf = None
            else:
                wf = gw.video_replace_model_scail(
                    vup, iup, scail_prompt,
                    mask_prompt=(p.get("mask_prompt") or "person"),
                    mask_threshold=_f("mask_threshold", 0.4), **sk,
                    **({"seedvr2_resolution": 1024}
                       if scail_seedvr2 else {}))
        else:
            iup = gc.upload_image(rpath)
            wf = gw.video_replace_model(
            vup, iup, prompt,
            reference_strength=_f("reference_strength", 1.0),
            mask_prompt=(p.get("mask_prompt") or "woman"),
            mask_expand=_i("mask_expand", 16),
            mask_threshold=_f("mask_threshold", 0.35),
            contact_reach=_i("contact_reach", 16),
            contact_pad=_i("contact_pad", 8),
            partner_threshold=_f("partner_threshold", 0.55),
            protect_contact=p.get("protect_contact", True)
                not in (False, 0, "false"), **knobs)
        suffix, timeout_s = "gpu_replace", 3600.0
        if use_gap:
            timeout_s = gap_timeout
    else:
        raise RuntimeError(f"unknown gpu generate option {gen_id!r}")

    JOBS.update(job_id, status="running", progress=25,
                phase="queued on your GPU pod")
    if scail_chunks is not None:
        saved = [_run_scail_chunks(
            gc, gw, job_id, jobdir, vpath, scail_chunks["ref_up"],
            scail_chunks["prompt"], scail_chunks["mask_prompt"],
            scail_chunks["mask_threshold"], scail_chunks["sk"],
            scail_chunks["info"], progress_cb=_progress)]
        if scail_seedvr2:
            # Long-video SeedVR2: post-pass over the stitched video
            # (see the note in the SCAIL branch). Falls back to the
            # un-finished stitch on any pod failure.
            try:
                JOBS.update(job_id, status="running", progress=88,
                            phase="SeedVR2 detail finish on your "
                                  "GPU pod")
                saved = [_run_seedvr2_finish(gc, gw, jobdir, saved[0])]
            except Exception:
                pass
    else:
        saved = gc.run_workflow(wf, jobdir, timeout_s=timeout_s,
                                poll_interval_s=10.0, progress_cb=_progress)
    if not saved:
        raise RuntimeError("GPU pod finished but produced no output files")
    gpu_lock_failed = False
    gpu_lock_error = None
    if gen_id == "gpu_replace" and gpu_src_video:
        # VACE renders silent video: put the original audio back,
        # then Face-Lock the reference's real face over the render.
        # Each step falls back to the previous file on any failure.
        cur = saved[0]
        try:
            aud = os.path.join(jobdir, "_with_audio.mp4")
            _mux_audio(cur, gpu_src_video, aud)
            if os.path.exists(aud):
                cur = aud
        except Exception:
            pass
        if gpu_face_ref and p.get("face_lock", True) not in (False, 0, "false"):
            try:
                locked = os.path.join(jobdir, "_facelocked.mp4")
                _face_lock(cur, gpu_face_ref, locked, jobdir)
                if os.path.exists(locked):
                    cur = locked
            except Exception as e:
                gpu_lock_failed = True
                gpu_lock_error = _lock_failure_note(e)
        saved[0] = cur
    JOBS.update(job_id, status="running", progress=95,
                phase="saving result")
    ext = os.path.splitext(saved[0])[1] or ".bin"
    result_name = f"{base}_{suffix}{ext}"
    dest = os.path.join(jobdir, result_name)
    if os.path.abspath(saved[0]) != os.path.abspath(dest):
        os.replace(saved[0], dest)
    done_extra = {"phase": "done"}
    if gpu_lock_failed:
        done_extra = {"phase": f"done — {FACE_LOCK_FAILED_NOTE}",
                      "warning": gpu_lock_error or FACE_LOCK_FAILED_NOTE}
    JOBS.update(job_id, status="done", progress=100,
                result_url=f"/api/result/{job_id}/{result_name}",
                result_name=result_name, **done_extra)


def _run_job(job_id: str) -> None:
    job = JOBS.get(job_id)
    if not job:
        return
    JOBS.update(job_id, status="running", progress=5, phase="starting")
    try:
        kind, name, params = job["kind"], job["name"], job["params"]
        if kind == "gpu_generate":
            # Self-hosted GPU path resolves its own media (t2v has none).
            _run_gpu_job(job_id, job)
            return
        mpath = safe_media_path(job["media"])
        if mpath is None:
            raise RuntimeError(f"media {job['media']!r} disappeared")
        jobdir = os.path.join(RESULTS_DIR, job_id)
        os.makedirs(jobdir, exist_ok=True)
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
        elif kind == "video_edit":
            if ve is None:
                raise RuntimeError("video editing not available")
            spec = ve.EDIT_OPS[name]
            inputs = [mpath]
            if spec.get("multi"):
                # merge: media2 / media3 params name additional videos
                for key in ("media2", "media3"):
                    extra = (params or {}).get(key)
                    if extra:
                        epath = safe_media_path(extra)
                        if epath is None:
                            raise RuntimeError(
                                f"merge input {extra!r} not found")
                        inputs.append(epath)
                if len(inputs) < 2:
                    raise RuntimeError("merge needs a second video "
                                       "(pick one in the Edit tab)")
            out = os.path.join(jobdir, f"{base}_{name}.{spec['ext']}")
            ve.apply_edit(name, inputs, out, params or None)
            if name == "extract_frame":
                # Land the still in the Gallery too, for AI reuse.
                gname = us.safe_name(f"{base}_frame.jpg")
                try:
                    import shutil as _sh
                    _sh.copy2(out, os.path.join(UPLOAD_DIR, gname))
                except OSError:
                    pass
        elif kind == "paid_generate":
            # Paid fal.ai generation. David approved fal.ai + funding.
            # mode "i2v": single-image animate. mode "r2v": Seedance 2.0
            # reference-to-video with 2+ image refs and/or video refs.
            try:
                from i2v_client import animate_image, reference_to_video
            except ImportError:
                import sys
                sys.path.insert(0, os.path.dirname(__file__))
                from i2v_client import animate_image, reference_to_video
            mode = job["params"].get("mode", "i2v")
            if mode == "r2v":
                ref_names = job["params"].get("reference_media") or []
                ref_paths = [p for p in
                             (safe_media_path(n) for n in ref_names) if p]
                image_paths = [p for p in ref_paths
                               if media_type(os.path.basename(p)) == "image"]
                video_paths = [p for p in ref_paths
                               if media_type(os.path.basename(p)) == "video"]
                manifest = reference_to_video(
                    image_paths=image_paths or None,
                    video_paths=video_paths or None,
                    prompt=job["params"].get("prompt", ""),
                    duration_s=float(job["params"].get("duration_s", 5.0)),
                    allow_third_party=True,  # David approved fal.ai
                    tos_ack=True,            # David approved the terms
                    jobs_root=RESULTS_DIR,
                )
                result_suffix = "r2v"
            else:
                manifest = animate_image(
                    mpath,
                    job["params"].get("prompt", ""),
                    model=job["params"].get("model", "seedance-mini"),
                    duration_s=float(job["params"].get("duration_s", 5.0)),
                    allow_third_party=True,  # David approved fal.ai
                    tos_ack=True,            # David approved the terms
                    jobs_root=RESULTS_DIR,
                )
                result_suffix = "i2v"
            out = manifest["output_path"]
            # Copy to jobdir with a clean name.
            import shutil
            result_name = f"{base}_{result_suffix}.mp4"
            dest = os.path.join(jobdir, result_name)
            shutil.copy2(out, dest)
            out = dest
            # True-to-reference face lock: the generative model redraws
            # the person and drifts; composite the source photo's real
            # face back onto the result. I2V only (source media is the
            # reference photo). Failure keeps the unlocked result.
            lock_failed = False
            lock_error = None
            if (job["params"].get("face_lock") and mode == "i2v"
                    and media_type(job["media"]) == "image"):
                try:
                    JOBS.update(job_id, status="running", progress=95,
                                phase="locking face to reference")
                    locked_name = f"{base}_{result_suffix}_facelock.mp4"
                    _face_lock(dest, mpath,
                               os.path.join(jobdir, locked_name), jobdir)
                    result_name = locked_name
                except Exception as e:
                    lock_failed = True
                    lock_error = _lock_failure_note(e)
            result_name = os.path.basename(
                os.path.join(jobdir, result_name))
            done_extra = {"phase": "done"}
            if lock_failed:
                done_extra = {"phase": f"done — {FACE_LOCK_FAILED_NOTE}",
                              "warning": lock_error or FACE_LOCK_FAILED_NOTE}
            JOBS.update(job_id, status="done", progress=100,
                        result_url=f"/api/result/{job_id}/{result_name}",
                        result_name=result_name, **done_extra)
            return
        elif kind == "v2v_generate":
            # Single AI restyle via Venice V2V/R2V. Prompt passes through
            # exactly as written — no filtering, ever.
            if vc is None:
                raise RuntimeError("venice_client not available")
            model_id = job["name"]
            p = job["params"]
            JOBS.update(job_id, status="running", progress=10,
                        phase="submitting to Venice")
            vjob = vc.submit_v2v(
                model_id,
                prompt=p.get("prompt", ""),
                video_path=mpath,
                duration_s=float(p.get("duration_s", 5.0)),
                resolution=p.get("resolution", "720p"),
            )
            JOBS.update(job_id, status="running", progress=30,
                        phase="Venice is rendering")
            url = vc.wait(vjob, timeout_s=1200.0, poll_interval_s=15.0)
            JOBS.update(job_id, status="running", progress=90,
                        phase="downloading result")
            result_name = f"{base}_restyled.mp4"
            vc.download_to_file(url, os.path.join(jobdir, result_name))
            JOBS.update(job_id, status="done", progress=100,
                        phase="done",
                        result_url=f"/api/result/{job_id}/{result_name}",
                        result_name=result_name)
            return
        elif kind == "face_replace":
            # Local face replacement: base video + reference photo.
            p = job["params"]
            ref_path = safe_media_path(p.get("ref_media", ""))
            if ref_path is None:
                raise RuntimeError("reference photo disappeared")
            ref_path = _ensure_photo_ref(ref_path, jobdir)
            JOBS.update(job_id, status="running", progress=30,
                        phase="replacing face (local)")
            result_name = f"{base}_facereplace.mp4"
            _face_lock(mpath, ref_path,
                       os.path.join(jobdir, result_name), jobdir)
            JOBS.update(job_id, status="done", progress=100,
                        phase="done",
                        result_url=f"/api/result/{job_id}/{result_name}",
                        result_name=result_name)
            return
        elif kind == "model_replace":
            # AI model replacement via Venice Seedance R2V: base video
            # is the performance; the reference defines the model.
            # Seedance takes at most 15s of reference video per call,
            # so longer bases are split into even <=14s parts, each
            # edited separately, then stitched back together with the
            # original audio restored.
            if vc is None:
                raise RuntimeError("venice_client not available")
            p = job["params"]
            ref_path = safe_media_path(p.get("ref_media", ""))
            if ref_path is None:
                raise RuntimeError("reference media disappeared")
            ref_type = media_type(p.get("ref_media", ""))
            prompt = p.get("prompt", "")
            quality = p.get("quality") or vc.DEFAULT_QUALITY
            q = vc.QUALITY_LEVELS[quality]
            base_info = _probe_video_info(mpath)
            base_dur = base_info.get("duration") or 0.0

            def _submit_edit(video_path, seconds):
                args = {
                    "prompt": prompt,
                    "duration": seconds,
                    "resolution": q["resolution"],
                    # Venice requires aspect_ratio on R2V submits;
                    # "adaptive" keeps the base video's own framing.
                    "aspect_ratio": "adaptive",
                    "reference_video_urls":
                        [vc.file_to_data_url(video_path)],
                }
                if ref_type == "image":
                    args["reference_image_urls"] = \
                        [vc.file_to_data_url(ref_path)]
                elif ref_type == "video":
                    # A video reference would eat the 15s combined
                    # budget; a still frame of the model anchors
                    # identity just as well and always fits.
                    frame = _ensure_photo_ref(ref_path, jobdir)
                    args["reference_image_urls"] = \
                        [vc.file_to_data_url(frame)]
                try:
                    return vc.submit("venice-seedance-r2v", args)
                except Exception as e:
                    info = _probe_video_info(video_path)
                    raise RuntimeError(
                        f"{e} [clip: {info.get('duration', 0):.1f}s, "
                        f"{(info.get('size') or 0) / 1e6:.1f} MB, "
                        f"{info.get('format', '?')}/"
                        f"{info.get('vcodec', '?')}]") from e

            result_name = f"{base}_modelreplace.mp4"
            if base_dur and base_dur > 15.0:
                chunks = _split_video_chunks(mpath, jobdir, 14.0,
                                             even=True)
                rendered = []
                for ci, chunk in enumerate(chunks):
                    JOBS.update(
                        job_id, status="running",
                        progress=int(5 + (ci / len(chunks)) * 80),
                        phase=f"AI-editing part {ci + 1} of {len(chunks)}")
                    cinfo = _probe_video_info(chunk)
                    vjob = _submit_edit(chunk,
                                        cinfo.get("duration") or 10.0)
                    url = vc.wait(vjob, timeout_s=1200.0,
                                  poll_interval_s=15.0)
                    out_chunk = os.path.join(jobdir,
                                             f"ai_out_{ci:03d}.mp4")
                    vc.download_to_file(url, out_chunk)
                    rendered.append(out_chunk)
                JOBS.update(job_id, status="running", progress=90,
                            phase="stitching parts + restoring audio")
                stitched = os.path.join(jobdir, f"{base}_stitched.mp4")
                _concat_normalized(rendered, stitched, mpath)
                _mux_audio(stitched, mpath,
                           os.path.join(jobdir, result_name))
                import glob as _glob
                for pat in ("chunk_*.mp4", "ai_out_*.mp4", "norm_*.mp4",
                            "concat_list.txt", f"{base}_stitched.mp4",
                            "ref_frame.jpg"):
                    for tmp in _glob.glob(os.path.join(jobdir, pat)):
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
            else:
                JOBS.update(job_id, status="running", progress=5,
                            phase="preparing video for Venice")
                send_base = _venice_ready_video(mpath, jobdir, "base")
                JOBS.update(job_id, status="running", progress=10,
                            phase="submitting to Venice")
                vjob = _submit_edit(send_base,
                                    float(p.get("duration_s", 5.0)))
                JOBS.update(job_id, status="running", progress=30,
                            phase="Venice is rendering")
                url = vc.wait(vjob, timeout_s=1200.0,
                              poll_interval_s=15.0)
                JOBS.update(job_id, status="running", progress=90,
                            phase="downloading result")
                vc.download_to_file(url,
                                    os.path.join(jobdir, result_name))
            JOBS.update(job_id, status="done", progress=100,
                        phase="done",
                        result_url=f"/api/result/{job_id}/{result_name}",
                        result_name=result_name)
            return
        elif kind == "animate_replace":
            # Full-body model replacement via Wan 2.2 Animate on
            # WaveSpeed: the reference PHOTO's person replaces the
            # performer in the base video, keeping the source motion,
            # scene and camera. Output has no audio track — the base
            # video's audio is muxed back (timing follows the source).
            # With face_lock the reference photo's REAL face is then
            # composited over the generative result (combo mode): the
            # body comes from the model, the face is pixel-true.
            if ac is None:
                raise RuntimeError("animate_client not available")
            p = job["params"]
            ref_path = safe_media_path(p.get("ref_media", ""))
            if ref_path is None:
                raise RuntimeError("reference photo disappeared")
            # A video reference works too: a still frame of the model
            # is pulled from it and used as the Animate reference.
            ref_path = _ensure_photo_ref(ref_path, jobdir)
            resolution = p.get("resolution", "720p")
            # Videos longer than Wan Animate's 120s per-job cap are
            # split into chunks, each replaced separately, then
            # stitched back into one full-length result.
            chunks = _split_video_chunks(mpath, jobdir, ANIMATE_CHUNK_S)
            rendered = []
            for ci, chunk in enumerate(chunks):
                frac = ci / max(1, len(chunks))
                JOBS.update(
                    job_id, status="running",
                    progress=int(5 + frac * 75),
                    phase=(f"replacing part {ci + 1} of {len(chunks)}"
                           if len(chunks) > 1 else
                           "Wan Animate is rendering"))
                ajob = ac.submit_replace(ref_path, chunk,
                                         resolution=resolution)
                url = ac.wait(ajob, timeout_s=1800.0,
                              poll_interval_s=10.0)
                out_chunk = os.path.join(jobdir,
                                         f"anim_out_{ci:03d}.mp4")
                ac.download_to_file(url, out_chunk)
                rendered.append(out_chunk)
            if len(rendered) == 1:
                gen = rendered[0]
            else:
                JOBS.update(job_id, status="running", progress=85,
                            phase="stitching parts together")
                gen = os.path.join(jobdir, f"{base}_animate_raw.mp4")
                _concat_normalized(rendered, gen, mpath)
            body_video = gen
            lock_failed = False
            lock_error = None
            if p.get("face_lock"):
                try:
                    JOBS.update(job_id, status="running", progress=93,
                                phase="locking face to reference")
                    locked = os.path.join(jobdir,
                                          f"{base}_animate_locked.mp4")
                    _face_lock(gen, ref_path, locked, jobdir)
                    body_video = locked
                except Exception as e:
                    body_video = gen  # keep the generative result
                    lock_failed = True
                    lock_error = _lock_failure_note(e)
            result_name = f"{base}_fullreplace.mp4"
            _mux_audio(body_video, mpath,
                       os.path.join(jobdir, result_name))
            # Temps are safe to drop: the result is a separate file.
            import glob as _glob
            for pat in ("chunk_*.mp4", "anim_out_*.mp4", "norm_*.mp4",
                        "concat_list.txt", f"{base}_animate_raw.mp4",
                        f"{base}_animate_locked.mp4"):
                for tmp in _glob.glob(os.path.join(jobdir, pat)):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            done_extra = {"phase": "done"}
            if lock_failed:
                done_extra = {"phase": f"done — {FACE_LOCK_FAILED_NOTE}",
                              "warning": lock_error or FACE_LOCK_FAILED_NOTE}
            JOBS.update(job_id, status="done", progress=100,
                        result_url=f"/api/result/{job_id}/{result_name}",
                        result_name=result_name, **done_extra)
            return
        elif kind == "longvideo":
            # Chained minutes-long video. See longvideo.py for the
            # continuity design (last-frame conditioning + beat prompts).
            if vc is None or lv is None:
                raise RuntimeError("venice/longvideo modules not available")
            p = job["params"]
            beats = p["beats"]
            n = len(beats)
            q = vc.QUALITY_LEVELS[p.get("quality", vc.DEFAULT_QUALITY)]
            chain_mode = p.get("chain_mode", "lastframe")
            chain_model = p.get("chain_model", "venice-wan27-uncensored")
            v2v_model = "venice-wan27-v2v"
            extend_model = "venice-seedance-r2v"

            def _progress(i: int, total: int, phase: str):
                # Check for user cancellation between clips
                if (JOBS.get(job_id) or {}).get("status") == "cancelled":
                    raise InterruptedError("Job cancelled by user")
                pct = 5 + int(90 * (i + 1) / max(total, 1))
                JOBS.update(job_id, status="running",
                            progress=min(pct, 98),
                            phase=f"Clip {i + 1} of {total}: {phase}"
                            if i < total else phase)

            def _wait_clip(vjob):
                return vc.wait(vjob, timeout_s=1200.0, poll_interval_s=15.0)

            def _download(url: str, dest: str):
                return vc.download_to_file(url, dest)

            from video_edit import extract_last_frame as _elf
            ref_image = p.get("ref_image")
            ref_video = p.get("ref_video")
            # Pre-convert reference image to data URL for R2V consistency.
            # Seedance R2V uses reference_image_urls (up to 30) to lock
            # character features across all clips in the chain.
            ref_image_url = None
            if ref_image:
                ref_path = safe_media_path(ref_image)
                if ref_path:
                    ref_image_url = vc.file_to_data_url(ref_path)
            # Make ref_image_url available to _submit_clip via closure.
            _r2v_ref_url = ref_image_url

            def _submit_clip(a: dict):
                mid = a.pop("model_id", chain_model)
                task_type = a.pop("task_type", None)
                if task_type or mid == extend_model:
                    return vc.submit_v2v(
                        mid, prompt=a["prompt"],
                        reference_video_paths=None,
                        video_data_url=(a.get("reference_video_urls") or
                                        [None])[0],
                        task_type=task_type or "extend",
                        duration_s=float(a.get("duration",
                                              p["clip_duration_s"])),
                        resolution=q["resolution"],
                        aspect_ratio=a.get("aspect_ratio"))
                if a.get("video_url"):
                    return vc.submit_v2v(
                        v2v_model, prompt=a["prompt"],
                        video_data_url=a["video_url"],
                        duration_s=float(a.get("duration",
                                              p["clip_duration_s"])),
                        resolution=q["resolution"])
                # I2V / R2V leg (first clip, or lastframe chaining).
                submit_args = {
                    "prompt": a["prompt"],
                    "duration": a.get("duration", p["clip_duration_s"]),
                    "resolution": q["resolution"],
                    "image_url": a.get("image_url"),
                }
                if a.get("negative_prompt"):
                    submit_args["negative_prompt"] = a["negative_prompt"]
                # Seedance R2V: pass original reference images every clip
                # for character consistency, in addition to the last frame.
                if mid == "venice-seedance-r2v" and _r2v_ref_url:
                    submit_args["reference_image_urls"] = [_r2v_ref_url]
                return vc.submit(mid, submit_args)

            # Per-clip face lock: each accepted clip is locked BEFORE
            # its last frame seeds the next clip, so generator drift
            # cannot compound down the chain. The stitched video still
            # gets the final lock pass below.
            _clip_lock_fn = None
            if p.get("face_lock") and ref_image:
                _lock_ref = safe_media_path(ref_image)
                if _lock_ref:
                    def _clip_lock(path, _ref=_lock_ref):
                        _tmp = path + ".locking.mp4"
                        _face_lock(path, _ref, _tmp, jobdir)
                        os.replace(_tmp, path)
                    _clip_lock_fn = _clip_lock
            final = lv.run_chain(
                beats=beats,
                clip_duration_s=float(p["clip_duration_s"]),
                chain_mode=chain_mode,
                submit_clip=_submit_clip,
                wait_clip=_wait_clip,
                download_clip=_download,
                extract_frame_fn=_elf,
                file_to_url_fn=vc.file_to_data_url,
                workdir=os.path.join(jobdir, "chain"),
                ref_image_path=(safe_media_path(ref_image)
                                if ref_image else None),
                ref_video_path=(safe_media_path(ref_video)
                                if ref_video else None),
                i2v_model_id=chain_model,
                v2v_model_id=v2v_model,
                extend_model_id=extend_model,
                resolution=q["resolution"],
                overlap_s=float(p.get("overlap_s", 0.0)),
                crf=int(q["crf"]),
                preset=q["preset"],
                progress_cb=_progress,
                clip_lock_fn=_clip_lock_fn,
            )
            result_name = f"{base}_longvideo.mp4"
            import shutil as _sh2
            _sh2.copy2(final, os.path.join(jobdir, result_name))
            # True-to-reference face lock on the stitched video: put the
            # reference photo's real face back over the AI's drifted one.
            lock_failed = False
            lock_error = None
            if p.get("face_lock") and ref_image:
                _ref_p = safe_media_path(ref_image)
                if _ref_p:
                    try:
                        JOBS.update(job_id, status="running", progress=97,
                                    phase="locking face to reference")
                        _locked_name = f"{base}_longvideo_facelock.mp4"
                        _face_lock(os.path.join(jobdir, result_name),
                                   _ref_p,
                                   os.path.join(jobdir, _locked_name),
                                   jobdir)
                        result_name = _locked_name
                    except Exception as e:
                        lock_failed = True
                        lock_error = _lock_failure_note(e)
            done_extra = {"phase": "done"}
            if lock_failed:
                done_extra = {"phase": f"done — {FACE_LOCK_FAILED_NOTE}",
                              "warning": lock_error or FACE_LOCK_FAILED_NOTE}
            JOBS.update(job_id, status="done", progress=100,
                        result_url=f"/api/result/{job_id}/{result_name}",
                        result_name=result_name, **done_extra)
            return
        else:
            raise RuntimeError(f"unknown kind {kind!r}")

        result_name = os.path.basename(out)
        JOBS.update(job_id, status="done", progress=100,
                    result_url=f"/api/result/{job_id}/{result_name}",
                    result_name=result_name)
    except InterruptedError:
        # User cancelled — status already set by _api_job_cancel
        pass
    except Exception as e:  # noqa: BLE001 - job errors are reported, not raised
        # Don't overwrite a user cancellation with an error
        if (JOBS.get(job_id) or {}).get("status") != "cancelled":
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
        elif path == "/manifest.json":
            self._serve_static_file("manifest.json", "application/manifest+json")
        elif path == "/icon-192.png":
            self._serve_static_file("icon-192.png", "image/png")
        elif path == "/icon-512.png":
            self._serve_static_file("icon-512.png", "image/png")
        elif path == "/apple-touch-icon.png":
            self._serve_static_file("apple-touch-icon.png", "image/png")
        elif path == "/api/media":
            self._api_media()
        elif path == "/api/effects":
            self._send_json(effects_catalog())
        elif path == "/api/generate_options":
            self._send_json(generate_options())
        elif path == "/api/fal_status":
            self._send_json(fal_status())
        elif path == "/api/v2v_models":
            self._api_v2v_models()
        elif path.startswith("/api/thumb/"):
            self._api_thumb(unquote(path[len("/api/thumb/"):]))
        elif path.startswith("/api/media/file/"):
            self._api_media_file(unquote(path[len("/api/media/file/"):]))
        elif path.startswith("/api/job/") and path.endswith("/clips"):
            # GET /api/job/<job_id>/clips — must precede the generic
            # /api/job/ route below or it is shadowed by it.
            job_id = path[len("/api/job/"):-len("/clips")]
            self._api_job_clips(unquote(job_id))
        elif path.startswith("/api/job/"):
            self._api_job(path[len("/api/job/"):])
        elif path == "/api/jobs":
            self._api_jobs()
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
        elif parsed.path == "/api/v2v":
            self._api_v2v()
        elif parsed.path == "/api/model_replace":
            self._api_model_replace()
        elif parsed.path == "/api/longvideo":
            self._api_longvideo()
        elif parsed.path == "/api/segment_prompt":
            self._api_segment_prompt()
        elif parsed.path == "/api/media/delete":
            self._api_media_delete()
        elif parsed.path == "/api/job/cancel":
            self._api_job_cancel()
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

    # PWA static files: fixed allowlist only, no user-controlled paths.
    _PWA_FILES = {
        "manifest.json": "application/manifest+json",
        "icon-192.png": "image/png",
        "icon-512.png": "image/png",
        "apple-touch-icon.png": "image/png",
    }

    def _serve_static_file(self, name: str, ctype: str):
        if name not in self._PWA_FILES or self._PWA_FILES[name] != ctype:
            self.send_error(404)
            return
        full = os.path.join(HERE, name)
        try:
            with open(full, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404)
            return
        self._send_bytes(body, ctype)

    def _api_media(self):
        items = []
        roles = _media_roles()
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
            try:
                st = os.stat(full)
                size, mtime = st.st_size, st.st_mtime
            except OSError:
                size, mtime = os.path.getsize(full), 0
            item = {
                "name": n,
                "type": mtype,
                "size": size,
                "mtime": mtime,
                "thumb": f"/api/thumb/{n}",
                "url": f"/api/media/file/{n}",
            }
            role = roles.get(n)
            if isinstance(role, dict):
                item["role"] = role.get("role", "")
                item["role_label"] = role.get("label", "")
            items.append(item)
        self._send_json({"media": items})

    def _api_media_file(self, name: str):
        """GET /api/media/file/<name> — serve an uploaded file itself
        (the Library viewer plays/shows the real file, not a thumb)."""
        path = safe_media_path(name)
        if path is None:
            self.send_error(404)
            return
        ctype, _ = mimetypes.guess_type(path)
        with open(path, "rb") as f:
            self._send_bytes(f.read(), ctype or "application/octet-stream",
                             filename=os.path.basename(path))

    def _api_media_delete(self):
        """POST /api/media/delete {names: [...]} -> {deleted: [...]}."""
        body = self._read_json()
        if not body:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        names = body.get("names")
        if not isinstance(names, list) or not names:
            self._send_json({"error": "names must be a non-empty list"}, 400)
            return
        deleted = []
        for name in names:
            if not isinstance(name, str):
                continue
            path = safe_media_path(name)
            if path is None:
                continue
            try:
                os.remove(path)
                # Also remove thumbnail cache if present
                deleted.append(os.path.basename(path))
            except OSError:
                continue
        self._send_json({"deleted": deleted})

    def _api_job_cancel(self):
        """POST /api/job/cancel {job_id} -> {status}. Marks job cancelled."""
        body = self._read_json()
        if not body:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        job_id = body.get("job_id", "")
        if not re.fullmatch(r"job-\d{4}", job_id or ""):
            self._send_json({"error": "invalid job_id"}, 400)
            return
        job = JOBS.get(job_id)
        if job is None:
            self._send_json({"error": "unknown job"}, 404)
            return
        if job["status"] in ("done", "failed", "error", "cancelled"):
            self._send_json({"job_id": job_id, "status": job["status"]})
            return
        JOBS.update(job_id, status="cancelled", phase="cancelled by user",
                     error="Cancelled by user")
        self._send_json({"job_id": job_id, "status": "cancelled"})

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
        out = dict(job)
        out["deliverables"] = job_deliverables(job_id,
                                               job.get("result_name"))
        out.update(job_primary(job_id, job.get("result_name"),
                               job.get("result_url")))
        out["has_clips"] = job_has_clips(job_id)
        self._send_json(out)

    def _api_jobs(self):
        """GET /api/jobs -> {jobs: [...]}. Recent jobs, newest first."""
        with JOBS._lock:
            jobs = sorted(JOBS._jobs.values(),
                          key=lambda j: j.get("created", 0), reverse=True)
            # Strip params to keep it light; keep status info. Primary
            # fields point at the file the job should present as its
            # result (a later post-step final outranks the raw render).
            out = []
            for j in jobs[:20]:
                entry = {
                    "job_id": j["job_id"], "kind": j["kind"],
                    "name": j["name"], "status": j["status"],
                    "progress": j["progress"], "phase": j.get("phase", ""),
                    "result_url": j["result_url"],
                    "result_name": j["result_name"], "error": j["error"],
                    "created": j["created"],
                    "has_clips": job_has_clips(j["job_id"]),
                }
                entry.update(job_primary(j["job_id"], j["result_name"],
                                         j["result_url"]))
                out.append(entry)
        self._send_json({"jobs": out})

    def _api_result(self, job_id: str, filename: str):
        if not re.fullmatch(r"job-\d{4}", job_id or ""):
            self.send_error(404)
            return
        # Allow chain/ subdirectory for clip previews; sanitize each part.
        parts = [us.safe_name(p) for p in filename.split("/")]
        # safe_name strips separators; reject if any part was altered to
        # prevent traversal (e.g. ".." becomes "").
        if any(not p or p in (".", "..") for p in parts):
            self.send_error(404)
            return
        clean = "/".join(parts)
        # Only allow top-level files or chain/ subdirectory.
        if "/" in clean and not clean.startswith("chain/"):
            self.send_error(404)
            return
        jobdir = os.path.realpath(os.path.join(RESULTS_DIR, job_id))
        if not jobdir.startswith(os.path.realpath(RESULTS_DIR) + os.sep):
            self.send_error(404)
            return
        fpath = os.path.realpath(os.path.join(jobdir, clean))
        if not fpath.startswith(jobdir + os.sep):
            self.send_error(404)
            return
        if not os.path.isfile(fpath):
            self.send_error(404)
            return
        ctype, _ = mimetypes.guess_type(fpath)
        with open(fpath, "rb") as f:
            self._send_bytes(f.read(), ctype or "application/octet-stream",
                             filename=clean)

    def _api_job_clips(self, job_id: str):
        """List completed chain clips for preview during a long-video job."""
        if not re.fullmatch(r"job-\d{4}", job_id or ""):
            self.send_error(404)
            return
        jobdir = os.path.realpath(os.path.join(RESULTS_DIR, job_id))
        if not jobdir.startswith(os.path.realpath(RESULTS_DIR) + os.sep):
            self.send_error(404)
            return
        chain_dir = os.path.join(jobdir, "chain")
        clips = []
        if os.path.isdir(chain_dir):
            for name in sorted(os.listdir(chain_dir)):
                if re.fullmatch(r"clip_\d{2}\.mp4", name):
                    fpath = os.path.join(chain_dir, name)
                    if os.path.getsize(fpath) > 0:
                        clips.append({
                            "name": name,
                            "url": f"/api/result/{job_id}/chain/{name}",
                            "size": os.path.getsize(fpath),
                        })
        self._send_json({"job_id": job_id, "clips": clips})

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
        rejected = []
        items = form["files"] if "files" in form else []
        if not isinstance(items, list):
            items = [items]
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        for item in items:
            if not getattr(item, "filename", None):
                continue
            name = us.safe_name(item.filename)
            if media_type(name) is None:
                # Never silently swallow a file — tell the user exactly
                # which file was refused and why.
                rejected.append({
                    "name": item.filename,
                    "reason": "unsupported file type "
                              f"({os.path.splitext(item.filename)[1] or 'no extension'})"
                })
                continue
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
        if not saved and rejected:
            self._send_json(
                {"error": "No usable file in that upload — " +
                          "; ".join(f"{r['name']}: {r['reason']}"
                                    for r in rejected),
                 "rejected": rejected}, 400)
            return
        self._send_json({"saved": saved, "rejected": rejected})

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
        # Self-hosted GPU options (RunPod ComfyUI pod): no fal.ai and no
        # third-party API — handled separately, media goes to David's
        # own pod only.
        if gen_id.startswith("gpu_"):
            self._api_generate_gpu(body, opt)
            return
        if opt["needs_media"]:
            mpath = safe_media_path(media)
            if mpath is None and not (body.get("reference_images") or
                                      body.get("reference_videos")):
                self._send_json({"error": f"media {media!r} not found"}, 404)
                return
            if mpath is not None and media_type(media) != opt["needs_media"]:
                self._send_json(
                    {"error": f"this needs an {opt['needs_media']}"}, 400)
                return
        info = fal_status()
        if not info["configured"]:
            self._send_json({"status": "needs_setup",
                             "message": info["message"]})
            return
        # Only image-to-video is wired up. Text-to-video needs a backend.
        if gen_id not in ("i2v", "i2v_venice", "i2v_venice_uncensored"):
            self._send_json({
                "status": "not_available",
                "message": ("Text-to-video isn't wired up yet — "
                            "image-to-video is ready now."),
            })
            return
        # Venice routes: single image only, no multi-reference.
        if gen_id in ("i2v_venice", "i2v_venice_uncensored"):
            # Allow frontend to specify model and duration (uncensored only)
            req_model = body.get("venice_model")
            req_duration = body.get("duration_s")
            if req_model == "venice-wan27-uncensored":
                venice_model = "venice-wan27-uncensored"
            else:
                venice_model = ("venice-wan3" if gen_id == "i2v_venice"
                                else "venice-wan27-uncensored")
            # Validate duration against model max
            try:
                from i2v_client import venice_client as _vc
                _spec = _vc.VENICE_MODELS.get(venice_model, {})
                _maxd = float(_spec.get("max_duration_s", 15.0))
            except Exception:
                _maxd = 15.0
            try:
                duration_s = float(req_duration) if req_duration else 5.0
            except (TypeError, ValueError):
                duration_s = 5.0
            duration_s = max(1.0, min(duration_s, _maxd))
            # Check Venice key is configured.
            try:
                from i2v_client import venice_client
            except ImportError:
                import sys
                sys.path.insert(0, os.path.dirname(__file__))
                from i2v_client import venice_client
            try:
                venice_client.get_api_key()
            except Exception as e:
                self._send_json({"status": "needs_setup",
                                 "message": str(e)})
                return
            mpath = safe_media_path(media)
            job_id = JOBS.new(media, "paid_generate", gen_id, {
                "prompt": prompt,
                "model": venice_model,
                "duration_s": duration_s,
                "face_lock": bool(body.get("face_lock")),
            })
            POOL.submit(_run_job, job_id)
            est = venice_client.estimate_cost(venice_model, duration_s)
            self._send_json({"job_id": job_id, "status": "started",
                             "estimate": f"~${est:.2f}/clip"})
            return
        # Reference media: 2-9 images and/or up to 3 short videos for
        # Seedance 2.0 reference-to-video (better character consistency).
        ref_names = []
        for key in ("reference_images", "reference_videos"):
            val = body.get(key) or []
            if not isinstance(val, list):
                self._send_json({"error": f"{key} must be a list"}, 400)
                return
            ref_names.extend(val)
        ref_paths = []
        for n in ref_names:
            p = safe_media_path(n)
            if p is None:
                self._send_json({"error": f"media {n!r} not found"}, 404)
                return
            ref_paths.append((n, p))
        ref_images = [p for n, p in ref_paths
                      if media_type(n) == "image"]
        ref_videos = [p for n, p in ref_paths
                      if media_type(n) == "video"]
        # Any non-image/non-video ref is rejected.
        if len(ref_images) + len(ref_videos) != len(ref_paths):
            self._send_json({"error": "reference media must be images "
                                      "or videos"}, 400)
            return
        if ref_images or ref_videos:
            # Multi-reference route (Seedance 2.0 reference-to-video).
            try:
                from i2v_client import estimate_reference_cost_usd
            except ImportError:
                import sys
                sys.path.insert(0, os.path.dirname(__file__))
                from i2v_client import estimate_reference_cost_usd
            if len(ref_images) > 9 or len(ref_videos) > 3:
                self._send_json(
                    {"error": "max 9 reference images and 3 reference "
                              "videos"}, 400)
                return
            est = estimate_reference_cost_usd(5.0,
                                              has_video_refs=bool(ref_videos))
            job_media = (ref_images + ref_videos)[0]
            job_id = JOBS.new(os.path.basename(job_media), "paid_generate",
                              gen_id, {
                                  "mode": "r2v",
                                  "prompt": prompt,
                                  "reference_media": ref_names,
                                  "duration_s": 5.0,
                              })
            POOL.submit(_run_job, job_id)
            self._send_json({"job_id": job_id, "status": "started",
                             "estimate": f"~${est:.2f}/clip"})
            return
        # Launch the paid single-image I2V job in the background.
        mpath = safe_media_path(media)
        job_id = JOBS.new(media, "paid_generate", gen_id, {
            "prompt": prompt,
            "model": "seedance-mini",  # approved route
            "duration_s": 5.0,
            "face_lock": bool(body.get("face_lock")),
        })
        POOL.submit(_run_job, job_id)
        self._send_json({"job_id": job_id, "status": "started",
                         "estimate": opt["cost"]})

    def _api_generate_gpu(self, body, opt):
        """GPU generate options: queue a gpu_generate job for the pod.

        Body: {id, prompt, media, reference_images?, reference_videos?,
        params?: {seed, steps, cfg, width, height, frames, fps, denoise,
        negative_prompt, reference_strength}}. The pod being off is NOT
        checked here — the job fails soft with a clear 'pod unreachable'
        error, so the studio never blocks on pod state.
        """
        gen_id = opt["id"]
        if gc is None or gw is None:
            self._send_json({"error": "gpu modules not available on server"},
                            500)
            return
        try:
            gc.get_base_url()
        except gc.GpuNotConfigured as e:
            self._send_json({"status": "needs_setup", "message": str(e)})
            return
        prompt = (body.get("prompt") or "").strip()
        raw = body.get("params") or {}
        if not isinstance(raw, dict):
            self._send_json({"error": "params must be an object"}, 400)
            return
        knobs = {k: raw[k] for k in
                 ("seed", "steps", "cfg", "width", "height", "frames",
                  "fps", "denoise", "negative_prompt", "reference_strength",
                  "mask_prompt", "mask_expand", "face_lock", "protect_contact",
                  "mask_threshold", "contact_reach", "contact_pad",
                  "partner_threshold", "engine", "ref_threshold",
                  "composite_expand", "composite_feather", "composite",
                  "seedvr2", "gap_swap", "person_lora",
                  "person_lora_version", "zoned", "closeup",
                  "person_lora_strength", "dit_fp16", "hero_prompt",
                  "ref_prompt")
                 if k in raw}
        media = body.get("media") or ""
        params = {"prompt": prompt, **knobs}
        if gen_id == "gpu_t2v":
            params["media_name"] = ""
            job_media = media or "gpu_t2v"
        elif gen_id == "gpu_replace":
            # Base video: the selected media if it is one, else the
            # first reference video (the UI multi-picker sends both).
            candidates = [media] + list(body.get("reference_videos") or [])
            base_name = next((n for n in candidates
                              if n and media_type(n) == "video"), "")
            ref_images = list(body.get("reference_images") or [])
            ref_videos = [n for n in (body.get("reference_videos") or [])
                          if n and n != base_name]
            # Up to 4 references in any mix: a reference video
            # contributes one extracted still frame and photos join
            # it in the sheet (tiled together on the pod path).
            ref_names = (ref_videos[:1] + ref_images)[:4]
            if not base_name or safe_media_path(base_name) is None:
                self._send_json({"error": "pick a base video"}, 400)
                return
            if not ref_names or any(safe_media_path(n) is None
                                    for n in ref_names):
                self._send_json(
                    {"error": "pick reference photo(s) or a video of "
                              "the new model"}, 400)
                return
            if any(media_type(n) not in ("image", "video")
                   for n in ref_names):
                self._send_json(
                    {"error": "references must be photos or a video"}, 400)
                return
            ref_name = ref_names[0]
            params["media_name"] = base_name
            params["ref_media_name"] = ref_name
            params["ref_media_names"] = ref_names
            job_media = base_name
        else:  # gpu_image_edit / gpu_i2v: one reference image
            if safe_media_path(media) is None:
                self._send_json({"error": f"media {media!r} not found"},
                                404)
                return
            if media_type(media) != "image":
                self._send_json({"error": "this needs an image"}, 400)
                return
            params["media_name"] = media
            job_media = media
        job_id = JOBS.new(job_media, "gpu_generate", gen_id, params)
        POOL.submit(_run_job, job_id)
        est = gc.estimate_cost_usd(gc.EST_RENDER_S.get(gen_id, 600.0))
        self._send_json({"job_id": job_id, "status": "started",
                         "estimate": f"~${est:.2f} of pod time"})

    # -- video-to-video / long-video --

    def _venice_key_or_setup(self):
        """Return (ok, message). Ensures Venice key is configured."""
        if vc is None:
            return False, "venice_client module not available on server."
        try:
            vc.get_api_key()
            return True, ""
        except Exception as e:
            return False, str(e)

    def _api_v2v_models(self):
        """Model picker data: V2V/R2V models + quality levels."""
        if vc is None:
            self._send_json({"models": [], "quality_levels": {},
                             "default_quality": "standard",
                             "error": "venice_client not available"})
            return
        models = []
        for mid, s in vc.v2v_models().items():
            models.append({
                "id": mid,
                "name": s.get("label", mid),
                "uncensored": bool(s.get("uncensored")),
                "workflow": s.get("workflow"),
                "supports_extend": bool(s.get("supports_extend")),
                "cost_per_s": s.get("cost_per_s"),
                "max_duration_s": s.get("max_duration_s"),
                "default_duration_s": s.get("default_duration_s"),
                "resolutions": s.get("resolutions", ["720p"]),
                "description": s.get("description", ""),
            })
        # Uncensored first — David's standing preference.
        models.sort(key=lambda m: (not m["uncensored"], m["name"]))
        chain_i2v = []
        for mid in ("venice-wan27-uncensored", "venice-wan3",
                    "venice-seedance-r2v"):
            s = vc.VENICE_MODELS.get(mid)
            if s:
                chain_i2v.append({
                    "id": mid, "name": s.get("label", mid),
                    "uncensored": bool(s.get("uncensored")),
                    "cost_per_s": s.get("cost_per_s"),
                    "max_duration_s": s.get("max_duration_s"),
                })
        self._send_json({
            "models": models,
            "chain_i2v_models": chain_i2v,
            "default_chain_i2v": "venice-wan27-uncensored",
            "quality_levels": vc.QUALITY_LEVELS,
            "default_quality": vc.DEFAULT_QUALITY,
            "chain_modes": [
                {"id": "lastframe",
                 "name": "Last-frame chain",
                 "description": ("Exact last frame of each clip starts the "
                                 "next. Works with any I2V model; default "
                                 "uses the uncensored Wan 2.7 line.")},
                {"id": "extend",
                 "name": "Seedance Extend",
                 "description": ("Native Extend workflow — built for seamless "
                                 "continuation. Note: public Seedance may "
                                 "reject clips with detectable persons.")},
            ],
        })

    def _api_v2v(self):
        """POST /api/v2v {media, model, prompt, quality, duration_s}
        -> {job_id, estimate}. Single AI restyle of one clip."""
        body = self._read_json()
        if not body:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        ok, msg = self._venice_key_or_setup()
        if not ok:
            self._send_json({"status": "needs_setup", "message": msg})
            return
        media = body.get("media", "")
        model_id = body.get("model", "")
        prompt = (body.get("prompt") or "").strip()
        quality = body.get("quality") or vc.DEFAULT_QUALITY
        if quality not in vc.QUALITY_LEVELS:
            self._send_json({"error": f"unknown quality {quality!r}"}, 400)
            return
        if model_id not in vc.v2v_models():
            self._send_json({"error": f"unknown V2V model {model_id!r}"}, 400)
            return
        if not prompt:
            self._send_json({"error": "a prompt is required"}, 400)
            return
        mpath = safe_media_path(media)
        if mpath is None:
            self._send_json({"error": f"media {media!r} not found"}, 404)
            return
        if media_type(media) != "video":
            self._send_json({"error": "AI restyle needs a video"}, 400)
            return
        spec = vc.VENICE_MODELS[model_id]
        duration_s = float(body.get("duration_s")
                           or spec.get("default_duration_s", 5.0))
        duration_s = min(duration_s, float(spec.get("max_duration_s", 15.0)))
        q = vc.QUALITY_LEVELS[quality]
        est = round(spec["cost_per_s"] * duration_s, 2)
        job_id = JOBS.new(media, "v2v_generate", model_id, {
            "prompt": prompt, "quality": quality,
            "resolution": q["resolution"],
            "duration_s": duration_s,
        })
        POOL.submit(_run_job, job_id)
        self._send_json({"job_id": job_id, "status": "started",
                         "estimate": f"~${est:.2f}/clip ({quality})"})

    def _api_model_replace(self):
        """POST /api/model_replace {media, ref_media, mode, prompt, quality}

        Replace the model in a base video with a reference model.
          media:     base video (the performance to keep)
          ref_media: reference photo OR reference video (the new model)
          mode:      "face" = local face replacement (free, photo only)
                     "ai"   = Venice Seedance R2V replacement (paid)
        -> {job_id, estimate}
        """
        body = self._read_json()
        if not body:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        media = body.get("media", "")
        ref_media = body.get("ref_media", "")
        mode = body.get("mode", "ai")
        prompt = (body.get("prompt") or "").strip()
        if mode not in ("face", "ai", "animate", "combo"):
            self._send_json({"error": f"unknown mode {mode!r}"}, 400)
            return
        mpath = safe_media_path(media)
        if mpath is None:
            self._send_json({"error": f"media {media!r} not found"}, 404)
            return
        if media_type(media) != "video":
            self._send_json({"error": "base media must be a video"}, 400)
            return
        rpath = safe_media_path(ref_media)
        if rpath is None:
            self._send_json({"error": f"reference {ref_media!r} not found"},
                            404)
            return
        ref_type = media_type(ref_media)
        if mode == "face":
            if fv is None:
                self._send_json(
                    {"error": "face replacement not available on server"},
                    500)
                return
            if ref_type not in ("image", "video"):
                self._send_json(
                    {"error": "face replacement needs a reference photo "
                              "or video"}, 400)
                return
            job_id = JOBS.new(media, "face_replace", "local", {
                "ref_media": ref_media,
            })
            POOL.submit(_run_job, job_id)
            self._send_json({"job_id": job_id, "status": "started",
                             "estimate": "$0 (local)"})
            return
        if mode in ("animate", "combo"):
            # Wan 2.2 Animate full-body replacement (WaveSpeed).
            # "combo" adds the free local face-lock on top: the body is
            # generative, the face ends up pixel-true to the photo.
            if ac is None:
                self._send_json(
                    {"error": "animate client not available on server"},
                    500)
                return
            if ref_type not in ("image", "video"):
                self._send_json(
                    {"error": "full-body replace needs a reference photo "
                              "or video of your model"}, 400)
                return
            try:
                ac.get_api_key()
            except Exception as e:
                self._send_json({"status": "needs_setup",
                                 "message": str(e)})
                return
            resolution = body.get("resolution") or "720p"
            if resolution not in ac.COST_PER_S:
                self._send_json(
                    {"error": f"unknown resolution {resolution!r}"}, 400)
                return
            # Estimate from the base video's real duration.
            duration_s = 5.0
            try:
                import subprocess as _sp2
                pr = _sp2.run(
                    ["ffprobe", "-v", "error", "-show_entries",
                     "format=duration", "-of", "csv=p=0", mpath],
                    capture_output=True, text=True, timeout=30)
                duration_s = float(pr.stdout.strip() or 5.0)
            except Exception:
                pass
            # Any length is fine: longer videos are split into
            # <=110s chunks server-side and stitched back together.
            est = ac.estimate_cost(duration_s, resolution)
            job_id = JOBS.new(media, "animate_replace", "wan-animate", {
                "ref_media": ref_media,
                "resolution": resolution,
                "face_lock": mode == "combo",
            })
            POOL.submit(_run_job, job_id)
            self._send_json({"job_id": job_id, "status": "started",
                             "estimate": f"~${est:.2f} "
                                         f"({duration_s:.0f}s at "
                                         f"{resolution})"})
            return
        # AI mode (Venice Seedance R2V)
        ok, msg = self._venice_key_or_setup()
        if not ok:
            self._send_json({"status": "needs_setup", "message": msg})
            return
        if ref_type not in ("image", "video"):
            self._send_json(
                {"error": "reference must be a photo or video"}, 400)
            return
        quality = body.get("quality") or vc.DEFAULT_QUALITY
        if quality not in vc.QUALITY_LEVELS:
            self._send_json({"error": f"unknown quality {quality!r}"}, 400)
            return
        spec = vc.VENICE_MODELS["venice-seedance-r2v"]
        # Seedance takes at most 15s of reference video per call;
        # longer bases are split into <=14s parts server-side and
        # stitched back together, so any length is fine. A reference
        # video is reduced to a still frame of the model (it would
        # otherwise eat the 15s combined budget).
        base_info = _probe_video_info(mpath)
        base_dur = base_info.get("duration") or 0.0
        if base_dur and base_dur < 2.0:
            self._send_json(
                {"error": "AI Replace needs at least 2 seconds of video."},
                400)
            return
        duration_s = float(body.get("duration_s") or base_dur
                           or spec.get("default_duration_s", 10.0))
        duration_s = min(duration_s, float(spec.get("max_duration_s", 15.0)))
        # Estimate on the FULL video length (chunked calls are billed
        # per output second).
        est = round(spec["cost_per_s"] * (base_dur or duration_s), 2)
        job_id = JOBS.new(media, "model_replace", "venice-seedance-r2v", {
            "ref_media": ref_media,
            "prompt": prompt or ("Replace the person in this video with "
                                 "the person from the reference, keeping "
                                 "the same actions and scene."),
            "quality": quality,
            "duration_s": duration_s,
        })
        POOL.submit(_run_job, job_id)
        self._send_json({"job_id": job_id, "status": "started",
                         "estimate": f"~${est:.2f}/clip ({quality})"})

    def _api_segment_prompt(self):
        """POST /api/segment_prompt {prompt, n_beats} -> {beats}. Free."""
        body = self._read_json()
        if not body:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        if lv is None:
            self._send_json({"error": "longvideo module not available"}, 500)
            return
        prompt = (body.get("prompt") or "").strip()
        n_beats = int(body.get("n_beats") or 1)
        if not prompt:
            self._send_json({"error": "prompt is empty"}, 400)
            return
        try:
            beats = lv.segment_prompt(prompt, n_beats)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
            return
        self._send_json({"beats": beats, "n_beats": len(beats)})

    def _api_longvideo(self):
        """POST /api/longvideo {media?, prompt, beats?, target_duration_s,
        chain_model?, chain_mode, quality, overlap_s} -> {job_id, ...}.

        Minutes-long chained video. Beats pre-segmented by the UI (or
        segmented server-side). Progress via /api/job phase field.
        """
        body = self._read_json()
        if not body:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        ok, msg = self._venice_key_or_setup()
        if not ok:
            self._send_json({"status": "needs_setup", "message": msg})
            return
        if lv is None:
            self._send_json({"error": "longvideo module not available"}, 500)
            return
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            self._send_json({"error": "a prompt is required"}, 400)
            return
        quality = body.get("quality") or vc.DEFAULT_QUALITY
        if quality not in vc.QUALITY_LEVELS:
            self._send_json({"error": f"unknown quality {quality!r}"}, 400)
            return
        chain_mode = body.get("chain_mode", "lastframe")
        if chain_mode not in ("lastframe", "extend"):
            self._send_json({"error": f"unknown chain_mode {chain_mode!r}"},
                            400)
            return
        target_s = float(body.get("target_duration_s") or 60.0)
        if not 15.0 <= target_s <= 600.0:
            self._send_json({"error": "target duration must be 15-600s"},
                            400)
            return
        overlap_s = float(body.get("overlap_s") or 0.0)
        overlap_s = max(0.0, min(overlap_s, 2.0))

        # Reference media: optional image (I2V start) or video (V2V start).
        media = body.get("media") or ""
        ref_image = ref_video = None
        if media:
            mpath = safe_media_path(media)
            if mpath is None:
                self._send_json({"error": f"media {media!r} not found"}, 404)
                return
            mt = media_type(media)
            if mt == "image":
                ref_image = media
            elif mt == "video":
                ref_video = media
            else:
                self._send_json({"error": "reference must be image/video"},
                                400)
                return
        if not ref_image and not ref_video:
            self._send_json({"error": "long video needs a reference photo "
                                      "or video to start from"}, 400)
            return

        # Clip model: uncensored Wan 2.7 I2V by default for lastframe mode.
        chain_model = body.get("chain_model") or "venice-wan27-uncensored"
        if chain_model not in vc.VENICE_MODELS:
            self._send_json({"error": f"unknown model {chain_model!r}"}, 400)
            return
        clip_s = float(vc.VENICE_MODELS[chain_model].get("max_duration_s",
                                                        15.0))
        plan = lv.plan_chain(target_s, clip_s, overlap_s)
        n = plan["n_clips"]

        beats = body.get("beats")
        if beats:
            if (not isinstance(beats, list) or not beats
                    or not all(isinstance(b, str) and b.strip()
                               for b in beats)):
                self._send_json({"error": "beats must be a non-empty list "
                                          "of strings"}, 400)
                return
            if len(beats) != n:
                self._send_json(
                    {"error": f"beats has {len(beats)} entries but the plan "
                              f"needs {n} clips"}, 400)
                return
        else:
            beats = lv.segment_prompt(prompt, n)

        # Cost estimate: N clips x per-clip cost at chosen quality.
        cps = float(vc.VENICE_MODELS[chain_model].get("cost_per_s", 0.13))
        if chain_mode == "extend":
            cps = float(vc.VENICE_MODELS["venice-seedance-r2v"]
                        .get("cost_per_s", 0.30))
        total_est = lv.estimate_cost_usd(n, clip_s, cps)

        job_id = JOBS.new(media or "longvideo", "longvideo", "chain", {
            "prompt": prompt, "beats": beats,
            "target_duration_s": target_s, "clip_duration_s": clip_s,
            "n_clips": n, "overlap_s": overlap_s,
            "chain_model": chain_model, "chain_mode": chain_mode,
            "quality": quality, "ref_image": ref_image,
            "ref_video": ref_video,
            "face_lock": bool(body.get("face_lock")),
        })
        POOL.submit(_run_job, job_id)
        self._send_json({"job_id": job_id, "status": "started",
                         "n_clips": n, "total_s": plan["total_s"],
                         "beats": beats,
                         "estimate": f"~${total_est:.2f} ({n} clips, "
                                     f"{quality})"})


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

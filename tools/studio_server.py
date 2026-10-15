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
                     "video AND a reference photo of the new model. "
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
             "format=duration,format_name,size:stream=codec_type,codec_name",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30)
        data = json.loads(pr.stdout or "{}")
    except Exception:
        return {}
    fmt = data.get("format", {})
    vcodec = ""
    for st in data.get("streams", []):
        if st.get("codec_type") == "video":
            vcodec = st.get("codec_name", "")
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
            "vcodec": vcodec, "size": size}


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


def _split_video_chunks(path: str, workdir: str, chunk_s: float) -> list:
    """Split a video into re-encoded MP4 chunks (no audio needed:
    the original audio is muxed back onto the final stitch)."""
    import subprocess as _sp
    info = _probe_video_info(path)
    duration = info.get("duration") or 0.0
    if duration <= chunk_s + 0.05:
        return [path]
    chunks = []
    for i, (start, length) in enumerate(_chunk_bounds(duration, chunk_s)):
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
    import cv2 as _cv2
    src_face = _cv2.imread(ref_photo_path)
    if src_face is None:
        raise RuntimeError("could not read reference photo")
    silent = os.path.join(workdir, "_facelock_silent.mp4")
    fv.replace_in_video(video_path, src_face, silent)
    _mux_audio(silent, video_path, out_path)
    try:
        os.remove(silent)
    except OSError:
        pass
    return out_path


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
        rpath = safe_media_path(p.get("ref_media_name") or "")
        if vpath is None or rpath is None:
            raise RuntimeError("base video or reference photo disappeared")
        vup = gc.upload_video(vpath)
        iup = gc.upload_image(rpath)
        wf = gw.video_replace_model(
            vup, iup, p.get("prompt", ""),
            reference_strength=_f("reference_strength", 1.0), **video_knobs)
        suffix, timeout_s = "gpu_replace", 3600.0
    else:
        raise RuntimeError(f"unknown gpu generate option {gen_id!r}")

    JOBS.update(job_id, status="running", progress=25,
                phase="queued on your GPU pod")
    saved = gc.run_workflow(wf, jobdir, timeout_s=timeout_s,
                            poll_interval_s=10.0, progress_cb=_progress)
    if not saved:
        raise RuntimeError("GPU pod finished but produced no output files")
    JOBS.update(job_id, status="running", progress=95,
                phase="saving result")
    ext = os.path.splitext(saved[0])[1] or ".bin"
    result_name = f"{base}_{suffix}{ext}"
    dest = os.path.join(jobdir, result_name)
    if os.path.abspath(saved[0]) != os.path.abspath(dest):
        os.replace(saved[0], dest)
    JOBS.update(job_id, status="done", progress=100, phase="done",
                result_url=f"/api/result/{job_id}/{result_name}",
                result_name=result_name)


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
            if (job["params"].get("face_lock") and mode == "i2v"
                    and media_type(job["media"]) == "image"):
                try:
                    JOBS.update(job_id, status="running", progress=95,
                                phase="locking face to reference")
                    locked_name = f"{base}_{result_suffix}_facelock.mp4"
                    _face_lock(dest, mpath,
                               os.path.join(jobdir, locked_name), jobdir)
                    result_name = locked_name
                except Exception:
                    pass
            result_name = os.path.basename(
                os.path.join(jobdir, result_name))
            JOBS.update(job_id, status="done", progress=100,
                        result_url=f"/api/result/{job_id}/{result_name}",
                        result_name=result_name)
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
            # AI model replacement via Venice Seedance R2V: base video is
            # the performance; reference photo/video defines the model.
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
            duration_s = float(p.get("duration_s", 5.0))
            JOBS.update(job_id, status="running", progress=5,
                        phase="preparing video for Venice")
            # Seedance only accepts MP4/MOV + H.264/HEVC reference
            # clips; anything else earns a generic Venice 500.
            # Convert off-spec files locally first.
            send_base = _venice_ready_video(mpath, jobdir, "base")
            send_ref = ref_path
            if ref_type == "video":
                send_ref = _venice_ready_video(ref_path, jobdir, "ref")
            JOBS.update(job_id, status="running", progress=10,
                        phase="submitting to Venice")
            args = {
                "prompt": prompt,
                "duration": duration_s,
                "resolution": q["resolution"],
                # Venice requires aspect_ratio on R2V submits; "adaptive"
                # keeps the base video's own framing.
                "aspect_ratio": "adaptive",
                "reference_video_urls": [vc.file_to_data_url(send_base)],
            }
            if ref_type == "image":
                args["reference_image_urls"] = [vc.file_to_data_url(ref_path)]
            elif ref_type == "video":
                args["reference_video_urls"].append(
                    vc.file_to_data_url(send_ref))
            try:
                vjob = vc.submit("venice-seedance-r2v", args)
            except Exception as e:
                info = _probe_video_info(send_base)
                raise RuntimeError(
                    f"{e} [base clip: {info.get('duration', 0):.1f}s, "
                    f"{(info.get('size') or 0) / 1e6:.1f} MB, "
                    f"{info.get('format', '?')}/{info.get('vcodec', '?')}]"
                ) from e
            JOBS.update(job_id, status="running", progress=30,
                        phase="Venice is rendering")
            url = vc.wait(vjob, timeout_s=1200.0, poll_interval_s=15.0)
            JOBS.update(job_id, status="running", progress=90,
                        phase="downloading result")
            result_name = f"{base}_modelreplace.mp4"
            vc.download_to_file(url, os.path.join(jobdir, result_name))
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
            if p.get("face_lock"):
                try:
                    JOBS.update(job_id, status="running", progress=93,
                                phase="locking face to reference")
                    locked = os.path.join(jobdir,
                                          f"{base}_animate_locked.mp4")
                    _face_lock(gen, ref_path, locked, jobdir)
                    body_video = locked
                except Exception:
                    body_video = gen  # keep the generative result
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
            JOBS.update(job_id, status="done", progress=100,
                        phase="done",
                        result_url=f"/api/result/{job_id}/{result_name}",
                        result_name=result_name)
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
            )
            result_name = f"{base}_longvideo.mp4"
            import shutil as _sh2
            _sh2.copy2(final, os.path.join(jobdir, result_name))
            # True-to-reference face lock on the stitched video: put the
            # reference photo's real face back over the AI's drifted one.
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
                    except Exception:
                        pass
            JOBS.update(job_id, status="done", progress=100,
                        phase="done",
                        result_url=f"/api/result/{job_id}/{result_name}",
                        result_name=result_name)
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
        elif path.startswith("/api/job/") and path.endswith("/clips"):
            # GET /api/job/<job_id>/clips — list completed chain clips
            # for preview during a running long-video job.
            job_id = path[len("/api/job/"):-len("/clips")]
            self._api_job_clips(unquote(job_id))
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
        self._send_json(job)

    def _api_jobs(self):
        """GET /api/jobs -> {jobs: [...]}. Recent jobs, newest first."""
        with JOBS._lock:
            jobs = sorted(JOBS._jobs.values(),
                          key=lambda j: j.get("created", 0), reverse=True)
            # Strip params to keep it light; keep status info
            out = [{
                "job_id": j["job_id"], "kind": j["kind"], "name": j["name"],
                "status": j["status"], "progress": j["progress"],
                "phase": j.get("phase", ""), "result_url": j["result_url"],
                "result_name": j["result_name"], "error": j["error"],
                "created": j["created"],
            } for j in jobs[:20]]
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
                  "fps", "denoise", "negative_prompt", "reference_strength")
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
            ref_name = ref_images[0] if ref_images else ""
            if not base_name or safe_media_path(base_name) is None:
                self._send_json({"error": "pick a base video"}, 400)
                return
            if not ref_name or safe_media_path(ref_name) is None:
                self._send_json(
                    {"error": "pick a reference photo of the new model"},
                    400)
                return
            if media_type(ref_name) != "image":
                self._send_json(
                    {"error": "reference must be a photo"}, 400)
                return
            params["media_name"] = base_name
            params["ref_media_name"] = ref_name
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
        # Seedance reference inputs have hard limits (Venice docs):
        # each clip 2-15s, combined <= 15s. An over-limit clip makes
        # Venice fail with a generic 500, so check here and say why.
        base_info = _probe_video_info(mpath)
        base_dur = base_info.get("duration") or 0.0
        if base_dur and not (2.0 <= base_dur <= 15.0):
            self._send_json(
                {"error": f"AI Replace accepts clips of 2-15 seconds "
                          f"(yours is {base_dur:.0f}s). Trim it in the "
                          "Edit tab first \u2014 or use Best Match / "
                          "Full-Body, which process videos of any "
                          "length automatically."}, 400)
            return
        if ref_type == "video":
            ref_info = _probe_video_info(ref_path)
            ref_dur = ref_info.get("duration") or 0.0
            if ref_dur and base_dur + ref_dur > 15.0:
                self._send_json(
                    {"error": f"base + reference videos total "
                              f"{base_dur + ref_dur:.0f}s; AI Replace "
                              "allows 15s combined. Use a photo reference "
                              "or shorter clips."}, 400)
                return
        duration_s = float(body.get("duration_s") or base_dur
                           or spec.get("default_duration_s", 10.0))
        duration_s = min(duration_s, float(spec.get("max_duration_s", 15.0)))
        est = round(spec["cost_per_s"] * duration_s, 2)
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

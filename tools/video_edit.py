#!/usr/bin/env python3
"""Local video editing operations — ffmpeg only, no API, no cost.

Every op here runs on the VPS with ffmpeg: nothing is uploaded, nothing
is billed, nothing leaves the machine.

Ops:
  trim          cut a segment  (start/end timestamps)
  speed         0.25x-4x with audio pitch correction (atempo chaining)
  crop          center-crop to aspect preset (16:9, 9:16, 1:1) or custom box
  rotate        90 / 180 / 270 degrees clockwise
  filter        grayscale | sepia | brightness | contrast | blur
  reverse       play backwards (video + audio)
  merge         concatenate 2+ videos (stream-copy when params match,
                re-encode fallback otherwise)
  extract_frame save a still JPG at a timestamp (feeds I2V / chaining)
  gif           short segment -> animated GIF (two-pass palette)

Interface mirrors local_effects.py:
  EDIT_OPS  name -> {description, params schema, output, multi}
  apply_edit(op, input_paths, output_path, params) -> output_path
  list_ops() -> [{name, description, params, output, multi}]

Pure stdlib + the ffmpeg/ffprobe binaries. No network, ever.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from typing import Any, Optional


# ------------------------------------------------------------------ probe ----

def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def probe(path: str) -> dict:
    """ffprobe -> {w, h, dur, fps, has_audio, codec, pix_fmt}."""
    if not _have("ffprobe"):
        raise RuntimeError("ffprobe not installed")
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries",
           "stream=width,height,codec_name,avg_frame_rate,pix_fmt",
           "-show_entries", "format=duration", "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    info = json.loads(out.stdout or "{}")
    s = (info.get("streams") or [{}])[0]
    fps_raw = s.get("avg_frame_rate", "0/1")
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0
    acmd = ["ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", path]
    has_audio = bool(subprocess.run(acmd, capture_output=True, text=True,
                                    timeout=30).stdout.strip())
    return {
        "w": int(s.get("width") or 0),
        "h": int(s.get("height") or 0),
        "dur": float(info.get("format", {}).get("duration") or 0),
        "fps": fps,
        "has_audio": has_audio,
        "codec": s.get("codec_name"),
        "pix_fmt": s.get("pix_fmt"),
    }


def _run_ffmpeg(cmd: list[str], timeout: int = 900) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-800:]}")


def _enc(cmd: list[str], crf: int = 20) -> list[str]:
    """Append standard high-quality h264 output encoding."""
    return cmd + ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                  "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


# -------------------------------------------------------------------- ops ----

def _op_trim(inputs: list[str], out: str, p: dict) -> str:
    src = inputs[0]
    info = probe(src)
    start = max(float(p.get("start", 0.0)), 0.0)
    end = float(p.get("end", info["dur"] or 0.0))
    if end <= 0:
        end = info["dur"]
    if end <= start:
        raise ValueError(f"trim: end ({end}) must be after start ({start})")
    cmd = ["ffmpeg", "-hide_banner", "-y",
           "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
           "-i", src]
    if info["has_audio"]:
        cmd += ["-c:a", "aac"]
    cmd = _enc(cmd) + [out]
    _run_ffmpeg(cmd)
    return out


def _atempo_chain(factor: float) -> str:
    # atempo only supports 0.5..2.0 — chain it for wider factors.
    chain = []
    f = factor
    while f > 2.0:
        chain.append("atempo=2.0000")
        f /= 2.0
    while f < 0.5:
        chain.append("atempo=0.5000")
        f /= 0.5
    chain.append(f"atempo={f:.4f}")
    return ",".join(chain)


def _op_speed(inputs: list[str], out: str, p: dict) -> str:
    src = inputs[0]
    factor = float(p.get("factor", 1.0))
    if not 0.25 <= factor <= 4.0:
        raise ValueError(f"speed: factor {factor} out of range 0.25-4.0")
    info = probe(src)
    vf = f"setpts={1.0 / factor:.6f}*PTS"
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", src,
           "-vf", vf]
    if info["has_audio"]:
        cmd += ["-af", _atempo_chain(factor), "-c:a", "aac"]
    cmd = _enc(cmd) + [out]
    _run_ffmpeg(cmd)
    return out


_ASPECTS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}


def _op_crop(inputs: list[str], out: str, p: dict) -> str:
    src = inputs[0]
    info = probe(src)
    w, h = info["w"], info["h"]
    preset = str(p.get("preset", "16:9"))
    if preset == "custom":
        cw, ch = int(p.get("width", w)), int(p.get("height", h))
        cx, cy = int(p.get("x", (w - cw) // 2)), int(p.get("y", (h - ch) // 2))
    elif preset in _ASPECTS:
        target = _ASPECTS[preset]
        src_ar = w / h if h else target
        if src_ar > target:
            # too wide: cut sides
            ch, cw = h, int(h * target)
        else:
            # too tall: cut top/bottom
            cw, cw_ = w, w
            cw, ch = w, int(w / target)
        cx, cy = (w - cw) // 2, (h - ch) // 2
    else:
        raise ValueError(f"crop: unknown preset {preset!r}; "
                         f"expected one of {sorted(_ASPECTS)} or 'custom'")
    cw = max(2, min(cw, w))
    ch = max(2, min(ch, h))
    cx = max(0, min(cx, w - cw))
    cy = max(0, min(cy, h - ch))
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", src,
           "-vf", f"crop={cw}:{ch}:{cx}:{cy}"]
    if info["has_audio"]:
        cmd += ["-c:a", "aac"]
    cmd = _enc(cmd) + [out]
    _run_ffmpeg(cmd)
    return out


def _op_rotate(inputs: list[str], out: str, p: dict) -> str:
    src = inputs[0]
    angle = int(p.get("angle", 90))
    if angle == 90:
        vf = "transpose=1"
    elif angle == 180:
        vf = "transpose=1,transpose=1"
    elif angle == 270:
        vf = "transpose=2"
    else:
        raise ValueError(f"rotate: angle {angle} not in (90, 180, 270)")
    info = probe(src)
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", src, "-vf", vf]
    if info["has_audio"]:
        cmd += ["-c:a", "aac"]
    cmd = _enc(cmd) + [out]
    _run_ffmpeg(cmd)
    return out


_FILTERS = ("grayscale", "sepia", "brightness", "contrast", "blur")


def _op_filter(inputs: list[str], out: str, p: dict) -> str:
    src = inputs[0]
    name = str(p.get("filter", "grayscale")).lower()
    amount = float(p.get("amount", 1.0))
    if name == "grayscale":
        vf = "hue=s=0"
    elif name == "sepia":
        vf = ("colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0"
              ":.272:.534:.131")
    elif name == "brightness":
        if not -1.0 <= amount <= 1.0:
            raise ValueError("brightness amount must be -1..1")
        vf = f"eq=brightness={amount:.3f}"
    elif name == "contrast":
        if not 0.0 <= amount <= 3.0:
            raise ValueError("contrast amount must be 0..3")
        vf = f"eq=contrast={amount:.3f}"
    elif name == "blur":
        sigma = max(0.5, min(amount, 40.0))
        vf = f"gblur=sigma={sigma:.1f}"
    else:
        raise ValueError(f"filter: unknown filter {name!r}; "
                         f"expected one of {_FILTERS}")
    vf += ",format=yuv420p"
    info = probe(src)
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", src, "-vf", vf]
    if info["has_audio"]:
        cmd += ["-c:a", "aac"]
    cmd = _enc(cmd) + [out]
    _run_ffmpeg(cmd)
    return out


def _op_reverse(inputs: list[str], out: str, p: dict) -> str:
    src = inputs[0]
    info = probe(src)
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", src,
           "-vf", "reverse"]
    if info["has_audio"]:
        cmd += ["-af", "areverse", "-c:a", "aac"]
    cmd = _enc(cmd) + [out]
    _run_ffmpeg(cmd)
    return out


def _op_merge(inputs: list[str], out: str, p: dict) -> str:
    if len(inputs) < 2:
        raise ValueError("merge needs at least 2 videos")
    infos = [probe(s) for s in inputs]
    # Stream-copy fast path when all params match.
    keys = [(i["codec"], i["w"], i["h"], i["fps"], i["pix_fmt"])
            for i in infos]
    list_path = out + ".filelist.txt"
    try:
        with open(list_path, "w") as f:
            for s in inputs:
                # concat demuxer: escape single quotes in path
                f.write("file '" + s.replace("'", "'\\''") + "'\n")
        if len(set(keys)) == 1:
            cmd = ["ffmpeg", "-hide_banner", "-y", "-f", "concat",
                   "-safe", "0", "-i", list_path, "-c", "copy", out]
        else:
            # Normalize everything, then concat re-encode.
            w, h = infos[0]["w"], infos[0]["h"]
            fps = infos[0]["fps"] or 30.0
            norm = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
                    f"setsar=1,fps={fps:.2f},format=yuv420p")
            n = len(inputs)
            cmd = ["ffmpeg", "-hide_banner", "-y"]
            for s in inputs:
                cmd += ["-i", s]
            filt = "".join(f"[{i}:v]{norm}[v{i}];" for i in range(n))
            filt += "".join(f"[v{i}]" for i in range(n))
            filt += f"concat=n={n}:v=1:a=0[vout]"
            cmd += ["-filter_complex", filt, "-map", "[vout]"]
            # also concat audio when present on all inputs
            if all(i["has_audio"] for i in infos):
                af = "".join(f"[{i}:a]aresample=48000[a{i}];"
                             for i in range(n))
                af += "".join(f"[a{i}]" for i in range(n))
                af += f"concat=n={n}:v=0:a=1[aout]"
                cmd[cmd.index("-filter_complex") + 1] += ";" + af
                cmd += ["-map", "[aout]", "-c:a", "aac"]
            cmd = _enc(cmd) + [out]
        _run_ffmpeg(cmd)
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass
    return out


def _op_extract_frame(inputs: list[str], out: str, p: dict) -> str:
    src = inputs[0]
    ts = max(float(p.get("timestamp", 0.0)), 0.0)
    cmd = ["ffmpeg", "-hide_banner", "-y", "-ss", f"{ts:.3f}",
           "-i", src, "-vframes", "1", "-q:v", "2", out]
    _run_ffmpeg(cmd)
    return out


def extract_frame_at(video_path: str, out_jpg: str, at_s: float = 0.0) -> str:
    """Extract a single frame at timestamp (seconds)."""
    if not _have("ffmpeg"):
        raise RuntimeError("ffmpeg not installed")
    ts = max(float(at_s), 0.0)
    cmd = ["ffmpeg", "-hide_banner", "-y", "-ss", f"{ts:.3f}",
           "-i", video_path, "-vframes", "1", "-q:v", "2", out_jpg]
    _run_ffmpeg(cmd)
    return out_jpg


def extract_last_frame(video_path: str, out_jpg: str) -> str:
    """Extract the EXACT last frame (for clip chaining continuity).

    Uses the frame count + select filter so the frame is precise, not
    an approximation from -sseof seeking.
    """
    if not _have("ffmpeg") or not _have("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe not installed")
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-count_frames", "-show_entries",
           "stream=nb_read_frames,avg_frame_rate", "-of", "json",
           video_path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=60)
        info = json.loads(out.stdout or "{}")
        s = (info.get("streams") or [{}])[0]
        n = int(s.get("nb_read_frames") or 0)
    except Exception:
        n = 0
    if n > 0:
        cmd = ["ffmpeg", "-hide_banner", "-y", "-i", video_path,
               "-vf", f"select='eq(n\\,{n - 1})'", "-vframes", "1",
               "-q:v", "2", out_jpg]
    else:
        # Fallback: seek to 50ms before EOF.
        cmd = ["ffmpeg", "-hide_banner", "-y", "-sseof", "-0.05",
               "-i", video_path, "-vframes", "1", "-q:v", "2", out_jpg]
    _run_ffmpeg(cmd)
    return out_jpg


def _op_gif(inputs: list[str], out: str, p: dict) -> str:
    src = inputs[0]
    start = max(float(p.get("start", 0.0)), 0.0)
    dur = float(p.get("duration", 3.0))
    if not 0.5 <= dur <= 15.0:
        raise ValueError("gif duration must be 0.5-15s")
    fps = int(p.get("fps", 12))
    fps = max(5, min(fps, 24))
    width = int(p.get("width", 480))
    width = max(160, min(width, 960))
    pal = out + ".palette.png"
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    try:
        _run_ffmpeg(["ffmpeg", "-hide_banner", "-y",
                     "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                     "-i", src, "-vf", vf + ",palettegen", pal])
        _run_ffmpeg(["ffmpeg", "-hide_banner", "-y",
                     "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                     "-i", src, "-i", pal,
                     "-lavfi", vf + "[x];[x][1:v]paletteuse",
                     out])
    finally:
        try:
            os.remove(pal)
        except OSError:
            pass
    return out


# ------------------------------------------------------------- registry ----

EDIT_OPS: dict[str, dict[str, Any]] = {
    "trim": {
        "description": "Cut a segment: keep only start..end.",
        "params": {"start": {"type": "float", "default": 0.0,
                             "help": "segment start, seconds"},
                   "end": {"type": "float", "default": 0.0,
                           "help": "segment end, seconds (0 = end of clip)"}},
        "apply": _op_trim, "output": "video", "multi": False, "ext": "mp4",
    },
    "speed": {
        "description": "Change playback speed 0.25x-4x. Audio pitch-corrected.",
        "params": {"factor": {"type": "float", "default": 1.0,
                              "help": "<1 slow-mo, >1 fast (0.25-4.0)"}},
        "apply": _op_speed, "output": "video", "multi": False, "ext": "mp4",
    },
    "crop": {
        "description": "Center-crop to an aspect ratio.",
        "params": {"preset": {"type": "string", "default": "16:9",
                              "help": "16:9, 9:16, 1:1, or custom"},
                   "x": {"type": "int", "default": 0,
                         "help": "custom: left px"},
                   "y": {"type": "int", "default": 0,
                         "help": "custom: top px"},
                   "width": {"type": "int", "default": 0,
                             "help": "custom: width px"},
                   "height": {"type": "int", "default": 0,
                              "help": "custom: height px"}},
        "apply": _op_crop, "output": "video", "multi": False, "ext": "mp4",
    },
    "rotate": {
        "description": "Rotate clockwise.",
        "params": {"angle": {"type": "int", "default": 90,
                             "help": "90, 180, or 270"}},
        "apply": _op_rotate, "output": "video", "multi": False, "ext": "mp4",
    },
    "filter": {
        "description": "Color/filter adjustment.",
        "params": {"filter": {"type": "string", "default": "grayscale",
                              "help": "grayscale, sepia, brightness, "
                                      "contrast, blur"},
                   "amount": {"type": "float", "default": 1.0,
                              "help": "brightness -1..1, contrast 0..3, "
                                      "blur = sigma px"}},
        "apply": _op_filter, "output": "video", "multi": False, "ext": "mp4",
    },
    "reverse": {
        "description": "Play the clip backwards (video + audio).",
        "params": {},
        "apply": _op_reverse, "output": "video", "multi": False, "ext": "mp4",
    },
    "merge": {
        "description": "Join 2+ videos end to end.",
        "params": {"media2": {"type": "string", "default": "",
                              "help": "second video filename"},
                   "media3": {"type": "string", "default": "",
                              "help": "optional third video filename"}},
        "apply": _op_merge, "output": "video", "multi": True, "ext": "mp4",
    },
    "extract_frame": {
        "description": "Save a still frame as JPG (also lands in Gallery "
                       "for AI use).",
        "params": {"timestamp": {"type": "float", "default": 0.0,
                                 "help": "frame time, seconds"}},
        "apply": _op_extract_frame, "output": "image", "multi": False,
        "ext": "jpg",
    },
    "gif": {
        "description": "Convert a short segment to animated GIF.",
        "params": {"start": {"type": "float", "default": 0.0,
                             "help": "segment start, seconds"},
                   "duration": {"type": "float", "default": 3.0,
                                "help": "0.5-15 seconds"},
                   "fps": {"type": "int", "default": 12,
                           "help": "5-24 fps"},
                   "width": {"type": "int", "default": 480,
                             "help": "160-960 px wide"}},
        "apply": _op_gif, "output": "image", "multi": False, "ext": "gif",
    },
}


def list_ops() -> list[dict]:
    return [{"name": n, "description": s["description"],
             "params": s["params"], "output": s["output"],
             "multi": s["multi"]}
            for n, s in EDIT_OPS.items()]


def apply_edit(op: str, input_paths: list[str], output_path: str,
               params: Optional[dict] = None, timeout: int = 900) -> str:
    """Apply an edit op. input_paths: 1 path normally, 2+ for merge.
    Returns output_path."""
    if not _have("ffmpeg"):
        raise RuntimeError("ffmpeg not installed")
    spec = EDIT_OPS.get(op)
    if spec is None:
        raise ValueError(f"unknown edit op {op!r}; "
                         f"expected one of {sorted(EDIT_OPS)}")
    if not input_paths:
        raise ValueError("apply_edit needs at least one input")
    if spec["multi"] and len(input_paths) < 2:
        raise ValueError(f"{op!r} needs at least 2 inputs")
    if not spec["multi"] and len(input_paths) != 1:
        raise ValueError(f"{op!r} takes exactly 1 input")
    for ip in input_paths:
        if not os.path.isfile(ip):
            raise ValueError(f"input not found: {ip}")
    return spec["apply"](input_paths, output_path, params or {})


# ------------------------------------------------------------------ CLI ----

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Local video editing ops")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list ops")
    a = sub.add_parser("apply", help="apply an op")
    a.add_argument("--op", required=True)
    a.add_argument("--in", dest="inputs", nargs="+", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--params", default="{}")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        print(json.dumps(list_ops(), indent=2))
    elif args.cmd == "apply":
        out = apply_edit(args.op, args.inputs, args.out,
                         json.loads(args.params))
        print(json.dumps({"output": out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

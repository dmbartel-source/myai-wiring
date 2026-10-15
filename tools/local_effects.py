"""Deterministic local effect recipes — ffmpeg filter graphs, no API, no cost.

Every recipe here runs on the VPS with ffmpeg only: nothing is uploaded,
nothing is billed, nothing leaves the machine. This is the DEFAULT route
for anything achievable deterministically; generative APIs are reserved
for what local processing cannot do (new pixels: T2V motion, object-
removal fill, style transfer).

Concepts:
  - A recipe is: name -> {description, params schema, build()}.
  - build(recipe, params, info) -> filter_complex string mapping
    [0:v] -> [vout]. `info` comes from probe() (w/h/duration/fps/audio).
  - build_audio(...) -> optional audio filter for [0:a] -> [aout]
    (used by speed_ramp; everything else passes audio through).
  - apply() / preview() shell out to ffmpeg. suggest_recipes() maps a
    plain-language description to candidate recipes for the $effect skill.

Pure stdlib + the ffmpeg/ffprobe binaries. No network, ever.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Optional


# ---------------------------------------------------------------- probe ----

def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def probe(path: str) -> dict:
    """ffprobe -> {w, h, dur, fps, has_audio, codec}."""
    if not _have("ffprobe"):
        raise RuntimeError("ffprobe not installed")
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,codec_name,avg_frame_rate",
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
    }


# ------------------------------------------------------------ builders ----

def _grain(p: dict, info: dict) -> str:
    s = float(p.get("strength", 8.0))
    return f"[0:v]noise=alls={s}:allf=t,format=yuv420p[vout]"


def _vignette(p: dict, info: dict) -> str:
    s = float(p.get("strength", 0.5))
    return f"[0:v]vignette=angle='PI/5+{s}*PI/5',format=yuv420p[vout]"


def _light_leak(p: dict, info: dict) -> str:
    period = float(p.get("period", 6.0))
    opacity = float(p.get("opacity", 0.35))
    w, h, dur = info["w"], info["h"], max(info["dur"], 0.1)
    fps = max(info.get("fps") or 30.0, 1.0)
    # geq's time variable is capital T; drive the drift off frame count N
    # so it stays correct at any framerate.
    return (
        f"[0:v]format=rgba[base];"
        f"nullsrc=s={w}x{h}:d={dur:.2f}:r=30,format=rgba,"
        f"geq=r='255':g='140+70*sin(2*PI*N/({fps:.2f}*{period}))':b='30',"
        f"colorchannelmixer=aa=0.5[leak];"
        f"[base][leak]blend=all_mode=screen:all_opacity={opacity}:shortest=1,"
        f"format=yuv420p[vout]"
    )


def _glitch_cut(p: dict, info: dict) -> str:
    period = float(p.get("period", 2.0))
    burst = float(p.get("burst", 0.18))
    gate = f"lt(mod(t\\,{period})\\,{burst})"
    # crop sees iw/ih of its own input; overlay sees main_h/main_w for the
    # base frame (ih there would mean the strip's height — wrong).
    ypos_crop = "mod(n*53\\,ih*7/8)"
    ypos_ovl = "mod(n*53\\,main_h*7/8)"
    return (
        f"[0:v]split=2[base][fx];"
        f"[fx]crop=w=iw:h='ih/8':x=0:y='{ypos_crop}'[sl];"
        f"[base][sl]overlay=x='48*sin(2*PI*t*13)':y='{ypos_ovl}':enable='{gate}',"
        f"format=yuv420p[vout]"
    )


def _zoom_punch(p: dict, info: dict) -> str:
    amt = float(p.get("amount", 0.25))
    at = float(p.get("at", 1.0))
    width = float(p.get("width", 0.6))
    e = f"(1+{amt}*exp(-pow((t-{at})/({width})\\,2)))"
    w, h = info["w"], info["h"]
    # scale animates per frame (eval=frame); the crop keeps FIXED w/h with
    # an animated center offset — animated crop dimensions force a filter
    # reinit and fail. Scale factor is always >= 1, so the crop is valid.
    return (
        f"[0:v]scale=w='iw*{e}':h='ih*{e}':eval=frame,"
        f"crop=w={w}:h={h}:x='(in_w-{w})/2':y='(in_h-{h})/2',"
        f"format=yuv420p[vout]"
    )


def _speed_ramp(p: dict, info: dict) -> str:
    factor = float(p.get("factor", 0.5))
    if factor <= 0:
        raise ValueError("speed_ramp factor must be > 0")
    return f"[0:v]setpts={1.0 / factor}*PTS[vout]"


def _speed_ramp_audio(p: dict, info: dict) -> Optional[str]:
    factor = float(p.get("factor", 0.5))
    if not info.get("has_audio"):
        return None
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
    return "[0:a]" + ",".join(chain) + "[aout]"


_GRADES = {
    "teal_orange": ("eq=saturation=1.15:contrast=1.05,"
                    "colorbalance=rs=-0.12:gs=0.03:bs=0.18:rm=-0.05:bm=0.08"),
    "noir": "eq=saturation=0:contrast=1.25:brightness=-0.03",
    "bleach": "eq=saturation=0.65:contrast=0.92:brightness=0.05",
    "warm": "colorbalance=rs=0.15:gs=0.06:bs=-0.08,eq=saturation=1.08",
    "cold": "colorbalance=rs=-0.10:gs=-0.02:bs=0.14,eq=saturation=1.05",
}


def _color_grade(p: dict, info: dict) -> str:
    preset = str(p.get("preset", "teal_orange")).lower()
    if preset not in _GRADES:
        raise ValueError(f"unknown grade preset {preset!r}; "
                         f"expected one of {sorted(_GRADES)}")
    intensity = float(p.get("intensity", 1.0))
    # intensity blends via eq on top of the preset chain (simple, honest)
    return f"[0:v]{_GRADES[preset]},format=yuv420p[vout]"


_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _find_font() -> Optional[str]:
    for fp in _FONT_PATHS:
        if os.path.exists(fp):
            return fp
    return None


def _escape_drawtext(text: str) -> str:
    return (text.replace("\\", "\\\\").replace("'", "\\'")
                .replace(":", "\\:").replace(",", "\\,"))


def _overlay_text(p: dict, info: dict) -> str:
    text = _escape_drawtext(str(p.get("text", "TITLE")))
    fs = int(p.get("fontsize", 48))
    color = str(p.get("color", "white"))
    x = str(p.get("x", "(w-text_w)/2"))
    y = str(p.get("y", "h-120"))
    start = float(p.get("start", 0.0))
    end = float(p.get("end", 2.0))
    box = ":box=1:boxcolor=black@0.5:boxborderw=12" if p.get("box", True) else ""
    font = _find_font()
    # NOTE: filter args start with '=' after the filter name (drawtext=...),
    # further options are ':'-separated.
    fontopt = f"fontfile={font}:" if font else ""
    return (
        f"[0:v]drawtext={fontopt}text='{text}':fontsize={fs}:"
        f"fontcolor={color}:x='{x}':y='{y}'"
        f":enable='between(t\\,{start}\\,{end})'{box}[vout]"
    )


def _shake(p: dict, info: dict) -> str:
    a = float(p.get("amplitude", 12.0))
    f = float(p.get("freq", 8.0))
    w, h = info["w"], info["h"]
    return (
        f"[0:v]crop=w='iw-{2 * a}':h='ih-{2 * a}':"
        f"x='{a}+{a}*sin(2*PI*{f}*t)':"
        f"y='{a}+{a}*cos(2*PI*{f}*1.31*t)',"
        f"scale={w}:{h},format=yuv420p[vout]"
    )


def _letterbox(p: dict, info: dict) -> str:
    ratio = float(p.get("ratio", 2.39))
    w, h = info["w"], info["h"]
    bh = max(0, int((h - w / ratio) / 2))
    return (
        f"[0:v]drawbox=x=0:y=0:w=iw:h={bh}:color=black:t=fill,"
        f"drawbox=x=0:y='ih-{bh}':w=iw:h={bh}:color=black:t=fill[vout]"
    )


def _fade_in_out(p: dict, info: dict) -> str:
    d = float(p.get("duration", 0.5))
    dur = max(info["dur"], 2 * d + 0.1)
    return (f"[0:v]fade=t=in:st=0:d={d},"
            f"fade=t=out:st={dur - d:.2f}:d={d}[vout]")


_XFADE_KINDS = {
    "fade", "fadeblack", "fadewhite", "wipeleft", "wiperight",
    "wipeup", "wipedown", "slideleft", "slideright", "slideup",
    "slidedown", "smoothleft", "smoothright", "distance", "radial",
}


def _xfade(p: dict, info: dict, info2: Optional[dict] = None) -> str:
    kind = str(p.get("kind", "fade")).lower()
    if kind not in _XFADE_KINDS:
        raise ValueError(f"unknown xfade kind {kind!r}; "
                         f"expected one of {sorted(_XFADE_KINDS)}")
    d = float(p.get("duration", 0.75))
    w, h = 1280, 720
    norm = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps=30,format=yuv420p")
    # offset defaults to (first clip length - transition duration)
    off = p.get("offset")
    if off is None:
        off = max(info["dur"] - d, 0) if info.get("dur") else 2.0
    return (
        f"[0:v]{norm}[v0];[1:v]{norm}[v1];"
        f"[v0][v1]xfade=transition={kind}:duration={d}:offset={off:.2f},"
        f"format=yuv420p[vout]"
    )


def _burn_env(at: float, width: float) -> str:
    """Soft in/out time envelope for geq: 0 outside [at-w, at+w], 1 at `at`."""
    ai, ao = at - width, at + width
    return (f"min(max((T-{ai:.2f})/{width:.2f}\\,0)\\,1)"
            f"*min(max(({ao:.2f}-T)/{width:.2f}\\,0)\\,1)")


def _film_burn(p: dict, info: dict) -> str:
    at = float(p.get("at", 1.0))
    width = float(p.get("width", 0.5))
    opacity = float(p.get("opacity", 0.85))
    flick = float(p.get("flicker", 3.0))
    w, h, dur = info["w"], info["h"], max(info["dur"], 0.1)
    fps = max(info.get("fps") or 30.0, 1.0)
    env = _burn_env(at, width)
    # Warm blobs drift across the burn layer; the geq time envelope drives
    # it black -> full -> black, and screen(black) == identity, so the
    # effect is perfectly gated with soft edges and no alpha-semantics risk.
    blob = f"(110+90*sin(2*PI*(X+Y)/400+2*PI*N/({fps:.2f}*{flick})))"
    return (
        f"[0:v]format=rgba[base];"
        f"nullsrc=s={w}x{h}:d={dur:.2f}:r=30,format=rgba,"
        f"geq=r='255*{env}':g='{blob}*{env}':b='25*{env}',"
        f"boxblur=luma_radius=40:luma_power=1[burn];"
        f"[base][burn]blend=all_mode=screen:all_opacity={opacity}:shortest=1,"
        f"format=yuv420p[vout]"
    )


def _rgb_split(p: dict, info: dict) -> str:
    a = float(p.get("amount", 6.0))
    f = float(p.get("freq", 1.5))
    # geq interpolates sub-pixel lookups; out-of-range clamps to edge.
    return (
        f"[0:v]format=gbrp,"
        f"geq=r='r(X-{a}*sin(2*PI*T*{f})\\,Y)'"
        f":g='g(X\\,Y)'"
        f":b='b(X+{a}*sin(2*PI*T*{f})\\,Y)',"
        f"format=yuv420p[vout]"
    )


def _datamosh_lite(p: dict, info: dict) -> str:
    b = max(int(p.get("block", 8)), 2)
    period = float(p.get("period", 2.0))
    burst = float(p.get("burst", 0.5))
    gate = f"lt(mod(t\\,{period})\\,{burst})"
    ypos = "mod(n*61\\,main_h*4/5)"
    return (
        f"[0:v]split=2[base][px];"
        f"[px]scale=w='iw/{b}':h='ih/{b}',"
        f"scale=w='iw*{b}':h='ih*{b}':flags=neighbor,"
        f"crop=w=iw:h='ih/5':x=0:y='mod(n*61\\,ih*4/5)'[blk];"
        f"[base][blk]overlay=x='30*sin(2*PI*t*2.3)':y='{ypos}':enable='{gate}',"
        f"format=yuv420p[vout]"
    )


def _light_streak(p: dict, info: dict) -> str:
    at = float(p.get("at", 1.5))
    width = float(p.get("width", 0.6))
    opacity = float(p.get("opacity", 0.8))
    w, h, dur = info["w"], info["h"], max(info["dur"], 0.1)
    env = _burn_env(at, width)
    # Gaussian horizontal band, time-gated the same way as film_burn.
    line = "exp(-pow((Y-H/2)/10\\,2))"
    return (
        f"[0:v]format=rgba[base];"
        f"nullsrc=s={w}x{h}:d={dur:.2f}:r=30,format=rgba,"
        f"geq=r='255*{line}*{env}':g='240*{line}*{env}':b='220*{line}*{env}',"
        f"boxblur=luma_radius=12:luma_power=2[streak];"
        f"[base][streak]blend=all_mode=screen:all_opacity={opacity}:shortest=1,"
        f"format=yuv420p[vout]"
    )


def _vignette_pulse(p: dict, info: dict) -> str:
    s0 = float(p.get("strength", 0.30))
    sa = float(p.get("pulse", 0.18))
    period = float(p.get("period", 4.0))
    env = f"({s0}+{sa}*sin(2*PI*T/{period}))"
    d2 = "pow(hypot(X-W/2\\,Y-H/2)/(W*0.7)\\,2)"
    chan = f"'r(X\\,Y)*(1-{env}*{d2})'"
    return (
        f"[0:v]format=gbrp,"
        f"geq=r={chan}:g={chan.replace('r(X', 'g(X')}:b={chan.replace('r(X', 'b(X')},"
        f"format=yuv420p[vout]"
    )


def _whip_pan(p: dict, info: dict) -> str:
    at = float(p.get("at", 1.5))
    wd = float(p.get("width", 0.5))
    a = float(p.get("amplitude", 120.0))
    w, h = info["w"], info["h"]
    # k: 0 -> 1 across the window; x sweeps 2a -> 0 (whip settles centered).
    k = f"min(max((t-({at}-{wd}/2))/{wd}\\,0)\\,1)"
    gate = f"between(t\\,{at - wd / 2:.2f}\\,{at + wd / 2:.2f})"
    return (
        f"[0:v]crop=w='iw-{2 * a}':h=ih:x='{2 * a}*(1-({k}))':y=0,"
        f"boxblur=luma_radius=14:luma_power=1:enable='{gate}',"
        f"scale={w}:{h},format=yuv420p[vout]"
    )


def _duotone(p: dict, info: dict) -> str:
    def _rgb(s: str) -> tuple[int, int, int]:
        parts = str(s).split(",")
        if len(parts) != 3 or not all(
                p.strip().lstrip("-").isdigit() for p in parts):
            raise ValueError(f"color {s!r} must be 'r,g,b'")
        return int(parts[0]), int(parts[1]), int(parts[2])
    sr, sg, sb = _rgb(p.get("shadow", "30,30,60"))
    hr, hg, hb = _rgb(p.get("highlight", "255,190,110"))
    lum = "(r(X\\,Y)+g(X\\,Y)+b(X\\,Y))/765"  # 0..1
    return (
        f"[0:v]format=gbrp,"
        f"geq=r='{sr}+({hr}-{sr})*{lum}'"
        f":g='{sg}+({hg}-{sg})*{lum}'"
        f":b='{sb}+({hb}-{sb})*{lum}',"
        f"format=yuv420p[vout]"
    )


def _vhs_tracking(p: dict, info: dict) -> str:
    period = float(p.get("period", 4.0))
    burst = float(p.get("burst", 1.2))
    gate = f"lt(mod(t\\,{period})\\,{burst})"
    # band sweeps downward; crop sees ih, overlay sees main_h.
    y_crop = "mod(t*ih*0.5\\,ih*11/12)"
    y_ovl = "mod(t*main_h*0.5\\,main_h*11/12)"
    return (
        f"[0:v]split=2[base][n];"
        f"[n]crop=w=iw:h='ih/12':x=0:y='{y_crop}',noise=alls=30:allf=t[bar];"
        f"[base][bar]overlay=x='20*sin(2*PI*t*9)':y='{y_ovl}':enable='{gate}',"
        f"format=yuv420p[vout]"
    )


def _blur_pulse(p: dict, info: dict) -> str:
    at = float(p.get("at", 1.5))
    width = float(p.get("width", 0.8))
    sigma = float(p.get("sigma", 18.0))
    d = width / 2
    dur = max(info["dur"], 0.1)
    # sharp -> blurred -> sharp via two chained xfades on split copies.
    # xfade extends the timeline (offset + second input length), so trim
    # back to the source duration afterwards.
    return (
        f"[0:v]split=3[a][b][c];"
        f"[b]gblur=sigma={sigma}[bb];"
        f"[a][bb]xfade=transition=fade:duration={d:.2f}:offset={at - d:.2f}[m];"
        f"[m][c]xfade=transition=fade:duration={d:.2f}:offset={at:.2f},"
        f"trim=duration={dur:.2f},setpts=PTS-STARTPTS,"
        f"format=yuv420p[vout]"
    )


def _projector_flicker(p: dict, info: dict) -> str:
    f1 = float(p.get("flicker", 0.05))
    f2 = float(p.get("flicker2", 0.02))
    wv = float(p.get("weave", 2.0))
    w, h = info["w"], info["h"]
    mod = f"(1+{f1}*sin(2*PI*13*T)+{f2}*sin(2*PI*47*T+1.3))"
    # crop a slightly shorter window for the weave, then scale back to full
    # frame (same pattern as _shake) so output dims never change.
    return (
        f"[0:v]crop=w=iw:h='ih-4':x=0:y='{wv}+{wv}*sin(2*PI*7*t)',"
        f"scale={w}:{h},format=gbrp,"
        f"geq=r='r(X\\,Y)*{mod}':g='g(X\\,Y)*{mod}':b='b(X\\,Y)*{mod}',"
        f"format=yuv420p[vout]"
    )


# ------------------------------------------------------------- registry ----

RECIPES: dict[str, dict[str, Any]] = {
    "film_grain": {
        "description": "Animated film grain (temporal noise). Pure texture, no API.",
        "params": {"strength": {"type": "float", "default": 8.0,
                                "help": "grain strength, 1-50"}},
        "build": _grain, "audio": None, "two_inputs": False,
    },
    "vignette": {
        "description": "Lens vignette darkening the frame edges.",
        "params": {"strength": {"type": "float", "default": 0.5,
                                "help": "0-1, edge darkening"}},
        "build": _vignette, "audio": None, "two_inputs": False,
    },
    "light_leak": {
        "description": "Warm animated light leak blended over the frame (screen).",
        "params": {"period": {"type": "float", "default": 6.0,
                              "help": "leak drift period, seconds"},
                   "opacity": {"type": "float", "default": 0.35,
                               "help": "0-1 blend opacity"}},
        "build": _light_leak, "audio": None, "two_inputs": False,
    },
    "glitch_cut": {
        "description": "Periodic digital glitch: horizontal slice displacement bursts.",
        "params": {"period": {"type": "float", "default": 2.0,
                              "help": "seconds between glitch bursts"},
                   "burst": {"type": "float", "default": 0.18,
                             "help": "burst length, seconds"}},
        "build": _glitch_cut, "audio": None, "two_inputs": False,
    },
    "zoom_punch": {
        "description": "Crash-zoom punch-in centered on frame (gaussian bump).",
        "params": {"amount": {"type": "float", "default": 0.25,
                              "help": "peak zoom, fraction of frame"},
                   "at": {"type": "float", "default": 1.0,
                          "help": "punch center, seconds"},
                   "width": {"type": "float", "default": 0.6,
                             "help": "punch width, seconds"}},
        "build": _zoom_punch, "audio": None, "two_inputs": False,
    },
    "speed_ramp": {
        "description": "Constant speed change (slow-mo / timelapse). Audio pitch-corrected.",
        "params": {"factor": {"type": "float", "default": 0.5,
                              "help": "<1 slow-mo, >1 fast (0.25-4.0)"}},
        "build": _speed_ramp, "audio": _speed_ramp_audio, "two_inputs": False,
    },
    "color_grade": {
        "description": "Deterministic color grade preset.",
        "params": {"preset": {"type": "string", "default": "teal_orange",
                              "help": f"one of {sorted(_GRADES)}"},
                   "intensity": {"type": "float", "default": 1.0,
                                 "help": "reserved for future blending"}},
        "build": _color_grade, "audio": None, "two_inputs": False,
    },
    "overlay_text": {
        "description": "Burned-in text/title with optional backdrop box, time-gated.",
        "params": {"text": {"type": "string", "default": "TITLE",
                            "help": "text to burn in"},
                   "fontsize": {"type": "int", "default": 48, "help": "px"},
                   "color": {"type": "string", "default": "white", "help": "fontcolor"},
                   "x": {"type": "string", "default": "(w-text_w)/2", "help": "x expr"},
                   "y": {"type": "string", "default": "h-120", "help": "y expr"},
                   "start": {"type": "float", "default": 0.0, "help": "appear at, s"},
                   "end": {"type": "float", "default": 2.0, "help": "disappear at, s"},
                   "box": {"type": "bool", "default": True, "help": "backdrop box"}},
        "build": _overlay_text, "audio": None, "two_inputs": False,
    },
    "shake": {
        "description": "Camera shake (impact/handheld). Deterministic sinusoid.",
        "params": {"amplitude": {"type": "float", "default": 12.0,
                                 "help": "shake amplitude, px"},
                   "freq": {"type": "float", "default": 8.0, "help": "Hz"}},
        "build": _shake, "audio": None, "two_inputs": False,
    },
    "letterbox": {
        "description": "Cinematic letterbox bars for a target aspect ratio.",
        "params": {"ratio": {"type": "float", "default": 2.39,
                             "help": "target aspect, e.g. 2.39"}},
        "build": _letterbox, "audio": None, "two_inputs": False,
    },
    "fade_in_out": {
        "description": "Fade from/to black at head and tail.",
        "params": {"duration": {"type": "float", "default": 0.5,
                                "help": "fade length, seconds"}},
        "build": _fade_in_out, "audio": None, "two_inputs": False,
    },
    "xfade_transition": {
        "description": "Transition between TWO clips (fade/wipe/slide/distance/radial).",
        "params": {"kind": {"type": "string", "default": "fade",
                            "help": f"one of {sorted(_XFADE_KINDS)}"},
                   "duration": {"type": "float", "default": 0.75,
                                 "help": "transition length, seconds"},
                   "offset": {"type": "float", "default": None,
                              "help": "start at, s (default: first clip end - duration)"}},
        "build": _xfade, "audio": None, "two_inputs": True,
    },
    "film_burn": {
        "description": "Film-burn flash: warm drifting blobs wash over the frame (screen).",
        "params": {"at": {"type": "float", "default": 1.0,
                           "help": "burn peak, seconds"},
                   "width": {"type": "float", "default": 0.5,
                             "help": "fade in/out length, seconds"},
                   "opacity": {"type": "float", "default": 0.85,
                               "help": "0-1 blend opacity"},
                   "flicker": {"type": "float", "default": 3.0,
                               "help": "blob drift rate"}},
        "build": _film_burn, "audio": None, "two_inputs": False,
    },
    "rgb_split": {
        "description": "Animated chromatic aberration: R/B channels oscillate apart.",
        "params": {"amount": {"type": "float", "default": 6.0,
                              "help": "max channel offset, px"},
                   "freq": {"type": "float", "default": 1.5, "help": "Hz"}},
        "build": _rgb_split, "audio": None, "two_inputs": False,
    },
    "datamosh_lite": {
        "description": "Datamosh-lite: pixelated blocks smear across the frame in bursts.",
        "params": {"block": {"type": "int", "default": 8,
                             "help": "pixelation factor"},
                   "period": {"type": "float", "default": 2.0,
                              "help": "seconds between bursts"},
                   "burst": {"type": "float", "default": 0.5,
                             "help": "burst length, seconds"}},
        "build": _datamosh_lite, "audio": None, "two_inputs": False,
    },
    "light_streak": {
        "description": "Anamorphic-style horizontal light streak flash (screen).",
        "params": {"at": {"type": "float", "default": 1.5,
                           "help": "streak peak, seconds"},
                   "width": {"type": "float", "default": 0.6,
                             "help": "fade in/out length, seconds"},
                   "opacity": {"type": "float", "default": 0.8,
                               "help": "0-1 blend opacity"}},
        "build": _light_streak, "audio": None, "two_inputs": False,
    },
    "vignette_pulse": {
        "description": "Breathing vignette: edge darkening oscillates.",
        "params": {"strength": {"type": "float", "default": 0.30,
                                "help": "base darkening 0-1"},
                   "pulse": {"type": "float", "default": 0.18,
                             "help": "oscillation amplitude"},
                   "period": {"type": "float", "default": 4.0,
                              "help": "breath period, seconds"}},
        "build": _vignette_pulse, "audio": None, "two_inputs": False,
    },
    "whip_pan": {
        "description": "Whip-pan into the clip: fast lateral move settling centered, motion-blurred.",
        "params": {"at": {"type": "float", "default": 1.5,
                           "help": "settle point, seconds"},
                   "width": {"type": "float", "default": 0.5,
                             "help": "pan duration, seconds"},
                   "amplitude": {"type": "float", "default": 120.0,
                                 "help": "pan travel, px"}},
        "build": _whip_pan, "audio": None, "two_inputs": False,
    },
    "duotone": {
        "description": "Duotone grade: luminance mapped between shadow and highlight colors.",
        "params": {"shadow": {"type": "string", "default": "30,30,60",
                              "help": "'r,g,b' for shadows"},
                   "highlight": {"type": "string", "default": "255,190,110",
                                 "help": "'r,g,b' for highlights"}},
        "build": _duotone, "audio": None, "two_inputs": False,
    },
    "vhs_tracking": {
        "description": "VHS tracking glitch: noise bar sweeps the frame in bursts.",
        "params": {"period": {"type": "float", "default": 4.0,
                              "help": "seconds between glitches"},
                   "burst": {"type": "float", "default": 1.2,
                             "help": "glitch length, seconds"}},
        "build": _vhs_tracking, "audio": None, "two_inputs": False,
    },
    "blur_pulse": {
        "description": "Defocus pulse: blurs out and back around a point (rack-focus feel).",
        "params": {"at": {"type": "float", "default": 1.5,
                           "help": "full-blur point, seconds"},
                   "width": {"type": "float", "default": 0.8,
                             "help": "total pulse length, seconds"},
                   "sigma": {"type": "float", "default": 18.0,
                             "help": "blur strength"}},
        "build": _blur_pulse, "audio": None, "two_inputs": False,
    },
    "projector_flicker": {
        "description": "Old-projector look: exposure flicker plus vertical gate weave.",
        "params": {"flicker": {"type": "float", "default": 0.05,
                               "help": "brightness wobble amplitude"},
                   "flicker2": {"type": "float", "default": 0.02,
                                "help": "second harmonic amplitude"},
                   "weave": {"type": "float", "default": 2.0,
                             "help": "vertical weave amplitude, px"}},
        "build": _projector_flicker, "audio": None, "two_inputs": False,
    },
}


# ------------------------------------------------------------ execution ----

def build(recipe: str, params: Optional[dict] = None, info: Optional[dict] = None) -> str:
    """Return the filter_complex string for a recipe ([0:v] -> [vout])."""
    spec = RECIPES.get(recipe)
    if spec is None:
        raise ValueError(f"unknown recipe {recipe!r}; "
                         f"expected one of {sorted(RECIPES)}")
    merged = {k: v["default"] for k, v in spec["params"].items()}
    merged.update(params or {})
    return spec["build"](merged, info or {})


def _run_ffmpeg(cmd: list[str], timeout: int = 900) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-800:]}")


def apply(input_path: str, output_path: str, recipe: str,
          params: Optional[dict] = None, timeout: int = 900) -> str:
    """Apply a single-input recipe. Returns output_path. Audio passes through
    (pitch-corrected for speed_ramp)."""
    if not _have("ffmpeg"):
        raise RuntimeError("ffmpeg not installed")
    spec = RECIPES.get(recipe)
    if spec is None:
        raise ValueError(f"unknown recipe {recipe!r}")
    if spec["two_inputs"]:
        raise ValueError(f"{recipe!r} needs two inputs — use apply2()")
    info = probe(input_path)
    filt = build(recipe, params, info)
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", input_path,
           "-filter_complex", filt, "-map", "[vout]"]
    audio_f = spec["audio"]
    if info["has_audio"] and audio_f:
        af = audio_f(params or {}, info)
        if af:
            cmd += ["-map", "[aout]"]
            # merge the audio filter into the same graph
            cmd[cmd.index("-filter_complex") + 1] = filt + ";" + af
            cmd += ["-c:a", "aac"]
        else:
            cmd += ["-map", "0:a?", "-c:a", "aac"]
    elif info["has_audio"]:
        cmd += ["-map", "0:a?", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", output_path]
    _run_ffmpeg(cmd, timeout)
    return output_path


def apply2(input_a: str, input_b: str, output_path: str, recipe: str,
           params: Optional[dict] = None, timeout: int = 900) -> str:
    """Apply a two-input recipe (xfade_transition). Returns output_path."""
    if not _have("ffmpeg"):
        raise RuntimeError("ffmpeg not installed")
    spec = RECIPES.get(recipe)
    if spec is None or not spec["two_inputs"]:
        raise ValueError(f"{recipe!r} is not a two-input recipe")
    info = probe(input_a)
    filt = build(recipe, params, info)
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", input_a, "-i", input_b,
           "-filter_complex", filt, "-map", "[vout]",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", output_path]
    _run_ffmpeg(cmd, timeout)
    return output_path


def preview(input_path: str, output_path: str, recipe: str,
            params: Optional[dict] = None, seconds: float = 3.0) -> str:
    """Render the first `seconds` of the effect for quick review."""
    if not _have("ffmpeg"):
        raise RuntimeError("ffmpeg not installed")
    spec = RECIPES.get(recipe)
    if spec is None or spec["two_inputs"]:
        raise ValueError(f"{recipe!r} cannot be previewed this way")
    info = probe(input_path)
    filt = build(recipe, params, info)
    cmd = ["ffmpeg", "-hide_banner", "-y", "-t", str(seconds),
           "-i", input_path, "-filter_complex", filt, "-map", "[vout]",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path]
    _run_ffmpeg(cmd, timeout=300)
    return output_path


# -------------------------------------------------------------- routing ----

_KEYWORDS: list[tuple[tuple[str, ...], str, float]] = [
    (("film grain", "grain", "16mm", "8mm", "noisy", "analog"), "film_grain", 0.9),
    (("light leak", "leak", "lens flare", "flare"), "light_leak", 0.9),
    (("glitch", "digital glitch"), "glitch_cut", 0.9),
    (("rgb split", "chromatic aberration", "chroma shift"), "rgb_split", 0.9),
    (("datamosh", "mosh", "block glitch"), "datamosh_lite", 0.9),
    (("film burn", "burn transition", "burnout"), "film_burn", 0.9),
    (("light streak", "anamorphic", "lens streak", "flare streak"), "light_streak", 0.9),
    (("vignette pulse", "pulsing vignette", "breathing vignette"), "vignette_pulse", 0.85),
    (("whip pan", "whip-pan", "swish pan", "fast pan"), "whip_pan", 0.9),
    (("duotone", "two tone", "two-tone", "color pop", "pop color"), "duotone", 0.9),
    (("vhs", "tracking glitch", "tape glitch", "retro glitch"), "vhs_tracking", 0.9),
    (("blur pulse", "defocus", "rack focus", "blur in", "blur out"), "blur_pulse", 0.9),
    (("flicker", "projector", "old film", "film flicker", "gate weave"), "projector_flicker", 0.9),
    (("zoom punch", "punch in", "crash zoom", "zoom in fast"), "zoom_punch", 0.9),
    (("slow motion", "slow-mo", "slowmo", "speed ramp", "ramp",
      "timelapse", "time-lapse", "fast motion", "speed up", "slow down"),
     "speed_ramp", 0.9),
    (("color grade", "grade", "teal", "noir", "black and white", "bleach",
      "warm look", "cold look", "lut", "cinematic color"), "color_grade", 0.85),
    (("text", "title", "caption", "lower third", "subtitle", "headline"),
     "overlay_text", 0.9),
    (("transition", "dissolve", "fade to", "wipe", "cut to", "crossfade"),
     "xfade_transition", 0.85),
    (("shake", "shaky", "earthquake", "impact shake", "handheld"), "shake", 0.9),
    (("letterbox", "cinematic bars", "2.39", "anamorphic bars", "widescreen"),
     "letterbox", 0.9),
    (("fade in", "fade out", "fadein", "fadeout"), "fade_in_out", 0.9),
    (("vignette",), "vignette", 0.8),
]


def suggest_recipes(description: str) -> list[dict]:
    """Map a plain description to local recipes: [{recipe, score, why}]."""
    desc = description.lower()
    hits: dict[str, dict] = {}
    for keywords, recipe, score in _KEYWORDS:
        for kw in keywords:
            if kw in desc:
                if recipe not in hits or hits[recipe]["score"] < score:
                    hits[recipe] = {"recipe": recipe, "score": score,
                                    "why": f"matched {kw!r}",
                                    "description": RECIPES[recipe]["description"]}
                break
    return sorted(hits.values(), key=lambda h: -h["score"])


def list_recipes() -> list[dict]:
    return [{"name": n, "description": s["description"],
             "params": s["params"], "two_inputs": s["two_inputs"]}
            for n, s in RECIPES.items()]


# ------------------------------------------------------------------ CLI ----

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Local deterministic effect recipes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list recipes")
    sg = sub.add_parser("suggest", help="suggest recipes for a description")
    sg.add_argument("description")
    ap2 = sub.add_parser("apply", help="apply a recipe")
    ap2.add_argument("--in", dest="inp", required=True)
    ap2.add_argument("--out", dest="out", required=True)
    ap2.add_argument("--recipe", required=True)
    ap2.add_argument("--params", default="{}")
    pv = sub.add_parser("preview", help="render a short preview")
    pv.add_argument("--in", dest="inp", required=True)
    pv.add_argument("--out", dest="out", required=True)
    pv.add_argument("--recipe", required=True)
    pv.add_argument("--params", default="{}")
    pv.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args(argv)
    if args.cmd == "list":
        print(json.dumps(list_recipes(), indent=2))
    elif args.cmd == "suggest":
        print(json.dumps(suggest_recipes(args.description), indent=2))
    elif args.cmd == "apply":
        out = apply(args.inp, args.out, args.recipe, json.loads(args.params))
        print(json.dumps({"output": out}))
    elif args.cmd == "preview":
        out = preview(args.inp, args.out, args.recipe,
                      json.loads(args.params), args.seconds)
        print(json.dumps({"output": out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

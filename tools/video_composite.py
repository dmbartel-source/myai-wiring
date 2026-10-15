"""Advanced video compositing — ffmpeg filter graphs, no API, no cost.

Multi-input and timeline-manipulation operations that don't fit in
local_effects.py (single-input recipes). Follows the same patterns:
  - VIDEO_COMPOSITE_OPS: name -> {description, params schema, build fn,
    n_inputs, audio fn}.
  - build(op, params, infos) -> filter_complex string ending in [vout].
    `infos` is a list of probe() dicts, one per input.
  - build_audio(op, params, infos) -> Optional[str] ending in [aout];
    None means "pass input 0's audio through".
  - apply() dispatches on n_inputs; stabilize gets its own two-pass path.

Pure stdlib + ffmpeg/ffprobe. No network, ever. Deterministic: no
unseeded randomness anywhere in this file.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Optional

try:
    from . import local_effects as le  # package import
except ImportError:  # pragma: no cover - direct tools/ sys.path usage
    import local_effects as le

probe = le.probe


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def _run_ffmpeg(cmd: list[str], timeout: int = 900) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-800:]}")


# ---------------------------------------------------------- chroma key ----

_KEY_COLORS = {"green": "0x00FF00", "blue": "0x0000FF", "red": "0xFF0000"}


def _chroma_key(p: dict, infos: list[dict]) -> str:
    color = str(p.get("color", "green")).lower()
    hexcol = _KEY_COLORS.get(color, color)
    sim = float(p.get("similarity", 0.15))
    blend = float(p.get("blend", 0.1))
    despill = max(0.0, min(1.0, float(p.get("despill", 0.5))))
    w, h = infos[0]["w"], infos[0]["h"]
    # Despill: pull green out of edge pixels left by the key.
    return (
        f"[0:v]format=yuva420p,chromakey={hexcol}:{sim}:{blend},"
        f"colorbalance=gs={-0.4 * despill:.2f}:bs={0.15 * despill:.2f}[keyed];"
        f"[1:v]format=yuv420p,scale={w}:{h}:flags=lanczos[bg];"
        f"[bg][keyed]overlay=0:0:format=yuv420,format=yuv420p[vout]"
    )


def _chroma_key_audio(p: dict, infos: list[dict]) -> Optional[str]:
    # Foreground carries the performance; keep its audio.
    return "[0:a]anull[aout]" if infos[0]["has_audio"] else None


# --------------------------------------------------- picture in picture ----

_PIP_POSITIONS = ("topleft", "topright", "bottomleft", "bottomright", "center")


def _pip(p: dict, infos: list[dict]) -> str:
    info = infos[0]
    w, h = info["w"], info["h"]
    pos = str(p.get("position", "bottomright")).lower()
    if pos not in _PIP_POSITIONS:
        pos = "bottomright"
    size = float(p.get("size", 0.25))
    margin = int(p.get("margin", 16))
    border = int(p.get("border", 3))
    bcolor = str(p.get("border_color", "white"))
    shadow = bool(p.get("shadow", True))
    so = int(p.get("shadow_offset", 8))
    sop = max(0.0, min(1.0, float(p.get("shadow_opacity", 0.4))))

    pw = max(2, int(w * size))
    if pos == "topleft":
        xe, ye = f"{margin}", f"{margin}"
    elif pos == "topright":
        xe, ye = f"W-w-{margin}", f"{margin}"
    elif pos == "bottomleft":
        xe, ye = f"{margin}", f"H-h-{margin}"
    elif pos == "center":
        xe, ye = "(W-w)/2", "(H-h)/2"
    else:  # bottomright
        xe, ye = f"W-w-{margin}", f"H-h-{margin}"
    # In overlay expressions W/H = main frame, w/h = overlay input.
    # Note: filtergraph kept simple for ffmpeg 6.x compatibility (no geq).
    # Calculate pip dimensions in Python for the shadow box.
    iw1, ih1 = infos[1]["w"], infos[1]["h"]
    ph = max(2, int(ih1 * pw / iw1))
    ph += ph % 2  # keep even
    shadow_w = pw + 2 * border
    shadow_h = ph + 2 * border

    # Calculate absolute pip position in Python (avoid expression issues).
    if pos == "topleft":
        px, py = margin, margin
    elif pos == "topright":
        px, py = w - shadow_w - margin, margin
    elif pos == "bottomleft":
        px, py = margin, h - shadow_h - margin
    elif pos == "center":
        px, py = (w - shadow_w) // 2, (h - shadow_h) // 2
    else:  # bottomright
        px, py = w - shadow_w - margin, h - shadow_h - margin

    g = (f"[1:v]scale={pw}:-2,setsar=1[pips];"
         f"[pips]pad=iw+{2 * border}:ih+{2 * border}:{border}:{border}"
         f":color={bcolor}[pipb];")
    base = "[0:v]"
    if shadow:
        # Shadow = black box drawn behind the pip position (offset).
        g += (f"{base}drawbox=x={px + so}:y={py + so}"
              f":w={shadow_w}:h={shadow_h}"
              f":color=black:t=fill[tmp];")
        base = "[tmp]"
    g += (f"{base}[pipb]overlay=x={px}:y={py}:eof_action=pass,"
          f"format=yuv420p[vout]")
    return g


def _pip_audio(p: dict, infos: list[dict]) -> Optional[str]:
    return "[0:a]anull[aout]" if infos[0]["has_audio"] else None


# -------------------------------------------------------- split screen ----

def _split_screen(p: dict, infos: list[dict]) -> str:
    layout = str(p.get("layout", "")).lower()
    n = len(infos)
    if not layout:
        layout = {2: "1x2", 4: "2x2"}.get(n, "")
    need = {"1x2": 2, "2x1": 2, "2x2": 4}.get(layout)
    if need is None:
        raise ValueError(f"unknown split_screen layout {layout!r}; "
                         "expected one of 1x2, 2x1, 2x2")
    if n != need:
        raise ValueError(f"layout {layout!r} needs {need} inputs, got {n}")
    w, h = infos[0]["w"], infos[0]["h"]
    div = int(p.get("divider", 4))
    dcolor = str(p.get("divider_color", "black"))
    if layout == "1x2":
        cw, ch, stack = w // 2, h, "0_0|w0_0"
        divf = (f"drawbox=x=iw/2-{div // 2}:y=0:w={div}:h=ih"
                f":color={dcolor}:t=fill")
    elif layout == "2x1":
        cw, ch, stack = w, h // 2, "0_0|0_h0"
        divf = (f"drawbox=x=0:y=ih/2-{div // 2}:w=iw:h={div}"
                f":color={dcolor}:t=fill")
    else:  # 2x2
        cw, ch = w // 2, h // 2
        stack = "0_0|w0_0|0_h0|w0_h0"
        divf = (f"drawbox=x=iw/2-{div // 2}:y=0:w={div}:h=ih"
                f":color={dcolor}:t=fill,"
                f"drawbox=x=0:y=ih/2-{div // 2}:w=iw:h={div}"
                f":color={dcolor}:t=fill")
    parts = []
    labels = []
    for i in range(n):
        parts.append(f"[{i}:v]scale={cw}:{ch}:flags=lanczos,setsar=1[v{i}]")
        labels.append(f"[v{i}]")
    g = ";".join(parts) + ";"
    g += "".join(labels) + f"xstack=inputs={n}:layout={stack}[stacked];"
    g += f"[stacked]{divf},format=yuv420p[vout]"
    return g


def _split_audio(p: dict, infos: list[dict]) -> Optional[str]:
    return "[0:a]anull[aout]" if infos[0]["has_audio"] else None


# -------------------------------------------------------- freeze frame ----

def _freeze_frame(p: dict, infos: list[dict]) -> str:
    info = infos[0]
    dur = info["dur"]
    fps = info["fps"] or 30.0
    at = max(0.0, float(p.get("at", 1.0)))
    hold = max(0.1, float(p.get("hold", 2.0)))
    if dur and at >= dur:
        at = max(0.0, dur - 0.5)
    frame_dur = 1.0 / fps
    segs = []
    if at > 0.01:
        segs.append(f"[0:v]trim=start=0:end={at:.3f},setpts=PTS-STARTPTS[pre]")
    segs.append(f"[0:v]trim=start={at:.3f}:duration={frame_dur:.4f},"
                f"setpts=PTS-STARTPTS[fr]")
    segs.append(f"[fr]loop=loop=-1:size=1:start=0,"
                f"trim=duration={hold:.3f},setpts=PTS-STARTPTS[frz]")
    if not dur or at < dur - 0.01:
        segs.append(f"[0:v]trim=start={at:.3f},setpts=PTS-STARTPTS[post]")
    labels = []
    if at > 0.01:
        labels.append("[pre]")
    labels.append("[frz]")
    if not dur or at < dur - 0.01:
        labels.append("[post]")
    g = ";".join(segs) + ";"
    g += "".join(labels) + f"concat=n={len(labels)}:v=1:a=0[vout]"
    return g


def _freeze_audio(p: dict, infos: list[dict]) -> Optional[str]:
    if not infos[0]["has_audio"]:
        return None
    info = infos[0]
    dur = info["dur"]
    at = max(0.0, float(p.get("at", 1.0)))
    hold = max(0.1, float(p.get("hold", 2.0)))
    if dur and at >= dur:
        at = max(0.0, dur - 0.5)
    parts = []
    if at > 0.01:
        parts.append(f"[0:a]atrim=start=0:end={at:.3f},asetpts=PTS-STARTPTS[a1]")
    parts.append(f"aevalsrc=0:duration={hold:.3f}:sample_rate=44100[asil]")
    if not dur or at < dur - 0.01:
        parts.append(f"[0:a]atrim=start={at:.3f},asetpts=PTS-STARTPTS[a2]")
    labels = []
    if at > 0.01:
        labels.append("[a1]")
    labels.append("[asil]")
    if not dur or at < dur - 0.01:
        labels.append("[a2]")
    return ";".join(parts) + ";" + "".join(labels) + \
        f"concat=n={len(labels)}:v=0:a=1[aout]"


# ------------------------------------------------------------- reverse ----

def _reverse(p: dict, infos: list[dict]) -> str:
    return "[0:v]reverse[vout]"


def _reverse_audio(p: dict, infos: list[dict]) -> Optional[str]:
    return "[0:a]areverse[aout]" if infos[0]["has_audio"] else None


# ----------------------------------------------------------- mask wipe ----

def _mask_wipe(p: dict, infos: list[dict]) -> str:
    w, h = infos[0]["w"], infos[0]["h"]
    dur1 = infos[0]["dur"] or 5.0
    direction = str(p.get("direction", "right")).lower()
    if direction not in ("left", "right", "up", "down"):
        direction = "right"
    tdur = max(0.2, float(p.get("duration", 1.0)))
    soft = max(0.05, min(1.0, float(p.get("softness", 0.3))))
    default_off = max(0.0, dur1 - tdur)
    off = p.get("offset")
    off = default_off if off is None else float(off)
    D = off + tdur
    u = {"right": "X/W", "left": "(1-X/W)",
         "down": "Y/H", "up": "(1-Y/H)"}[direction]
    # Wipe position p sweeps -s/2 -> 1+s/2 over the transition; the soft
    # edge is s wide. At t=0 all black (clip A), at t=1 all white (clip B).
    expr = (f"255*clip((T/{tdur:.3f}*(1+{soft})-{soft}/2-({u}))"
            f"/{soft}+0.5,0,1)")
    return (
        f"[0:v]trim=duration={D:.3f},setpts=PTS-STARTPTS[a];"
        f"[1:v]trim=duration={D:.3f},setpts=PTS-STARTPTS[b];"
        f"nullsrc=s={w}x{h}:d={tdur:.3f},format=gray,"
        f"geq=lum='{expr}'[mask];"
        f"[a][b][mask]maskedmerge,format=yuv420p[vout]"
    )


def _mask_wipe_audio(p: dict, infos: list[dict]) -> Optional[str]:
    if not (infos[0]["has_audio"] and infos[1]["has_audio"]):
        return "[0:a]anull[aout]" if infos[0]["has_audio"] else None
    dur1 = infos[0]["dur"] or 5.0
    tdur = max(0.2, float(p.get("duration", 1.0)))
    default_off = max(0.0, dur1 - tdur)
    off = p.get("offset")
    off = default_off if off is None else float(off)
    D = off + tdur
    return (f"[0:a]atrim=duration={D:.3f},asetpts=PTS-STARTPTS[aa];"
            f"[1:a]atrim=duration={D:.3f},asetpts=PTS-STARTPTS[ab];"
            f"[aa][ab]acrossfade=d={tdur:.3f}[aout]")


# ------------------------------------------------- speed ramp advanced ----

def _speed_ramp_adv(p: dict, infos: list[dict]) -> str:
    dur = infos[0]["dur"]
    if not dur:
        raise ValueError("speed_ramp_advanced needs a known input duration")
    slow = max(0.1, min(2.0, float(p.get("slow_factor", 0.5))))
    center = max(0.0, min(1.0, float(p.get("center", 0.5))))
    width = max(0.05, min(1.0, float(p.get("width", 0.3))))
    t1 = dur * max(0.0, center - width / 2)
    t2 = dur * min(1.0, center + width / 2)
    mult = 1.0 / slow  # setpts multiplier: slow=0.5 -> 2x timestamps
    return (
        f"[0:v]trim=start=0:end={t1:.3f},setpts=PTS-STARTPTS[v1];"
        f"[0:v]trim=start={t1:.3f}:end={t2:.3f},"
        f"setpts={mult:.4f}*(PTS-STARTPTS)[v2];"
        f"[0:v]trim=start={t2:.3f},setpts=PTS-STARTPTS[v3];"
        f"[v1][v2][v3]concat=n=3:v=1:a=0[vout]"
    )


def _speed_ramp_adv_audio(p: dict, infos: list[dict]) -> Optional[str]:
    if not infos[0]["has_audio"]:
        return None
    dur = infos[0]["dur"]
    slow = max(0.1, min(2.0, float(p.get("slow_factor", 0.5))))
    center = max(0.0, min(1.0, float(p.get("center", 0.5))))
    width = max(0.05, min(1.0, float(p.get("width", 0.3))))
    t1 = dur * max(0.0, center - width / 2)
    t2 = dur * min(1.0, center + width / 2)
    # atempo only accepts 0.5..100; chain two stages for slower factors.
    stages = []
    s = slow
    while s < 0.5:
        stages.append("atempo=0.5")
        s /= 0.5
    stages.append(f"atempo={s:.4f}")
    tempo = ",".join(stages)
    return (
        f"[0:a]atrim=start=0:end={t1:.3f},asetpts=PTS-STARTPTS[a1];"
        f"[0:a]atrim=start={t1:.3f}:end={t2:.3f},asetpts=PTS-STARTPTS,"
        f"{tempo}[a2];"
        f"[0:a]atrim=start={t2:.3f},asetpts=PTS-STARTPTS[a3];"
        f"[a1][a2][a3]concat=n=3:v=0:a=1[aout]"
    )


# ------------------------------------------------------------ stabilize ----

_STABILIZER: Optional[str] = None


def detect_stabilizer() -> str:
    """Return 'vidstab' (2-pass, better) or 'deshake' (1-pass fallback)."""
    global _STABILIZER
    if _STABILIZER is None:
        try:
            out = subprocess.run(
                ["ffmpeg", "-hide_banner", "-h", "filter=vidstabdetect"],
                capture_output=True, text=True, timeout=30)
            ok = out.returncode == 0 and "vidstabdetect" in out.stdout
        except Exception:
            ok = False
        _STABILIZER = "vidstab" if ok else "deshake"
    return _STABILIZER


def _stabilize(p: dict, infos: list[dict]) -> str:
    method = detect_stabilizer()
    if method == "vidstab":
        trf = p.get("_trf", "transforms.trf")
        zoom = float(p.get("zoom", 0))
        return (f"[0:v]vidstabtransform=input={trf}:zoom={zoom}:"
                f"smoothing=10,format=yuv420p[vout]")
    # deshake fallback: single pass, in-graph.
    return "[0:v]deshake,format=yuv420p[vout]"


def _stabilize_audio(p: dict, infos: list[dict]) -> Optional[str]:
    return "[0:a]anull[aout]" if infos[0]["has_audio"] else None


# ------------------------------------------------------------- registry ----

VIDEO_COMPOSITE_OPS: dict[str, dict[str, Any]] = {
    "chroma_key": {
        "description": "Remove a green/blue screen and composite over a "
                       "background clip. Includes despill.",
        "params": {"color": {"type": "string", "default": "green",
                             "help": "green|blue|red|0xRRGGBB"},
                   "similarity": {"type": "float", "default": 0.15,
                                  "help": "0-1 color match tolerance"},
                   "blend": {"type": "float", "default": 0.1,
                             "help": "0-1 edge blend"},
                   "despill": {"type": "float", "default": 0.5,
                               "help": "0-1 green spill removal"}},
        "build": _chroma_key, "audio": _chroma_key_audio, "n_inputs": 2,
    },
    "picture_in_picture": {
        "description": "Overlay a smaller video on the main frame with "
                       "border and drop shadow.",
        "params": {"position": {"type": "string", "default": "bottomright",
                                "help": "topleft|topright|bottomleft|"
                                        "bottomright|center"},
                   "size": {"type": "float", "default": 0.25,
                            "help": "inset width as fraction of main"},
                   "margin": {"type": "int", "default": 16,
                              "help": "px from frame edge"},
                   "border": {"type": "int", "default": 3,
                              "help": "border px"},
                   "border_color": {"type": "string", "default": "white"},
                   "shadow": {"type": "bool", "default": True},
                   "shadow_offset": {"type": "int", "default": 8},
                   "shadow_opacity": {"type": "float", "default": 0.4}},
        "build": _pip, "audio": _pip_audio, "n_inputs": 2,
    },
    "split_screen": {
        "description": "Grid of 2-4 clips: 1x2 side-by-side, 2x1 stacked, "
                       "or 2x2 quad, with dividers.",
        "params": {"layout": {"type": "string", "default": "",
                              "help": "1x2|2x1|2x2 (auto from input count)"},
                   "divider": {"type": "int", "default": 4,
                               "help": "divider px"},
                   "divider_color": {"type": "string", "default": "black"}},
        "build": _split_screen, "audio": _split_audio,
        "n_inputs": (2, 4),
    },
    "freeze_frame": {
        "description": "Hold the frame at time `at` for `hold` seconds "
                       "(silence fills the audio gap).",
        "params": {"at": {"type": "float", "default": 1.0,
                          "help": "freeze point, seconds"},
                   "hold": {"type": "float", "default": 2.0,
                            "help": "hold length, seconds"}},
        "build": _freeze_frame, "audio": _freeze_audio, "n_inputs": 1,
    },
    "reverse": {
        "description": "Play the clip backwards (video + audio).",
        "params": {},
        "build": _reverse, "audio": _reverse_audio, "n_inputs": 1,
    },
    "mask_wipe": {
        "description": "Reveal transition between two clips using an "
                       "animated gradient mask (directional wipe).",
        "params": {"direction": {"type": "string", "default": "right",
                                 "help": "left|right|up|down"},
                   "duration": {"type": "float", "default": 1.0,
                                "help": "wipe length, seconds"},
                   "softness": {"type": "float", "default": 0.3,
                                "help": "0-1 edge softness"},
                   "offset": {"type": "float", "default": None,
                              "help": "wipe start, s (default: end-duration)"}},
        "build": _mask_wipe, "audio": _mask_wipe_audio, "n_inputs": 2,
    },
    "speed_ramp_advanced": {
        "description": "Variable speed: slow-motion zone in the middle of "
                       "the clip (pitch-corrected audio).",
        "params": {"slow_factor": {"type": "float", "default": 0.5,
                                   "help": "0.1-2.0; <1 is slow-mo"},
                   "center": {"type": "float", "default": 0.5,
                              "help": "0-1 slow zone center"},
                   "width": {"type": "float", "default": 0.3,
                             "help": "0-1 slow zone width"}},
        "build": _speed_ramp_adv, "audio": _speed_ramp_adv_audio,
        "n_inputs": 1,
    },
    "stabilize": {
        "description": "Stabilize shaky footage: 2-pass vidstab when "
                       "available, deshake fallback otherwise.",
        "params": {"zoom": {"type": "float", "default": 0,
                            "help": "vidstab extra zoom %"}},
        "build": _stabilize, "audio": _stabilize_audio, "n_inputs": 1,
        "two_pass": True,
    },
}


def _check_inputs(op: str, n: int) -> None:
    spec = VIDEO_COMPOSITE_OPS[op]
    want = spec["n_inputs"]
    ok = (n == want) if isinstance(want, int) else (n in want)
    if not ok:
        raise ValueError(f"{op!r} needs {want} input(s), got {n}")


def build(op: str, params: Optional[dict] = None,
          infos: Optional[list[dict]] = None) -> str:
    """Return the filter_complex video graph ending in [vout]."""
    spec = VIDEO_COMPOSITE_OPS.get(op)
    if spec is None:
        raise ValueError(f"unknown composite op {op!r}; "
                         f"expected one of {sorted(VIDEO_COMPOSITE_OPS)}")
    _check_inputs(op, len(infos or []))
    merged = {k: v["default"] for k, v in spec["params"].items()}
    merged.update(params or {})
    return spec["build"](merged, infos or [])


def build_audio(op: str, params: Optional[dict] = None,
                infos: Optional[list[dict]] = None) -> Optional[str]:
    """Return the audio graph ending in [aout], or None for passthrough."""
    spec = VIDEO_COMPOSITE_OPS.get(op)
    if spec is None:
        raise ValueError(f"unknown composite op {op!r}")
    _check_inputs(op, len(infos or []))
    merged = {k: v["default"] for k, v in spec["params"].items()}
    merged.update(params or {})
    fn = spec.get("audio")
    return fn(merged, infos or []) if fn else None


def list_ops() -> list[dict]:
    return [{"name": n, "description": s["description"],
             "params": s["params"], "n_inputs": s["n_inputs"]}
            for n, s in VIDEO_COMPOSITE_OPS.items()]


# ------------------------------------------------------------- execution ----

def _base_cmd(inputs: list[str], filt: str, audio_filt: Optional[str],
              has_audio: bool, output_path: str) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    for inp in inputs:
        cmd += ["-i", inp]
    cmd += ["-filter_complex", filt]
    if audio_filt:
        cmd += ["-map", "[vout]", "-map", "[aout]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-movflags", "+faststart", output_path]
    elif has_audio:
        cmd += ["-map", "[vout]", "-map", "0:a?",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-movflags", "+faststart", output_path]
    else:
        cmd += ["-map", "[vout]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", output_path]
    return cmd


def _apply_generic(inputs: list[str], output_path: str, op: str,
                   params: Optional[dict], timeout: int) -> str:
    if not _have("ffmpeg"):
        raise RuntimeError("ffmpeg not installed")
    if op not in VIDEO_COMPOSITE_OPS:
        raise ValueError(f"unknown composite op {op!r}")
    _check_inputs(op, len(inputs))
    infos = [probe(p) for p in inputs]
    filt = build(op, params, infos)
    afilt = build_audio(op, params, infos)
    cmd = _base_cmd(inputs, filt, afilt, infos[0]["has_audio"], output_path)
    if afilt:
        # Merge the audio graph into the same filter_complex.
        idx = cmd.index("-filter_complex") + 1
        cmd[idx] = filt + ";" + afilt
    try:
        _run_ffmpeg(cmd, timeout)
    except RuntimeError as e:
        # Some ffmpeg builds/environments disagree with the probe about
        # audio streams ("matches no streams"). Fall back to video-only.
        if afilt and "matches no streams" in str(e):
            cmd2 = _base_cmd(inputs, filt, None, False, output_path)
            _run_ffmpeg(cmd2, timeout)
        else:
            raise
    return output_path


def _apply_stabilize(input_path: str, output_path: str,
                     params: Optional[dict], timeout: int) -> str:
    if not _have("ffmpeg"):
        raise RuntimeError("ffmpeg not installed")
    info = probe(input_path)
    method = detect_stabilizer()
    if method == "deshake":
        filt = build("stabilize", params, [info])
        afilt = build_audio("stabilize", params, [info])
        cmd = _base_cmd([input_path], filt, afilt,
                        info["has_audio"], output_path)
        if afilt:
            idx = cmd.index("-filter_complex") + 1
            cmd[idx] = filt + ";" + afilt
        _run_ffmpeg(cmd, timeout)
        return output_path
    # vidstab: pass 1 detects motion into a transforms file.
    tmpdir = tempfile.mkdtemp(prefix="stab_")
    try:
        trf = os.path.join(tmpdir, "transforms.trf")
        p1 = dict(params or {})
        shakiness = int(p1.get("shakiness", 5))
        cmd1 = ["ffmpeg", "-hide_banner", "-y", "-i", input_path,
                "-vf", f"vidstabdetect=shakiness={shakiness}:result={trf}",
                "-f", "null", "-"]
        _run_ffmpeg(cmd1, timeout)
        p2 = dict(params or {})
        p2["_trf"] = trf
        filt = build("stabilize", p2, [info])
        afilt = build_audio("stabilize", p2, [info])
        cmd2 = _base_cmd([input_path], filt, afilt,
                         info["has_audio"], output_path)
        if afilt:
            idx = cmd2.index("-filter_complex") + 1
            cmd2[idx] = filt + ";" + afilt
        _run_ffmpeg(cmd2, timeout)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return output_path


def apply(inputs: list[str] | str, output_path: str, op: str,
          params: Optional[dict] = None, timeout: int = 900) -> str:
    """Apply a composite op. `inputs` is a path or list of paths."""
    if isinstance(inputs, str):
        inputs = [inputs]
    if op == "stabilize":
        if len(inputs) != 1:
            raise ValueError("stabilize takes exactly 1 input")
        return _apply_stabilize(inputs[0], output_path, params, timeout)
    return _apply_generic(inputs, output_path, op, params, timeout)


# -------------------------------------------------------------- routing ----

_KEYWORDS: list[tuple[tuple[str, ...], str, float]] = [
    (("green screen", "chroma key", "chroma", "blue screen",
      "remove background video", "key out"), "chroma_key", 0.95),
    (("picture in picture", "picture-in-picture", "pip",
      "corner video", "overlay video", "inset video"), "picture_in_picture",
     0.9),
    (("split screen", "side by side", "2x2", "quad split",
      "stacked video"), "split_screen", 0.9),
    (("freeze frame", "freeze", "hold frame", "still frame",
      "stop motion hold"), "freeze_frame", 0.9),
    (("reverse", "backwards", "rewind", "play backwards"), "reverse", 0.9),
    (("mask wipe", "wipe reveal", "gradient wipe", "wipe transition"),
     "mask_wipe", 0.85),
    (("speed ramp", "slow-mo", "slowmo", "slow motion", "ramp",
      "variable speed"), "speed_ramp_advanced", 0.85),
    (("stabilize", "stabilise", "stabilization", "deshake", "shaky",
      "steady cam", "steadicam fix"), "stabilize", 0.9),
]


def suggest_composite(description: str) -> list[dict]:
    """Map a plain description to composite ops: [{op, score, why}]."""
    desc = description.lower()
    hits: dict[str, dict] = {}
    for keywords, op, score in _KEYWORDS:
        for kw in keywords:
            if kw in desc:
                if op not in hits or hits[op]["score"] < score:
                    hits[op] = {"op": op, "score": score,
                                "why": f"matched {kw!r}",
                                "description":
                                    VIDEO_COMPOSITE_OPS[op]["description"]}
                break
    return sorted(hits.values(), key=lambda h: -h["score"])

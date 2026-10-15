"""Ken Burns effect: animate a still image with smooth zoom/pan.

Local-first path for $animate ("bring this picture to life"). Pure ffmpeg
zoompan — deterministic, zero network, $0, fully private. Renders in
seconds on the VPS. If David wants true generative motion (not just camera
moves), escalate to tools/i2v_client.py.

Any input aspect is handled: the image is scaled to cover 2x the output
size (cropped to the output aspect), giving headroom for zoom and pan
without ever hitting the frame edge.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Optional, Tuple

try:
    from . import local_effects as _le  # reuse probe()/_have()/_run_ffmpeg()
except ImportError:
    import local_effects as _le


# Pan is (dx, dy) as a fraction of the output frame: 0.2 = pan 20% of the
# frame width over the full duration. Positive dx pans right, positive dy
# pans down. Zoom presets hold a fixed zoom > 1.0 so there is room to pan.
PRESETS: dict[str, dict] = {
    "zoom_in":  {"zoom_from": 1.00, "zoom_to": 1.30, "pan": ( 0.00,  0.00)},
    "zoom_out": {"zoom_from": 1.30, "zoom_to": 1.00, "pan": ( 0.00,  0.00)},
    "pan_left":  {"zoom_from": 1.25, "zoom_to": 1.25, "pan": (-0.20,  0.00)},
    "pan_right": {"zoom_from": 1.25, "zoom_to": 1.25, "pan": ( 0.20,  0.00)},
    "pan_up":    {"zoom_from": 1.25, "zoom_to": 1.25, "pan": ( 0.00, -0.20)},
    "pan_down":  {"zoom_from": 1.25, "zoom_to": 1.25, "pan": ( 0.00,  0.20)},
    "slow_push": {"zoom_from": 1.00, "zoom_to": 1.15, "pan": ( 0.00,  0.00)},
}

RESOLUTIONS = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
}


def list_presets() -> list[dict]:
    return [{"name": n, **p} for n, p in PRESETS.items()]


def _resolve_size(resolution: str, input_path: str) -> Tuple[int, int]:
    if resolution in RESOLUTIONS:
        return RESOLUTIONS[resolution]
    if resolution == "source":
        info = _le.probe(input_path)
        w, h = int(info["w"]), int(info["h"])
        # h264 needs even dimensions.
        return (w - w % 2, h - h % 2)
    raise ValueError(
        f"unknown resolution {resolution!r}; "
        f"expected one of {sorted(RESOLUTIONS) + ['source']}"
    )


def build_filter(zoom_from: float, zoom_to: float, pan: Tuple[float, float],
                 n_frames: int, width: int, height: int) -> str:
    """Return the ffmpeg filter chain for the Ken Burns move."""
    px = pan[0] * width
    py = pan[1] * height
    # 2x canvas: scale to cover, crop to output aspect. Headroom for moves.
    pre = (f"scale={2*width}:{2*height}:force_original_aspect_ratio=increase,"
           f"crop={2*width}:{2*height}")
    zp = (
        f"zoompan=z='{zoom_from}+({zoom_to}-{zoom_from})*on/{n_frames}':"
        f"x='iw/2-(iw/zoom/2)+({px})*on/{n_frames}':"
        f"y='ih/2-(ih/zoom/2)+({py})*on/{n_frames}':"
        f"d={n_frames}:s={width}x{height}:fps=30"
    )
    return f"{pre},{zp},format=yuv420p"


def kenburns(
    input_image: str,
    output_video: str,
    duration: float = 5.0,
    zoom_from: float = 1.0,
    zoom_to: float = 1.3,
    pan: Tuple[float, float] = (0.0, 0.0),
    fps: int = 30,
    resolution: str = "1080p",
    preset: Optional[str] = None,
    timeout: int = 300,
) -> str:
    """Animate a still image. Returns output_video.

    preset (one of list_presets()) overrides zoom_from/zoom_to/pan.
    Deterministic: same inputs always produce the same video.
    """
    if not os.path.isfile(input_image):
        raise FileNotFoundError(f"input image not found: {input_image}")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if preset is not None:
        spec = PRESETS.get(preset)
        if spec is None:
            raise ValueError(
                f"unknown preset {preset!r}; expected one of {sorted(PRESETS)}")
        zoom_from, zoom_to, pan = spec["zoom_from"], spec["zoom_to"], spec["pan"]
    if zoom_from < 1.0 or zoom_to < 1.0:
        raise ValueError("zoom values must be >= 1.0")
    if not (0.0 <= abs(pan[0]) <= 0.5 and 0.0 <= abs(pan[1]) <= 0.5):
        raise ValueError("pan components must be in [-0.5, 0.5]")

    width, height = _resolve_size(resolution, input_image)
    n_frames = max(int(round(duration * fps)), 1)
    filt = build_filter(zoom_from, zoom_to, pan, n_frames, width, height)

    cmd = ["ffmpeg", "-hide_banner", "-y",
           "-loop", "1", "-i", input_image,
           "-vf", filt,
           "-frames:v", str(n_frames),
           "-r", str(fps),
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart",
           output_video]
    _le._run_ffmpeg(cmd, timeout=timeout)
    return output_video


def probe_video(path: str) -> dict:
    """ffprobe a rendered video: duration, fps, dimensions."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,duration",
         "-of", "json", path],
        capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out.stderr[:300]}")
    s = json.loads(out.stdout)["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": float(num) / float(den),
            "duration": float(s.get("duration", 0) or 0)}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Ken Burns still-image animation")
    ap.add_argument("input", help="input image path")
    ap.add_argument("output", help="output video path")
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--preset", default=None, choices=sorted(PRESETS))
    ap.add_argument("--zoom-from", type=float, default=1.0)
    ap.add_argument("--zoom-to", type=float, default=1.3)
    ap.add_argument("--pan", type=float, nargs=2, default=(0.0, 0.0),
                    metavar=("DX", "DY"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resolution", default="1080p",
                    choices=sorted(RESOLUTIONS) + ["source"])
    a = ap.parse_args(argv)
    kenburns(a.input, a.output, duration=a.duration, preset=a.preset,
             zoom_from=a.zoom_from, zoom_to=a.zoom_to, pan=tuple(a.pan),
             fps=a.fps, resolution=a.resolution)
    print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

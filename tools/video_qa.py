"""4-tier video QC gates for the MYAI media pipeline.

Each tier is a function returning a JSON-serializable verdict dict:
  {"tier": ..., "pass": bool, "skipped": bool, "details": {...}, "errors": [...]}

The CLI (`python video_qa.py <file> [--reference <ref>]`) exits 0 only if
ALL applicable tiers pass; any failure exits non-zero with the verdict JSON
on stdout. Missing ffmpeg/ffprobe binaries -> tiers that need them are
marked "skipped" (never silently passed).

Tiers:
  0 - structural: ffprobe resolution/codec/fps/duration/audio streams.
      `expect` may also assert fps (within 0.5) and audio_required.
  1 - artifacts: single decode pass with blackdetect/freezedetect/silencedetect,
      plus a volumedetect loudness pass (near-clipping peaks >= -1 dBFS fail).
  2 - metrics: SSIM >= 0.95 / PSNR >= 30 dB vs a reference file (VMAF skipped:
      needs a libvmaf build + heavy CPU; one reference project formally
      removed it for this reason).
  3 - check frames: extract stills at DECISION POINTS (hook, seams, overlays,
      ending) for vision-model review -- never an even grid.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any, Optional


SSIM_MIN = 0.95
PSNR_MIN_DB = 30.0


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def _run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _verdict(tier: str, passed: bool, details: dict, errors: Optional[list] = None,
             skipped: bool = False) -> dict:
    return {
        "tier": tier,
        "pass": bool(passed),
        "skipped": bool(skipped),
        "details": details,
        "errors": errors or [],
    }


# ---------------------------------------------------------------- Tier 0 ---
def tier0_structural(path: str, expect: Optional[dict] = None) -> dict:
    """ffprobe: resolution, codec, fps, duration, audio streams, file sanity."""
    if not os.path.exists(path):
        return _verdict("tier0_structural", False, {"path": path},
                        [f"file not found: {path}"])
    if not _have("ffprobe"):
        return _verdict("tier0_structural", False, {"path": path},
                        ["ffprobe not installed"], skipped=True)
    size = os.path.getsize(path)
    if size == 0:
        return _verdict("tier0_structural", False, {"path": path, "bytes": 0},
                        ["file is empty (0 bytes)"])
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,codec_name,avg_frame_rate,duration",
           "-show_entries", "format=duration,size", "-of", "json", path]
    try:
        proc = _run(cmd, timeout=60)
    except (subprocess.TimeoutExpired, OSError) as e:
        return _verdict("tier0_structural", False, {"path": path}, [f"ffprobe failed: {e}"])
    try:
        info = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return _verdict("tier0_structural", False, {"path": path},
                        [f"ffprobe unparseable output: {proc.stderr[:200]}"])
    streams = info.get("streams", [])
    if not streams:
        return _verdict("tier0_structural", False, {"path": path},
                        ["no video stream found"])
    s = streams[0]
    details: dict[str, Any] = {
        "path": path,
        "bytes": size,
        "width": s.get("width"),
        "height": s.get("height"),
        "codec": s.get("codec_name"),
        "avg_frame_rate": s.get("avg_frame_rate"),
        "duration_s": float(info.get("format", {}).get("duration") or s.get("duration") or 0),
    }
    # audio stream presence
    acmd = ["ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", path]
    try:
        details["has_audio"] = bool(_run(acmd, timeout=30).stdout.strip())
    except (subprocess.TimeoutExpired, OSError):
        details["has_audio"] = None

    errors = []
    if expect:
        for key in ("width", "height", "codec"):
            if key in expect and details.get(key) != expect[key]:
                errors.append(f"{key}: expected {expect[key]!r}, got {details.get(key)!r}")
        if "duration_s" in expect:
            got, want = details["duration_s"], expect["duration_s"]
            if abs(got - want) > 3.0:  # ±3s tolerance
                errors.append(f"duration_s: expected ~{want}, got {got:.2f}")
        if "fps" in expect:
            got_fps = _parse_fps(details.get("avg_frame_rate"))
            want_fps = float(expect["fps"])
            if got_fps is None:
                errors.append("fps: could not parse avg_frame_rate "
                              f"{details.get('avg_frame_rate')!r}")
            elif abs(got_fps - want_fps) > 0.5:
                errors.append(f"fps: expected ~{want_fps}, got {got_fps:.2f}")
        if "audio_required" in expect:
            if expect["audio_required"] and not details.get("has_audio"):
                errors.append("audio_required: no audio stream found")
            if expect["audio_required"] is False and details.get("has_audio"):
                errors.append("audio_required=false: unexpected audio stream found")
    return _verdict("tier0_structural", not errors, details, errors)


def _parse_fps(raw: Any) -> Optional[float]:
    """'30000/1001' -> 29.97 ; '30' -> 30.0 ; None on garbage."""
    if not raw:
        return None
    try:
        if "/" in str(raw):
            num, den = str(raw).split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else None
        return float(raw)
    except (ValueError, ZeroDivisionError):
        return None


# Loudness: peaks at/above -1 dBFS risk clipping on consumer playback.
PEAK_WARN_DB = -1.0


def _loudness(path: str) -> dict:
    """volumedetect pass: max/mean volume in dB. Returns {} if no audio."""
    cmd = ["ffmpeg", "-hide_banner", "-v", "info", "-vn", "-i", path,
           "-af", "volumedetect", "-f", "null", "-"]
    try:
        proc = _run(cmd, timeout=300)
    except (subprocess.TimeoutExpired, OSError):
        return {}
    import re
    stderr = proc.stderr or ""
    if "does not contain any stream" in stderr or "Stream map" in stderr:
        return {}
    out: dict[str, Any] = {}
    m = re.search(r"max_volume:\s*(-?[0-9.]+|n/a)\s*dB", stderr)
    if m and m.group(1) != "n/a":
        out["max_volume_db"] = float(m.group(1))
    m = re.search(r"mean_volume:\s*(-?[0-9.]+|n/a)\s*dB", stderr)
    if m and m.group(1) != "n/a":
        out["mean_volume_db"] = float(m.group(1))
    return out


# ---------------------------------------------------------------- Tier 1 ---
def tier1_artifacts(path: str) -> dict:
    """Single decode pass: blackdetect / freezedetect / silencedetect."""
    if not _have("ffmpeg"):
        return _verdict("tier1_artifacts", False, {"path": path},
                        ["ffmpeg not installed"], skipped=True)
    filt = ("blackdetect=d=0.5:pix_th=0.10,"
            "freezedetect=n=-60dB:d=0.5,"
            "silencedetect=noise=-30dB:d=1.0")
    cmd = ["ffmpeg", "-hide_banner", "-v", "info", "-i", path,
           "-vf", filt, "-af", "silencedetect=noise=-30dB:d=1.0",
           "-f", "null", "-"]
    try:
        proc = _run(cmd, timeout=600)
    except (subprocess.TimeoutExpired, OSError) as e:
        return _verdict("tier1_artifacts", False, {"path": path}, [f"ffmpeg failed: {e}"])
    stderr = proc.stderr or ""
    blacks = stderr.count("black_start")
    freezes = stderr.count("freeze_start")
    silences = stderr.count("silence_start")
    details = {
        "path": path,
        "black_segments": blacks,
        "freeze_segments": freezes,
        "silence_segments": silences,
    }
    errors = []
    if blacks:
        errors.append(f"{blacks} black segment(s) detected")
    if freezes:
        errors.append(f"{freezes} freeze segment(s) detected")
    if silences:
        # silence is a warning, not a failure (dialogue pauses are normal)
        details["silence_note"] = f"{silences} silent segment(s) — review, not fail"
    if proc.returncode != 0 and "Invalid data" in stderr:
        errors.append("decode errors in stream")

    # Loudness: near-clipping peaks are a failure (consumer playback clips).
    loud = _loudness(path)
    details.update(loud)
    peak = loud.get("max_volume_db")
    if peak is not None and peak >= PEAK_WARN_DB:
        errors.append(f"near-clipping peak {peak:.1f} dBFS (>= {PEAK_WARN_DB} dBFS)")
    return _verdict("tier1_artifacts", not errors, details, errors)


# ---------------------------------------------------------------- Tier 2 ---
def tier2_metrics(path: str, reference: str) -> dict:
    """SSIM >= 0.95 and PSNR >= 30 dB vs a reference file.

    VMAF is deliberately skipped: it needs a libvmaf build and heavy CPU,
    and reference projects removed it for exactly that reason.
    """
    if not _have("ffmpeg"):
        return _verdict("tier2_metrics", False, {"path": path},
                        ["ffmpeg not installed"], skipped=True)
    cmd = ["ffmpeg", "-hide_banner", "-v", "info", "-i", path, "-i", reference,
           "-lavfi", "ssim;[0:v][1:v]psnr", "-f", "null", "-"]
    try:
        proc = _run(cmd, timeout=600)
    except (subprocess.TimeoutExpired, OSError) as e:
        return _verdict("tier2_metrics", False, {"path": path}, [f"ffmpeg failed: {e}"])
    stderr = proc.stderr or ""
    import re
    ssim_m = re.search(r"SSIM\s+\w+:\s*([0-9.]+)", stderr)
    psnr_m = re.search(r"PSNR\s+\w+:\s*([0-9.]+|inf)", stderr)
    details: dict[str, Any] = {"path": path, "reference": reference}
    errors = []
    if ssim_m:
        details["ssim_y"] = float(ssim_m.group(1))
        if details["ssim_y"] < SSIM_MIN:
            errors.append(f"SSIM {details['ssim_y']:.4f} < {SSIM_MIN}")
    else:
        errors.append("SSIM not measurable (resolution mismatch?)")
    if psnr_m:
        raw = psnr_m.group(1)
        details["psnr_y_db"] = float("inf") if raw == "inf" else float(raw)
        if details["psnr_y_db"] != float("inf") and details["psnr_y_db"] < PSNR_MIN_DB:
            errors.append(f"PSNR {details['psnr_y_db']:.2f} dB < {PSNR_MIN_DB} dB")
    else:
        errors.append("PSNR not measurable")
    return _verdict("tier2_metrics", not errors, details, errors)


# ---------------------------------------------------------------- Tier 3 ---
def tier3_check_frames(path: str, out_dir: str,
                       decision_points: Optional[list[float]] = None) -> dict:
    """Extract stills at DECISION POINTS for vision-model review.

    decision_points: seconds into the clip (hook, seams, overlays, ending).
    Never an even grid — uniform sampling misses the seams where bad joins
    and clipped captions show.
    """
    if not _have("ffmpeg"):
        return _verdict("tier3_check_frames", False, {"path": path},
                        ["ffmpeg not installed"], skipped=True)
    os.makedirs(out_dir, exist_ok=True)
    points = decision_points or [0.5]  # default: the hook frame
    frames = []
    errors = []
    for i, t in enumerate(points):
        out = os.path.join(out_dir, f"check_{i:02d}_t{t:.1f}s.png")
        cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y",
               "-ss", str(t), "-i", path, "-frames:v", "1", out]
        try:
            proc = _run(cmd, timeout=60)
            if proc.returncode == 0 and os.path.exists(out):
                frames.append(out)
            else:
                errors.append(f"frame at t={t}s failed: {proc.stderr[:120]}")
        except (subprocess.TimeoutExpired, OSError) as e:
            errors.append(f"frame at t={t}s failed: {e}")
    details = {"path": path, "frames": frames,
               "note": ("extracted at decision points (hook, seams, overlays, ending) — "
                        "hand these frames to the vision model with the written rubric")}
    return _verdict("tier3_check_frames", not errors and bool(frames), details, errors)


# ------------------------------------------------------------------ CLI ----
def run_all(path: str, reference: Optional[str] = None,
            expect: Optional[dict] = None,
            decision_points: Optional[list[float]] = None,
            frames_dir: Optional[str] = None) -> dict:
    """Run all applicable tiers; return the full report dict."""
    verdicts = [tier0_structural(path, expect), tier1_artifacts(path)]
    if reference:
        verdicts.append(tier2_metrics(path, reference))
    verdicts.append(tier3_check_frames(
        path, frames_dir or os.path.join(os.path.dirname(path) or ".", "check_frames"),
        decision_points))
    failed = [v for v in verdicts if not v["skipped"] and not v["pass"]]
    return {
        "path": path,
        "overall_pass": not failed,
        "verdicts": verdicts,
    }


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="4-tier video QC gates")
    ap.add_argument("path", help="video file to check")
    ap.add_argument("--reference", help="reference file for tier-2 metrics")
    ap.add_argument("--expect", help="JSON of expected tier-0 specs")
    ap.add_argument("--points", help="comma-separated decision-point seconds for tier 3")
    ap.add_argument("--frames-dir", help="where to write check frames")
    args = ap.parse_args(argv)
    report = run_all(
        args.path,
        reference=args.reference,
        expect=json.loads(args.expect) if args.expect else None,
        decision_points=[float(x) for x in args.points.split(",")] if args.points else None,
        frames_dir=args.frames_dir,
    )
    print(json.dumps(report, indent=2))
    # Skipped tiers (missing ffmpeg) do not fail the run, but are loud in JSON.
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Long-video chaining: beat segmentation + clip chaining + stitching.

David's bar: chained clips MUST look like one continuous video, not clips
stuck together. Three mechanisms enforce this:

1. LAST-FRAME CONDITIONING — the EXACT last frame of clip N (frame-count
   precise via video_edit.extract_last_frame) becomes the first-frame
   image input of clip N+1. No gaps, no jumps.
2. PROMPT CONTINUITY — every chained prompt carries the same subject /
   clothing / lighting / camera context, plus a "seamless continuation"
   tail so the model doesn't reinterpret the scene.
3. OVERLAP + XFADE — optional 1-2s overlap between clips, crossfaded with
   ffmpeg xfade so join points are invisible in normal playback.

Plus: PROMPT SEGMENTATION — the full prompt is a narrative arc, not
repeated per clip. segment_prompt() splits it into N sequential beats;
the UI shows the beats for review/edit before generating.

Chain modes:
  lastframe  extract last frame -> I2V next clip (works with any I2V
             model; default with the uncensored Wan 2.7 line)
  extend     Seedance native Extend workflow (reference_video_urls +
             omni_reference_task_type="extend"); designed for this, but
             public Seedance may reject clips with detectable persons.

No prompt filtering anywhere — prompts pass through exactly as written.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from typing import Any, Callable, Optional


# ---------------------------------------------------------------- quality ----

QUALITY_LEVELS = {
    "draft": {
        "label": "Draft",
        "resolution": "480p",
        "crf": 23,
        "preset": "veryfast",
        "description": "Fast + cheap. For iterating on prompts.",
    },
    "standard": {
        "label": "Standard",
        "resolution": "720p",
        "crf": 20,
        "preset": "medium",
        "description": "Balanced quality and cost.",
    },
    "maximum": {
        "label": "Maximum",
        "resolution": "1080p",
        "crf": 18,
        "preset": "slow",
        "description": "Best quality. Pro output.",
    },
}
DEFAULT_QUALITY = "standard"

# Continuity tail appended to every chained beat after the first.
CONTINUITY_TAIL = (
    " STRICT CONTINUITY LOCK: This is a direct continuation of the exact same"
    " shot. Preserve with pixel-level fidelity: the SAME person's face, facial"
    " features, skin tone, hair style and color; the SAME clothing, colors and"
    " textures; the SAME body type and proportions; the SAME room, background,"
    " objects, lighting direction and color temperature; the SAME camera angle"
    " and framing as the reference image. Do NOT redesign, beautify, or"
    " reinterpret anything. Do NOT change the person's appearance in any way."
)

# Identity anchor for the first clip (its input IS the reference image).
IDENTITY_ANCHOR = (
    " The person on screen is EXACTLY the person in the starting image —"
    " identical face, hair, body type and outfit. Do not change their"
    " appearance in any way."
)

# Anti-loop negative prompt sent with every chained clip. Generative
# models fall into looping motion when unsteered; naming the failure
# modes in the negative prompt measurably suppresses them.
CHAIN_NEGATIVE_PROMPT = (
    "looping motion, repeating the same movement, cyclic motion, "
    "static pose, frozen frame, no movement, rocking back and forth, "
    "morphing face, changing identity, different person, face change, "
    "wardrobe change, extra limbs, distorted hands"
)

# Escalation appended when QC rejects a clip for looping / not advancing.
_QC_ESCALATION = (
    " CRITICAL CORRECTION: the previous attempt was rejected because the"
    " motion looped or repeated the earlier action instead of advancing."
    " This time the action MUST visibly progress from its start to a"
    " clearly different end position. No repeated or cyclic movement."
)


# ------------------------------------------------------- prompt beats ----

_BEAT_SPLIT = re.compile(
    r"\s+then\s+|\s+and then\s+|\s*;\s*|(?<=[.!?])\s+"
)


def segment_prompt(prompt: str, n_beats: int) -> list[str]:
    """Split a full prompt into n_beats sequential narrative beats.

    Heuristic: split on "then"/"and then"/";"/sentence boundaries, then
    distribute the pieces across beats. If fewer pieces than beats, split
    the longest pieces on commas. Beats never repeat the whole prompt —
    each advances the story.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt is empty")
    n_beats = max(1, int(n_beats))

    pieces = [p.strip(" ,") for p in _BEAT_SPLIT.split(prompt) if p.strip(" ,")]
    if not pieces:
        pieces = [prompt]

    # Too few pieces: subdivide the longest on commas.
    while len(pieces) < n_beats:
        idx = max(range(len(pieces)), key=lambda i: len(pieces[i]))
        sub = [s.strip(" ,") for s in pieces[idx].split(",") if s.strip(" ,")]
        if len(sub) < 2:
            break  # can't subdivide further
        pieces[idx:idx + 1] = sub

    # Distribute pieces across beats as evenly as possible.
    beats: list[str] = []
    if len(pieces) <= n_beats:
        beats = pieces + [""] * (n_beats - len(pieces))
    else:
        per, extra = divmod(len(pieces), n_beats)
        i = 0
        for b in range(n_beats):
            take = per + (1 if b < extra else 0)
            beats.append(" ".join(pieces[i:i + take]))
            i += take

    # Fill any empty beats by developing the previous beat further.
    # (The old filler repeated the beat + "(continues)", which taught
    # the model to loop the same motion — the exact failure David hit.)
    for b in range(n_beats):
        if not beats[b].strip():
            prev = beats[b - 1] if b else prompt
            beats[b] = (prev + " — the action develops further with new,"
                        " non-repeating motion")
    return [b.strip() for b in beats]


def beat_prompt(beat: str, idx: int) -> str:
    """Build a clip prompt that the model can actually follow.

    Structure matters for prompt adherence:
      1. the action, stated first and as a complete performance with a
         defined end state ("ends with ...") — models follow concrete
         end states far better than open-ended motion descriptions;
      2. an explicit no-loop instruction;
      3. identity/continuity lock (anchor on clip 1, full continuity
         lock on later clips).
    """
    beat = beat.strip()
    action = (
        f"Perform this action: {beat}. Carry it through from its natural"
        " beginning to a completed end state — the clip finishes with the"
        " action done and the subject settled in the resulting position."
        " One continuous performance: never loop, repeat, or cycle any"
        " movement within the clip."
    )
    if idx == 0:
        return action + IDENTITY_ANCHOR
    return action + CONTINUITY_TAIL


# ------------------------------------------------------------ planning ----

def plan_chain(target_duration_s: float, clip_duration_s: float,
               overlap_s: float = 0.0) -> dict:
    """How many clips for the target duration.

    With overlap o and clip length c, n clips cover n*c - (n-1)*o.
    Returns {n_clips, clip_duration_s, overlap_s, total_s}.
    """
    target = max(float(target_duration_s), 1.0)
    clip = max(float(clip_duration_s), 1.0)
    overlap = min(max(float(overlap_s), 0.0), clip * 0.5)
    step = clip - overlap
    n = max(1, math.ceil((target - overlap) / step))
    total = n * clip - (n - 1) * overlap
    return {"n_clips": n, "clip_duration_s": clip,
            "overlap_s": overlap, "total_s": round(total, 2)}


def estimate_cost_usd(n_clips: int, clip_duration_s: float,
                      cost_per_s: float) -> float:
    return round(n_clips * clip_duration_s * cost_per_s, 2)


# ------------------------------------------------------------ stitching ----

def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def _probe(path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,avg_frame_rate",
           "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    info = json.loads(out.stdout or "{}")
    s = (info.get("streams") or [{}])[0]
    try:
        num, den = s.get("avg_frame_rate", "30/1").split("/")
        fps = float(num) / float(den) if float(den) else 30.0
    except Exception:
        fps = 30.0
    return {"w": int(s.get("width") or 1280),
            "h": int(s.get("height") or 720), "fps": fps}


# ------------------------------------------------------- anti-loop QC ----
#
# Generative clips fail in two measurable ways: (1) static / internally
# looping — first and last frames nearly identical; (2) not advancing —
# the clip ends where the previous clip ended. Both are detectable by
# comparing tiny grayscale frame signatures, no imaging library needed
# (ffmpeg emits raw gray bytes; the comparison is pure Python).

_SIG_SIZE = 32  # signature frames are _SIG_SIZE x _SIG_SIZE gray bytes


def _signature_from_seek(video_path: str, seek: list) -> Optional[bytes]:
    """32x32 grayscale signature of the frame at a seek position."""
    if not _have("ffmpeg"):
        return None
    cmd = (["ffmpeg", "-v", "error"] + seek + ["-i", video_path,
           "-frames:v", "1", "-vf", f"scale={_SIG_SIZE}:{_SIG_SIZE}",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=60)
    except Exception:
        return None
    data = out.stdout or b""
    if len(data) < _SIG_SIZE * _SIG_SIZE:
        return None
    return data[:_SIG_SIZE * _SIG_SIZE]


def _gray_signature(video_path: str, which: str) -> Optional[bytes]:
    """32x32 grayscale signature of the first or last frame, or None."""
    seek = ["-ss", "0"] if which == "first" else ["-sseof", "-0.25"]
    return _signature_from_seek(video_path, seek)


def _probe_duration_s(video_path: str) -> Optional[float]:
    """Clip duration in seconds via ffprobe, or None."""
    if not _have("ffprobe"):
        return None
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "csv=p=0", video_path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return float((out.stdout or "").strip())
    except Exception:
        return None


def _gray_signature_at(video_path: str, frac: float) -> Optional[bytes]:
    """32x32 grayscale signature of the frame at fraction frac (0..1)."""
    if frac <= 0.0:
        return _gray_signature(video_path, "first")
    if frac >= 1.0:
        return _gray_signature(video_path, "last")
    dur = _probe_duration_s(video_path)
    if not dur or dur <= 0:
        return None
    t = min(max(dur * frac, 0.0), max(dur - 0.05, 0.0))
    return _signature_from_seek(video_path, ["-ss", f"{t:.3f}"])


def _signature_diff(a: Optional[bytes], b: Optional[bytes]) -> Optional[float]:
    """Mean absolute per-pixel difference (0..255), or None if unknown."""
    if not a or not b or len(a) != len(b):
        return None
    total = 0
    for x, y in zip(a, b):
        total += x - y if x >= y else y - x
    return total / len(a)


# Below these mean-abs-diff values a clip is treated as looping/static
# (motion) or as not having advanced from the previous clip (advance).
# Deliberately conservative: a false rejection costs a paid re-render.
QC_MIN_MOTION = 1.5
QC_MIN_ADVANCE = 2.5
# Two frames this close (mean abs diff, 0..255) are effectively the same
# frame. At NON-adjacent points of a clip that means the generator
# looped back to an earlier composition mid-clip — a failure the
# first-vs-last check cannot see when the clip still ends somewhere new.
QC_LOOP_DIFF = 0.8


def _clip_qc(video_path: str, prev_last_sig: Optional[bytes]) -> tuple:
    """Check a downloaded clip. Returns (ok, reason, last_sig).

    Fail-open: when signatures can't be computed (no ffmpeg, odd file),
    the clip is accepted — QC must never break a chain it can't judge.
    """
    first_sig = _gray_signature(video_path, "first")
    last_sig = _gray_signature(video_path, "last")
    motion = _signature_diff(first_sig, last_sig)
    if motion is not None and motion < QC_MIN_MOTION:
        return False, "static/looping clip (no visible motion)", last_sig
    # Internal-loop check: sample the interior and look for a near-exact
    # repeat of a frame from a distant earlier point. Adjacent samples
    # are exempt — a brief hold is legitimate; a distant repeat is a loop.
    samples = [first_sig,
               _gray_signature_at(video_path, 0.25),
               _gray_signature_at(video_path, 0.5),
               _gray_signature_at(video_path, 0.75),
               last_sig]
    for a in range(len(samples)):
        for b in range(a + 2, len(samples)):
            d = _signature_diff(samples[a], samples[b])
            if d is not None and d < QC_LOOP_DIFF:
                return False, "clip loops back on itself mid-clip", last_sig
    advance = _signature_diff(prev_last_sig, last_sig)
    if advance is not None and advance < QC_MIN_ADVANCE:
        return False, "clip ends where the previous clip ended", last_sig
    return True, "", last_sig


def stitch_clips(clip_paths: list[str], out_path: str,
                 overlap_s: float = 0.0, crf: int = 20,
                 preset: str = "medium") -> str:
    """Stitch clips into one continuous MP4.

    overlap_s = 0  -> concat demuxer (hard cuts, exact).
    overlap_s > 0  -> normalize + chained xfade crossfades over the
                      overlap region (join points invisible).

    Re-encodes at the given quality (David: high quality, crf 18-20).
    """
    if not _have("ffmpeg"):
        raise RuntimeError("ffmpeg not installed")
    if len(clip_paths) < 1:
        raise ValueError("stitch_clips needs at least 1 clip")
    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], out_path)
        return out_path

    if overlap_s <= 0:
        list_path = out_path + ".filelist.txt"
        try:
            with open(list_path, "w") as f:
                for c in clip_paths:
                    f.write("file '" + c.replace("'", "'\\''") + "'\n")
            # Re-encode for uniform params (clips come from the same
            # model but concatenation must be sample-exact).
            cmd = ["ffmpeg", "-hide_banner", "-y", "-f", "concat",
                   "-safe", "0", "-i", list_path,
                   "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
                   "-pix_fmt", "yuv420p", "-c:a", "aac",
                   "-movflags", "+faststart", out_path]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=3600)
            if proc.returncode != 0:
                raise RuntimeError(f"stitch failed: {proc.stderr[-800:]}")
        finally:
            try:
                os.remove(list_path)
            except OSError:
                pass
        return out_path

    # Overlap path: normalize all clips, then chained xfade.
    ref = _probe(clip_paths[0])
    w, h, fps = ref["w"], ref["h"], ref["fps"]
    norm = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={fps:.2f},format=yuv420p")
    # Per-clip durations for xfade offsets.
    durs = []
    for c in clip_paths:
        cmd = ["ffprobe", "-v", "error", "-show_entries",
               "format=duration", "-of", "csv=p=0", c]
        try:
            durs.append(float(subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=30).stdout.strip() or 0))
        except Exception:
            durs.append(0.0)

    n = len(clip_paths)
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    for c in clip_paths:
        cmd += ["-i", c]
    filt = "".join(f"[{i}:v]{norm}[v{i}];" for i in range(n))
    # chained xfade: offset_k = sum(durs[0..k]) - overlap*(k+1)
    acc = 0.0
    cur = "[v0]"
    for k in range(1, n):
        acc += durs[k - 1]
        off = max(acc - overlap_s * k, 0.0)
        nxt = f"[x{k}]"
        filt += (f"{cur}[v{k}]xfade=transition=fade:"
                 f"duration={overlap_s:.2f}:offset={off:.2f}{nxt};")
        cur = nxt
    filt = filt.rstrip(";")
    cmd += ["-filter_complex", filt, "-map", cur,
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"xfade stitch failed: {proc.stderr[-800:]}")
    return out_path


# --------------------------------------------------------------- chain ----

def run_chain(
    *,
    beats: list[str],
    clip_duration_s: float,
    chain_mode: str,
    submit_clip: Callable[[dict], Any],
    wait_clip: Callable[[Any], str],
    download_clip: Callable[[str, str], str],
    extract_frame_fn: Callable[[str, str], str],
    file_to_url_fn: Callable[[str], str],
    workdir: str,
    ref_image_path: Optional[str] = None,
    ref_video_path: Optional[str] = None,
    i2v_model_id: str = "venice-wan27-uncensored",
    v2v_model_id: str = "venice-wan27-v2v",
    extend_model_id: str = "venice-seedance-r2v",
    resolution: str = "720p",
    overlap_s: float = 0.0,
    crf: int = 20,
    preset: str = "medium",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    clip_lock_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """Generate N chained clips and stitch them. Returns final MP4 path.

    submit_clip(args) -> job handle; args: {prompt, duration, resolution,
      image_url?, video_url?, reference_video_urls?, task_type?}.
    wait_clip(job) -> result video URL.
    download_clip(url, dest) -> local path.
    extract_frame_fn(video, jpg) -> jpg path (exact last frame).
    file_to_url_fn(path) -> data: URL.
    clip_lock_fn(path) -> None: optional in-place transform applied to
      each accepted clip BEFORE it seeds the next clip (the Studio passes
      a face lock, so last-frame conditioning starts from the reference's
      real face instead of the generator's drifted one). A lock failure
      is reported via progress_cb and never kills the chain.

    Prompts pass through exactly as written — no filtering.
    """
    os.makedirs(workdir, exist_ok=True)
    n = len(beats)
    if n < 1:
        raise ValueError("run_chain needs at least 1 beat")
    if chain_mode not in ("lastframe", "extend"):
        raise ValueError(f"unknown chain_mode {chain_mode!r}")

    def progress(i: int, phase: str):
        if progress_cb:
            progress_cb(i, n, phase)

    def _apply_clip_lock(path: str, i: int):
        """Run the caller's per-clip lock in place, best-effort.

        The clip is already paid for, so a lock failure is surfaced as
        a progress warning only — never a chain failure.
        """
        if clip_lock_fn is None:
            return
        try:
            clip_lock_fn(path)
        except Exception as e:
            progress(i, f"clip {i + 1}/{n}: per-clip lock failed "
                        f"({e}) — continuing unlocked")

    clips: list[str] = []
    prev_last_sig: Optional[bytes] = None
    for i, beat in enumerate(beats):
        dest = os.path.join(workdir, f"clip_{i:02d}.mp4")
        # Checkpoint/resume: skip clips already completed in a prior run.
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            _apply_clip_lock(dest, i)
            clips.append(dest)
            # Keep the advance-check baseline current for the next clip.
            prev_last_sig = _gray_signature(dest, "last")
            progress(i, f"clip {i + 1}/{n} resumed from checkpoint")
            continue
        progress(i, f"clip {i + 1}/{n}")
        prompt = beat_prompt(beat, i)
        args: dict = {"prompt": prompt, "duration": clip_duration_s,
                      "resolution": resolution,
                      "negative_prompt": CHAIN_NEGATIVE_PROMPT}
        if i == 0:
            if ref_image_path:
                args["model_id"] = i2v_model_id
                args["image_url"] = file_to_url_fn(ref_image_path)
            elif ref_video_path:
                # V2V proved unreliable (422s, hangs) — extract first frame
                # and use I2V instead. Sacrifices source motion for reliability.
                from video_edit import extract_frame_at as _efa
                first_frame = os.path.join(workdir, "ref_first_frame.jpg")
                _efa(ref_video_path, first_frame, at_s=0.0)
                args["model_id"] = i2v_model_id
                args["image_url"] = file_to_url_fn(first_frame)
            else:
                raise ValueError("long video needs a reference image or video")
        elif chain_mode == "extend":
            args["model_id"] = extend_model_id
            args["reference_video_urls"] = [file_to_url_fn(clips[-1])]
            args["task_type"] = "extend"
            args["aspect_ratio"] = "adaptive"
        else:  # lastframe
            frame_jpg = os.path.join(workdir, f"lastframe_{i:02d}.jpg")
            extract_frame_fn(clips[-1], frame_jpg)
            args["model_id"] = i2v_model_id
            args["image_url"] = file_to_url_fn(frame_jpg)

        # Per-clip retry: transient failures retry the clip, not the chain.
        # A downloaded clip also goes through anti-loop QC; a clip that
        # loops or fails to advance the action is retried with an
        # escalated prompt. On the final attempt the clip is accepted
        # regardless — a flawed clip beats a dead chain.
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                job = submit_clip(dict(args))  # copy: submit pops keys
                url = wait_clip(job)
                download_clip(url, dest)
                ok, reason, last_sig = _clip_qc(dest, prev_last_sig)
                if ok or attempt == 2:
                    if not ok:
                        progress(i, f"clip {i + 1}/{n}: QC warning "
                                    f"({reason}) — accepted on final try")
                    prev_last_sig = last_sig
                    last_err = None
                    break
                # QC rejected: escalate the prompt and regenerate.
                progress(i, f"clip {i + 1}/{n}: redoing — {reason}")
                args["prompt"] = beat_prompt(beat, i) + _QC_ESCALATION
                try:
                    os.remove(dest)
                except OSError:
                    pass
                last_err = None
                continue
            except (InterruptedError, KeyboardInterrupt):
                raise
            except Exception as e:
                last_err = e
                # Clean up partial download before retry.
                try:
                    if os.path.isfile(dest):
                        os.remove(dest)
                except OSError:
                    pass
                if attempt < 2:
                    import time as _t
                    _t.sleep(5 * (attempt + 1))
        if last_err is not None:
            raise RuntimeError(
                f"clip {i + 1}/{n} failed after 3 attempts: {last_err}") from last_err
        _apply_clip_lock(dest, i)
        if clip_lock_fn is not None:
            # The lock changed the clip's frames; keep the advance
            # baseline describing the file the next clip builds on.
            prev_last_sig = _gray_signature(dest, "last") or prev_last_sig
        clips.append(dest)
        progress(i, f"clip {i + 1}/{n} done")

    progress(n - 1, "stitching")
    final = os.path.join(workdir, "longvideo_final.mp4")
    stitch_clips(clips, final, overlap_s=overlap_s, crf=crf, preset=preset)
    progress(n - 1, "done")
    return final


# ------------------------------------------------------------------ CLI ----

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Long-video chaining helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sg = sub.add_parser("segment", help="split prompt into beats")
    sg.add_argument("prompt")
    sg.add_argument("--beats", type=int, default=4)
    pl = sub.add_parser("plan", help="plan clip count")
    pl.add_argument("--target", type=float, required=True)
    pl.add_argument("--clip", type=float, default=15.0)
    pl.add_argument("--overlap", type=float, default=0.0)
    st = sub.add_parser("stitch", help="stitch clips")
    st.add_argument("--out", required=True)
    st.add_argument("--overlap", type=float, default=0.0)
    st.add_argument("clips", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "segment":
        print(json.dumps(segment_prompt(args.prompt, args.beats), indent=2))
    elif args.cmd == "plan":
        print(json.dumps(plan_chain(args.target, args.clip,
                                    args.overlap), indent=2))
    elif args.cmd == "stitch":
        out = stitch_clips(args.clips, args.out, overlap_s=args.overlap)
        print(json.dumps({"output": out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

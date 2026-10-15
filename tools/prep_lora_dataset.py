#!/usr/bin/env python3
"""Build a person-LoRA training dataset for MYAI (runs on David's
server, next to his media library — his files never leave it).

Recipe (research sweep, Oct 15): 15-40 varied images of the model,
captions carrying a trigger token + the generic class only. Frames
are sampled from her reference video (varied poses/expressions come
free from video) and can be topped up with still photos.

Usage:
    python3 prep_lora_dataset.py --video 4707.mp4 --out ~/lora_dataset
    python3 prep_lora_dataset.py --video 4707.mp4 --photos ~/refs --out ~/lora_dataset

Output: out/img_000.jpg + img_000.txt pairs, ready to upload to the
pod for the SCAIL-2 musubi-tuner fork.
"""
import argparse
import os
import subprocess
import sys

TRIGGER = "skswoman"
CAPTION = f"{TRIGGER}, a woman"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{r.stderr[-400:]}")
    return r


def duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float((r.stdout or "0").strip() or 0)
    except ValueError:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="reference video of the model")
    ap.add_argument("--photos", default="", help="folder of reference photos (optional)")
    ap.add_argument("--out", required=True, help="dataset output folder")
    ap.add_argument("--count", type=int, default=28,
                    help="target frame count from the video (15-40)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dur = duration(args.video)
    if dur <= 0:
        print("could not probe the video duration", file=sys.stderr)
        return 1
    idx = 0

    def emit(src_args, label):
        nonlocal idx
        img = os.path.join(args.out, f"img_{idx:03d}.jpg")
        cmd = ["ffmpeg", "-y", "-v", "error", *src_args,
               "-vf", "scale=768:-2", "-frames:v", "1", "-q:v", "3", img]
        run(cmd)
        with open(os.path.join(args.out, f"img_{idx:03d}.txt"), "w") as fh:
            fh.write(CAPTION + "\n")
        idx += 1
        print(f"  {label} -> img_{idx - 1:03d}.jpg")

    # Evenly spaced frames across the whole video (skip head/tail 5%).
    print(f"video {dur:.1f}s -> {args.count} frames")
    for i in range(args.count):
        t = dur * (0.05 + 0.9 * (i / max(1, args.count - 1)))
        emit(["-ss", f"{t:.2f}", "-i", args.video], f"frame @ {t:.1f}s")

    if args.photos and os.path.isdir(args.photos):
        for name in sorted(os.listdir(args.photos)):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                emit(["-i", os.path.join(args.photos, name)], f"photo {name}")

    print(f"dataset ready: {idx} images in {args.out} "
          f"(trigger '{TRIGGER}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

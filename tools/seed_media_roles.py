#!/usr/bin/env python3
"""Seed <UPLOAD_DIR>/.media_roles.json from the library records.

Roles reflect David's judgments (Oct 18, 2026; see library_build/
JUDGMENTS.txt on the server). Run on the server:
    python3 seed_media_roles.py /home/nanobot/uploads
Idempotent; only assigns roles to files that exist in the uploads dir.
"""
import json
import os
import sys

FIXED = {
    "faceref_5785_neutral.jpeg": ("face_ref", "Face reference — lock source"),
    "facecand_5785.png": ("face_ref", "Face reference — identity support"),
    "template_5812.png": ("template", "Template — scene & body"),
    "template_5812_crop.png": ("template", "Template — scene & body (cropped)"),
    "5766.png": ("body_ref", "Body reference — primary full-body"),
    "5760.png": ("body_ref", "Body reference"),
    "5761.png": ("body_ref", "Body reference"),
    "5762.png": ("still360", "360 still — ungraded"),
    "5763.png": ("still360", "360 still — ungraded"),
    "5764.png": ("still360", "360 still — ungraded"),
    "5769.png": ("still360", "360 still — ungraded"),
    "retest_0054_locked.mp4": ("result", "Result — job-0054 face-locked"),
    "lib_5765_f_0012.jpg": ("excluded", "Excluded — face drift"),
}
UNGRADED_PREFIXES = ("lib_4707_", "lib_4700-1_", "lib_4701_")


def main() -> int:
    upload_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/nanobot/uploads"
    roles = {}
    for name in sorted(os.listdir(upload_dir)):
        if name in FIXED:
            role, label = FIXED[name]
            roles[name] = {"role": role, "label": label}
        elif name.startswith(UNGRADED_PREFIXES):
            roles[name] = {"role": "frame_ungraded",
                           "label": "Frame — ungraded"}
        elif name.startswith("lib_"):
            roles[name] = {"role": "graded_frame",
                           "label": "Graded frame — approved"}
    out = os.path.join(upload_dir, ".media_roles.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(roles, f, indent=1, sort_keys=True)
    print(f"wrote {out}: {len(roles)} roles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

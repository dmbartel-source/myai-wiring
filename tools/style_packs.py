"""Style packs: curated multi-effect chains for local media.

A style pack is a named, ordered list of (recipe, params) applied in
sequence through tools.local_effects.apply(). Packs answer the question
"make it look like X" with one name instead of hand-assembling chains.

`match()` maps a plain-English description to the best pack.
`apply_pack()` runs the chain with temp-file intermediates.

All packs reference only recipes registered in local_effects.RECIPES;
unknown names fail loudly at import time, not at render time.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

try:
    from . import local_effects as le  # package import
except ImportError:  # pragma: no cover - direct tools/ sys.path usage
    import local_effects as le


STYLE_PACKS: dict[str, dict] = {
    "vhs": {
        "blurb": "VHS tape: tracking glitches, heavy grain, faded color, soft vignette.",
        "effects": [
            ("glitch_cut", {"period": 3.0, "burst": 0.12}),
            ("film_grain", {"strength": 18.0}),
            ("color_grade", {"preset": "bleach"}),
            ("vignette", {"strength": 0.45}),
        ],
        "keywords": ("vhs", "vhs tape", "80s video", "90s video", "tracking",
                     "retro tv", "old videotape", "camcorder"),
    },
    "vintage_film": {
        "blurb": "Old cinema: animated grain, warm grade, light leaks, deep vignette.",
        "effects": [
            ("film_grain", {"strength": 14.0}),
            ("color_grade", {"preset": "warm"}),
            ("light_leak", {"period": 7.0, "opacity": 0.25}),
            ("vignette", {"strength": 0.6}),
        ],
        "keywords": ("vintage film", "old film", "8mm", "16mm", "silent film",
                     "1940s", "1950s", "1960s", "retro film", "celluloid"),
    },
    "noir": {
        "blurb": "Film noir: black-and-white, crushed contrast, grain, vignette.",
        "effects": [
            ("color_grade", {"preset": "noir"}),
            ("film_grain", {"strength": 10.0}),
            ("vignette", {"strength": 0.65}),
        ],
        "keywords": ("noir", "film noir", "black and white", "black-and-white",
                     "b&w", "monochrome", "detective", "shadowy"),
    },
    "cyberpunk": {
        "blurb": "Cyberpunk: digital glitches, teal-orange grade, neon light leaks.",
        "effects": [
            ("glitch_cut", {"period": 1.6, "burst": 0.2}),
            ("color_grade", {"preset": "teal_orange"}),
            ("light_leak", {"period": 4.0, "opacity": 0.3}),
        ],
        "keywords": ("cyberpunk", "neon", "synthwave", "retrowave", "digital",
                     "hacker", "futuristic glitch", "blade runner"),
    },
    "cinematic": {
        "blurb": "Modern cinema: letterbox, teal-orange grade, gentle vignette, fades.",
        "effects": [
            ("letterbox", {"ratio": 2.39}),
            ("color_grade", {"preset": "teal_orange"}),
            ("vignette", {"strength": 0.35}),
            ("fade_in_out", {"duration": 0.6}),
        ],
        "keywords": ("cinematic", "movie look", "film look", "letterbox",
                     "anamorphic", "blockbuster", "epic"),
    },
    "dreamy": {
        "blurb": "Dreamy: warm glow, soft light leaks, gentle vignette.",
        "effects": [
            ("light_leak", {"period": 5.0, "opacity": 0.4}),
            ("color_grade", {"preset": "warm"}),
            ("vignette", {"strength": 0.3}),
        ],
        "keywords": ("dreamy", "dreamlike", "ethereal", "soft glow", "romantic",
                     "hazy", "nostalgic glow", "golden hour"),
    },
    "action": {
        "blurb": "Action: impact shake, crash zoom, cold grade.",
        "effects": [
            ("shake", {"amplitude": 10.0, "freq": 7.0}),
            ("zoom_punch", {"amount": 0.22, "at": 1.0, "width": 0.5}),
            ("color_grade", {"preset": "cold"}),
        ],
        "keywords": ("action", "intense", "impact", "explosive", "adrenaline",
                     "fight", "chase", "punchy"),
    },
    "music_video": {
        "blurb": "Music video: rhythmic zoom punches, subtle glitch, bold grade.",
        "effects": [
            ("zoom_punch", {"amount": 0.18, "at": 0.8, "width": 0.4}),
            ("glitch_cut", {"period": 4.0, "burst": 0.1}),
            ("color_grade", {"preset": "teal_orange"}),
        ],
        "keywords": ("music video", "mtv", "concert", "performance", "rap video",
                     "pop video", "stage"),
    },
}


# Fail fast if a pack references a recipe that does not exist.
for _pack_name, _pack in STYLE_PACKS.items():
    for _recipe, _params in _pack["effects"]:
        if _recipe not in le.RECIPES:
            raise ImportError(
                f"style pack {_pack_name!r} references unknown recipe {_recipe!r}"
            )
        if le.RECIPES[_recipe]["two_inputs"]:
            raise ImportError(
                f"style pack {_pack_name!r} uses two-input recipe {_recipe!r}; "
                "packs are single-input chains only"
            )


def list_packs() -> list[dict]:
    """Return pack metadata: name, blurb, effect chain."""
    return [
        {"name": name, "blurb": p["blurb"],
         "effects": [{"recipe": r, "params": pr} for r, pr in p["effects"]]}
        for name, p in STYLE_PACKS.items()
    ]


def match(description: str) -> Optional[dict]:
    """Map a plain description to the best style pack.

    Returns {pack, score, why, blurb} or None when nothing matches.
    Also falls back to single-recipe suggestions from local_effects.
    """
    desc = description.lower()
    best: Optional[dict] = None
    for name, pack in STYLE_PACKS.items():
        for kw in pack["keywords"]:
            if kw in desc:
                # Longer keyword = more specific = higher score.
                score = 0.7 + min(len(kw) / 40.0, 0.25)
                if best is None or score > best["score"]:
                    best = {"pack": name, "score": round(score, 2),
                            "why": f"matched {kw!r}", "blurb": pack["blurb"]}
                break
    return best


def apply_pack(input_path: str, output_path: str, pack_name: str,
               timeout: int = 900) -> str:
    """Apply a style pack chain. Returns output_path."""
    pack = STYLE_PACKS.get(pack_name)
    if pack is None:
        raise ValueError(f"unknown style pack {pack_name!r}; "
                         f"expected one of {sorted(STYLE_PACKS)}")
    effects = pack["effects"]
    if not effects:
        raise ValueError(f"style pack {pack_name!r} has no effects")
    tmpdir = tempfile.mkdtemp(prefix="stylepack_")
    try:
        current = input_path
        for i, (recipe, params) in enumerate(effects):
            is_last = (i == len(effects) - 1)
            dest = output_path if is_last else os.path.join(
                tmpdir, f"step{i:02d}.mp4")
            le.apply(current, dest, recipe, params, timeout=timeout)
            current = dest
    finally:
        for f in os.listdir(tmpdir):
            try:
                os.remove(os.path.join(tmpdir, f))
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
    return output_path


def describe(description: str) -> dict:
    """One-call helper: match a description, return pack or recipe fallback.

    Returns {"kind": "pack", ...} or {"kind": "recipes", ...} or
    {"kind": "none"}.
    """
    hit = match(description)
    if hit:
        return {"kind": "pack", **hit}
    recipes = le.suggest_recipes(description)
    if recipes:
        return {"kind": "recipes", "suggestions": recipes[:3]}
    return {"kind": "none"}

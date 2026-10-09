"""Prompt director: effect description -> engineered video-generation prompt.

Pure functions, no network, no API keys. Implements the prompt-engineering
patterns from the research:
  - 6-part formula: [SUBJECT] + [ACTION] + [ENVIRONMENT] + [CAMERA] + [STYLE]
    + [CONSTRAINTS], 60-100 words; first 20-30 words carry subject + action.
  - Direct-don't-describe: what changes over time + how the camera observes.
  - Camera vocabulary with one camera move per clip.
  - Negative prompts are load-bearing in video: 8-12 targeted terms.
  - Seed management: uint32 seeds + a stable house seed for series.
  - I2V-from-generated-still guidance for exact compositional control.
  - Model-specific builders (fal.ai research, Oct 2026):
      * Seedance: subject+action -> camera -> sound cues; one action and one
        camera move per shot; 2-4 sentences.
      * Wan 2.5: grammatical sentences, Subject -> Environment -> Lighting
        -> Camera -> Action; 100-150 words.
  - I2V fidelity: motion-only prompts (the still holds identity — never
    re-describe face/body/clothing/background); fidelity locks appended;
    optional Lock Block pasted byte-identical for cross-clip consistency.

No content filtering anywhere: prompts pass through exactly as specified.
Fidelity locks are visual-consistency guards, not content restrictions.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, asdict


# House seed: stable bias toward a consistent palette/lighting for shots
# that must cut together. Same seed != identical output, but keeps
# generations in the same neighborhood.
HOUSE_SEED = 20261007

CAMERA_MOVES = {
    "push-in": "slow push-in toward the subject",
    "pull-back": "slow pull-back revealing the environment",
    "tracking-left": "smooth tracking shot moving left",
    "tracking-right": "smooth tracking shot moving right",
    "crane-up": "crane shot rising above the scene",
    "orbit": "slow 360-degree orbit around the subject",
    "whip-pan": "fast whip pan across the scene",
    "static": "static locked-off camera",
    "steadicam": "steadicam follow shot",
    "handheld": "subtle handheld camera movement",
    "gimbal": "smooth gimbal glide",
    "rack-focus": "rack focus from foreground to background",
    "fpv": "continuous FPV long take",
    "worms-eye": "dramatic worm's-eye view",
    "birds-eye": "bird's-eye view looking straight down",
}

BASE_NEGATIVE = [
    "morphing face", "extra limbs", "extra fingers", "flickering",
    "warping background", "distorted hands", "text", "watermark",
    "sudden scene change", "deformed body", "disappearing objects",
]

# Extra negative terms keyed by common failure mode.
NEGATIVE_EXTENSIONS = {
    "human": ["cloned face", "asymmetric eyes"],
    "fire": ["smoke looking like cotton", "cartoon flames"],
    "water": ["water looking like glass sheet", "repeating wave pattern"],
    "explosion": ["slow-motion look", "video game explosion"],
    "night": ["daylight leaking in", "overexposed highlights"],
}


def build_negative(extra_terms: Optional[list[str]] = None,
                   categories: Optional[list[str]] = None) -> str:
    """Assemble an 8-12 term negative prompt. Load-bearing in video."""
    terms = list(BASE_NEGATIVE)
    for cat in categories or []:
        terms.extend(NEGATIVE_EXTENSIONS.get(cat, []))
    terms.extend(extra_terms or [])
    # De-dupe, keep order, cap at 12.
    seen: list[str] = []
    for t in terms:
        if t not in seen:
            seen.append(t)
    return ", ".join(seen[:12])


def make_seed(stable: bool = False, salt: str = "") -> int:
    """uint32 seed. stable=True -> deterministic from salt (reproducible)."""
    if stable:
        digest = hashlib.sha256(f"{HOUSE_SEED}:{salt}".encode()).hexdigest()
        return int(digest[:8], 16)
    return random.randint(0, 2 ** 32 - 1)


@dataclass
class ShotPrompt:
    positive: str
    negative: str
    camera_move: str
    seed: int
    duration_s: float
    aspect: str = "16:9"
    i2v_source: Optional[str] = None  # path to keyframe still for I2V
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def direct(
    description: str,
    camera: str = "static",
    duration_s: float = 5.0,
    aspect: str = "16:9",
    style: str = "cinematic photoreal",
    negative_categories: Optional[list[str]] = None,
    house_seed: bool = False,
    i2v_source: Optional[str] = None,
) -> ShotPrompt:
    """Turn a plain description into an engineered shot prompt.

    `description` should name the subject and the action; this function adds
    camera, style, constraints, and the negative block. Keeps prompts in the
    60-100 word band with subject + core action up front.
    """
    cam_key = camera if camera in CAMERA_MOVES else "static"
    cam_phrase = CAMERA_MOVES[cam_key]

    desc = description.strip().rstrip(".")
    positive = (
        f"{desc}. {cam_phrase}. "
        f"Style: {style}. Photographically correct lighting, coherent physics, "
        f"no camera shake beyond the described move, clean edges on the effect, "
        f"consistent color grade across the shot."
    )
    # Word-count guard: keep 60-100 words.
    words = positive.split()
    if len(words) > 100:
        positive = " ".join(words[:100])

    return ShotPrompt(
        positive=positive,
        negative=build_negative(categories=negative_categories),
        camera_move=cam_key,
        seed=make_seed(stable=house_seed, salt=desc),
        duration_s=duration_s,
        aspect=aspect,
        i2v_source=i2v_source,
        notes=("I2V mode: motion-only prompt; composition comes from the still. "
               if i2v_source else
               "T2V mode: one continuous action, one camera move, nothing else moves."),
    )


def six_part(subject: str, action: str, environment: str, camera: str,
             style: str, constraints: str, **kwargs) -> ShotPrompt:
    """Explicit 6-part formula for full control."""
    description = (f"{subject} {action} in {environment}. "
                   f"Constraints: {constraints}")
    return direct(description, camera=camera, style=style, **kwargs)


def motion_only(i2v_source: str, action: str, camera: str = "static",
                model: str = "seedance", fidelity_locks: bool = True,
                lock_block: Optional[str] = None,
                audio: Optional[str] = None,
                subject: Optional[str] = None,
                environment: Optional[str] = None,
                lighting: Optional[str] = None,
                **kwargs) -> ShotPrompt:
    """I2V variant: the still carries composition; the prompt carries ONLY
    motion. Routes to the model-specific builder (Seedance default).

    `model`: "seedance" (default) or "wan"/"wan-25"/"wan-2.5".
    For Wan, subject/environment/lighting default to referencing the still
    (never re-describing it) to preserve source fidelity.

    No content filtering: prompts pass through exactly as specified.
    Fidelity locks are visual-consistency guards, not content restrictions.
    """
    model_key = (model or "seedance").strip().lower()
    if model_key in ("wan", "wan-25", "wan_25", "wan2.5", "wan-2.5"):
        return build_wan_prompt(
            subject=subject or "The subject in the still image",
            environment=environment or "The environment in the still image",
            lighting=lighting or "The lighting in the still image",
            camera=camera,
            action=f"Animate this still image: {action}",
            fidelity_locks=fidelity_locks,
            lock_block=lock_block,
            i2v_source=i2v_source,
            **kwargs)
    # Default: Seedance structure (David's primary I2V route).
    return build_seedance_prompt(
        f"Animate this still image: {action}",
        camera=camera,
        audio=audio,
        fidelity_locks=fidelity_locks,
        lock_block=lock_block,
        i2v_source=i2v_source,
        **kwargs)


# ------------------------------------------------- model-specific builders ----

# Visual-consistency guards for I2V. These freeze identity across the
# generation; they are NOT content restrictions.
FIDELITY_LOCKS = ("No face reshaping, no outfit change, no age change, "
                  "no new characters.")


def build_seedance_prompt(action: str, camera: str = "static",
                          audio: Optional[str] = None,
                          fidelity_locks: bool = True,
                          lock_block: Optional[str] = None,
                          duration_s: float = 5.0, aspect: str = "16:9",
                          negative_categories: Optional[list[str]] = None,
                          house_seed: bool = False,
                          i2v_source: Optional[str] = None,
                          **kwargs) -> ShotPrompt:
    """Seedance-optimized prompt: subject+action -> camera -> sound cues.

    Research: Seedance wants cinematic direction, not keyword lists.
    ONE action + ONE camera move per shot — stacking actions is the #1
    prompt-following failure mode. 2-4 sentences for single shots.
    For I2V, keep it motion-only: the still holds identity, so never
    re-describe face, body, clothing, or background here.
    """
    cam_key = camera if camera in CAMERA_MOVES else "static"
    cam_phrase = CAMERA_MOVES[cam_key]

    act = action.strip().rstrip(".")
    sentences = [f"{act}.", f"Camera: {cam_phrase}."]
    if audio:
        sentences.append(f"Audio: {audio.strip().rstrip('.')}.")
    if fidelity_locks:
        sentences.append(FIDELITY_LOCKS)
    if lock_block:
        sentences.append(lock_block.strip())
    positive = " ".join(sentences)

    return ShotPrompt(
        positive=positive,
        negative=build_negative(categories=negative_categories),
        camera_move=cam_key,
        seed=make_seed(stable=house_seed, salt=act),
        duration_s=duration_s,
        aspect=aspect,
        i2v_source=i2v_source,
        notes=("Seedance structure: subject+action, one camera move, sound cue. "
               "Motion-only for I2V; fidelity locks preserve identity."),
    )


def build_wan_prompt(subject: str, environment: str, lighting: str,
                     camera: str, action: str,
                     fidelity_locks: bool = True,
                     lock_block: Optional[str] = None,
                     lens: Optional[str] = None,
                     duration_s: float = 5.0, aspect: str = "16:9",
                     negative_categories: Optional[list[str]] = None,
                     house_seed: bool = False,
                     i2v_source: Optional[str] = None,
                     **kwargs) -> ShotPrompt:
    """Wan 2.5-optimized prompt: Subject -> Environment -> Lighting ->
    Camera -> Action, written as grammatical sentences (target 100-150
    words). Professional cinematography terminology; never keyword lists.

    For I2V, reference the still instead of re-describing it
    (e.g. subject="The subject in the still image") so the model does not
    reinterpret identity. The Lock Block (if given) is pasted byte-identical
    and is never trimmed.
    """
    cam_key = camera if camera in CAMERA_MOVES else "static"
    cam_phrase = CAMERA_MOVES[cam_key]
    cam_sentence = f"Camera: {cam_phrase}"
    if lens:
        cam_sentence += f", shot on {lens.strip().rstrip('.')}"
    cam_sentence += "."

    core = " ".join([
        f"{subject.strip().rstrip('.')}.",
        f"{environment.strip().rstrip('.')}.",
        f"{lighting.strip().rstrip('.')}.",
        cam_sentence,
        f"{action.strip().rstrip('.')}.",
    ])
    # Keep the descriptive core bounded; locks and lock block are sacred.
    words = core.split()
    if len(words) > 120:
        core = " ".join(words[:120])

    parts = [core]
    if fidelity_locks:
        parts.append(FIDELITY_LOCKS)
    if lock_block:
        parts.append(lock_block.strip())
    positive = " ".join(parts)

    return ShotPrompt(
        positive=positive,
        negative=build_negative(categories=negative_categories),
        camera_move=cam_key,
        seed=make_seed(stable=house_seed, salt=core),
        duration_s=duration_s,
        aspect=aspect,
        i2v_source=i2v_source,
        notes=("Wan 2.5 structure: Subject -> Environment -> Lighting -> "
               "Camera -> Action, grammatical sentences. Fidelity locks "
               "preserve identity; disable fal's prompt expansion for "
               "precise prompts."),
    )

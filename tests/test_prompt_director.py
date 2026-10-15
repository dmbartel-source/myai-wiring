"""Tests for tools/prompt_director.py — pure functions, no network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import prompt_director as pd


def test_direct_prompt_structure():
    sp = pd.direct("a car explodes in a desert", camera="push-in")
    words = sp.positive.split()
    assert 20 <= len(words) <= 100
    assert "car explodes" in sp.positive
    assert "push-in" in sp.positive
    assert sp.camera_move == "push-in"


def test_negative_prompt_load_bearing():
    neg = pd.build_negative()
    terms = [t.strip() for t in neg.split(",")]
    assert 8 <= len(terms) <= 12
    for must in ("text", "watermark", "extra fingers"):
        assert must in neg


def test_negative_categories_extend():
    neg = pd.build_negative(categories=["explosion"])
    assert "slow-motion look" in neg  # category terms appended, capped at 12
    assert len([t.strip() for t in neg.split(",")]) <= 12


def test_unknown_camera_falls_back_to_static():
    sp = pd.direct("waves crash", camera="teleport")
    assert sp.camera_move == "static"


def test_camera_vocabulary_coverage():
    for move in ("orbit", "crane-up", "whip-pan", "rack-focus", "birds-eye"):
        assert move in pd.CAMERA_MOVES


def test_seed_determinism():
    a = pd.make_seed(stable=True, salt="effect-1")
    b = pd.make_seed(stable=True, salt="effect-1")
    c = pd.make_seed(stable=True, salt="effect-2")
    assert a == b and a != c
    assert 0 <= a < 2 ** 32


def test_house_seed_constant():
    assert pd.HOUSE_SEED == 20261007


def test_motion_only_i2v():
    sp = pd.motion_only("plate.png", "flames spread left to right",
                        camera="tracking-right")
    assert sp.i2v_source == "plate.png"
    assert "I2V" in sp.notes
    assert "flames" in sp.positive


def test_six_part_formula():
    sp = pd.six_part(subject="a robot", action="walks forward",
                     environment="a neon alley", camera="steadicam",
                     style="cyberpunk", constraints="rain reflections")
    assert "robot" in sp.positive and "neon alley" in sp.positive
    assert sp.camera_move == "steadicam"


def test_no_text_in_video_guidance():
    # The negative block must always ban generated text.
    sp = pd.direct("a title card appears")
    assert "text" in sp.negative


# ------------------------------------------------- Seedance/Wan builders ----

def test_fidelity_locks_constant():
    assert pd.FIDELITY_LOCKS == ("No face reshaping, no outfit change, "
                                 "no age change, no new characters.")


def test_build_seedance_prompt_structure():
    sp = pd.build_seedance_prompt("a hero leaps across rooftops",
                                  camera="push-in", audio="wind rushing")
    assert "a hero leaps across rooftops" in sp.positive
    assert "Camera:" in sp.positive
    assert "push-in" in sp.positive
    assert "Audio: wind rushing." in sp.positive
    assert pd.FIDELITY_LOCKS in sp.positive
    assert sp.camera_move == "push-in"
    # Base structure is 2-4 sentences.
    sents = [s for s in sp.positive.split(".") if s.strip()]
    assert 2 <= len(sents) <= 4


def test_build_seedance_no_audio_no_locks():
    sp = pd.build_seedance_prompt("waves roll", camera="static",
                                  fidelity_locks=False)
    assert "Audio:" not in sp.positive
    assert pd.FIDELITY_LOCKS not in sp.positive
    assert "waves roll" in sp.positive


def test_build_seedance_lock_block_verbatim():
    lb = "Character: tall hero in a red suit; Wardrobe: matte crimson."
    sp = pd.build_seedance_prompt("he runs", lock_block=lb)
    assert lb in sp.positive  # byte-identical, never paraphrased


def test_build_seedance_unknown_camera_falls_back():
    sp = pd.build_seedance_prompt("something moves", camera="teleport")
    assert sp.camera_move == "static"


def test_build_wan_prompt_structure():
    sp = pd.build_wan_prompt(
        subject="A superhero in a red suit",
        environment="On a neon-lit rooftop at night",
        lighting="Hard rim lighting with deep blue shadows",
        camera="orbit",
        action="He leaps across the gap between buildings")
    p = sp.positive
    # Grammatical order: Subject -> Environment -> Lighting -> Camera -> Action.
    assert p.index("superhero") < p.index("rooftop") < p.index("rim lighting")
    assert "Camera:" in p and "orbit" in p
    assert pd.FIDELITY_LOCKS in p
    assert sp.camera_move == "orbit"


def test_build_wan_prompt_lens_and_lock_block():
    lb = "Style: warm 3200K key, anamorphic flares."
    sp = pd.build_wan_prompt("A hero", "A rooftop", "Night lighting",
                             "push-in", "He runs", lens="85mm f/1.4",
                             lock_block=lb, fidelity_locks=False)
    assert "85mm f/1.4" in sp.positive
    assert lb in sp.positive  # byte-identical
    assert pd.FIDELITY_LOCKS not in sp.positive


def test_build_wan_prompt_long_core_trimmed():
    long_subject = " ".join(["word"] * 200)
    sp = pd.build_wan_prompt(long_subject, "env", "light", "static", "act")
    core_words = sp.positive.split(pd.FIDELITY_LOCKS)[0].split()
    assert len(core_words) <= 120
    # Locks and lock block survive trimming.
    assert pd.FIDELITY_LOCKS in sp.positive


def test_motion_only_defaults_to_seedance():
    sp = pd.motion_only("plate.png", "flames spread",
                        camera="tracking-right")
    assert sp.i2v_source == "plate.png"
    assert "I2V" in sp.notes
    assert "flames spread" in sp.positive
    assert "Camera:" in sp.positive
    assert pd.FIDELITY_LOCKS in sp.positive


def test_motion_only_wan_routing():
    sp = pd.motion_only("plate.png", "flames spread", model="wan-25")
    assert sp.i2v_source == "plate.png"
    assert pd.FIDELITY_LOCKS in sp.positive
    # Still referenced, never re-described (fidelity).
    assert "still image" in sp.positive


def test_motion_only_backward_compat():
    # Old call style (camera/duration/aspect kwargs) still works.
    sp = pd.motion_only("plate.png", "flames spread left to right",
                        camera="tracking-right", duration_s=7.0,
                        aspect="21:9")
    assert sp.i2v_source == "plate.png"
    assert sp.duration_s == 7.0
    assert sp.aspect == "21:9"
    assert "flames" in sp.positive
    assert "I2V" in sp.notes


def test_motion_only_lock_block_and_no_locks():
    lb = "Character: hero in red."
    sp = pd.motion_only("plate.png", "he runs", lock_block=lb,
                        fidelity_locks=False, audio="city ambience")
    assert lb in sp.positive
    assert pd.FIDELITY_LOCKS not in sp.positive
    assert "Audio: city ambience." in sp.positive


def test_builders_do_not_filter_content():
    # Prompts pass through verbatim: no content filtering, no moralizing.
    raw = "two consenting adults embrace passionately"
    sp = pd.build_seedance_prompt(raw, camera="static")
    assert raw in sp.positive
    sp2 = pd.build_wan_prompt(raw, "a bedroom", "soft lamplight",
                              "static", "they move closer")
    assert raw in sp2.positive
    sp3 = pd.motion_only("plate.png", raw, model="wan")
    assert raw in sp3.positive


# ---------------------------------------------------------------------------
# build_reference_prompt: Seedance 2.0 reference-to-video prompts.
# ---------------------------------------------------------------------------

def test_build_reference_prompt_image_refs():
    sp = pd.build_reference_prompt(
        image_roles=["character front view", "character side view",
                     "outfit detail"],
        motion="she walks toward the camera")
    assert "@Image1 (character front view)" in sp.positive
    assert "@Image2 (character side view)" in sp.positive
    assert "@Image3 (outfit detail)" in sp.positive
    assert "she walks toward the camera" in sp.positive
    # character-consistency fidelity locks present
    assert "same face" in sp.positive
    assert "same outfit" in sp.positive
    assert "No face reshaping" in sp.positive


def test_build_reference_prompt_mixed_refs():
    sp = pd.build_reference_prompt(
        image_roles=["hero front view"], video_roles=["walk cycle sample"],
        motion="the hero walks forward")
    assert "@Image1 (hero front view)" in sp.positive
    assert "@Video1 (walk cycle sample)" in sp.positive


def test_build_reference_prompt_default_roles():
    # counts without roles get generic labels, still tagged
    sp = pd.build_reference_prompt(n_images=2, n_videos=1, motion="spin")
    assert "@Image1" in sp.positive
    assert "@Image2" in sp.positive
    assert "@Video1" in sp.positive


def test_build_reference_prompt_requires_refs():
    with pytest.raises(ValueError, match="at least one"):
        pd.build_reference_prompt(motion="x")


def test_build_reference_prompt_no_filtering():
    # Prompts pass through verbatim: no content filtering, no moralizing.
    raw = "two consenting adults embrace passionately"
    sp = pd.build_reference_prompt(image_roles=["the couple"], motion=raw)
    assert raw in sp.positive
    assert "@Image1 (the couple)" in sp.positive


def test_build_reference_prompt_lock_block_and_camera():
    lb = "Character: hero in red cape."
    sp = pd.build_reference_prompt(image_roles=["hero front"], motion="he runs",
                                   camera="tracking-right", lock_block=lb,
                                   fidelity_locks=False)
    assert lb in sp.positive
    assert "same face" not in sp.positive  # locks disabled
    assert sp.camera_move == "tracking-right"

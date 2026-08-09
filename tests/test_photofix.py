"""Tests for the photo pipeline.

The scenes in tools/make_samples.py are tilted by a known angle, so the
straightening step can be scored rather than eyeballed. They are built once per
session and reused.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photofix import Config, io_utils, process, resolve_ratio          # noqa: E402
from photofix.config import RATIO_ALIASES                              # noqa: E402
from photofix.framing import plan_crop, subject_bbox                   # noqa: E402
from photofix.geometry import _circular_mean_mod90, rotate             # noqa: E402
from photofix.segment import segment                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import make_samples as ms                                              # noqa: E402


# The straightening is accurate to well under a tenth of a degree on these
# scenes; half a degree is the point where a tilt starts to be visible on a
# phone's long edge, so that is the line the tests hold.
ANGLE_TOLERANCE = 0.5


def _codes(result) -> set[str]:
    """The stable note codes a run produced, ignoring their wording."""
    return {n.code for n in result.report.warnings}


@pytest.fixture(scope="session")
def scenes() -> dict[str, tuple[np.ndarray, float]]:
    out = {}
    for name, fn in ms.SCENES.items():
        angle = ms.ANGLES[name]
        out[name] = (fn(angle), angle)
    return out


# --------------------------------------------------------------------------
# straightening
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(ms.SCENES))
def test_straightens_to_the_known_tilt(scenes, name):
    image, expected = scenes[name]
    result = process(image, Config())
    assert result.report.ok, result.report.warnings
    assert abs(result.report.rotated_deg - expected) < ANGLE_TOLERANCE, (
        f"{name}: corrected {result.report.rotated_deg:+.2f}, "
        f"expected {expected:+.2f}"
    )


def test_already_straight_photo_is_not_rotated():
    image = ms.scene_single_phone(0.0)
    result = process(image, Config())
    assert abs(result.report.rotated_deg) < 0.2


def test_straightening_can_be_switched_off(scenes):
    image, _ = scenes["single_phone"]
    result = process(image, Config(straighten=False))
    assert result.report.rotated_deg == 0.0


def test_an_absurd_tilt_is_refused_rather_than_guessed():
    """A 40 degree reading means the subject was found badly. Leaving the photo
    alone beats rotating it on a bad guess."""
    cfg = Config(max_rotation=5.0)
    image = ms.scene_single_phone(12.0)
    result = process(image, cfg)
    assert result.report.rotated_deg == 0.0
    assert "rotation_refused" in _codes(result)


def test_circular_mean_handles_the_90_degree_wrap():
    # -44 and +44 are two degrees apart across the wrap, not 88.
    mean = _circular_mean_mod90(np.array([-44.0, 44.0]), np.array([1.0, 1.0]))
    assert abs(abs(mean) - 45.0) < 0.01


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------
def test_clutter_is_cropped_out(scenes):
    """The mug straddling the corner must not survive into the listing."""
    image, _ = scenes["with_distractor"]
    result = process(image, Config())
    assert result.report.objects_found > result.report.objects_kept
    assert result.report.objects_kept == 1


def test_keeping_extras_is_opt_in(scenes):
    image, _ = scenes["with_distractor"]
    result = process(image, Config(drop_distractors=False))
    assert result.report.objects_kept == result.report.objects_found


def test_accessories_beside_the_product_are_kept(scenes):
    """The charging cable is part of what is being sold, not clutter."""
    image, _ = scenes["phone_and_cable"]
    result = process(image, Config())
    assert result.report.objects_kept == 2


def test_auto_ratio_follows_the_subject(scenes):
    tall, _ = scenes["single_phone"]
    wide, _ = scenes["full_kit"]
    portrait = process(tall, Config()).image
    landscape = process(wide, Config()).image
    assert portrait.shape[0] > portrait.shape[1], "a single phone should come out portrait"
    assert landscape.shape[1] > landscape.shape[0], "a row of items should come out landscape"


@pytest.mark.parametrize("alias", ["1:1", "4:5", "2:3"])
def test_a_fixed_ratio_is_honoured(scenes, alias):
    image, _ = scenes["single_phone"]
    out = process(image, Config(ratio=alias)).image
    got = out.shape[1] / out.shape[0]
    assert abs(got - RATIO_ALIASES[alias]) < 0.02, f"{alias}: got {got:.3f}"


def test_an_impossible_ratio_is_reported_not_faked(scenes):
    """A tall phone photo does not contain enough floor to make a landscape
    frame. Returning a different shape without saying so would silently break
    a batch that is supposed to be uniform."""
    image, _ = scenes["single_phone"]
    result = process(image, Config(ratio="3:2"))
    got = result.image.shape[1] / result.image.shape[0]
    assert abs(got - 1.5) > 0.02
    assert "ratio_unreachable" in _codes(result)


def test_padding_forces_an_impossible_ratio(scenes):
    image, _ = scenes["single_phone"]
    result = process(image, Config(ratio="3:2", pad_to_ratio=True))
    got = result.image.shape[1] / result.image.shape[0]
    assert abs(got - 1.5) < 0.02, f"got {got:.3f}"
    assert result.report.padded > 0


def test_no_crop_keeps_the_original_framing(scenes):
    image, _ = scenes["single_phone"]
    result = process(image, Config(crop=False, straighten=False, max_side=0))
    assert result.image.shape[:2] == image.shape[:2]


def test_the_product_is_never_cut_by_the_crop():
    """Whatever else it trades away, the framer must keep the whole subject."""
    valid = np.full((1000, 800), 255, np.uint8)
    subject = (300, 200, 200, 600)
    crop = plan_crop((1000, 800), subject, valid, Config(), original_ratio=0.8)
    sx, sy, sw, sh = subject
    assert crop.x <= sx and crop.y <= sy
    assert crop.x + crop.w >= sx + sw
    assert crop.y + crop.h >= sy + sh


def test_the_product_gets_the_margin_it_was_asked_for():
    valid = np.full((1400, 1400), 255, np.uint8)
    subject = (600, 500, 200, 400)
    cfg = Config(fill=0.8)
    crop = plan_crop((1400, 1400), subject, valid, cfg, original_ratio=1.0)
    # The tight dimension should end up close to the requested fill.
    filled = max(subject[2] / crop.w, subject[3] / crop.h)
    assert filled == pytest.approx(cfg.fill, abs=0.06)


def test_the_crop_stays_off_the_rotated_corners(scenes):
    """Rotation exposes empty wedges; the crop should mostly avoid them."""
    image, _ = scenes["single_phone"]
    seg = segment(image, Config())
    h, w = image.shape[:2]
    mask = cv2.resize(seg.mask, (w, h), interpolation=cv2.INTER_NEAREST)
    rot = rotate(image, mask, 7.4, (150, 176, 198))
    box = subject_bbox(rot.mask)
    crop = plan_crop(rot.image.shape, box, rot.valid, Config(), original_ratio=w / h)
    assert crop.filled_corners <= Config().max_filled


# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------
def test_a_blank_photo_is_left_alone_not_mangled():
    flat = np.full((900, 700, 3), 180, np.uint8)
    result = process(flat, Config())
    assert not result.report.ok
    assert result.report.warnings
    assert result.image.shape[0] > 0 and result.image.shape[1] > 0


def test_a_photo_that_is_all_product_is_left_alone():
    """No visible surface means no background model, so there is nothing to
    align to and nothing to crop away."""
    busy = np.random.default_rng(0).integers(0, 255, (600, 600, 3), dtype=np.uint8)
    result = process(busy, Config())
    assert result.image.shape[0] > 0


def test_a_tiny_image_does_not_crash():
    tiny = np.full((40, 30, 3), 200, np.uint8)
    tiny[10:30, 8:22] = 30
    result = process(tiny, Config())
    assert result.image.size > 0


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def test_max_side_is_respected(scenes):
    image, _ = scenes["single_phone"]
    out = process(image, Config(max_side=800)).image
    assert max(out.shape[:2]) == 800


def test_small_images_are_not_upscaled():
    small = np.full((200, 150, 3), 190, np.uint8)
    assert io_utils.fit_max_side(small, 2000).shape[:2] == (200, 150)


@pytest.mark.parametrize("fmt", ["jpg", "png", "webp"])
def test_every_output_format_round_trips(tmp_path, fmt):
    image = np.full((80, 60, 3), 120, np.uint8)
    path = tmp_path / f"x.{fmt}"
    io_utils.save(image, path, fmt=fmt, quality=95)
    assert path.exists() and path.stat().st_size > 0
    assert io_utils.load_bgr(path).shape == image.shape


def test_exif_orientation_is_applied(tmp_path):
    """An iPhone stores a sideways photo upright plus a rotate tag. Reading it
    without honouring the tag is the classic 'photo on its side' bug."""
    from PIL import Image

    rgb = np.zeros((40, 90, 3), np.uint8)
    rgb[:, :10] = 255                       # a marker on the left edge
    path = tmp_path / "rotated.jpg"
    pil = Image.fromarray(rgb)
    exif = pil.getexif()
    exif[274] = 6                           # orientation: rotate 90 CW on show
    pil.save(path, exif=exif)

    loaded = io_utils.load_bgr(path)
    assert loaded.shape[:2] == (90, 40), "the tag should have been applied"


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [("4:5", 0.8), ("1:1", 1.0), ("1.5", 1.5)])
def test_ratio_parsing(text, expected):
    assert resolve_ratio(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["auto", "original", "free", ""])
def test_non_fixed_ratios_resolve_to_nothing(text):
    assert resolve_ratio(text) is None


def test_a_nonsense_ratio_is_rejected():
    with pytest.raises(ValueError):
        resolve_ratio("banana")

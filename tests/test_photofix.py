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


# --------------------------------------------------------------------------
# replacing the surface
# --------------------------------------------------------------------------
from photofix import background, cutout                                # noqa: E402
from photofix.segment import segment                                   # noqa: E402

needs_model = pytest.mark.skipif(
    not cutout.model_available(download=True),
    reason="the cut-out model could not be fetched",
)


def test_generated_surface_is_the_size_asked_for():
    s = background.surface(300, 500)
    assert s.shape == (300, 500, 3)


def test_a_photo_of_your_own_surface_tiles_to_any_size(tmp_path):
    swatch = background.wood_surface(120, 90)
    path = tmp_path / "floor.jpg"
    io_utils.save(swatch, path)
    s = background.surface(700, 400, str(path))
    assert s.shape == (700, 400, 3)


def test_a_missing_surface_photo_is_an_error():
    with pytest.raises(FileNotFoundError):
        background.surface(50, 50, "/nope/not/here.jpg")


def test_the_shadow_falls_outside_the_product_not_on_it():
    """It hugs the object: the blur radius scales with the frame, so on a small
    test image the shadow only reaches a few pixels past the edge."""
    alpha = np.zeros((200, 200), np.uint8)
    alpha[60:140, 60:140] = 255
    s = background.contact_shadow(alpha)
    assert s[100, 100] == pytest.approx(0.0, abs=1e-6), "none under the object"
    assert s[141:150, 100].max() > 0.02, "some just below it"
    assert s[190:, 100].max() < 0.01, "none far away"
    # Offset down and to the right, as in the reference photos.
    assert s[141:150, 100].max() > s[50:59, 100].max()


def test_a_flat_lay_keeps_its_own_surface(scenes):
    """Replacing a perfectly good wooden floor with a synthetic one only loses
    detail, so 'auto' should leave these alone."""
    for name in ("single_phone", "phone_and_cable", "full_kit"):
        image, _ = scenes[name]
        assert segment(image, Config()).uniform_surface, name


def test_a_photo_with_no_usable_surface_is_spotted(scenes):
    image, _ = scenes["in_hand"]
    assert not segment(image, Config()).uniform_surface


@needs_model
def test_auto_replaces_only_the_background_that_needs_it(scenes):
    replaced = {}
    for name in ("single_phone", "in_hand"):
        image, _ = scenes[name]
        result = process(image, Config(background="auto"))
        replaced[name] = "background_replaced" in _codes(result)
    assert replaced == {"single_phone": False, "in_hand": True}


@needs_model
def test_the_hand_is_removed(scenes):
    image, _ = scenes["in_hand"]
    result = process(image, Config(background="wood"))
    assert "background_replaced" in _codes(result)
    assert "hand_removed" in _codes(result)


@needs_model
def test_a_hand_over_the_product_is_reported_not_hidden(scenes):
    """What a finger covers cannot be recovered. The tool rebuilds the outline
    so the result is not full of holes, and says how much it had to invent."""
    image, _ = scenes["in_hand"]
    result = process(image, Config(background="wood"))
    assert "hand_covers_product" in _codes(result)


@needs_model
def test_skin_survives_on_a_product_that_is_merely_warm_coloured():
    """A beige product must not be mistaken for a hand and deleted."""
    beige = np.full((900, 700, 3), 40, np.uint8)
    beige[200:700, 220:480] = (150, 176, 205)      # a skin-toned rectangle
    result = process(beige, Config(background="wood"))
    assert result.image.size > 0


@needs_model
def test_a_replaced_surface_hits_the_exact_ratio(scenes):
    """With a generated surface nothing constrains the frame, so a fixed ratio
    is always achievable -- unlike when cropping inside the original photo."""
    image, _ = scenes["in_hand"]
    for alias in ("1:1", "4:5", "3:2"):
        out = process(image, Config(background="wood", ratio=alias)).image
        got = out.shape[1] / out.shape[0]
        assert abs(got - RATIO_ALIASES[alias]) < 0.02, f"{alias}: got {got:.3f}"


def test_without_the_model_the_original_background_is_kept(scenes, tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOFIX_MODEL_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(cutout, "_NET", None)
    image, _ = scenes["single_phone"]
    result = process(image, Config(background="wood", allow_model_download=False))
    assert "cutout_unavailable" in _codes(result)
    assert result.image.size > 0


# --------------------------------------------------------------------------
# turning upright
# --------------------------------------------------------------------------
from photofix import orient                                            # noqa: E402


def _phone_on_floor(face: str, body=(60, 64, 72)) -> np.ndarray:
    """A phone lying upright in the middle of a wooden floor."""
    H = W = 1500
    canvas = ms.wood_floor(H, W, seed=5)
    ph = 900
    pw = int(ph * ms.PHONE_RATIO)
    patch, mask = ms.phone(ph, pw, face=face, body=body)
    ms._paste(canvas, patch, mask, (H - ph) // 2, (W - pw) // 2)
    return canvas


@pytest.mark.parametrize("face", ["back", "front"])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_a_phone_is_turned_so_its_top_is_up(face, k):
    image = np.ascontiguousarray(np.rot90(_phone_on_floor(face), k))
    result = process(image, Config())
    # np.rot90 turned it k quarters anticlockwise; undoing that is k clockwise.
    assert result.report.turned_deg == (k * 90) % 360, (
        face, k, result.report.turned_deg, result.report.turn_source)
    h, w = result.image.shape[:2]
    assert h > w, "an upright phone makes a portrait photo"
    if k:
        assert result.report.turn_source == ("camera" if face == "back" else "island")


def test_the_turn_is_judged_by_the_feature_not_the_shape():
    # A plain box has a phone's outline but neither a camera nor an island:
    # nothing says which end is up, so nothing turns.
    H = W = 1400
    canvas = ms.wood_floor(H, W, seed=6)
    bh, bw = 860, 410
    patch, mask = ms.retail_box(bh, bw)
    ms._paste(canvas, patch, mask, (H - bh) // 2, (W - bw) // 2)
    result = process(np.ascontiguousarray(np.rot90(canvas)), Config())
    assert result.report.turned_deg == 0


def test_turning_upright_can_be_switched_off():
    image = np.ascontiguousarray(np.rot90(_phone_on_floor("back")))
    assert process(image, Config(upright=False)).report.turned_deg == 0


def test_a_turn_chosen_by_hand_overrides_the_guess():
    image = np.ascontiguousarray(np.rot90(_phone_on_floor("back")))   # top now left
    result = process(image, Config(turn="270"))
    assert result.report.turned_deg == 270
    assert result.report.turn_source == "manual"


def test_a_turn_that_is_not_a_quarter_is_refused():
    with pytest.raises(ValueError):
        process(_phone_on_floor("back"), Config(turn="45"))


def test_quarter_turn_arithmetic():
    up, right, down, left = (0, -1), (1, 0), (0, 1), (-1, 0)
    assert orient._turns_for(np.array(up)) == 0
    assert orient._turns_for(np.array(right)) == 1     # anticlockwise brings it up
    assert orient._turns_for(np.array(down)) == 2
    assert orient._turns_for(np.array(left)) == 3


# --------------------------------------------------------------------------
# objects cut off by the frame
# --------------------------------------------------------------------------
def test_something_poking_in_from_the_edge_is_not_part_of_the_product():
    # Only a sliver of the intruder shows. GrabCut is told the outer ring is
    # background, so this used to stop short of the edge and be kept.
    H, W = 1200, 1400
    canvas = ms.wood_floor(H, W, seed=7)
    ph = 760
    pw = int(ph * ms.PHONE_RATIO)
    ms._paste(canvas, *ms.phone(ph, pw, face="back", body=(60, 64, 72)),
              (H - ph) // 2, (W - pw) // 2)
    sliver = np.full((300, 200), 255, np.uint8)
    ms._paste(canvas, np.full((300, 200, 3), 245, np.float32), sliver, 200, W - 90)
    seg = segment(canvas, Config())
    assert len(seg.components) == 1
    assert len(seg.all_components) >= 2


# --------------------------------------------------------------------------
# web interface: turning one photo by hand
# --------------------------------------------------------------------------
def test_the_web_turn_button_sticks(tmp_path, monkeypatch):
    import io as _io
    monkeypatch.setenv("PHOTOFIX_WORKDIR", str(tmp_path))
    import importlib
    import webapp.app as webapp_app
    webapp_app = importlib.reload(webapp_app)
    client = webapp_app.app.test_client()

    ok, buf = cv2.imencode(".jpg", _phone_on_floor("back"))
    assert ok
    res = client.post("/api/upload", data={
        "photos": (_io.BytesIO(buf.tobytes()), "p.jpg"),
    }, content_type="multipart/form-data")
    assert res.status_code == 200, res.get_json()
    job = res.get_json()["job"]
    first = res.get_json()["items"][0]
    assert first["turned"] == 0

    res = client.post(f"/api/turn/{job}", data={"name": "p.jpg", "turn": "90"})
    assert res.status_code == 200, res.get_json()
    item = res.get_json()["item"]
    assert item["turned"] == 90 and item["turn_manual"]

    # Changing the settings afterwards keeps the hand-made turn.
    res = client.post(f"/api/rerender/{job}", data={"fill": "0.7"})
    assert res.get_json()["items"][0]["turned"] == 90

    assert client.post(f"/api/turn/{job}", data={"name": "p.jpg", "turn": "45"}).status_code == 400
    assert client.post(f"/api/turn/{job}", data={"name": "../x", "turn": "90"}).status_code == 404

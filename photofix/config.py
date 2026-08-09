"""Configuration for the photo pipeline.

Every knob the pipeline uses lives here so the CLI, the web UI and the tests
all speak the same language.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Sequence


# Aspect ratios (width / height) the "auto" framer is allowed to choose from.
# They match the shapes seen in typical marketplace listings: tall portrait for
# a single phone, 4:5 for a phone plus an accessory, landscape for a row of
# items.
DEFAULT_RATIOS: tuple[float, ...] = (
    2 / 3,   # 0.667  tall portrait, single upright product
    3 / 4,   # 0.750
    4 / 5,   # 0.800
    1.0,     # square
    5 / 4,   # 1.250
    4 / 3,   # 1.333
    3 / 2,   # 1.500  landscape, several items side by side
)

RATIO_ALIASES: dict[str, float] = {
    "2:3": 2 / 3,
    "3:4": 3 / 4,
    "4:5": 4 / 5,
    "1:1": 1.0,
    "5:4": 5 / 4,
    "4:3": 4 / 3,
    "3:2": 3 / 2,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
}


@dataclass
class Config:
    # ---- geometry -------------------------------------------------------
    straighten: bool = True
    """Rotate the picture so the product's edges are perfectly vertical."""

    max_rotation: float = 25.0
    """Refuse corrections larger than this (degrees). A bigger angle almost
    always means the subject was detected badly, and rotating on a bad guess
    is worse than leaving the photo alone."""

    min_rotation: float = 0.15
    """Below this the rotation is not worth the resampling."""

    # ---- framing --------------------------------------------------------
    crop: bool = True
    ratio: str = "auto"
    """'auto', 'original', 'free', or an alias such as '4:5'."""

    ratios: Sequence[float] = field(default_factory=lambda: DEFAULT_RATIOS)

    pad_to_ratio: bool = False
    """When a fixed ratio does not fit the photo, extend the frame with the
    surface mirrored outwards instead of settling for a different shape. Useful
    when a marketplace needs every image the same shape."""

    fill: float = 0.86
    """Fraction of the tight dimension the product should occupy. 0.86 leaves
    a ~7% breathing margin on each side, like the reference shots."""

    center_bias_y: float = 0.0
    """Positive pushes the product above centre (0.05 = 5% of crop height)."""

    drop_distractors: bool = True
    """Discard objects that are cut off by the frame edge or far too small."""

    background: str = ""
    """Empty keeps the surface the photo was taken on -- the default, and the
    right answer when the product is already lying on a decent surface. Set to
    "wood" for the built-in generated floor, or to the path of a photo of your
    own surface. Anything else in the frame, including a hand, disappears."""

    remove_hands: bool = True
    """When replacing the background, drop skin-coloured regions so a product
    held up in a hand comes out on its own. What a finger covers *on top of*
    the product cannot be recovered, and is reported rather than invented."""

    shadow: bool = True
    """Lay a soft contact shadow under the product on the new surface. Without
    one it reads as a sticker."""

    allow_model_download: bool = True
    """The cut-out needs a 4.5 MB model file, fetched once and cached."""

    max_filled: float = 0.015
    """How much of the crop may fall outside the original photo before the
    framer starts sacrificing margin. Rotating always exposes wedges at the
    corners; a sliver this small is reconstructed from the surrounding surface
    and is invisible, whereas shrinking the crop to avoid it would squeeze the
    product against the frame edge."""

    min_object_ratio: float = 0.04
    """Keep a secondary object only if its area is at least this fraction of
    the main object's area."""

    # ---- look -----------------------------------------------------------
    enhance: str = "light"       # none | light | strong
    white_balance: bool = False
    """Off by default: neutralising the colour cast also drains the warmth out
    of a wooden surface, which the reference shots keep."""

    # ---- output ---------------------------------------------------------
    max_side: int = 2000
    fmt: str = "jpg"             # jpg | png | webp
    quality: int = 95
    suffix: str = ""
    strip_metadata: bool = True

    # ---- segmentation internals ----------------------------------------
    work_size: int = 900
    """Long side used for the analysis pass. Full resolution is only touched
    when the final rotation and crop are applied."""

    border_frac: float = 0.05
    """Width of the frame band sampled to model the background."""

    threshold_k: float = 3.2
    """Robust z-score above which a pixel is considered foreground."""

    grabcut: bool = True
    grabcut_iters: int = 3

    def with_(self, **kwargs) -> "Config":
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ratios"] = list(self.ratios)
        return d


def resolve_ratio(name: str) -> float | None:
    """Turn a ratio string into a number. None means 'not a fixed ratio'."""
    key = (name or "").strip().lower()
    if key in ("auto", "original", "free", ""):
        return None
    if key in RATIO_ALIASES:
        return RATIO_ALIASES[key]
    if ":" in key:
        a, _, b = key.partition(":")
        try:
            w, h = float(a), float(b)
            if w > 0 and h > 0:
                return w / h
        except ValueError:
            pass
    try:
        v = float(key)
        if v > 0:
            return v
    except ValueError:
        pass
    raise ValueError(f"Unrecognised aspect ratio: {name!r}")

"""The surface the product gets placed on, and how it gets placed there.

Two jobs: produce a background of any size, and composite a cut-out product
onto it so it looks photographed rather than pasted -- which mostly comes down
to the shadow. A product with no shadow reads as a sticker instantly.

The default surface is generated, not photographed, so the repository carries
no image files. Point `Config.background` at a photo of your own floor to use
that instead; it will be tiled and colour-matched to fill any frame.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


BUILTIN = "wood"


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def wood_surface(h: int, w: int, seed: int = 7) -> np.ndarray:
    """Light oak laminate seen from above, in the style of the reference shots.

    Warm base, grain running down the frame, plank seams, and a soft unevenness
    in the light so it does not read as a flat swatch.
    """
    r = _rng(seed)
    base = np.array([150.0, 176.0, 199.0])          # BGR: warm, light
    img = np.tile(base, (h, w, 1))

    x = np.arange(w, dtype=np.float32)[None, :]
    y = np.arange(h, dtype=np.float32)[:, None]

    # Grain: a few sine trains of different frequency, gently bent along the
    # plank so the lines are not perfectly straight.
    bend = 2.4 * np.sin(y * 0.0035 + r.uniform(0, 6))
    grain = (
        8.5 * np.sin(x * 0.34 + bend + r.uniform(0, 6))
        + 5.5 * np.sin(x * 0.115 + bend * 1.7 + r.uniform(0, 6))
        + 3.5 * np.sin(x * 1.63 + r.uniform(0, 6))
        + 2.0 * np.sin(x * 3.10 + r.uniform(0, 6))
    )
    grain = grain + r.normal(0, 3.4, size=(h, w))
    grain = cv2.GaussianBlur(grain.astype(np.float32), (0, 0), 1.15)
    img += grain[:, :, None] * np.array([0.88, 1.0, 1.06])

    # Plank seams, a shade darker and slightly irregular.
    period = max(140, int(w * 0.36))
    for sx in range(period, w, period):
        jitter = int(r.integers(-10, 10))
        cv2.line(img, (sx + jitter, 0), (sx + jitter, h), (120, 142, 163), 3, cv2.LINE_AA)

    # Daylight falling off towards one corner.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fall = 1.0 - 0.6 * (((xx / w) - 0.42) ** 2 + ((yy / h) - 0.36) ** 2)
    img *= np.clip(fall, 0.74, 1.05)[:, :, None]

    img += r.normal(0, 1.8, size=(h, w, 1))         # a little sensor grain
    return np.clip(img, 0, 255).astype(np.uint8)


@lru_cache(maxsize=4)
def _load_texture(path: str) -> np.ndarray:
    from . import io_utils

    return io_utils.load_bgr(path)


def _tile_to(texture: np.ndarray, h: int, w: int) -> np.ndarray:
    """Cover an h x w frame with a texture, mirroring so seams do not show."""
    th, tw = texture.shape[:2]
    # Scale the texture up if it is smaller than the frame in either direction,
    # so we never tile a tiny image dozens of times.
    scale = max(h / th, w / tw, 1.0)
    if scale > 1.0:
        texture = cv2.resize(texture, (int(np.ceil(tw * scale)), int(np.ceil(th * scale))),
                             interpolation=cv2.INTER_CUBIC)
        th, tw = texture.shape[:2]

    reps_y = int(np.ceil(h / th))
    reps_x = int(np.ceil(w / tw))
    if reps_y == 1 and reps_x == 1:
        return texture[:h, :w].copy()

    # Mirror alternate tiles so edges line up.
    rows = []
    for j in range(reps_y):
        cols = []
        for i in range(reps_x):
            t = texture
            if i % 2:
                t = t[:, ::-1]
            if j % 2:
                t = t[::-1, :]
            cols.append(t)
        rows.append(np.concatenate(cols, axis=1))
    return np.concatenate(rows, axis=0)[:h, :w].copy()


def surface(h: int, w: int, source: str = BUILTIN, seed: int = 7) -> np.ndarray:
    """Background of the requested size, generated or from a photo."""
    if not source or source == BUILTIN:
        return wood_surface(h, w, seed)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Background image not found: {source}")
    return _tile_to(_load_texture(str(path)), h, w)


def contact_shadow(alpha: np.ndarray, strength: float = 0.42,
                   softness: float = 0.018, offset: float = 0.010) -> np.ndarray:
    """A soft shadow under the product, as a 0..1 darkening map.

    Modelled on the reference photos: a diffuse shadow, slightly offset down
    and to the right, with a tighter darker core right where the object meets
    the surface.
    """
    h, w = alpha.shape[:2]
    a = alpha.astype(np.float32) / 255.0
    span = max(h, w)

    wide = cv2.GaussianBlur(a, (0, 0), max(2.0, span * softness))
    tight = cv2.GaussianBlur(a, (0, 0), max(1.0, span * softness * 0.35))
    shadow = np.clip(0.62 * wide + 0.38 * tight, 0, 1)

    dy, dx = int(round(span * offset)), int(round(span * offset * 0.55))
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    shadow = cv2.warpAffine(shadow, m, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Only darken where the product is not, or the object's own edge goes muddy.
    return np.clip(shadow - a, 0, 1) * strength


def composite(product: np.ndarray, alpha: np.ndarray, back: np.ndarray,
              shadow: bool = True) -> np.ndarray:
    """Lay the cut-out product over the background, shadow first."""
    if back.shape[:2] != product.shape[:2]:
        back = cv2.resize(back, (product.shape[1], product.shape[0]),
                          interpolation=cv2.INTER_AREA)

    out = back.astype(np.float32)
    if shadow:
        out *= (1.0 - contact_shadow(alpha))[:, :, None]

    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    out = out * (1.0 - a) + product.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)

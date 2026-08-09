"""Choosing the final crop: what to keep, how much air, which shape."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import cv2

from . import notes
from .config import Config, resolve_ratio


@dataclass
class Crop:
    x: int
    y: int
    w: int
    h: int
    ratio: float
    filled_corners: float   # fraction of the crop that was outside the original
    notes: list           # notes.Note entries explaining any compromises

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


def subject_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight bounding box around everything still in the mask.

    boundingRect walks the mask in place. The obvious np.nonzero version builds
    two index arrays first, which on a full-resolution mask means tens of
    megabytes to answer a question with four numbers in it.
    """
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return None
    return int(x), int(y), int(w), int(h)


# The framer asks "how much of this rectangle fell outside the original?"
# thousands of times, which wants a summed-area table. Built at full size that
# table is enormous -- a 12 megapixel photo needs over 100 MB, and the two
# cumulative sums each need another copy of it. The answer only steers a crop
# by a pixel or two, so it is computed on a reduced copy instead.
PLAN_MAX_SIDE = 1400


class _InvalidIndex:
    """O(1) lookup of the invalid area inside any rectangle.

    Coordinates stay in full resolution; only the lookup grid is reduced. A
    reduced cell counts as invalid unless it is *entirely* valid, so the answer
    never under-reports the damage.
    """

    def __init__(self, valid: np.ndarray, max_side: int = PLAN_MAX_SIDE):
        h, w = valid.shape[:2]
        self.scale = min(1.0, max_side / max(h, w, 1))

        if self.scale < 1.0:
            small = cv2.resize(
                valid,
                (max(1, round(w * self.scale)), max(1, round(h * self.scale))),
                interpolation=cv2.INTER_AREA,
            )
            small = np.where(small >= 255, 0, 1).astype(np.int32)
        else:
            small = (valid == 0).astype(np.int32)

        self.h, self.w = small.shape[:2]
        self.integral = np.pad(
            small.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32),
            ((1, 0), (1, 0)),
        )

    def fraction(self, x: int, y: int, w: int, h: int) -> float:
        s = self.scale
        x0, y0 = max(0, int(np.floor(x * s))), max(0, int(np.floor(y * s)))
        x1 = min(self.w, max(x0 + 1, int(np.ceil((x + w) * s))))
        y1 = min(self.h, max(y0 + 1, int(np.ceil((y + h) * s))))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        g = self.integral
        bad = int(g[y1, x1] - g[y0, x1] - g[y1, x0] + g[y0, x0])
        return bad / float((x1 - x0) * (y1 - y0))


def _candidate_ratios(cfg: Config, original_ratio: float, subject_ratio: float) -> list[float]:
    mode = (cfg.ratio or "auto").strip().lower()
    if mode == "auto":
        return list(cfg.ratios)
    if mode == "original":
        return [original_ratio]
    if mode == "free":
        return [subject_ratio]
    fixed = resolve_ratio(mode)
    return [fixed] if fixed else list(cfg.ratios)


def choose_canvas(
    sw: int, sh: int, cfg: Config, original_ratio: float
) -> tuple[int, int, float]:
    """Size of a fresh frame built around a subject of sw x sh.

    Used when the background is being replaced, where nothing constrains the
    frame: the surface is generated to whatever size suits the product, so the
    margin and the aspect ratio always come out exactly as asked.
    """
    fill = float(np.clip(cfg.fill, 0.30, 1.0))
    mode = (cfg.ratio or "auto").strip().lower()

    best: tuple[float, int, int, float] | None = None
    for r in _candidate_ratios(cfg, original_ratio, sw / max(sh, 1)):
        if mode == "free":
            cw, ch = sw / fill, sh / fill
            r = cw / ch
        else:
            cw = max(sw / fill, r * (sh / fill))
            ch = cw / r
        waste = cw * ch - sw * sh
        if best is None or waste < best[0]:
            best = (waste, max(1, int(round(cw))), max(1, int(round(ch))), r)

    assert best is not None
    return best[1], best[2], best[3]


def plan_crop(
    frame_shape: tuple[int, int],
    subject: tuple[int, int, int, int],
    valid: np.ndarray,
    cfg: Config,
    original_ratio: float,
) -> Crop:
    """Pick the crop rectangle.

    Order of priorities, highest first: keep the whole product; respect the
    requested aspect ratio; leave no rotation-filled corners showing; land as
    close as possible to the ideal margin.
    """
    H, W = frame_shape[:2]
    sx, sy, sw, sh = subject
    subject_ratio = sw / max(sh, 1)

    fill = float(np.clip(cfg.fill, 0.30, 1.0))
    mode = (cfg.ratio or "auto").strip().lower()
    invalid = _InvalidIndex(valid)

    ideal_cx = sx + sw / 2.0
    ideal_cy = sy + sh / 2.0

    scored: list[tuple[float, float, float, float]] = []  # waste, ratio, cw, ch
    for r in _candidate_ratios(cfg, original_ratio, subject_ratio):
        if mode == "free":
            cw = sw / fill
            ch = sh / fill
            r = cw / ch
        else:
            cw = max(sw / fill, r * (sh / fill))
            ch = cw / r
        if cw < 1 or ch < 1:
            continue
        waste = cw * ch - sw * sh
        # A crop that does not fit the frame has to be shrunk later, which
        # eats the margin; penalise it rather than forbidding it outright.
        overflow = max(cw / W, ch / H, 1.0)
        scored.append((waste * (overflow ** 3), r, cw, ch))

    if not scored:
        return Crop(0, 0, W, H, W / max(H, 1), 0.0, [])

    scored.sort(key=lambda t: t[0])
    candidates = scored[:4]

    # Shrinking is the last resort: it eats the very margin we are trying to
    # create. So the outer loop walks the crop size down slowly and the inner
    # loop tries every shape at that size, which means the biggest crop that
    # works always beats a smaller one with a nicer aspect ratio.
    shrink_steps = [1.0 - i * 0.02 for i in range(0, 11)]
    tolerance = max(0.0, cfg.max_filled)
    fallback: Crop | None = None

    for shrink in shrink_steps:
        for _, ratio, cw0, ch0 in candidates:
            cw = int(round(min(cw0 * shrink, W)))
            ch = int(round(min(ch0 * shrink, H)))
            if cw < 8 or ch < 8:
                continue
            # Never shrink so far that the product itself gets cut.
            if cw < sw or ch < sh:
                continue

            # Positions that still contain the whole subject, then as close to
            # the ideal centre as the frame allows.
            x_lo, x_hi = max(0, sx + sw - cw), min(W - cw, sx)
            y_lo, y_hi = max(0, sy + sh - ch), min(H - ch, sy)
            if x_lo > x_hi or y_lo > y_hi:
                continue

            want_x = ideal_cx - cw / 2.0
            want_y = ideal_cy - ch / 2.0 - cfg.center_bias_y * ch

            best: tuple[float, float, int, int] | None = None
            for y in _search_positions(want_y, y_lo, y_hi):
                for x in _search_positions(want_x, x_lo, x_hi):
                    frac = invalid.fraction(x, y, cw, ch)
                    dist = abs(x - want_x) + abs(y - want_y)
                    key = (frac, dist)
                    if best is None or key < (best[0], best[1]):
                        best = (frac, dist, x, y)
                    if frac == 0.0 and dist <= 1.0:
                        break
                if best is not None and best[0] == 0.0 and best[1] <= 1.0:
                    break

            if best is None:
                continue

            frac, _, x, y = best
            if frac <= tolerance:
                found = []
                if shrink < 1.0:
                    found.append(notes.margin_trimmed(1.0 - shrink))
                if frac >= 0.001:   # below this it rounds to "0.0%", so say nothing
                    found.append(notes.edge_rebuilt(frac))
                return Crop(x, y, cw, ch, cw / ch, frac, found)

            if fallback is None or frac < fallback.filled_corners:
                fallback = Crop(x, y, cw, ch, cw / ch, frac, [notes.edge_rebuilt(frac)])

    if fallback is not None:
        return fallback
    return Crop(0, 0, W, H, W / max(H, 1), 0.0, [])


def _search_positions(want: float, lo: int, hi: int, samples: int = 17) -> list[int]:
    """Candidate offsets, nearest to the ideal one first."""
    lo, hi = int(np.floor(lo)), int(np.ceil(hi))
    if hi <= lo:
        return [lo]
    grid = np.unique(
        np.clip(
            np.round(np.linspace(lo, hi, samples)).astype(int)
            .tolist() + [int(round(min(max(want, lo), hi)))],
            lo, hi,
        )
    )
    return sorted(grid.tolist(), key=lambda v: abs(v - want))


def pad_to_ratio(
    image: np.ndarray,
    subject: tuple[int, int, int, int],
    target: float,
) -> tuple[np.ndarray, float, list]:
    """Widen or heighten a finished crop to hit an exact aspect ratio.

    The extra strip is the surface mirrored outwards, which continues a floor
    or a table convincingly. That only holds while the strip is made purely of
    surface, so the pad is capped at the distance from the frame edge to the
    product -- mirroring any further would duplicate part of the product into
    the margin. Returns the image, the fraction that is new, and a note.
    """
    h, w = image.shape[:2]
    current = w / h
    if abs(current - target) / target < 0.005:
        return image, 0.0, []

    sx, sy, sw, sh = subject
    if current < target:                       # too narrow -> pad left/right
        need = int(round(h * target)) - w
        room = min(sx, w - (sx + sw))
        axis = "sides"
    else:                                      # too tall -> pad top/bottom
        need = int(round(w / target)) - h
        room = min(sy, h - (sy + sh))
        axis = "top and bottom"

    if need <= 0:
        return image, 0.0, []

    pad = min(need // 2, max(0, room))
    if pad <= 0:
        return image, 0.0, [notes.pad_blocked(target)]

    if current < target:
        out = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_REFLECT_101)
    else:
        out = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_REFLECT_101)

    added = 1.0 - (w * h) / float(out.shape[0] * out.shape[1])
    found = [notes.padded(axis, added)]
    reached = out.shape[1] / out.shape[0]
    if abs(reached - target) / target > 0.01:   # ignore integer rounding
        found.append(notes.pad_short(target, reached))
    return out, added, found


def apply_crop(image: np.ndarray, crop: Crop) -> np.ndarray:
    x, y, w, h = crop.box
    H, W = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return image[y0:y1, x0:x1].copy()

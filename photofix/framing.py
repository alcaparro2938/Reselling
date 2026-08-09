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
    """Tight bounding box around everything still in the mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _integral_invalid(valid: np.ndarray) -> np.ndarray:
    """Summed-area table of invalid pixels, for O(1) rectangle queries."""
    inv = (valid == 0).astype(np.int64)
    return np.pad(inv.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def _invalid_in(integral: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    x2, y2 = x + w, y + h
    return int(
        integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x]
    )


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
    integral = _integral_invalid(valid)

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

            area = float(cw * ch)
            best: tuple[float, float, int, int] | None = None
            for y in _search_positions(want_y, y_lo, y_hi):
                for x in _search_positions(want_x, x_lo, x_hi):
                    frac = _invalid_in(integral, x, y, cw, ch) / area
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

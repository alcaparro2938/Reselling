"""Finding the product against the surface it was photographed on.

The reference shots are all the same situation: one or more objects laid on a
plain-ish surface (a wooden floor), shot from directly above. That gives us a
strong prior -- the frame border is background -- which is far more reliable
here than any generic saliency model, and it needs no model download.

The mask is built in two passes:

1. A robust colour model of the border band. Every pixel gets a per-channel
   z-score against it in CIELAB. Wood grain moves a lot in lightness but very
   little in chroma, and the robust scale absorbs that automatically.
2. GrabCut, seeded from pass 1, to snap the outline onto the real edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config


@dataclass
class Component:
    """One detected object."""

    label: int
    area: int
    bbox: tuple[int, int, int, int]      # x, y, w, h
    centroid: tuple[float, float]
    touches_border: bool
    rectangularity: float                # 1.0 == a perfect rectangle
    angle: float                         # minAreaRect angle in (-45, 45]


@dataclass
class Segmentation:
    mask: np.ndarray                     # uint8 0/255, at working scale
    components: list[Component]          # kept objects, largest first
    all_components: list[Component]      # before distractor filtering
    scale: float                         # working size / original size
    coverage: float                      # kept mask area / frame area
    confident: bool
    note: str = ""
    work: np.ndarray | None = None       # the downscaled image analysed
    spread: float = 0.0                  # how far the border is from one surface

    @property
    def uniform_surface(self) -> bool:
        """True when the product is lying on a single, even surface."""
        return self.spread <= UNIFORM_SURFACE_MAX


def _robust_stats(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and MAD-based sigma per channel, floored so we never divide by 0."""
    med = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - med), axis=0)
    sigma = 1.4826 * mad
    return med, np.maximum(sigma, 1.5)


def _border_samples(lab: np.ndarray, frac: float) -> np.ndarray:
    h, w = lab.shape[:2]
    by = max(2, int(round(h * frac)))
    bx = max(2, int(round(w * frac)))
    bands = [
        lab[:by, :, :].reshape(-1, 3),
        lab[-by:, :, :].reshape(-1, 3),
        lab[:, :bx, :].reshape(-1, 3),
        lab[:, -bx:, :].reshape(-1, 3),
    ]
    return np.concatenate(bands, axis=0).astype(np.float32)


# Above this the frame border is not one surface but several things -- a wall
# and a floor, a room behind a raised hand -- and the whole border-colour idea
# stops applying. Measured spread is around 0.6 on clean synthetic flat-lays and
# 2.4 on a phone held up in a room. Real flat-lays on a wooden table measure
# 1.2-1.8: the seam between two boards, the corner of the box or a cable at the
# edge all add to it. At 1.0 every one of those had its real wood swapped for a
# generated one, which looks worse and bites into the phone's outline.
UNIFORM_SURFACE_MAX = 2.0


def surface_spread(bgr: np.ndarray, cfg: Config) -> float:
    """How far the frame border is from being a single flat surface.

    Combines the robust spread of the border in each CIELAB channel with the
    share of border pixels that its own colour model calls outliers. 0 is a
    perfectly even surface; 1 is the edge of usable.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    band = _border_samples(lab, cfg.border_frac)
    med, sigma = _robust_stats(band)

    weights = np.array([0.6, 1.0, 1.0], dtype=np.float32)
    z = np.sqrt(np.sum((((band - med) / sigma) * weights) ** 2, axis=1))
    outliers = float((z > cfg.threshold_k).mean())

    return float(max(
        sigma[0] / 25.0,        # lightness: grain and shadow live here
        sigma[1] / 2.5,         # chroma barely moves on one real surface
        sigma[2] / 2.5,
        outliers / 0.12,
    ))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes (a phone screen can read as 'background')."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _distance_mask(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    med, sigma = _robust_stats(_border_samples(lab, cfg.border_frac))

    z = (lab - med) / sigma
    # Lightness is the noisiest channel on a textured surface, so it counts
    # less than chroma when deciding what is an object.
    weights = np.array([0.6, 1.0, 1.0], dtype=np.float32)
    dist = np.sqrt(np.sum((z * weights) ** 2, axis=2))

    raw = (dist > cfg.threshold_k).astype(np.uint8) * 255

    h, w = bgr.shape[:2]
    k = max(3, (int(round(min(h, w) * 0.012)) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel, iterations=1)
    return _fill_holes(raw)


def _grabcut_ring(bgr: np.ndarray, cfg: Config) -> int:
    """Width of the frame band GrabCut is forced to treat as background."""
    h, w = bgr.shape[:2]
    return max(2, int(round(min(h, w) * cfg.border_frac * 0.5)))


def _grabcut(bgr: np.ndarray, seed: np.ndarray, cfg: Config) -> np.ndarray:
    """Refine `seed` with GrabCut. Returns the seed unchanged on any failure."""
    h, w = bgr.shape[:2]
    k = max(3, (int(round(min(h, w) * 0.03)) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    sure_fg = cv2.erode(seed, kernel, iterations=1)
    near = cv2.dilate(seed, kernel, iterations=2)

    gc = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc[near > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    gc[near == 0] = cv2.GC_BGD
    # The outermost ring is background by construction.
    b = _grabcut_ring(bgr, cfg)
    gc[:b, :] = cv2.GC_BGD
    gc[-b:, :] = cv2.GC_BGD
    gc[:, :b] = cv2.GC_BGD
    gc[:, -b:] = cv2.GC_BGD

    if not np.any(gc == cv2.GC_FGD):
        return seed

    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, gc, None, bgd, fgd, cfg.grabcut_iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return seed

    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    area = out.mean() / 255.0
    if area < 0.005 or area > 0.92:
        return seed  # GrabCut collapsed or swallowed the frame

    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kk, iterations=2)
    return _fill_holes(out)


def _describe(mask: np.ndarray, margin: int | None = None) -> tuple[list[Component], np.ndarray]:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    h, w = mask.shape[:2]
    if margin is None:
        margin = max(2, int(round(min(h, w) * 0.005)))
    out: list[Component] = []

    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        blob = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        (_, _), (rw, rh), ang = cv2.minAreaRect(cnt)
        rect_area = max(rw * rh, 1.0)
        rectangularity = float(cv2.contourArea(cnt) / rect_area)

        # minAreaRect reports (-90, 0]; a rectangle is symmetric every 90
        # degrees, so fold it into the shortest correction.
        ang = ang % 90.0
        if ang > 45.0:
            ang -= 90.0

        touches = (
            x <= margin or y <= margin
            or x + cw >= w - margin or y + ch >= h - margin
        )
        out.append(
            Component(
                label=i,
                area=int(area),
                bbox=(int(x), int(y), int(cw), int(ch)),
                centroid=(float(centroids[i][0]), float(centroids[i][1])),
                touches_border=bool(touches),
                rectangularity=rectangularity,
                angle=float(ang),
            )
        )

    out.sort(key=lambda c: c.area, reverse=True)
    return out, labels


def _filter(components: list[Component], frame_area: int, cfg: Config) -> list[Component]:
    """Drop the things the seller does not want in the photo.

    Two rules cover nearly every real case: anything cut off by the frame edge
    is furniture or a hand, not the product; anything tiny next to the main
    object is a crumb or a shadow speck.
    """
    if not components:
        return []

    significant = [c for c in components if c.area >= frame_area * 0.0015]
    if not significant:
        significant = components[:1]

    main = significant[0]
    kept = [main]
    if not cfg.drop_distractors:
        return significant

    for c in significant[1:]:
        if c.area < main.area * cfg.min_object_ratio:
            continue
        if c.touches_border:
            continue  # cut off by the frame -> not part of the listing
        kept.append(c)

    kept.sort(key=lambda c: c.area, reverse=True)
    return kept


def segment(bgr: np.ndarray, cfg: Config) -> Segmentation:
    """Locate the product(s). Always returns a result; check `.confident`."""
    h, w = bgr.shape[:2]
    scale = min(1.0, cfg.work_size / max(h, w))
    work = (
        cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else bgr.copy()
    )

    spread = surface_spread(work, cfg)
    seed = _distance_mask(work, cfg)
    coverage_seed = float(seed.mean() / 255.0)

    note = ""
    if coverage_seed < 0.004:
        return Segmentation(seed, [], [], scale, coverage_seed, False,
                            "no object stood out from the background", work, spread)
    if coverage_seed > 0.90:
        return Segmentation(seed, [], [], scale, coverage_seed, False,
                            "the background is not uniform enough to isolate the product",
                            work, spread)

    mask = _grabcut(work, seed, cfg) if cfg.grabcut else seed

    # GrabCut is told the outermost ring is background, so after it nothing
    # can reach the true frame edge. Measured against the frame itself, the
    # "cut off by the edge" rule would never fire and a box corner or a cable
    # poking in from the side would be kept as part of the listing. Measure
    # against the edge of the ring instead.
    ring = _grabcut_ring(work, cfg) if cfg.grabcut else 0
    all_comps, labels = _describe(mask, margin=ring + max(2, int(round(min(work.shape[:2]) * 0.005))))
    frame_area = work.shape[0] * work.shape[1]
    kept = _filter(all_comps, frame_area, cfg)

    if not kept:
        return Segmentation(mask, [], all_comps, scale, 0.0, False,
                            "no object survived filtering", work, spread)

    keep_mask = np.zeros_like(mask)
    for c in kept:
        keep_mask[labels == c.label] = 255

    coverage = float(keep_mask.mean() / 255.0)
    confident = 0.004 <= coverage <= 0.85
    if not confident:
        note = "product coverage looks implausible"

    dropped = len(all_comps) - len(kept)
    if dropped > 0 and confident:
        note = f"ignored {dropped} object(s) outside the product"

    return Segmentation(keep_mask, kept, all_comps, scale, coverage, confident, note, work, spread)

"""Turn a phone upright: the camera or the Dynamic Island at the top.

Straightening fixes a few degrees. This fixes the quarter turns: a phone lying
across the frame, or upside down. Nothing in the pixels says which way is up
in general, but a phone says it twice:

- from the back, the camera module sits in the top-left corner. It is by far
  the busiest part of an otherwise flat back, so the corner with the most
  edges is the top-left one;
- from the front, the Dynamic Island is a dark pill centred near the top
  edge, separated from the bezel by a strip of screen.

The phone is first located as a rounded rectangle with a phone's proportions
(inside a box tray or on its own), then cut out and laid along its long axis
both ways round. Whichever way round shows the feature where it belongs is
up. When neither shows clearly, nothing turns: a photo turned the wrong way is
worse than one left as it was, and the interface lets the seller turn it with
one tap.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# iPhones from the X on are 2.05-2.17 long to wide; the range is wide enough
# for the outline to include a bit of shadow or tray.
PHONE_ASPECT = (1.8, 2.45)
CANON_WIDTH = 300
# Edge density across the middle of the body. A matte back measures about
# 0.01, or 0.025 when the outline found includes the edge of a box tray;
# a screen showing anything measures 0.03 and up, though a dark one can come
# lower -- which is why the island is looked for first.
BACK_MAX_MID_EDGES = 0.03


@dataclass
class Orientation:
    turns: int          # quarter turns counter-clockwise, as np.rot90 takes them
    reason: str         # "camera", "island", or why nothing was decided
    confident: bool


def _candidates(gray: np.ndarray, mask: np.ndarray):
    """Rectangles inside the product: the product itself, and closed outlines
    within it (a phone lying in its box tray)."""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return [], None, 0.0
    main = max(cs, key=cv2.contourArea)
    main_area = cv2.contourArea(main)
    if main_area <= 0:
        return [], None, 0.0

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    regions = cv2.bitwise_not(edges)
    regions[mask == 0] = 0
    inner, _ = cv2.findContours(regions, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for c in [main, *inner]:
        a = cv2.contourArea(c)
        if a < 0.08 * main_area:
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if min(rw, rh) < 12:
            continue
        out.append((a, max(rw, rh) / min(rw, rh), a / (rw * rh), rect))
    main_rect = cv2.minAreaRect(main)
    fill = main_area / max(main_rect[1][0] * main_rect[1][1], 1.0)
    return out, main_rect, fill


def _canonical(gray: np.ndarray, rect, end: int, width: int = CANON_WIDTH):
    """The rectangle's contents, long axis vertical, with short edge `end` at
    the top. Also returns the direction from the centre to that edge."""
    pts = cv2.boxPoints(rect)
    centre = np.asarray(rect[0], np.float32)
    edges = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    lengths = [float(np.linalg.norm(a - b)) for a, b in edges]
    shorts = sorted(sorted(range(4), key=lambda i: lengths[i])[:2])
    p, q = edges[shorts[end]]
    # Keep the handedness: going top-left -> top-right -> centre must turn
    # clockwise in image coordinates, or the patch comes out mirrored.
    v1, v2 = q - p, centre - p
    if v1[0] * v2[1] - v1[1] * v2[0] < 0:
        p, q = q, p
    rest = [x for x in pts if not (np.allclose(x, p) or np.allclose(x, q))]
    bl = min(rest, key=lambda x: np.linalg.norm(x - p))
    br = next(x for x in rest if not np.allclose(x, bl))
    aspect = max(rect[1]) / max(min(rect[1]), 1e-6)
    height = int(round(width * aspect))
    m = cv2.getPerspectiveTransform(
        np.float32([p, q, br, bl]),
        np.float32([[0, 0], [width, 0], [width, height], [0, height]]),
    )
    patch = cv2.warpPerspective(gray, m, (width, height), flags=cv2.INTER_AREA,
                                borderMode=cv2.BORDER_REPLICATE)
    return patch, (p + q) / 2.0 - centre


def _corner_edges(patch: np.ndarray) -> tuple[float, float]:
    """Edge density of the top-left corner, and of the middle of the body."""
    h, w = patch.shape
    e = cv2.Canny(cv2.GaussianBlur(patch, (3, 3), 0), 30, 90) > 0
    tl = e[int(0.03 * h):int(0.25 * h), int(0.05 * w):int(0.55 * w)].mean()
    mid = e[int(0.35 * h):int(0.65 * h), int(0.10 * w):int(0.90 * w)].mean()
    return float(tl), float(mid)


def _has_island(patch: np.ndarray) -> bool:
    """A dark pill near the top, centred, not joined to the bezel.

    Tried at several darkness levels: on a bright screen the island separates
    at almost any level, on a dark one only at the very darkest, where the
    bezel and the strip of screen between them still differ.
    """
    h, w = patch.shape
    y1 = int(0.14 * h)
    x0, x1 = int(0.10 * w), int(0.90 * w)
    region = patch[:y1, x0:x1]
    if region.size == 0:
        return False
    for t in (10, 16, 24, 34, 46, 60):
        dark = (region < t).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(dark, 8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if x <= 1 or y <= 1 or x + bw >= region.shape[1] - 1:
                continue        # joined to the bezel or the outline
            cx = (x + x0 + bw / 2.0) / w
            if not (0.30 <= cx <= 0.70):
                continue
            if not (0.16 <= bw / w <= 0.50 and 0.15 <= bh / bw <= 0.55):
                continue
            if area / float(bw * bh) < 0.65:
                continue        # a pill is solid
            blob = labels == i
            ring = cv2.dilate(blob.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            ring &= ~blob
            if not ring.any():
                continue
            if float(region[ring].mean()) - float(region[blob].mean()) < 15:
                continue
            return True
    return False


def _turns_for(direction: np.ndarray) -> int:
    """Quarter turns (counter-clockwise) that point `direction` up."""
    best, best_up = 0, -np.inf
    x, y = float(direction[0]), float(direction[1])
    for k in range(4):
        up = -y
        if up > best_up:
            best, best_up = k, up
        x, y = y, -x          # one more counter-clockwise quarter turn
    return best


def detect(bgr: np.ndarray, mask: np.ndarray) -> Orientation:
    """Decide how many quarter turns make the phone upright.

    `mask` is the product mask at the same size as `bgr`.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cands, main_rect, main_fill = _candidates(gray, mask)
    if main_rect is None:
        return Orientation(0, "no product", False)

    phones = [c for c in cands
              if PHONE_ASPECT[0] <= c[1] <= PHONE_ASPECT[1] and c[2] >= 0.85]
    if phones:
        rect, whole = max(phones, key=lambda c: c[0])[3], True
    else:
        # Not a whole phone in view -- a close-up of the camera, say. The
        # camera corner still says which way is up; the island cannot, since
        # it is only meaningful relative to the whole front.
        rect, whole = main_rect, False
        rw, rh = rect[1]
        if main_fill < 0.8 or min(rw, rh) < 12 or max(rw, rh) / min(rw, rh) > 3.0:
            # A phone seen edge-on in a hand is neither: nothing to read.
            return Orientation(0, "no phone outline", False)

    views = [_canonical(gray, rect, end) for end in (0, 1)]
    stats = [_corner_edges(p) for p, _ in views]

    # Front first: the island is a much more specific shape than "a busy
    # corner", and a dark screen can be as quiet across the middle as a back.
    if whole:
        islands = [_has_island(p) for p, _ in views]
        if islands[0] != islands[1]:
            end = 0 if islands[0] else 1
            return Orientation(_turns_for(views[end][1]), "island", True)

    # Back: a flat body with one busy corner. Which end puts that corner
    # top-left decides it.
    mids = [m for _, m in stats]
    tls = [t for t, _ in stats]
    hi, lo = (0, 1) if tls[0] >= tls[1] else (1, 0)
    if whole:
        # A whole back is matte and nearly featureless across the middle; a
        # screen is not. Without this a busy bottom-left of a screen (the
        # Control Centre toggles) would pass for a camera.
        back = max(mids) < BACK_MAX_MID_EDGES and tls[hi] >= 2.0 * max(tls[lo], 0.01)
    else:
        # A close-up has no "middle" to judge by; ask for a wider margin.
        back = tls[hi] >= 2.5 * max(tls[lo], 0.01)
    if back and tls[hi] >= 2.5 * max(mids):
        return Orientation(_turns_for(views[hi][1]), "camera", True)

    return Orientation(0, "no clear camera or island", False)

"""Working out how crooked the photo is, and rotating it without damage."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .segment import Component


@dataclass
class Rotation:
    angle: float                 # degrees, positive = image rotated CCW to fix
    image: np.ndarray            # rotated, full resolution
    mask: np.ndarray             # subject mask, same size as `image`
    valid: np.ndarray            # 255 where pixels came from the original
    source: str                  # how the angle was decided


def _circular_mean_mod90(angles: np.ndarray, weights: np.ndarray) -> float:
    """Average angles that are only meaningful modulo 90 degrees.

    A rectangle tilted by 1 degree and one tilted by 89 are the same tilt, so a
    plain arithmetic mean would be nonsense. Mapping to the complex unit circle
    at 4x the angle makes 90 degrees a full turn and the mean well defined.
    """
    if len(angles) == 0:
        return 0.0
    vec = np.sum(weights * np.exp(1j * np.deg2rad(angles * 4.0)))
    if abs(vec) < 1e-9:
        return 0.0
    mean = np.rad2deg(np.angle(vec)) / 4.0
    return float(mean)


# A circle fills exactly pi/4 = 0.785 of its bounding rectangle, and the
# "angle" of a round object is meaningless -- a coiled cable would otherwise
# vote on how to straighten the photo. Phones, boxes and cases all sit above
# 0.94, so this cuts cleanly between the two.
MIN_RECTANGULARITY = 0.85

# Two objects lying on the same surface should agree to well within a degree.
INLIER_TOLERANCE = 1.2


def _deviation(angles: np.ndarray, reference: float) -> np.ndarray:
    """Smallest angular distance to `reference`, modulo 90 degrees."""
    return np.abs(((angles - reference + 45.0) % 90.0) - 45.0)


def _angle_from_components(components: list[Component]) -> tuple[float, float, str]:
    """Angle from the rectangular objects in the frame, with a 0..1 confidence.

    A rounded corner can make minAreaRect snap its answer to the bounding box
    instead of the true edge, which produces one wildly wrong vote among
    several good ones. So the consensus is found first and the outliers are
    dropped, rather than averaging everything together.
    """
    rects = [c for c in components if c.rectangularity >= MIN_RECTANGULARITY]
    if not rects:
        return 0.0, 0.0, "no straight-edged object"

    angles = np.array([c.angle for c in rects], dtype=np.float64)
    # Big, crisply rectangular objects (the phone, the box) dominate.
    weights = np.array(
        [np.sqrt(c.area) * (c.rectangularity ** 2) for c in rects], dtype=np.float64
    )

    # Weighted geometric median: the candidate closest to all the others.
    costs = [float(np.sum(weights * _deviation(angles, a))) for a in angles]
    consensus = float(angles[int(np.argmin(costs))])

    inliers = _deviation(angles, consensus) <= INLIER_TOLERANCE
    inlier_weight = float(np.sum(weights[inliers]) / np.sum(weights))
    angle = _circular_mean_mod90(angles[inliers], weights[inliers])

    spread = _deviation(angles[inliers], angle)
    disagreement = float(np.average(spread, weights=weights[inliers]))
    confidence = inlier_weight * float(np.clip(1.0 - disagreement / 3.0, 0.0, 1.0))
    if len(rects) == 1:
        confidence = min(1.0, rects[0].rectangularity / 0.95) * 0.9

    n_out = int(np.sum(~inliers))
    detail = f"{int(np.sum(inliers))} of {len(rects)} straight-edged object(s)"
    if n_out:
        detail += f", {n_out} outlier(s) discarded"
    return angle, confidence, detail


def _angle_from_edges(bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float, str]:
    """The dominant edge direction around the product's outline.

    Measured from image gradients rather than detected line segments: a line
    detector quantises its angles and votes in whole pixels, which costs a few
    tenths of a degree, and on a phone's long side that is visible. Gradient
    orientation is continuous, and weighting by magnitude squared means the
    hard product edge outvotes the soft shadow beside it.
    """
    h, w = mask.shape[:2]
    k = max(3, (int(round(min(h, w) * 0.02)) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ring = cv2.subtract(cv2.dilate(mask, kernel), cv2.erode(mask, kernel))
    if not ring.any():
        return 0.0, 0.0, "no outline"

    # Scharr rather than Sobel: the 3x3 Sobel kernel is not rotationally
    # symmetric and drags orientation estimates toward the pixel axes, which
    # measured as a consistent ~0.4 degree shortfall on tilted test scenes.
    sigma = max(1.2, min(h, w) / 300.0)
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), sigma)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)

    sel = ring > 0
    gx, gy = gx[sel], gy[sel]
    mag2 = gx * gx + gy * gy
    if mag2.size == 0 or not np.any(mag2 > 0):
        return 0.0, 0.0, "no gradient"

    # Only the strongest edges get a say; soft shading gradients are dropped.
    cutoff = np.percentile(mag2, 90.0)
    strong = mag2 >= max(cutoff, 1e-6)
    if np.count_nonzero(strong) < 32:
        return 0.0, 0.0, "too few strong edges"
    gx, gy, mag2 = gx[strong], gy[strong], mag2[strong]

    # A rectangle's gradients point along two perpendicular directions, so the
    # signal repeats every 90 degrees; multiplying the angle by 4 turns that
    # into a full turn and makes the mean well defined.
    phi = np.arctan2(gy, gx)
    resultant = np.sum(mag2 * np.exp(1j * 4.0 * phi))
    total = float(np.sum(mag2))
    if total <= 0 or abs(resultant) <= 0:
        return 0.0, 0.0, "no dominant direction"

    orientation = np.rad2deg(np.angle(resultant)) / 4.0
    # Gradients run perpendicular to the edge they sit on.
    angle = orientation % 90.0
    if angle > 45.0:
        angle -= 90.0

    concentration = float(abs(resultant) / total)   # 1.0 == perfectly aligned
    confidence = float(np.clip(concentration, 0.0, 1.0))
    return float(angle), confidence, f"{int(np.count_nonzero(strong))} edge pixels"


def estimate_angle(
    work_bgr: np.ndarray,
    mask: np.ndarray,
    components: list[Component],
    cfg: Config,
) -> tuple[float, str, float | None]:
    """Return the correction angle, a short explanation, and -- when the
    correction was refused for being implausibly large -- the angle that was
    rejected, so the caller can report it.

    A positive angle means the content must be turned counter-clockwise to
    stand up straight.
    """
    a_comp, c_comp, why_comp = _angle_from_components(components)
    a_edge, c_edge, why_edge = _angle_from_edges(work_bgr, mask)

    if c_comp <= 0.0 and c_edge <= 0.0:
        return 0.0, "could not find any straight edge to align to", None

    if c_comp > 0.0 and c_edge > 0.0:
        diff = abs(((a_comp - a_edge + 45.0) % 90.0) - 45.0)
        if diff <= 2.5:
            # Both agree, so average them -- but the outline of the product is
            # the thing that has to end up square with the frame, and fitting a
            # rectangle to it measured more accurately than the gradient
            # statistic on every test scene. The gradient estimate is here
            # mainly to catch a rectangle fit that snapped to the wrong side,
            # so it gets the smaller share.
            angle = _circular_mean_mod90(
                np.array([a_comp, a_edge]),
                np.array([2.0 * c_comp ** 2, c_edge ** 2]),
            )
            source = f"object shape + edges agree ({why_comp}, {why_edge})"
        elif c_comp >= c_edge:
            angle, source = a_comp, f"object shape ({why_comp})"
        else:
            angle, source = a_edge, f"edge direction ({why_edge})"
    elif c_comp > 0.0:
        angle, source = a_comp, f"object shape ({why_comp})"
    else:
        angle, source = a_edge, f"edge direction ({why_edge})"

    if abs(angle) > cfg.max_rotation:
        return 0.0, "refused: tilt beyond the safety limit", float(angle)
    if abs(angle) < cfg.min_rotation:
        return 0.0, "already straight", None

    return float(angle), source, None


def rotate(
    bgr: np.ndarray,
    mask_full: np.ndarray,
    angle: float,
    fill_bgr: tuple[int, int, int],
) -> Rotation:
    """Rotate image and mask about the centre, growing the canvas to keep
    every original pixel.

    Corners the rotation exposes are filled with the surface's own colour, and
    `valid` records exactly where those are so the framing step can keep them
    out of the final crop.
    """
    h, w = bgr.shape[:2]
    if abs(angle) < 1e-6:
        return Rotation(0.0, bgr, mask_full, np.full((h, w), 255, np.uint8), "no rotation")

    center = (w / 2.0, h / 2.0)
    # OpenCV's positive angle is counter-clockwise, which matches our sign.
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))
    m[0, 2] += new_w / 2.0 - center[0]
    m[1, 2] += new_h / 2.0 - center[1]

    img = cv2.warpAffine(
        bgr, m, (new_w, new_h), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(int(v) for v in fill_bgr),
    )
    msk = cv2.warpAffine(
        mask_full, m, (new_w, new_h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    valid = cv2.warpAffine(
        np.full((h, w), 255, np.uint8), m, (new_w, new_h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    # Nearest-neighbour leaves a ragged one-pixel fringe on the valid edge.
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8), iterations=1)
    return Rotation(float(angle), img, msk, valid, "rotated")


def reconstruct_edges(bgr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Rebuild the wedges that rotating exposed at the frame edge.

    They are filled with a flat colour by `rotate`, which reads as a painted
    border. Inpainting from the neighbouring pixels continues the grain of the
    surface instead, and for the thin slivers the framer allows the seam is not
    visible.
    """
    holes = (valid == 0).astype(np.uint8)
    if not holes.any():
        return bgr
    # Reach a little into the real pixels so the inpainter has something to
    # copy the texture from.
    holes = cv2.dilate(holes, np.ones((5, 5), np.uint8), iterations=1)
    radius = max(3, int(round(min(bgr.shape[:2]) * 0.01)))
    return cv2.inpaint(bgr, holes, radius, cv2.INPAINT_TELEA)


def background_color(bgr: np.ndarray, border_frac: float = 0.05) -> tuple[int, int, int]:
    """Median colour of the frame border -- the surface the product sits on."""
    h, w = bgr.shape[:2]
    by = max(1, int(round(h * border_frac)))
    bx = max(1, int(round(w * border_frac)))
    samples = np.concatenate(
        [
            bgr[:by, :, :].reshape(-1, 3),
            bgr[-by:, :, :].reshape(-1, 3),
            bgr[:, :bx, :].reshape(-1, 3),
            bgr[:, -bx:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    med = np.median(samples, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))

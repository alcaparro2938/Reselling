"""Cutting the product out of a photo, whatever it was shot against.

`segment.py` assumes the product is lying on a plain surface and reads the
background off the frame border. That is the good case and it stays the
default. It cannot cope with a product held in a hand, or shot against a
cluttered room -- there is no border colour to model.

This module handles those, with a small saliency network (u2netp, 4.5 MB) run
through OpenCV's own ONNX support. Using OpenCV rather than onnxruntime is not
incidental: onnxruntime costs about 300 MB of resident memory just to import,
which does not fit the machines this is meant to run on, and produces the same
mask to four decimal places.
"""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import Config

MODEL_URL = (
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
)
MODEL_SHA256 = "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"
# u2netp was trained at 320, but it is fully convolutional and the mask is
# upscaled afterwards anyway. Measured against the 320 result, 256 agrees to an
# IoU of 0.998 while the convolution buffers drop from 170 MB to 71 MB, which
# is the difference between fitting on a small host and being killed on it.
MODEL_SIZE = int(os.environ.get("PHOTOFIX_MODEL_SIZE", "256"))

_NET = None
_NET_LOCK = threading.Lock()


@dataclass
class Cutout:
    alpha: np.ndarray                 # uint8, 0..255, full resolution
    covered: np.ndarray               # where a finger sat on top of the product
    method: str                       # "model" | "surface"
    hand_removed: bool = False
    hand_overlap: float = 0.0         # of the product, how much a hand covered
    confident: bool = True
    note: str = ""


def model_dir() -> Path:
    return Path(
        os.environ.get("PHOTOFIX_MODEL_DIR")
        or Path.home() / ".cache" / "photofix"
    )


def model_path() -> Path:
    return model_dir() / "u2netp.onnx"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(download: bool = True) -> Path | None:
    """Return the model file, fetching it once if allowed.

    The hash is checked on arrival: this file is executed as a network, and a
    truncated download would fail in a confusing way much later.
    """
    path = model_path()
    if path.exists():
        if _sha256(path) == MODEL_SHA256:
            return path
        path.unlink()                 # corrupt or truncated, fetch it again

    if not download:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        if _sha256(tmp) != MODEL_SHA256:
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(path)
        return path
    except Exception:
        tmp.unlink(missing_ok=True)
        return None


def model_available(download: bool = False) -> bool:
    return ensure_model(download=download) is not None


def _net(download: bool = True):
    global _NET
    with _NET_LOCK:
        if _NET is None:
            path = ensure_model(download=download)
            if path is None:
                return None
            _NET = cv2.dnn.readNetFromONNX(str(path))
        return _NET


def saliency_alpha(bgr: np.ndarray, download: bool = True) -> np.ndarray | None:
    """Probability that each pixel belongs to the subject, 0..255."""
    net = _net(download=download)
    if net is None:
        return None

    h, w = bgr.shape[:2]
    # Resized while still 8-bit: a full-resolution float copy of a 12 megapixel
    # photo is 150 MB, and the network only ever sees MODEL_SIZE square.
    peak = float(bgr.max())
    small = cv2.resize(bgr, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / max(peak, 1e-6)

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    blob = ((rgb - mean) / std).transpose(2, 0, 1)[None]

    net.setInput(blob)
    d = net.forward(net.getUnconnectedOutLayersNames())[0][0, 0]
    lo, hi = float(d.min()), float(d.max())
    d = (d - lo) / max(hi - lo, 1e-8)
    return cv2.resize((d * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Where the picture looks like skin.

    Two colour spaces have to agree. YCrCb alone calls warm wood skin, which
    would eat the very surface we are trying to keep.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    a = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    b = cv2.inRange(hsv, (0, 30, 60), (25, 170, 255))
    c = cv2.inRange(hsv, (160, 30, 60), (180, 170, 255))    # hue wraps at red
    mask = cv2.bitwise_and(a, cv2.bitwise_or(b, c))

    k = max(3, (int(round(min(bgr.shape[:2]) * 0.012)) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def _largest_regions(mask: np.ndarray, keep_ratio: float = 0.08) -> np.ndarray:
    """Keep the biggest blob plus anything comparable to it."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(areas.max())
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= biggest * keep_ratio:
            out[labels == i] = 255
    return out


def _drop_specks(mask: np.ndarray, min_frac: float = 0.001) -> np.ndarray:
    """Remove blobs smaller than a fraction of the whole frame.

    Used for skin rather than `_largest_regions`, which measures against the
    biggest blob: a palm dwarfs a fingertip, so relative filtering throws the
    fingers away and leaves them sitting on the product.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    floor = mask.shape[0] * mask.shape[1] * min_frac
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= floor:
            out[labels == i] = 255
    return out


def _feather(alpha: np.ndarray, span: int) -> np.ndarray:
    """Soften the cut by a pixel or two so it does not look die-cut."""
    r = max(1.0, span * 0.0015)
    return cv2.GaussianBlur(alpha, (0, 0), r)


def remove_hand(
    alpha: np.ndarray, bgr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Take the hand out, keeping the product's own outline intact.

    Subtracting skin from the mask is only half the job. A hand *behind* the
    product just disappears, but fingers curled over the front leave bites out
    of the silhouette and holes in the middle, which looks far worse than the
    hand did.

    So the product's true outline is recovered as the convex hull of the part
    that is definitely not skin -- a phone, a box or a case is convex, and the
    hull of one with a finger across it is still the object. Skin inside that
    hull is a finger on top of the product, and comes back as a region to
    reconstruct; skin outside it is the hand, and simply goes.

    Returns the product mask, the area a finger was covering, whether a hand
    was found, and how much of the product it hid.
    """
    empty = np.zeros_like(alpha)
    skin = cv2.bitwise_and(skin_mask(bgr), alpha)
    total = int((alpha > 127).sum())
    if total == 0:
        return alpha, empty, False, 0.0

    skin = _drop_specks(skin, min_frac=0.001)
    if int((skin > 127).sum()) < total * 0.02:     # a warm reflection, not a hand
        return alpha, empty, False, 0.0

    k = max(3, (int(round(min(bgr.shape[:2]) * 0.006)) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    skin = cv2.dilate(skin, kernel, iterations=1)  # catch the shaded rim

    core = _largest_regions(cv2.bitwise_and(alpha, cv2.bitwise_not(skin)), 0.08)
    if int((core > 127).sum()) < total * 0.15:
        # Almost all of the subject read as skin. Believing that would delete
        # the product, so change nothing.
        return alpha, empty, False, 0.0

    contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return alpha, empty, False, 0.0
    hull = np.zeros_like(alpha)
    cv2.drawContours(hull, [cv2.convexHull(np.vstack(contours))], -1, 255, cv2.FILLED)

    product = cv2.bitwise_or(core, cv2.bitwise_and(alpha, hull))
    product = _largest_regions(product, keep_ratio=0.08)

    covered = cv2.bitwise_and(skin, product)
    product_area = max(int((product > 127).sum()), 1)
    return product, covered, True, int((covered > 127).sum()) / product_area


# Everything here -- skin detection, morphology, connected components -- is
# decided on shapes, not on fine detail, and every one of those operations
# scales with the pixel count. A 12 megapixel frame takes half a minute and
# hundreds of megabytes to reach the same answer this does in under a second.
WORK_MAX_SIDE = 1200


def cutout(bgr: np.ndarray, cfg: Config, download: bool = True) -> Cutout | None:
    """Alpha for the product alone. None when no model is available."""
    full_h, full_w = bgr.shape[:2]
    scale = min(1.0, WORK_MAX_SIDE / max(full_h, full_w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (max(1, round(full_w * scale)), max(1, round(full_h * scale))),
                         interpolation=cv2.INTER_AREA)

    alpha = saliency_alpha(bgr, download=download)
    if alpha is None:
        return None

    span = max(bgr.shape[:2])
    hard = (alpha > 127).astype(np.uint8) * 255
    hard = _largest_regions(hard, keep_ratio=0.08)

    hand_removed, overlap = False, 0.0
    covered = np.zeros_like(hard)
    if cfg.remove_hands:
        hard, covered, hand_removed, overlap = remove_hand(hard, bgr)

    contours, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(hard)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    coverage = float((filled > 127).mean())
    confident = 0.005 <= coverage <= 0.92
    note = "" if confident else "the cut-out covers an implausible area"

    out_alpha = _feather(filled, span)
    out_covered = cv2.bitwise_and(covered, filled)
    if scale < 1.0:
        full = (full_w, full_h)
        out_alpha = cv2.resize(out_alpha, full, interpolation=cv2.INTER_LINEAR)
        out_covered = cv2.resize(out_covered, full, interpolation=cv2.INTER_NEAREST)

    return Cutout(
        alpha=out_alpha,
        covered=out_covered,
        method="model",
        hand_removed=hand_removed,
        hand_overlap=overlap,
        confident=confident,
        note=note,
    )

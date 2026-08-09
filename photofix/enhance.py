"""Gentle finishing so a batch of photos looks like one consistent set.

Deliberately conservative: the reference shots keep the warm tone of the wooden
floor, so nothing here tries to neutralise the surface colour unless asked.
"""

from __future__ import annotations

import cv2
import numpy as np


PRESETS: dict[str, dict[str, float]] = {
    "none":   {"levels": 0.00, "saturation": 1.00, "sharpen": 0.00},
    "light":  {"levels": 0.50, "saturation": 1.04, "sharpen": 0.25},
    "strong": {"levels": 0.85, "saturation": 1.10, "sharpen": 0.50},
}


def auto_levels(bgr: np.ndarray, strength: float = 0.5,
                low_pct: float = 0.4, high_pct: float = 99.6) -> np.ndarray:
    """Stretch the tonal range, luminance only so colours do not shift."""
    if strength <= 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    lo, hi = np.percentile(L, [low_pct, high_pct])
    if hi - lo < 8:
        return bgr

    stretched = np.clip((L - lo) * (255.0 / (hi - lo)), 0, 255)
    blended = L * (1.0 - strength) + stretched * strength
    lab[:, :, 0] = np.clip(blended, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def white_balance(bgr: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """Shades-of-grey illuminant correction, applied partially.

    Off by default: on a warm wooden surface a full correction drains the
    colour the reference photos rely on.
    """
    if strength <= 0:
        return bgr
    img = bgr.astype(np.float32) + 1e-6
    p = 6.0
    illum = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p)
    gain = illum.mean() / illum
    gain = 1.0 + (gain - 1.0) * strength
    return np.clip(img * gain, 0, 255).astype(np.uint8)


def saturate(bgr: np.ndarray, factor: float = 1.0) -> np.ndarray:
    if abs(factor - 1.0) < 1e-3:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def unsharp(bgr: np.ndarray, amount: float = 0.25) -> np.ndarray:
    """Restore the crispness that resampling costs. Radius scales with size."""
    if amount <= 0:
        return bgr
    sigma = max(0.8, min(bgr.shape[:2]) / 900.0)
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigma)
    return cv2.addWeighted(bgr, 1.0 + amount, blurred, -amount, 0)


def finish(bgr: np.ndarray, preset: str = "light", do_white_balance: bool = False) -> np.ndarray:
    params = PRESETS.get((preset or "none").lower(), PRESETS["light"])
    out = bgr
    if do_white_balance:
        out = white_balance(out, 0.4)
    out = auto_levels(out, params["levels"])
    out = saturate(out, params["saturation"])
    out = unsharp(out, params["sharpen"])
    return out

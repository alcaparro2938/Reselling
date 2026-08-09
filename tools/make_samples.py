"""Generate test scenes that imitate the reference shots.

A wooden floor, one or more objects laid on it, photographed from above and
tilted by a known angle. Because the tilt is known, the straightening step can
be scored instead of eyeballed.

    python tools/make_samples.py --out samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

rng = np.random.default_rng(20240624)


# --------------------------------------------------------------------------
# surface
# --------------------------------------------------------------------------
def wood_floor(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Light oak laminate: warm base, vertical grain, plank seams."""
    r = np.random.default_rng(seed)
    base = np.array([150.0, 176.0, 198.0])  # BGR, warm
    img = np.tile(base, (h, w, 1))

    x = np.arange(w)[None, :]
    y = np.arange(h)[:, None]

    grain = (
        9.0 * np.sin(x * 0.35 + r.uniform(0, 6))
        + 6.0 * np.sin(x * 0.11 + 2.0 * np.sin(y * 0.004))
        + 4.0 * np.sin(x * 1.7 + r.uniform(0, 6))
    )
    grain = grain + r.normal(0, 4.0, size=(h, w))
    grain = cv2.GaussianBlur(grain.astype(np.float32), (0, 0), 1.2)
    img += grain[:, :, None] * np.array([0.9, 1.0, 1.05])

    # plank seams every ~38% of the width
    seam_period = max(120, int(w * 0.38))
    for sx in range(seam_period, w, seam_period):
        jitter = int(r.integers(-8, 8))
        cv2.line(img, (sx + jitter, 0), (sx + jitter, h), (118, 140, 160), 3, cv2.LINE_AA)

    # soft, uneven daylight
    yy, xx = np.mgrid[0:h, 0:w]
    fall = 1.0 - 0.16 * (
        ((xx - w * 0.42) / w) ** 2 + ((yy - h * 0.38) / h) ** 2
    ) * 4.0
    img *= np.clip(fall, 0.72, 1.06)[:, :, None]

    return np.clip(img, 0, 255).astype(np.uint8)


def _rounded_rect_mask(h: int, w: int, radius: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(m, (0, radius), (w, h - radius), 255, -1)
    for cx, cy in ((radius, radius), (w - radius, radius),
                   (radius, h - radius), (w - radius, h - radius)):
        cv2.circle(m, (cx, cy), radius, 255, -1)
    return m


def _paste(canvas: np.ndarray, patch: np.ndarray, mask: np.ndarray,
           top: int, left: int, shadow: int = 14) -> None:
    """Drop an object onto the floor with a soft contact shadow.

    Objects may hang off the edge of the canvas -- that is exactly the case the
    distractor scene needs -- so everything is clipped to the visible area.
    """
    ch, cw = canvas.shape[:2]
    h, w = mask.shape
    pad = shadow * 3
    y0, x0 = max(0, top - pad), max(0, left - pad)
    y1, x1 = min(ch, top + h + pad), min(cw, left + w + pad)
    if y1 <= y0 or x1 <= x0:
        return

    # Portion of the object that actually lands inside the region.
    sy0, sx0 = max(0, y0 - top), max(0, x0 - left)
    sy1, sx1 = min(h, y1 - top), min(w, x1 - left)
    if sy1 <= sy0 or sx1 <= sx0:
        return
    ty, tx = top + sy0 - y0, left + sx0 - x0

    region = canvas[y0:y1, x0:x1].astype(np.float32)
    local = np.zeros(region.shape[:2], np.uint8)
    local[ty:ty + (sy1 - sy0), tx:tx + (sx1 - sx0)] = mask[sy0:sy1, sx0:sx1]

    blur = cv2.GaussianBlur(local.astype(np.float32) / 255.0, (0, 0), shadow)
    shifted = np.roll(np.roll(blur, shadow // 2, axis=0), shadow // 3, axis=1)
    region *= (1.0 - 0.34 * np.clip(shifted - local / 255.0, 0, 1))[:, :, None]

    obj = np.zeros(region.shape, np.float32)
    obj[ty:ty + (sy1 - sy0), tx:tx + (sx1 - sx0)] = patch[sy0:sy1, sx0:sx1]
    alpha = (local.astype(np.float32) / 255.0)[:, :, None]
    region = region * (1 - alpha) + obj * alpha
    canvas[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# objects
# --------------------------------------------------------------------------
def phone(h: int, w: int, face: str = "back", body=(238, 240, 242)) -> tuple[np.ndarray, np.ndarray]:
    mask = _rounded_rect_mask(h, w, radius=int(min(h, w) * 0.13))
    patch = np.zeros((h, w, 3), np.float32)
    patch[:, :] = body

    if face == "back":
        # brushed sheen across the back glass
        grad = np.linspace(-10, 12, w)[None, :]
        patch += grad[:, :, None]
        # camera island, top-left
        iw = int(w * 0.42)
        ix, iy = int(w * 0.06), int(h * 0.03)
        island = patch[iy:iy + iw, ix:ix + iw]
        island[:] = np.clip(island * 0.99 + 3, 0, 255)
        cv2.rectangle(patch, (ix, iy), (ix + iw, iy + iw),
                      (int(body[0]) - 16, int(body[1]) - 16, int(body[2]) - 16), 2)
        for cx, cy in ((0.28, 0.28), (0.28, 0.70), (0.70, 0.42)):
            c = (ix + int(iw * cx), iy + int(iw * cy))
            cv2.circle(patch, c, int(iw * 0.17), (150, 152, 156), -1, cv2.LINE_AA)
            cv2.circle(patch, c, int(iw * 0.13), (28, 28, 30), -1, cv2.LINE_AA)
        cv2.circle(patch, (int(w * 0.5), int(h * 0.47)), int(w * 0.09),
                   (int(body[0]) - 12, int(body[1]) - 12, int(body[2]) - 12), -1, cv2.LINE_AA)
    else:
        # lock screen: dark bezel, colourful wallpaper
        patch[:, :] = (34, 32, 32)
        bz = int(min(h, w) * 0.035)
        yy, xx = np.mgrid[0:h - 2 * bz, 0:w - 2 * bz]
        u = xx / max(w - 2 * bz, 1)
        v = yy / max(h - 2 * bz, 1)
        screen = np.stack(
            [
                200 - 90 * v + 40 * np.sin(u * 3.1),
                150 + 60 * np.sin(v * 2.6) - 30 * u,
                120 + 70 * np.cos(v * 2.2 + u),
            ],
            axis=2,
        )
        patch[bz:h - bz, bz:w - bz] = np.clip(screen, 20, 255)
        # putText only accepts 8-bit buffers
        tmp = patch.astype(np.uint8)
        cv2.putText(tmp, "4:30", (int(w * 0.22), int(h * 0.30)),
                    cv2.FONT_HERSHEY_DUPLEX, w / 260.0, (250, 250, 250),
                    max(1, int(w / 160)), cv2.LINE_AA)
        cv2.line(tmp, (int(w * 0.33), int(h * 0.955)), (int(w * 0.67), int(h * 0.955)),
                 (250, 250, 250), max(2, int(h / 260)), cv2.LINE_AA)
        patch = tmp.astype(np.float32)

    patch[mask == 0] = 0
    return patch, mask


def cable_coil(size: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((size, size), np.uint8)
    patch = np.zeros((size, size, 3), np.float32)
    c = size // 2
    cv2.circle(mask, (c, c), int(size * 0.46), 255, int(size * 0.11))
    cv2.circle(patch, (c, c), int(size * 0.46), (243, 244, 246), int(size * 0.11), cv2.LINE_AA)
    cv2.circle(patch, (c, c), int(size * 0.46), (215, 216, 220), int(size * 0.03), cv2.LINE_AA)
    cv2.ellipse(mask, (int(size * 0.72), int(size * 0.30)), (int(size * 0.13), int(size * 0.06)),
                -35, 0, 360, 255, -1)
    cv2.ellipse(patch, (int(size * 0.72), int(size * 0.30)), (int(size * 0.13), int(size * 0.06)),
                -35, 0, 360, (238, 239, 242), -1, cv2.LINE_AA)
    patch[mask == 0] = 0
    return patch, mask


def retail_box(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.full((h, w), 255, np.uint8)
    patch = np.full((h, w, 3), 246.0, np.float32)
    cv2.rectangle(patch, (0, 0), (w - 1, h - 1), (214, 216, 220), 3)
    ph, pw = int(h * 0.74), int(w * 0.70)
    py, px = int(h * 0.13), int(w * 0.15)
    yy, xx = np.mgrid[0:ph, 0:pw]
    art = 90 + 70 * np.sin(yy / (ph / 6.0) + np.cos(xx / (pw / 3.0)))
    patch[py:py + ph, px:px + pw] = np.stack([art * 0.95, art, art * 1.05], axis=2)
    return patch, mask


def silicone_case(h: int, w: int, color=(210, 60, 40)) -> tuple[np.ndarray, np.ndarray]:
    mask = _rounded_rect_mask(h, w, radius=int(min(h, w) * 0.15))
    patch = np.zeros((h, w, 3), np.float32)
    patch[:, :] = color
    cut = _rounded_rect_mask(int(h * 0.30), int(w * 0.46), radius=int(w * 0.12))
    cy, cx = int(h * 0.04), int(w * 0.07)
    sub = patch[cy:cy + cut.shape[0], cx:cx + cut.shape[1]]
    sub[cut > 0] = np.array(color) * 0.72
    patch[mask == 0] = 0
    return patch, mask


# --------------------------------------------------------------------------
# scenes
# --------------------------------------------------------------------------
def _inset_for(h: int, w: int, angle: float) -> int:
    """How far in the tilt crop has to sit to stay on real pixels."""
    return int(round(max(h, w) * abs(np.sin(np.deg2rad(angle))) * 1.15)) + 8


def visible_rect(h: int, w: int, angle: float) -> tuple[int, int, int, int]:
    """The part of the canvas that survives the tilt, as x, y, width, height.

    Objects are positioned against this rather than the raw canvas, so a scene
    never accidentally has its subject sliced by the simulated crop -- that
    would test the wrong thing.
    """
    inset = _inset_for(h, w, angle)
    return inset, inset, w - 2 * inset, h - 2 * inset


def _tilt(canvas: np.ndarray, angle: float) -> np.ndarray:
    """Simulate a hand-held tilt: rotate the scene and crop back inside it."""
    h, w = canvas.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    rotated = cv2.warpAffine(canvas, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    x, y, vw, vh = visible_rect(h, w, angle)
    return rotated[y:y + vh, x:x + vw]


def _grain(img: np.ndarray, sigma: float = 2.2) -> np.ndarray:
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


PHONE_RATIO = 0.483   # an iPhone is about 147 x 71 mm


def scene_single_phone(angle: float) -> np.ndarray:
    H, W = 2400, 1800
    canvas = wood_floor(H, W, seed=1)
    vx, vy, vw, vh = visible_rect(H, W, angle)

    ph = int(vh * 0.70)
    pw = int(ph * PHONE_RATIO)
    patch, mask = phone(ph, pw, face="back")
    _paste(canvas, patch, mask, vy + (vh - ph) // 2, vx + (vw - pw) // 2)
    return _grain(_tilt(canvas, angle))


def scene_phone_and_cable(angle: float) -> np.ndarray:
    H, W = 2300, 2000
    canvas = wood_floor(H, W, seed=2)
    vx, vy, vw, vh = visible_rect(H, W, angle)

    ph = int(vh * 0.60)
    pw = int(ph * PHONE_RATIO)
    patch, mask = phone(ph, pw, face="front")
    _paste(canvas, patch, mask, vy + int(vh * 0.18), vx + int(vw * 0.14))

    d = int(vw * 0.26)
    cp, cm = cable_coil(d)
    _paste(canvas, cp, cm, vy + int(vh * 0.36), vx + int(vw * 0.58))
    return _grain(_tilt(canvas, angle))


def scene_with_distractor(angle: float) -> np.ndarray:
    """A phone plus a mug intruding from the corner -- the thing the crop is
    supposed to get rid of. The mug is deliberately placed half outside the
    visible area, which is what makes it identifiable as clutter."""
    H, W = 2400, 2000
    canvas = wood_floor(H, W, seed=3)
    vx, vy, vw, vh = visible_rect(H, W, angle)

    ph = int(vh * 0.66)
    pw = int(ph * PHONE_RATIO)
    _paste(canvas, *phone(ph, pw, face="back", body=(60, 58, 56)),
           vy + int(vh * 0.15), vx + int(vw * 0.20))

    d = int(vw * 0.34)
    intruder = np.zeros((d, d), np.uint8)
    cv2.circle(intruder, (d // 2, d // 2), d // 2 - 2, 255, -1)
    ip = np.zeros((d, d, 3), np.float32)
    cv2.circle(ip, (d // 2, d // 2), d // 2 - 2, (70, 110, 190), -1, cv2.LINE_AA)
    cv2.circle(ip, (d // 2, d // 2), int(d * 0.37), (240, 242, 245), -1, cv2.LINE_AA)
    ip[intruder == 0] = 0
    # Straddling the bottom-right corner of the frame, so it reads as cut off.
    _paste(canvas, ip, intruder, vy + vh - d // 2, vx + vw - d // 2)
    return _grain(_tilt(canvas, angle))


def scene_full_kit(angle: float) -> np.ndarray:
    H, W = 1900, 2600
    canvas = wood_floor(H, W, seed=4)
    vx, vy, vw, vh = visible_rect(H, W, angle)

    oh = int(vh * 0.58)
    top = vy + (vh - oh) // 2
    items = [
        silicone_case(oh, int(oh * 0.52)),
        phone(oh, int(oh * PHONE_RATIO), face="front"),
        retail_box(oh, int(oh * 0.55)),
        retail_box(oh, int(oh * 0.55)),
    ]

    # Lay the row out from the actual item widths so the end items keep a real
    # margin instead of landing flush against the frame.
    widths = [item[1].shape[1] for item in items]
    span = int(vw * 0.88)
    gap = (span - sum(widths)) // (len(items) - 1)
    left = vx + (vw - span) // 2
    for patch, mask in items:
        _paste(canvas, patch, mask, top, left)
        left += mask.shape[1] + gap
    return _grain(_tilt(canvas, angle))


SCENES = {
    "single_phone": scene_single_phone,
    "phone_and_cable": scene_phone_and_cable,
    "with_distractor": scene_with_distractor,
    "full_kit": scene_full_kit,
}

ANGLES = {
    "single_phone": 7.4,
    "phone_and_cable": -4.1,
    "with_distractor": 12.8,
    "full_kit": 2.3,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("samples"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    truth = {}
    for name, fn in SCENES.items():
        angle = ANGLES[name]
        img = fn(angle)
        path = args.out / f"{name}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 94])
        # The scene was tilted by `angle`, so the fix is to rotate it back.
        # `_tilt` turns the scene clockwise by `angle`, so putting it straight
        # again means turning it counter-clockwise by the same amount, which is
        # the positive direction the pipeline reports.
        truth[path.name] = {"expected_correction": angle}
        print(f"{path}  ({img.shape[1]}x{img.shape[0]})  tilt {angle:+.1f} deg")

    (args.out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

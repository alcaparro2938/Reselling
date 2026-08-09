"""Loading and saving images, including iPhone HEIC files."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # iPhone photos are HEIC unless the phone was set to "Most compatible"
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the install
    HEIF_AVAILABLE = False


SUPPORTED_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
    ".heic", ".heif", ".avif",
}


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


def load_bgr(source: str | Path | bytes, max_side: int = 0) -> np.ndarray:
    """Read an image as BGR uint8 with the EXIF orientation already applied.

    iPhones almost never rotate the pixels themselves; they set an EXIF tag and
    let the viewer do it. Applying it here is what fixes the "photo lying on
    its side" case before anything else runs.

    `max_side` bounds the result. For JPEG it is passed to the decoder, which
    can produce a half or quarter size image directly -- far cheaper in both
    time and memory than decoding twelve megapixels and throwing most of them
    away, which matters on a small server.
    """
    if isinstance(source, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(bytes(source)))
    else:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() in (".heic", ".heif", ".avif") and not HEIF_AVAILABLE:
            raise RuntimeError(
                f"{p.name}: HEIC support is missing. Install it with "
                "`pip install pillow-heif`."
            )
        pil = Image.open(p)

    if max_side > 0:
        # A no-op unless the decoder can actually hit the target; it only
        # halves, quarters and so on.
        try:
            pil.draft("RGB", (max_side, max_side))
        except (AttributeError, ValueError):
            pass

    pil = ImageOps.exif_transpose(pil)  # honour the orientation tag
    if pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")
    elif pil.mode == "L":
        pil = pil.convert("RGB")

    rgb = np.asarray(pil)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return fit_max_side(bgr, max_side) if max_side > 0 else bgr


def encode(bgr: np.ndarray, fmt: str = "jpg", quality: int = 95) -> bytes:
    """Encode a BGR array to bytes in the requested format."""
    fmt = fmt.lower().lstrip(".")
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    if fmt in ("jpg", "jpeg"):
        pil.save(buf, "JPEG", quality=quality, subsampling=0, optimize=True)
    elif fmt == "png":
        pil.save(buf, "PNG", optimize=True)
    elif fmt == "webp":
        pil.save(buf, "WEBP", quality=quality, method=5)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")
    return buf.getvalue()


def save(bgr: np.ndarray, path: str | Path, fmt: str = "jpg", quality: int = 95) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(bgr, fmt=fmt, quality=quality))
    return path


def iter_images(root: str | Path, recursive: bool = True):
    """Yield every supported image under `root` (or the file itself)."""
    root = Path(root)
    if root.is_file():
        if is_supported(root):
            yield root
        return
    pattern = "**/*" if recursive else "*"
    for p in sorted(root.glob(pattern)):
        if p.is_file() and is_supported(p):
            yield p


def fit_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest side is `max_side`. Never upscales."""
    if max_side <= 0:
        return bgr
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr
    scale = max_side / longest
    interp = cv2.INTER_AREA
    return cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=interp)

"""End-to-end: a crooked snapshot in, a straight, tidy listing photo out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import background, cutout, enhance, framing, geometry, io_utils, notes, orient
from .config import Config, resolve_ratio
from .segment import Segmentation, segment
from .segment import _describe as describe_components


@dataclass
class Report:
    """What the pipeline did, in terms a person can check."""

    ok: bool = True
    turned_deg: int = 0              # quarter turn, clockwise
    turn_source: str = ""
    rotated_deg: float = 0.0
    rotation_source: str = ""
    cropped: bool = False
    crop_box: tuple[int, int, int, int] | None = None
    output_ratio: float = 0.0
    objects_found: int = 0
    objects_kept: int = 0
    coverage: float = 0.0
    padded: float = 0.0
    confident: bool = False
    input_size: tuple[int, int] = (0, 0)
    output_size: tuple[int, int] = (0, 0)
    elapsed_ms: float = 0.0
    warnings: list = field(default_factory=list)   # notes.Note entries

    def summary(self) -> str:
        bits = []
        if self.turned_deg:
            bits.append(f"turned {self.turned_deg}°")
        if abs(self.rotated_deg) >= 0.01:
            bits.append(f"straightened {self.rotated_deg:+.2f}°")
        else:
            bits.append("no rotation needed")
        if self.cropped:
            dropped = self.objects_found - self.objects_kept
            frame = f"cropped to {self.output_size[0]}x{self.output_size[1]}"
            if dropped > 0:
                frame += f", removed {dropped} unwanted object(s)"
            bits.append(frame)
        else:
            bits.append("not cropped")
        return "; ".join(bits)


@dataclass
class Result:
    image: np.ndarray
    report: Report
    segmentation: Segmentation | None = None
    debug: dict = field(default_factory=dict)


def _try_new_surface(bgr: np.ndarray, cfg: Config, report: Report) -> np.ndarray | None:
    """Cut the product out and place it on a fresh surface.

    Returns None when the cut-out cannot be trusted, so the caller falls back
    to keeping the photo's own background. A bad cut-out deletes the product;
    an unchanged background is merely disappointing.
    """
    # On this path the product is cut out and re-photographed onto a generated
    # surface at most `max_side` across, and the mask itself comes from a
    # 320-pixel network. Full sensor resolution buys nothing here and costs a
    # great deal, so it goes first rather than being carried through.
    if cfg.max_side > 0:
        cap = max(1600, int(cfg.max_side * 1.6))
        if max(bgr.shape[:2]) > cap:
            k = cap / max(bgr.shape[:2])
            bgr = cv2.resize(
                bgr,
                (max(1, round(bgr.shape[1] * k)), max(1, round(bgr.shape[0] * k))),
                interpolation=cv2.INTER_AREA,
            )

    cut = cutout.cutout(bgr, cfg, download=cfg.allow_model_download)
    if cut is None:
        report.warnings.append(notes.cutout_unavailable())
        return None
    if not cut.confident:
        report.warnings.append(notes.cutout_unsure(cut.note or "unclear subject"))
        return None

    # The angle is measured on a reduced copy, as everywhere else: gradients
    # over a 12 megapixel frame cost a lot of memory and tell you nothing a
    # 900 pixel one does not.
    h, w = bgr.shape[:2]
    scale = min(1.0, cfg.work_size / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    work = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    work_mask = cv2.resize((cut.alpha > 127).astype(np.uint8) * 255, size,
                           interpolation=cv2.INTER_NEAREST)

    components, _ = describe_components(work_mask)
    report.objects_found = len(components)
    report.objects_kept = len(components)
    report.coverage = float((work_mask > 0).mean())
    report.confident = True

    if cut.hand_removed:
        report.warnings.append(notes.hand_removed(cut.hand_overlap))
        if cut.hand_overlap >= 0.05:
            report.warnings.append(notes.hand_covers_product(cut.hand_overlap))

    angle, source, refused = (0.0, "straightening disabled", None)
    if cfg.straighten:
        angle, source, refused = geometry.estimate_angle(work, work_mask, components, cfg)
    report.rotated_deg = angle
    report.rotation_source = source
    if angle == 0.0 and refused is not None:
        report.warnings.append(notes.rotation_refused(refused, cfg.max_rotation))

    report.cropped = True
    report.warnings.append(
        notes.background_replaced("wood" if cfg.background == "wood" else "custom")
    )
    return _place_on_new_surface(bgr, cut, cfg, report)


# Telea fills inward from the hole boundary; the radius is only the size of the
# neighbourhood each pixel averages, and a large one costs dearly for no gain.
INPAINT_RADIUS = 6
REBUILD_REGION_MAX = 700


def _rebuild_covered(bgr: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Fill what a finger was hiding, using the product around it.

    Confined to the region that actually needs rebuilding. Inpainting and
    blending across the whole frame cost several seconds and a few hundred
    megabytes of float temporaries to change a patch the size of a fingertip.
    """
    h, w = bgr.shape[:2]
    holes = cv2.dilate(covered, np.ones((5, 5), np.uint8), iterations=1)
    if not holes.any():
        return bgr

    pad = 24
    hx, hy, hw, hh = cv2.boundingRect(holes)
    x0, y0 = max(0, hx - pad), max(0, hy - pad)
    x1, y1 = min(w, hx + hw + pad), min(h, hy + hh + pad)

    region = bgr[y0:y1, x0:x1]
    region_holes = holes[y0:y1, x0:x1]

    # Inpainting cost climbs steeply with both the radius and the area, and
    # neither buys anything here: Telea propagates inward from the edge of the
    # hole whatever the radius, and these pixels are invented, so a small
    # neighbourhood on a reduced copy gives the same soft fill in a fraction of
    # a second instead of a minute and a half.
    rh, rw = region.shape[:2]
    k = min(1.0, REBUILD_REGION_MAX / max(rh, rw, 1))
    if k < 1.0:
        size = (max(8, round(rw * k)), max(8, round(rh * k)))
        small = cv2.resize(region, size, interpolation=cv2.INTER_AREA)
        small_holes = cv2.resize(region_holes, size, interpolation=cv2.INTER_NEAREST)
        patched = cv2.inpaint(small, small_holes, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        patched = cv2.resize(patched, (rw, rh), interpolation=cv2.INTER_CUBIC)
    else:
        patched = cv2.inpaint(region, region_holes, INPAINT_RADIUS, cv2.INPAINT_TELEA)

    blend = cv2.GaussianBlur(region_holes.astype(np.float32) / 255.0, (0, 0),
                             max(1.0, min(rh, rw) * 0.01))[:, :, None]
    out = bgr.copy()
    out[y0:y1, x0:x1] = np.clip(
        region * (1.0 - blend) + patched * blend, 0, 255
    ).astype(np.uint8)
    return out


def _place_on_new_surface(
    bgr: np.ndarray, cut: "cutout.Cutout", cfg: Config, report: Report
) -> np.ndarray:
    """Straighten the cut-out product and set it down on a generated surface.

    Nothing here is constrained by the original frame: the surface is made to
    whatever size suits the product, so the margin and the aspect ratio come
    out exactly as asked, and the corners the rotation exposes simply become
    more surface.
    """
    H, W = bgr.shape[:2]
    alpha, covered = cut.alpha, cut.covered

    # The product ends up occupying `fill` of a frame capped at max_side, so
    # anything beyond that resolution is rotated and resampled only to be
    # thrown away. Scaling down first is the difference between seconds and
    # half a minute on a phone-sized photo.
    if cfg.max_side > 0:
        box = framing.subject_bbox((alpha > 8).astype(np.uint8) * 255)
        if box is not None:
            longest = max(box[2], box[3])
            wanted = cfg.fill * cfg.max_side * 1.15   # headroom for resampling
            if longest > wanted > 0:
                k = wanted / longest
                size = (max(1, round(W * k)), max(1, round(H * k)))
                bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
                alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR)
                covered = cv2.resize(covered, size, interpolation=cv2.INTER_NEAREST)
                H, W = bgr.shape[:2]

    if covered.any():
        # A finger sat on the product. Its outline has already been restored;
        # the pixels underneath are rebuilt from the surrounding surface so the
        # result is not a hole. Whatever was really there is unknown, which is
        # why the report says how much of the product this was.
        bgr = _rebuild_covered(bgr, covered)

    angle = report.rotated_deg
    if abs(angle) > 1e-6:
        centre = (W / 2.0, H / 2.0)
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(round(H * sin + W * cos)), int(round(H * cos + W * sin))
        m[0, 2] += nw / 2.0 - centre[0]
        m[1, 2] += nh / 2.0 - centre[1]
        bgr = cv2.warpAffine(bgr, m, (nw, nh), flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(alpha, m, (nw, nh), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    box = framing.subject_bbox((alpha > 8).astype(np.uint8) * 255)
    if box is None:
        return bgr
    sx, sy, sw, sh = box
    product = bgr[sy:sy + sh, sx:sx + sw]
    mask = alpha[sy:sy + sh, sx:sx + sw]

    cw, ch, _ = framing.choose_canvas(sw, sh, cfg, original_ratio=W / max(H, 1))
    # The surface is generated, so there is no reason to build it larger than
    # the picture will ever be shown at.
    limit = cfg.max_side if cfg.max_side > 0 else max(cw, ch)
    if max(cw, ch) > limit:
        k = limit / max(cw, ch)
        cw, ch = max(1, int(cw * k)), max(1, int(ch * k))
        tw, th = max(1, int(sw * k)), max(1, int(sh * k))
        product = cv2.resize(product, (tw, th), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_AREA)
        sw, sh = tw, th

    canvas = background.surface(ch, cw, cfg.background)
    layer = np.zeros_like(canvas)
    layer_alpha = np.zeros((ch, cw), np.uint8)

    x = int(round((cw - sw) / 2.0))
    y = int(round((ch - sh) / 2.0 - cfg.center_bias_y * ch))
    x, y = max(0, min(x, cw - sw)), max(0, min(y, ch - sh))
    layer[y:y + sh, x:x + sw] = product
    layer_alpha[y:y + sh, x:x + sw] = mask

    report.output_ratio = cw / ch
    return background.composite(layer, layer_alpha, canvas, shadow=cfg.shadow)


# Orientation is read at a fixed size so that its edge-density thresholds mean
# the same thing for a 12 MP original and a compressed copy from a chat app.
ORIENT_SIZE = 1200


def _turn_upright(
    bgr: np.ndarray, cfg: Config, report: Report
) -> tuple[np.ndarray, Segmentation | None]:
    """Apply the quarter turn, fixed or detected. Returns the image and, when
    it was not turned, the segmentation already made, so it is not redone."""
    turn = str(cfg.turn).strip().lower()
    if turn not in ("", "auto"):
        deg = int(float(turn)) % 360
        if deg % 90:
            raise ValueError(f"turn must be a multiple of 90 degrees, not {cfg.turn!r}")
        report.turned_deg = deg
        report.turn_source = "manual"
        k = (4 - deg // 90) % 4        # np.rot90 turns counter-clockwise
        return (np.ascontiguousarray(np.rot90(bgr, k)) if k else bgr), None

    if not cfg.upright:
        return bgr, None

    seg = segment(bgr, cfg)
    if not (seg.confident and seg.components):
        return bgr, seg

    h, w = bgr.shape[:2]
    k = min(1.0, ORIENT_SIZE / max(h, w))
    small = bgr if k >= 1.0 else cv2.resize(
        bgr, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(seg.mask, (small.shape[1], small.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    found = orient.detect(small, mask)
    report.turn_source = found.reason
    if not found.confident or found.turns == 0:
        return bgr, seg

    report.turned_deg = ((4 - found.turns) % 4) * 90
    return np.ascontiguousarray(np.rot90(bgr, found.turns)), None


def process(bgr: np.ndarray, cfg: Config | None = None) -> Result:
    """Straighten, de-clutter and reframe a single product photo.

    The analysis runs on a downscaled copy; the geometry it decides on is then
    applied once, at full resolution, so the output is resampled a single time.
    """
    cfg = cfg or Config()
    started = time.perf_counter()
    report = Report(input_size=(bgr.shape[1], bgr.shape[0]))

    # "auto" is the useful default: a photo already taken flat on a decent
    # surface is better served by keeping that surface, since the cut-out can
    # only lose detail at the edges. It is the photos with no usable surface --
    # held in a hand, shot against a room -- that need a new one.
    seg: Segmentation | None = None
    bgr, seg = _turn_upright(bgr, cfg, report)

    source = (cfg.background or "").strip()
    if source == "auto":
        if seg is None:
            seg = segment(bgr, cfg)
        source = "" if (seg.confident and seg.uniform_surface) else "wood"

    if source:
        placed = _try_new_surface(bgr, cfg.with_(background=source), report)
        if placed is not None:
            placed = io_utils.fit_max_side(placed, cfg.max_side)
            placed = enhance.finish(placed, cfg.enhance, cfg.white_balance)
            report.output_size = (placed.shape[1], placed.shape[0])
            report.elapsed_ms = (time.perf_counter() - started) * 1000
            return Result(placed, report)
        # Falls through to keeping the original surface, having already said
        # in the report why.

    if seg is None:
        seg = segment(bgr, cfg)
    report.objects_found = len(seg.all_components)
    report.objects_kept = len(seg.components)
    report.coverage = seg.coverage
    report.confident = seg.confident

    H, W = bgr.shape[:2]
    fill_color = geometry.background_color(bgr, cfg.border_frac)

    if not seg.confident or not seg.components:
        # Nothing trustworthy to align to. Leave the framing alone rather than
        # guess -- a wrongly rotated photo is worse than an untouched one.
        report.warnings.append(notes.not_detected(seg.note or "unknown reason"))
        out = enhance.finish(bgr, cfg.enhance, cfg.white_balance)
        out = io_utils.fit_max_side(out, cfg.max_side)
        report.ok = False
        report.output_size = (out.shape[1], out.shape[0])
        report.elapsed_ms = (time.perf_counter() - started) * 1000
        return Result(out, report, seg)

    # --- straighten ------------------------------------------------------
    angle, source, refused = (0.0, "straightening disabled", None)
    if cfg.straighten:
        angle, source, refused = geometry.estimate_angle(
            seg.work, seg.mask, seg.components, cfg
        )
    report.rotated_deg = angle
    report.rotation_source = source
    if angle == 0.0 and refused is not None:
        report.warnings.append(notes.rotation_refused(refused, cfg.max_rotation))

    mask_full = cv2.resize(seg.mask, (W, H), interpolation=cv2.INTER_NEAREST)
    rot = geometry.rotate(bgr, mask_full, angle, fill_color)

    # --- reframe ---------------------------------------------------------
    out = rot.image
    if cfg.crop:
        box = framing.subject_bbox(rot.mask)
        if box is None:
            report.warnings.append(notes.lost_subject())
        else:
            crop = framing.plan_crop(
                rot.image.shape, box, rot.valid, cfg, original_ratio=W / max(H, 1)
            )
            out = framing.apply_crop(rot.image, crop)
            if crop.filled_corners > 0:
                out = geometry.reconstruct_edges(
                    out, framing.apply_crop(rot.valid, crop)
                )
            report.cropped = True
            report.crop_box = crop.box
            report.warnings.extend(crop.notes)

            # A pinned ratio does not always fit: a tall phone photo simply may
            # not contain enough floor to make a landscape frame. Say so rather
            # than quietly returning a different shape.
            target = resolve_ratio(cfg.ratio)
            if target is not None and abs(crop.ratio - target) / target > 0.005:
                if cfg.pad_to_ratio:
                    subject_in_crop = (box[0] - crop.x, box[1] - crop.y, box[2], box[3])
                    out, added, found = framing.pad_to_ratio(out, subject_in_crop, target)
                    report.padded = added
                    report.warnings.extend(found)
                else:
                    report.warnings.append(
                        notes.ratio_unreachable(cfg.ratio, crop.w, crop.h)
                    )
            report.output_ratio = out.shape[1] / out.shape[0]

    # --- finish ----------------------------------------------------------
    # Resize first: sharpening belongs at the size the photo will be seen at,
    # and everything downstream then works on a much smaller array.
    out = io_utils.fit_max_side(out, cfg.max_side)
    out = enhance.finish(out, cfg.enhance, cfg.white_balance)

    report.output_size = (out.shape[1], out.shape[0])
    report.elapsed_ms = (time.perf_counter() - started) * 1000
    return Result(out, report, seg, debug={"rotation": rot})


def process_file(path: str | Path, cfg: Config | None = None) -> Result:
    return process(io_utils.load_bgr(path), cfg)


def debug_overlay(bgr: np.ndarray, seg: Segmentation, angle: float = 0.0) -> np.ndarray:
    """Visualise what the detector saw: kept objects green, discarded red."""
    work = seg.work if seg.work is not None else bgr
    canvas = work.copy()

    tint = np.zeros_like(canvas)
    tint[seg.mask > 0] = (0, 200, 0)
    canvas = cv2.addWeighted(canvas, 1.0, tint, 0.35, 0)

    kept_labels = {c.label for c in seg.components}
    for c in seg.all_components:
        x, y, w, h = c.bbox
        kept = c.label in kept_labels
        color = (0, 220, 0) if kept else (0, 0, 230)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        label = f"{c.rectangularity:.2f} {c.angle:+.1f}deg"
        cv2.putText(canvas, label, (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.putText(canvas, f"correction {angle:+.2f} deg", (10, canvas.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas

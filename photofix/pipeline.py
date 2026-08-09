"""End-to-end: a crooked snapshot in, a straight, tidy listing photo out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import enhance, framing, geometry, io_utils, notes
from .config import Config, resolve_ratio
from .segment import Segmentation, segment


@dataclass
class Report:
    """What the pipeline did, in terms a person can check."""

    ok: bool = True
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


def process(bgr: np.ndarray, cfg: Config | None = None) -> Result:
    """Straighten, de-clutter and reframe a single product photo.

    The analysis runs on a downscaled copy; the geometry it decides on is then
    applied once, at full resolution, so the output is resampled a single time.
    """
    cfg = cfg or Config()
    started = time.perf_counter()
    report = Report(input_size=(bgr.shape[1], bgr.shape[0]))

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
    out = enhance.finish(out, cfg.enhance, cfg.white_balance)
    out = io_utils.fit_max_side(out, cfg.max_side)

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

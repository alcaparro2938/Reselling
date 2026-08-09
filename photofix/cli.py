"""Command line entry point: photofix IN [IN ...] -o OUT"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, io_utils
from .config import Config, RATIO_ALIASES
from .pipeline import debug_overlay, process
from .enhance import PRESETS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="photofix",
        description="Straighten product photos, crop out what does not belong, "
                    "and export them at a consistent size and shape.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", type=Path,
                   help="image files or folders")
    p.add_argument("-o", "--output", type=Path, default=Path("out"),
                   help="output folder")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="descend into subfolders")

    g = p.add_argument_group("framing")
    g.add_argument("--ratio", default="auto",
                   help="auto, original, free, or one of: "
                        + ", ".join(sorted(RATIO_ALIASES)))
    g.add_argument("--fill", type=float, default=0.86,
                   help="how much of the frame the product fills (0.3-1.0)")
    g.add_argument("--bias-y", type=float, default=0.0,
                   help="shift the product up (+) or down (-), as a fraction "
                        "of the crop height")
    g.add_argument("--pad-to-ratio", action="store_true",
                   help="if a fixed --ratio does not fit, extend the frame with "
                        "the surface mirrored outwards instead of using a "
                        "different shape")
    g.add_argument("--no-crop", action="store_true", help="keep the original framing")
    g.add_argument("--keep-extras", action="store_true",
                   help="do not discard secondary objects")

    g = p.add_argument_group("surface")
    g.add_argument("--background", default="",
                   help="'auto' replaces the background only when the photo has "
                        "no usable surface (held in a hand, shot in a room); "
                        "'wood' always replaces it; or give the path to a photo "
                        "of your own surface. Empty keeps the original")
    g.add_argument("--keep-hands", action="store_true",
                   help="do not try to remove a hand holding the product")
    g.add_argument("--no-shadow", action="store_true",
                   help="do not lay a shadow under the product on a new surface")
    g.add_argument("--offline", action="store_true",
                   help="never fetch the cut-out model; keep the original "
                        "background if it is not already cached")

    g = p.add_argument_group("straightening")
    g.add_argument("--no-straighten", action="store_true")
    g.add_argument("--max-rotation", type=float, default=25.0,
                   help="refuse corrections larger than this, in degrees")

    g = p.add_argument_group("look")
    g.add_argument("--enhance", choices=sorted(PRESETS), default="light")
    g.add_argument("--white-balance", action="store_true",
                   help="neutralise the colour cast (drains warm surfaces)")

    g = p.add_argument_group("output")
    g.add_argument("--max-side", type=int, default=2000,
                   help="longest edge of the exported image; 0 keeps full size")
    g.add_argument("--format", dest="fmt", choices=["jpg", "png", "webp"], default="jpg")
    g.add_argument("--quality", type=int, default=95)
    g.add_argument("--suffix", default="", help="appended to each output filename")
    g.add_argument("--overwrite", action="store_true",
                   help="replace existing files instead of skipping them")

    p.add_argument("--debug", action="store_true",
                   help="also write a detection overlay next to each result")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--version", action="version", version=f"photofix {__version__}")
    return p


def config_from_args(a: argparse.Namespace) -> Config:
    return Config(
        straighten=not a.no_straighten,
        max_rotation=a.max_rotation,
        crop=not a.no_crop,
        ratio=a.ratio,
        pad_to_ratio=a.pad_to_ratio,
        fill=a.fill,
        center_bias_y=a.bias_y,
        drop_distractors=not a.keep_extras,
        background=a.background,
        remove_hands=not a.keep_hands,
        shadow=not a.no_shadow,
        allow_model_download=not a.offline,
        enhance=a.enhance,
        white_balance=a.white_balance,
        max_side=a.max_side,
        fmt=a.fmt,
        quality=a.quality,
        suffix=a.suffix,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    files: list[Path] = []
    for src in args.inputs:
        if not src.exists():
            print(f"warning: {src} does not exist", file=sys.stderr)
            continue
        files.extend(io_utils.iter_images(src, recursive=args.recursive))

    if not files:
        print("No supported images found.", file=sys.stderr)
        return 1

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = skipped = failed = 0
    for i, path in enumerate(files, 1):
        dest = out_dir / f"{path.stem}{cfg.suffix}.{cfg.fmt}"
        if dest.exists() and not args.overwrite:
            skipped += 1
            if not args.quiet:
                print(f"[{i}/{len(files)}] {path.name}: exists, skipped "
                      f"(use --overwrite to replace)")
            continue

        try:
            result = process(io_utils.load_bgr(path), cfg)
        except Exception as exc:  # keep the batch going
            failed += 1
            print(f"[{i}/{len(files)}] {path.name}: FAILED -- {exc}", file=sys.stderr)
            continue

        io_utils.save(result.image, dest, fmt=cfg.fmt, quality=cfg.quality)
        processed += 1

        if args.debug and result.segmentation is not None:
            overlay = debug_overlay(
                io_utils.load_bgr(path), result.segmentation, result.report.rotated_deg
            )
            io_utils.save(overlay, out_dir / f"{path.stem}{cfg.suffix}.debug.jpg",
                          fmt="jpg", quality=88)

        if not args.quiet:
            rep = result.report
            line = f"[{i}/{len(files)}] {path.name} -> {dest.name}: {rep.summary()}"
            print(line)
            for w in rep.warnings:
                print(f"    note: {w.text}")

    if not args.quiet:
        parts = [f"{processed} processed"]
        if skipped:
            parts.append(f"{skipped} skipped")
        if failed:
            parts.append(f"{failed} failed")
        print(f"\nDone: {', '.join(parts)}. Output in {out_dir}/")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

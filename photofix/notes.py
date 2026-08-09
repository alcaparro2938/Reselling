"""Structured notes about what the pipeline did or refused to do.

Each note carries a stable `code` plus its parameters, so a front end can phrase
it in its own language instead of pattern-matching English sentences. `text` is
the English rendering the command line prints.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Note:
    code: str
    text: str
    params: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


def not_detected(reason: str) -> Note:
    return Note("not_detected", f"product not detected: {reason}", {"reason": reason})


def rotation_refused(angle: float, limit: float) -> Note:
    return Note(
        "rotation_refused",
        f"skipped straightening: measured tilt {angle:.1f} deg exceeds the "
        f"{limit:.0f} deg safety limit",
        {"angle": round(angle, 1), "limit": round(limit)},
    )


def lost_subject() -> Note:
    return Note("lost_subject", "lost the product after rotating; skipped the crop")


def margin_trimmed(fraction: float) -> Note:
    pct = round(fraction * 100)
    return Note("margin_trimmed", f"margin trimmed {pct}%", {"percent": pct})


def edge_rebuilt(fraction: float) -> Note:
    pct = round(fraction * 100, 1)
    return Note(
        "edge_rebuilt",
        f"{pct}% of the edge rebuilt from the surface",
        {"percent": pct},
    )


def ratio_unreachable(requested: str, width: int, height: int) -> Note:
    return Note(
        "ratio_unreachable",
        f"could not reach the {requested} ratio without cutting the product; "
        f"used {width}x{height} instead",
        {"requested": requested, "width": width, "height": height},
    )


def padded(axis: str, fraction: float) -> Note:
    pct = round(fraction * 100)
    return Note(
        "padded",
        f"extended the {axis} by {pct}% of mirrored surface",
        {"axis": axis, "percent": pct},
    )


def pad_blocked(target: float) -> Note:
    return Note(
        "pad_blocked",
        f"cannot pad to {target:.3f} without mirroring the product itself",
        {"target": round(target, 3)},
    )


def pad_short(target: float, reached: float) -> Note:
    return Note(
        "pad_short",
        f"still short of {target:.3f}, reached {reached:.3f}",
        {"target": round(target, 3), "reached": round(reached, 3)},
    )


def background_replaced(source: str) -> Note:
    return Note("background_replaced", f"placed on a new {source} surface",
                {"source": source})


def hand_removed(overlap: float) -> Note:
    pct = round(overlap * 100)
    return Note("hand_removed", f"removed the hand holding it ({pct}% of the "
                                f"product was against skin)", {"percent": pct})


def hand_covers_product(overlap: float) -> Note:
    pct = round(overlap * 100)
    return Note(
        "hand_covers_product",
        f"a hand covers about {pct}% of the product itself; that part cannot "
        f"be recovered, so retake this one with the product put down",
        {"percent": pct},
    )


def cutout_unavailable() -> Note:
    return Note("cutout_unavailable",
                "the cut-out model is not available, so the background was kept")


def cutout_unsure(reason: str) -> Note:
    return Note("cutout_unsure",
                f"the cut-out looks wrong ({reason}), so the background was kept",
                {"reason": reason})

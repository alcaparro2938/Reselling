"""Spanish wording for the web interface.

The pipeline reports what it did as note codes plus parameters, so the phrasing
lives here rather than being parsed back out of English sentences.
"""

from __future__ import annotations

from photofix.pipeline import Report


NOTE_TEXT: dict[str, str] = {
    "not_detected":
        "No se ha localizado el producto, así que la foto se ha dejado como estaba.",
    "rotation_refused":
        "Se ha medido una inclinación de {angle}°, por encima del límite de "
        "seguridad de {limit}°: no se ha girado para no estropearla.",
    "lost_subject":
        "El producto se ha perdido al girar la foto; no se ha recortado.",
    "margin_trimmed":
        "Se ha reducido el margen un {percent}% para que cupiera el encuadre.",
    "edge_rebuilt":
        "Se ha reconstruido un {percent}% del borde a partir de la superficie.",
    "ratio_unreachable":
        "No cabe el formato {requested} sin cortar el producto; se ha usado "
        "{width}x{height}. Marca «Forzar el formato» para conseguirlo.",
    "padded":
        "Se ha ampliado el encuadre un {percent}% reflejando la superficie.",
    "pad_blocked":
        "No se puede ampliar hasta {target} sin reflejar el propio producto.",
    "background_replaced":
        "La foto no ten\u00eda una superficie aprovechable, as\u00ed que el producto "
        "se ha recortado y colocado sobre madera.",
    "hand_removed":
        "Se ha quitado la mano que lo sujetaba.",
    "hand_covers_product":
        "Ojo: la mano tapaba un {percent}% del producto. Esa parte se ha "
        "reconstruido y no es real \u2014 mejor repetir la foto con el m\u00f3vil "
        "apoyado.",
    "cutout_unavailable":
        "No se ha podido cargar el recortador, as\u00ed que se ha dejado el fondo "
        "original.",
    "cutout_unsure":
        "El recorte no ha salido fiable, as\u00ed que se ha dejado el fondo original.",
    "pad_short":
        "No se ha llegado a {target}; se ha quedado en {reached}.",
}

AXIS_TEXT = {"sides": "los lados", "top and bottom": "arriba y abajo"}


def note_to_spanish(note) -> str:
    template = NOTE_TEXT.get(note.code)
    if template is None:
        return note.text          # an unmapped code still says something useful
    params = dict(note.params)
    if "axis" in params:
        params["axis"] = AXIS_TEXT.get(params["axis"], params["axis"])
    for key, value in params.items():
        if isinstance(value, float):
            params[key] = f"{value:g}".replace(".", ",")   # decimal comma
    try:
        return template.format(**params)
    except (KeyError, IndexError):
        return note.text


def summarise(report: Report) -> str:
    """One line describing the result, in the same order the eye checks it."""
    bits = []
    if report.turned_deg:
        how = "a mano" if report.turn_source == "manual" else {
            "camera": "por la cámara", "island": "por la Dynamic Island",
        }.get(report.turn_source, "")
        bits.append(f"girada {report.turned_deg}° {how}".strip())
    if abs(report.rotated_deg) >= 0.01:
        bits.append(f"enderezada {report.rotated_deg:+.2f}°".replace(".", ","))
    else:
        bits.append("ya estaba recta")

    if report.cropped:
        w, h = report.output_size
        piece = f"recortada a {w}x{h}"
        removed = max(0, report.objects_found - report.objects_kept)
        if removed == 1:
            piece += ", fuera 1 objeto ajeno"
        elif removed > 1:
            piece += f", fuera {removed} objetos ajenos"
        bits.append(piece)
    else:
        bits.append("sin recortar")

    return "; ".join(bits)

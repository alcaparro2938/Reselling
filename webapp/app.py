"""Drag-and-drop front end for photofix.

    python -m webapp.app          then open http://127.0.0.1:5000

Photos are uploaded once into a scratch folder and kept there for the life of
the job, so the settings can be changed and everything re-rendered without
uploading again -- which is the normal way of working: drop the batch, nudge
the margin, look, adjust.
"""

from __future__ import annotations

import atexit
import io
import shutil
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photofix import Config, __version__, io_utils, process   # noqa: E402
from photofix.config import RATIO_ALIASES                     # noqa: E402
from photofix.enhance import PRESETS                          # noqa: E402

from .spanish import note_to_spanish, summarise               # noqa: E402


MAX_UPLOAD_MB = 200
JOB_LIFETIME_S = 60 * 60          # an hour is plenty for one editing session
MAX_JOBS = 24

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

ROOT = Path(tempfile.mkdtemp(prefix="photofix-"))
atexit.register(lambda: shutil.rmtree(ROOT, ignore_errors=True))


@dataclass
class Job:
    id: str
    created: float
    originals: list[str] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return ROOT / self.id

    @property
    def in_dir(self) -> Path:
        return self.dir / "in"

    @property
    def out_dir(self) -> Path:
        return self.dir / "out"


JOBS: dict[str, Job] = {}


def _sweep() -> None:
    """Drop jobs that are old or simply too many to keep around."""
    now = time.time()
    for jid in [j for j, job in JOBS.items() if now - job.created > JOB_LIFETIME_S]:
        shutil.rmtree(JOBS.pop(jid).dir, ignore_errors=True)
    while len(JOBS) > MAX_JOBS:
        oldest = min(JOBS.values(), key=lambda j: j.created)
        shutil.rmtree(JOBS.pop(oldest.id).dir, ignore_errors=True)


def _config_from_form(form) -> Config:
    def num(name: str, default: float) -> float:
        try:
            return float(form.get(name, default))
        except (TypeError, ValueError):
            return default

    def flag(name: str, default: bool = False) -> bool:
        v = form.get(name)
        if v is None:
            return default
        return str(v).lower() in ("1", "true", "on", "yes")

    ratio = form.get("ratio", "auto")
    if ratio not in RATIO_ALIASES and ratio not in ("auto", "original", "free"):
        ratio = "auto"

    enhance = form.get("enhance", "light")
    if enhance not in PRESETS:
        enhance = "light"

    fmt = form.get("format", "jpg")
    if fmt not in ("jpg", "png", "webp"):
        fmt = "jpg"

    return Config(
        straighten=flag("straighten", True),
        crop=flag("crop", True),
        ratio=ratio,
        pad_to_ratio=flag("pad_to_ratio", False),
        fill=min(max(num("fill", 0.86), 0.3), 1.0),
        center_bias_y=min(max(num("bias_y", 0.0), -0.3), 0.3),
        drop_distractors=flag("drop_distractors", True),
        enhance=enhance,
        white_balance=flag("white_balance", False),
        max_side=int(min(max(num("max_side", 2000), 200), 6000)),
        fmt=fmt,
        quality=int(min(max(num("quality", 95), 40), 100)),
    )


def _render(job: Job, cfg: Config) -> list[dict]:
    """Run every original in the job through the pipeline."""
    shutil.rmtree(job.out_dir, ignore_errors=True)
    job.out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for name in job.originals:
        src = job.in_dir / name
        entry: dict = {"name": name, "original": f"/file/{job.id}/in/{name}"}
        try:
            result = process(io_utils.load_bgr(src), cfg)
        except Exception as exc:
            entry.update(ok=False, error=str(exc))
            items.append(entry)
            continue

        out_name = f"{Path(name).stem}.{cfg.fmt}"
        io_utils.save(result.image, job.out_dir / out_name,
                      fmt=cfg.fmt, quality=cfg.quality)

        rep = result.report
        entry.update(
            ok=True,
            output=f"/file/{job.id}/out/{out_name}?v={int(time.time() * 1000)}",
            out_name=out_name,
            rotated=round(rep.rotated_deg, 2),
            summary=summarise(rep),
            warnings=[note_to_spanish(n) for n in rep.warnings],
            size=f"{rep.output_size[0]} x {rep.output_size[1]}",
            removed=max(0, rep.objects_found - rep.objects_kept),
            detected=rep.confident,
            ms=round(rep.elapsed_ms),
        )
        items.append(entry)
    return items


@app.get("/")
def index():
    return render_template(
        "index.html",
        version=__version__,
        ratios=["auto", "original", "free"] + sorted(RATIO_ALIASES),
        presets=sorted(PRESETS),
    )


@app.post("/api/upload")
def upload():
    _sweep()
    files = request.files.getlist("photos")
    if not files:
        return jsonify(error="No se ha recibido ninguna foto."), 400

    job = Job(id=uuid.uuid4().hex[:12], created=time.time())
    job.in_dir.mkdir(parents=True, exist_ok=True)

    rejected = []
    for f in files:
        name = secure_filename(f.filename or "")
        if not name or not io_utils.is_supported(name):
            rejected.append(f.filename or "(unnamed)")
            continue
        # Two files can reduce to the same safe name; keep both.
        target, n = job.in_dir / name, 1
        while target.exists():
            target = job.in_dir / f"{Path(name).stem}-{n}{Path(name).suffix}"
            n += 1
        f.save(target)
        job.originals.append(target.name)

    if not job.originals:
        shutil.rmtree(job.dir, ignore_errors=True)
        return jsonify(
            error="Ninguno de esos archivos es una imagen que se pueda leer."
        ), 400

    JOBS[job.id] = job
    items = _render(job, _config_from_form(request.form))
    return jsonify(job=job.id, items=items, rejected=rejected)


@app.post("/api/rerender/<job_id>")
def rerender(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify(
            error="Ese lote ha caducado. Vuelve a subir las fotos."
        ), 404
    job.created = time.time()
    return jsonify(job=job.id, items=_render(job, _config_from_form(request.form)))


@app.get("/file/<job_id>/<kind>/<path:name>")
def serve(job_id: str, kind: str, name: str):
    job = JOBS.get(job_id)
    if job is None or kind not in ("in", "out"):
        abort(404)
    base = (job.in_dir if kind == "in" else job.out_dir).resolve()
    path = (base / name).resolve()
    if not path.is_file() or base not in path.parents:
        abort(404)
    # HEIC has no browser support, so previews of the original are converted.
    if path.suffix.lower() in (".heic", ".heif", ".avif"):
        data = io_utils.encode(io_utils.fit_max_side(io_utils.load_bgr(path), 1400),
                               "jpg", 88)
        return send_file(io.BytesIO(data), mimetype="image/jpeg")
    return send_file(path)


@app.get("/api/download/<job_id>")
def download(job_id: str):
    job = JOBS.get(job_id)
    if job is None or not job.out_dir.exists():
        abort(404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(job.out_dir.iterdir()):
            if p.is_file():
                z.write(p, p.name)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name="fotos-listas.zip")


@app.errorhandler(413)
def too_large(_):
    return jsonify(
        error=f"El lote supera el límite de {MAX_UPLOAD_MB} MB."
    ), 413


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="photofix web interface")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    print(f"photofix {__version__} -- open http://{a.host}:{a.port}")
    app.run(host=a.host, port=a.port, debug=a.debug)


if __name__ == "__main__":
    main()

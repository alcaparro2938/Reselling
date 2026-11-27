"""Drag-and-drop front end for photofix.

    python -m webapp.app          then open http://127.0.0.1:5000

Photos are uploaded once into a scratch folder and kept there for the life of
the job, so the settings can be changed and everything re-rendered without
uploading again -- which is the normal way of working: drop the batch, nudge
the margin, look, adjust.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
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

# A batch lives on disk under a shared root rather than in a dictionary in
# memory. Under a real server there are several worker processes, and the one
# that answers "re-render batch X" is rarely the one that received the upload,
# so anything held in a process is invisible half the time.
ROOT = Path(
    os.environ.get("PHOTOFIX_WORKDIR") or Path(tempfile.gettempdir()) / "photofix-jobs"
)
ROOT.mkdir(parents=True, exist_ok=True)

JOB_ID = re.compile(r"^[0-9a-f]{12}$")

# Transforming one 12 megapixel photo peaks at roughly 280 MB, and the small
# hosts this is meant to run on have 512 MB in total. So only one photo is
# transformed at a time; receiving uploads and sending results back still
# happen concurrently on the other threads.
CPU_SLOTS = threading.Semaphore(int(os.environ.get("PHOTOFIX_CONCURRENCY", "1")))

# Photos are bounded on the way in. A 12 megapixel iPhone photo costs about
# 120 MB just to decode, before any work happens, and the export is 2000 pixels
# across at most -- so nothing above this is ever seen by the person looking at
# the result. Raise it where there is memory to spare.
MAX_INPUT = int(os.environ.get("PHOTOFIX_MAX_INPUT", "2600"))


@dataclass
class Job:
    id: str

    @property
    def dir(self) -> Path:
        return ROOT / self.id

    @property
    def in_dir(self) -> Path:
        return self.dir / "in"

    @property
    def out_dir(self) -> Path:
        return self.dir / "out"

    @property
    def originals(self) -> list[str]:
        if not self.in_dir.is_dir():
            return []
        return sorted(p.name for p in self.in_dir.iterdir() if p.is_file())

    def touch(self) -> None:
        self.dir.touch(exist_ok=True)

    # A turn chosen by hand for one photo outlives "Aplicar cambios": having
    # to turn the same photo again after changing the margin would be absurd.
    @property
    def turns(self) -> dict[str, str]:
        try:
            data = json.loads((self.dir / "turns.json").read_text())
        except (OSError, ValueError):
            return {}
        return {k: str(v) for k, v in data.items() if isinstance(k, str)}

    def set_turn(self, name: str, deg: str) -> None:
        turns = self.turns
        if deg == "auto":
            turns.pop(name, None)
        else:
            turns[name] = deg
        (self.dir / "turns.json").write_text(json.dumps(turns))


def find_job(job_id: str) -> Job | None:
    """Look a batch up by id, rejecting anything that is not one of ours."""
    if not JOB_ID.match(job_id or ""):
        return None
    job = Job(job_id)
    return job if job.in_dir.is_dir() else None


def _sweep() -> None:
    """Delete batches that are past their hour, and the oldest beyond the cap."""
    now = time.time()
    dirs = [d for d in ROOT.iterdir() if d.is_dir()] if ROOT.is_dir() else []
    kept = []
    for d in dirs:
        try:
            age = now - d.stat().st_mtime
        except OSError:
            continue
        if age > JOB_LIFETIME_S:
            shutil.rmtree(d, ignore_errors=True)
        else:
            kept.append(d)

    kept.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for d in kept[MAX_JOBS:]:
        shutil.rmtree(d, ignore_errors=True)


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

    bg = form.get("background", "")
    if bg not in ("", "auto", "wood"):
        bg = ""

    fmt = form.get("format", "jpg")
    if fmt not in ("jpg", "png", "webp"):
        fmt = "jpg"

    return Config(
        straighten=flag("straighten", True),
        crop=flag("crop", True),
        ratio=ratio,
        pad_to_ratio=flag("pad_to_ratio", False),
        fill=min(max(num("fill", 0.78), 0.3), 1.0),
        center_bias_y=min(max(num("bias_y", 0.0), -0.3), 0.3),
        drop_distractors=flag("drop_distractors", True),
        background=bg,
        remove_hands=flag("remove_hands", True),
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

    turns = job.turns
    return [_render_one(job, name, cfg, turns.get(name, "auto"))
            for name in job.originals]


def _render_one(job: Job, name: str, cfg: Config, turn: str = "auto") -> dict:
    """Run one original through the pipeline and describe the result."""
    src = job.in_dir / name
    entry: dict = {"name": name, "original": f"/file/{job.id}/in/{name}"}
    try:
        with CPU_SLOTS:
            result = process(io_utils.load_bgr(src, MAX_INPUT), cfg.with_(turn=turn))
    except Exception as exc:
        entry.update(ok=False, error=str(exc))
        return entry

    out_name = f"{Path(name).stem}.{cfg.fmt}"
    io_utils.save(result.image, job.out_dir / out_name,
                  fmt=cfg.fmt, quality=cfg.quality)

    rep = result.report
    entry.update(
        ok=True,
        output=f"/file/{job.id}/out/{out_name}?v={int(time.time() * 1000)}",
        out_name=out_name,
        rotated=round(rep.rotated_deg, 2),
        turned=rep.turned_deg,
        turn_manual=rep.turn_source == "manual",
        summary=summarise(rep),
        warnings=[note_to_spanish(n) for n in rep.warnings],
        size=f"{rep.output_size[0]} x {rep.output_size[1]}",
        removed=max(0, rep.objects_found - rep.objects_kept),
        detected=rep.confident,
        ms=round(rep.elapsed_ms),
        # The before/after slider sizes its frame from these, so neither
        # picture ends up letterboxed inside the other's shape.
        ratio_before=round(rep.input_size[0] / max(rep.input_size[1], 1), 4),
        ratio_after=round(rep.output_size[0] / max(rep.output_size[1], 1), 4),
    )
    return entry


@app.get("/")
def index():
    return render_template(
        "index.html",
        version=__version__,
        ratios=["auto", "original", "free"] + sorted(RATIO_ALIASES),
        presets=sorted(PRESETS),
    )


@app.get("/manifest.webmanifest")
def manifest():
    """Lets iOS "Añadir a pantalla de inicio" open this like an app."""
    return jsonify({
        "name": "photofix",
        "short_name": "photofix",
        "description": "Fotos de producto rectas y recortadas, listas para publicar",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#f6f5f3",
        "theme_color": "#2f6f4e",
        "lang": "es",
        "icons": [
            {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png",
             "purpose": "any maskable"},
        ],
    })


@app.get("/sw.js")
def service_worker():
    """Served from the root so it can control the whole site.

    Deliberately thin: the photos live on the server, so there is nothing
    useful to cache. Its only job is to satisfy the installability check and
    to show a readable message when the phone is off the network.
    """
    js = (
        "self.addEventListener('install', e => self.skipWaiting());\n"
        "self.addEventListener('activate', e => e.waitUntil(clients.claim()));\n"
        "self.addEventListener('fetch', e => {\n"
        "  if (e.request.mode !== 'navigate') return;\n"
        "  e.respondWith(fetch(e.request).catch(() => new Response(\n"
        "    '<meta charset=utf-8><p style=\"font:16px system-ui;padding:24px\">"
        "No hay conexión con photofix. Comprueba que el ordenador sigue "
        "encendido y en la misma wifi.</p>',\n"
        "    { headers: { 'Content-Type': 'text/html; charset=utf-8' } })));\n"
        "});\n"
    )
    return app.response_class(js, mimetype="application/javascript")


@app.post("/api/upload")
def upload():
    _sweep()
    files = request.files.getlist("photos")
    if not files:
        return jsonify(error="No se ha recibido ninguna foto."), 400

    job = Job(uuid.uuid4().hex[:12])
    job.in_dir.mkdir(parents=True, exist_ok=True)

    rejected: list[str] = []
    saved = 0
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
        saved += 1

    if not saved:
        shutil.rmtree(job.dir, ignore_errors=True)
        return jsonify(
            error="Ninguno de esos archivos es una imagen que se pueda leer."
        ), 400

    job.touch()
    items = _render(job, _config_from_form(request.form))
    return jsonify(job=job.id, items=items, rejected=rejected)


@app.post("/api/rerender/<job_id>")
def rerender(job_id: str):
    job = find_job(job_id)
    if job is None:
        return jsonify(
            error="Ese lote ha caducado. Vuelve a subir las fotos."
        ), 404
    job.touch()
    return jsonify(job=job.id, items=_render(job, _config_from_form(request.form)))


@app.post("/api/turn/<job_id>")
def turn(job_id: str):
    """Fix the turn of one photo by hand and process just that one again."""
    job = find_job(job_id)
    if job is None:
        return jsonify(
            error="Ese lote ha caducado. Vuelve a subir las fotos."
        ), 404
    name = request.form.get("name", "")
    if name not in job.originals:
        return jsonify(error="Esa foto no está en el lote."), 404
    deg = request.form.get("turn", "auto")
    if deg != "auto":
        try:
            deg = int(deg) % 360
        except ValueError:
            return jsonify(error="Giro no válido."), 400
        if deg % 90:
            return jsonify(error="Giro no válido."), 400
        deg = str(deg)
    job.touch()
    job.set_turn(name, deg)
    job.out_dir.mkdir(parents=True, exist_ok=True)
    return jsonify(job=job.id, item=_render_one(job, name, _config_from_form(request.form), deg))


@app.get("/file/<job_id>/<kind>/<path:name>")
def serve(job_id: str, kind: str, name: str):
    job = find_job(job_id)
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
    job = find_job(job_id)
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


def lan_address() -> str | None:
    """This machine's address on the local network.

    Opening a UDP socket towards an outside address makes the OS pick the
    interface it would really route through, which is the one the phone can
    reach. Nothing is actually sent.
    """
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))     # reserved, never routed anywhere
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="photofix web interface")
    ap.add_argument("--host", default=None,
                    help="address to listen on (default: this machine only)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--phone", action="store_true",
                    help="also accept connections from the local network, so a "
                         "phone on the same wifi can use it")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    host = a.host or ("0.0.0.0" if a.phone else "127.0.0.1")
    _sweep()        # clear out anything left behind by a previous session

    print(f"\n  photofix {__version__}")
    print(f"  En este ordenador:  http://127.0.0.1:{a.port}")
    if a.phone:
        ip = lan_address()
        if ip:
            print(f"  En el móvil:        http://{ip}:{a.port}")
            print("  (el móvil tiene que estar en la misma wifi)")
        else:
            print("  No se ha podido averiguar la IP de esta red.")
    else:
        print("  Para abrirlo en el móvil, añade  --phone")
    print("\n  Ctrl+C para parar.\n")

    app.run(host=host, port=a.port, debug=a.debug)


if __name__ == "__main__":
    main()

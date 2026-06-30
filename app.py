"""
Web UI para OCR benchmark.
Uso: python app.py  →  http://localhost:5000
"""
import datetime, json, os, shutil, tempfile, threading, zipfile
from pathlib import Path
from flask import Flask, Response, render_template, request, send_file

import sys
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_ocr import ENGINES, run_engine, _start_key_listener
from check import analizar_csv

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = None  # sin límite de upload

IMG_EXTS     = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
OUT_DIR      = Path(__file__).parent / "benchmark_results"
OUT_DIR.mkdir(exist_ok=True)
HISTORY_FILE = OUT_DIR / "history.json"


def _append_history(entry):
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    history.insert(0, entry)
    HISTORY_FILE.write_text(json.dumps(history[:200], ensure_ascii=False, indent=2), encoding="utf-8")

# ── sesiones de upload en batches ───────────────────────────────────────────
_sessions: dict = {}  # session_id → tmpdir Path

# ── estado global persistente ────────────────────────────────────────────────
_job_lock    = threading.Lock()
_job_engine  = None        # nombre del engine corriendo
_job_running = False       # True mientras corre
_job_log     = []          # todos los eventos emitidos (persiste en memoria)
_job_cond    = threading.Condition(_job_lock)

MODEL_META = {
    "tesseract": ("Tesseract", "Estandar"),
    "doctr":     ("Doctr",    "Lento - Bueno"),
    "rapidocr":  ("Rapidocr", "No Recomendado"),
    "easyocr":   ("EasyOCR",  "Foco en Banco"),
    "paddleocr": ("PaddleOCR","Alternativa"),
    "surya":     ("Surya",    "Experimental"),
}

# ── rutas ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    engines = [e["name"] for e in ENGINES]
    return render_template("index.html", engines=engines, meta=MODEL_META)


def _push(data):
    """Agrega evento al log y notifica a los streams SSE conectados."""
    with _job_cond:
        _job_log.append(data)
        _job_cond.notify_all()


@app.get("/status")
def status():
    with _job_lock:
        return json.dumps({
            "running": _job_running,
            "engine":  _job_engine,
            "log_len": len(_job_log),
        }), 200, {"Content-Type": "application/json"}


@app.get("/files")
def files():
    csvs = sorted(OUT_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return json.dumps([
        {"name": p.name, "engine": p.stem,
         "size": round(p.stat().st_size / 1024, 1),
         "mtime": int(p.stat().st_mtime)}
        for p in csvs[:10]
    ]), 200, {"Content-Type": "application/json"}


def _start_job(engine_name, imagenes, cleanup_fn=None):
    """Lanza job de procesamiento. cleanup_fn se llama al terminar (para borrar tmpdir)."""
    global _job_engine, _job_running, _job_log

    cfg = next((e for e in ENGINES if e["name"] == engine_name), None)
    if cfg is None:
        return "Engine no existe", 400

    with _job_lock:
        if _job_running:
            return "Job ya corriendo", 409
        _job_log.clear()
        _job_engine  = engine_name
        _job_running = True

    _push({"done": 0, "n": len(imagenes), "avg_ms": 0, "eta": 0,
           "imagen": f"{len(imagenes)} imagenes encontradas, iniciando {engine_name}...",
           "CBU": "", "CUIT": "", "titular": "", "banco": "", "_status": "init"})

    def job():
        global _job_engine, _job_running
        _t0 = datetime.datetime.now()
        try:
            def cb(data):
                data["engine"] = engine_name
                _push(data)

            imagenes_a_procesar = imagenes
            preproc_dir = None
            if engine_name == "tesseract":
                from benchmark_ocr import opencv_preprocess
                import tempfile
                from PIL import Image as _Image
                _push({"done": 0, "n": len(imagenes), "avg_ms": 0, "eta": 0,
                       "imagen": f"preprocesando 0/{len(imagenes)} imágenes...",
                       "CBU": "", "CUIT": "", "titular": "", "banco": "", "_status": "init"})
                preproc_dir = Path(tempfile.mkdtemp())
                preproc_imgs = []
                from benchmark_ocr import _skip as _skip_ref
                _skip_ref.clear()
                for i, img_path in enumerate(imagenes):
                    if _skip_ref.is_set():
                        break
                    out_path = preproc_dir / img_path.name
                    opencv_preprocess(_Image.open(img_path)).save(str(out_path))
                    preproc_imgs.append(out_path)
                    _push({"done": i + 1, "n": len(imagenes), "avg_ms": 0, "eta": 0,
                           "imagen": f"preprocesando {i + 1}/{len(imagenes)} imágenes...",
                           "CBU": "", "CUIT": "", "titular": "", "banco": "", "_status": "init"})
                imagenes_a_procesar = preproc_imgs

            run_engine(cfg, imagenes_a_procesar, OUT_DIR, on_progress=cb)
            from repair import reparar_csv, postprocesar_csv
            from benchmark_ocr import _skip
            csv_path = OUT_DIR / f"{engine_name}.csv"
            _append_history({
                "engine":     engine_name,
                "fecha":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "n":          len(imagenes),
                "csv":        f"{engine_name}.csv",
                "duracion_s": round((datetime.datetime.now() - _t0).total_seconds()),
            })
            carpeta_imagenes = imagenes[0].parent if imagenes else None
            if carpeta_imagenes and not _skip.is_set() and csv_path.exists():
                _push({"done": 0, "n": 0, "avg_ms": 0, "eta": 0,
                       "imagen": "reparando campos vacíos...", "CBU": "", "CUIT": "",
                       "titular": "", "banco": "", "_status": "init"})
                def _log_repair(msg):
                    _push({"_log": msg, "done": 0, "n": 0, "avg_ms": 0, "eta": 0,
                           "imagen": msg, "CBU": "", "CUIT": "", "titular": "", "banco": "",
                           "_status": "repair"})
                reparar_csv(csv_path, carpeta_imagenes, on_log=_log_repair, stop_event=_skip,
                            debug_dir=OUT_DIR / "debug_banco")
            def _log_post(msg):
                _push({"_log": msg, "done": 0, "n": 0, "avg_ms": 0, "eta": 0,
                       "imagen": msg, "CBU": "", "CUIT": "", "titular": "", "banco": "",
                       "_status": "repair"})
            if csv_path.exists():
                postprocesar_csv(csv_path, on_log=_log_post)
            _push({"done": -1, "msg": f"Listo — {engine_name}", "csv": True, "engine": engine_name})
        except Exception as e:
            _push({"done": -1, "msg": f"Error: {e}", "csv": False})
        finally:
            if cleanup_fn:
                cleanup_fn()
            if preproc_dir:
                shutil.rmtree(preproc_dir, ignore_errors=True)
            with _job_lock:
                _job_running = False
                _job_engine  = None

    threading.Thread(target=job, daemon=True).start()
    return "OK"


@app.post("/upload-batch")
def upload_batch():
    session_id = request.form.get("session") or request.args.get("session")
    if not session_id:
        return "session requerido", 400
    if session_id not in _sessions:
        _sessions[session_id] = Path(tempfile.mkdtemp())
    tmpdir = _sessions[session_id]
    for f in request.files.getlist("files"):
        name = Path(f.filename).name
        if Path(name).suffix.lower() in IMG_EXTS:
            f.save(str(tmpdir / name))
    return "OK"


@app.post("/upload-commit")
def upload_commit():
    data        = request.json or {}
    session_id  = data.get("session")
    engine_name = data.get("engine", ENGINES[0]["name"])
    tmpdir      = _sessions.pop(session_id, None)
    if not tmpdir:
        return "Sesion no encontrada", 400
    imagenes = sorted(p for p in tmpdir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not imagenes:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return "Sin imagenes", 400
    result = _start_job(engine_name, imagenes, cleanup_fn=lambda: shutil.rmtree(tmpdir, ignore_errors=True))
    if result != "OK":
        return (result[0], result[1]) if isinstance(result, tuple) else (result, 500)
    return "OK"


@app.post("/process-local")
def process_local():
    """Procesa carpeta local sin upload — solo acepta rutas del servidor."""
    engine_name = request.json.get("engine", ENGINES[0]["name"])
    carpeta     = request.json.get("carpeta", "").strip()

    if not carpeta:
        return "Carpeta no especificada", 400

    p = Path(carpeta)
    if not p.exists() or not p.is_dir():
        return f"Carpeta no encontrada: {carpeta}", 400

    imagenes = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
    if not imagenes:
        return "No se encontraron imagenes en esa carpeta", 400

    result = _start_job(engine_name, imagenes, cleanup_fn=None)
    if result != "OK":
        return result if isinstance(result, str) else result[0], result[1] if isinstance(result, tuple) else 500
    return "OK"


@app.post("/upload")
def upload():
    engine_name = request.form.get("engine", ENGINES[0]["name"])

    files = request.files.getlist("files")
    if not files:
        return "Sin archivos", 400

    tmpdir = Path(tempfile.mkdtemp())
    for f in files:
        name = Path(f.filename).name
        ext  = Path(name).suffix.lower()
        if ext == ".zip":
            zip_path = tmpdir / name
            f.save(str(zip_path))
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    if Path(member).suffix.lower() in IMG_EXTS:
                        zf.extract(member, tmpdir / "zip_out")
        elif ext in IMG_EXTS:
            f.save(str(tmpdir / name))

    imagenes = sorted(p for p in tmpdir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not imagenes:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return "No se encontraron imagenes", 400

    result = _start_job(engine_name, imagenes, cleanup_fn=lambda: shutil.rmtree(tmpdir, ignore_errors=True))
    if result != "OK":
        return result if isinstance(result, str) else result[0], result[1] if isinstance(result, tuple) else 500
    return "OK"


@app.get("/stream")
def stream():
    offset = int(request.args.get("offset", 0))

    def generate():
        idx = offset
        import time
        deadline = time.time() + 10
        while True:
            with _job_cond:
                # replay buffered events
                while idx < len(_job_log):
                    data = _job_log[idx]
                    yield f"data: {json.dumps(data)}\n\n"
                    idx += 1
                    if data.get("done") == -1:
                        return
                # si el job terminó y no hay más eventos, cerrar stream
                if not _job_running and idx >= len(_job_log):
                    return
                # esperar nuevos eventos
                _job_cond.wait(timeout=30)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/stop")
def stop():
    from benchmark_ocr import _skip
    _skip.set()
    return "OK"


@app.get("/history")
def history():
    if not HISTORY_FILE.exists():
        return "[]", 200, {"Content-Type": "application/json"}
    return HISTORY_FILE.read_text(encoding="utf-8"), 200, {"Content-Type": "application/json"}


@app.post("/check-csv")
def check_csv():
    f = request.files.get("file")
    if not f:
        return {"error": "sin archivo"}, 400
    tmp = Path(tempfile.mktemp(suffix=".csv"))
    try:
        f.save(str(tmp))
        result = analizar_csv(tmp)
    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        tmp.unlink(missing_ok=True)
    from flask import jsonify
    return jsonify(result)


@app.get("/csv/<engine>")
def csv_download(engine):
    p = OUT_DIR / f"{engine}.csv"
    if not p.exists():
        return "CSV no encontrado", 404
    fecha = request.args.get("fecha", "")
    if fecha:
        # "2026-06-28 20:26:00" → "Extraccion 28-6 20:26"
        try:
            dt = datetime.datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
            name = f"Extraccion {dt.day}-{dt.month} {dt.strftime('%H:%M')}.csv"
        except Exception:
            name = f"{engine}.csv"
    else:
        name = f"{engine}.csv"
    return send_file(str(p), as_attachment=True, download_name=name)


@app.get("/csv-json/<engine>")
def csv_json(engine):
    import csv as _csv
    p = OUT_DIR / f"{engine}.csv"
    if not p.exists():
        return "CSV no encontrado", 404
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    return json.dumps(rows, ensure_ascii=False), 200, {"Content-Type": "application/json"}


if __name__ == "__main__":
    _start_key_listener()
    port = int(os.environ.get("PORT", 5000))
    print(f"Abriendo en http://localhost:{port}")
    app.run(debug=False, threaded=True, host="0.0.0.0", port=port)

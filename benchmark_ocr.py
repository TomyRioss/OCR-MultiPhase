"""
Benchmark OCR engines. CSV escrito en tiempo real. Tecla S = skip engine actual.
4 workers en engines thread-safe, 1 en los que no lo son.

Uso:
    python benchmark_ocr.py
"""

import re, csv, time, sys, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageOps
import cv2
import numpy as np

WORKERS = 4

# ── Datos ─────────────────────────────────────────────────────────────────────

BANCOS_CONOCIDOS = [
    "Nación", "Galicia", "Santander", "BBVA", "Macro", "HSBC", "Citibank",
    "Provincia BA", "Provincia", "Patagonia", "Supervielle", "Comafi",
    "Credicoop", "Industrial", "Brubank", "Mercado Pago", "Uala", "Ualá",
    "NaranjaX", "Naranja X", "Nuevo Banco de Santa Fe", "Santa Fe",
    "Bind", "Wilobank", "Reba", "Lemon", "Personal Pay", "Modo", "Bapro",
    "Claro Pay",
]

def _solo_letras(linea):
    return re.sub(r'^[^a-zA-ZáéíóúÁÉÍÓÚñÑ]+', '', linea).strip()

def _es_nombre(s):
    palabras = s.replace(",", " ").split()
    return (len(palabras) >= 2
            and all(re.match(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ'\-]+$", w) for w in palabras)
            and not re.search(r'\d', s))

def _limpiar_titular(raw):
    palabras = [w for w in raw.replace(",", " ").split()
                if re.match(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ'\-]{2,}$", w)]
    return " ".join(w.capitalize() for w in palabras)

def extraer_datos(texto):
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    titular = ""
    for i, linea in enumerate(lineas):
        if re.search(r'TU[LT]AR\s+DE\s+LA\s+CUENTA', linea, re.IGNORECASE):
            partes = []
            j = i + 1
            while j < len(lineas):
                sig = lineas[j]
                if re.match(r'CUIT\s*[:\-]', sig, re.IGNORECASE): break
                if re.search(r'\d{10,}', sig): break
                if any(b.lower() in _solo_letras(sig).lower() for b in BANCOS_CONOCIDOS): break
                partes.append(sig)
                j += 1
            titular = _limpiar_titular(" ".join(partes))
            break
    if not titular:
        for i, linea in enumerate(lineas):
            if "TRANSFERENCIA" in linea.upper():
                for j in range(i + 1, min(i + 5, len(lineas))):
                    if _es_nombre(lineas[j]):
                        titular = _limpiar_titular(lineas[j])
                        break
                break
    cuit = ""
    m = re.search(r'CUIT\s*[:\-]?\s*(\d{2})\s*[-–]\s*(\d{8})\s*[-–]\s*([\dO])', texto, re.IGNORECASE)
    if m:
        cuit = f"{m.group(1)}-{m.group(2)}-{m.group(3).replace('O','0')}"
    tipo_cuenta = "CVU" if re.search(r'\bCVU\b', texto, re.IGNORECASE) else \
                  "CBU" if re.search(r'\bCBU\b', texto, re.IGNORECASE) else ""
    cbu = ""
    m = re.search(r'(?<!\d)(\d{22})(?!\d)', texto)
    if m:
        cbu = m.group(1) if tipo_cuenta != "CVU" else "N/A"
    elif tipo_cuenta == "CVU":
        cbu = "N/A"
    banco = ""
    for linea in lineas:
        limpia = _solo_letras(linea)
        for b in BANCOS_CONOCIDOS:
            if b.lower() in limpia.lower():
                banco = b
                break
        if banco: break
    return {"titular": titular, "banco": banco, "CBU": cbu, "CUIT": cuit}

def preprocess(img):
    gris = img.convert("L")
    media = sum(gris.getdata()) / (gris.width * gris.height)
    return ImageOps.invert(gris) if media < 128 else gris


def opencv_preprocess(pil_img):
    """
    Min-channel + Otsu: preserva texto coloreado (badges bancarios verde/azul/naranja)
    que el adaptive threshold destruía. Escala 2x + CLAHE suave.
    """
    arr = np.array(pil_img.convert("RGB"))
    # min de canales: texto coloreado sobre blanco → oscuro; fondo blanco → 255
    gray = np.min(arr, axis=2).astype(np.uint8)

    if gray.mean() < 128:
        gray = cv2.bitwise_not(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation=cv2.INTER_LANCZOS4)

    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(gray)

# ── Skip listener ─────────────────────────────────────────────────────────────

_skip = threading.Event()
_stop_listener = threading.Event()

def _start_key_listener():
    def _loop():
        try:
            import msvcrt
            while not _stop_listener.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                    if ch == "s":
                        _skip.set()
                time.sleep(0.05)
        except Exception:
            pass  # no-op en entornos sin tty
    t = threading.Thread(target=_loop, daemon=True)
    t.start()

# ── Engine wrappers ───────────────────────────────────────────────────────────

_easyocr_reader = None
def ocr_easyocr(img):
    global _easyocr_reader
    import easyocr, numpy as np
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['es', 'en'], gpu=False)
    return "\n".join(_easyocr_reader.readtext(np.array(img), detail=0))

_paddle_ocr = None
def ocr_paddleocr(img):
    global _paddle_ocr
    from paddleocr import PaddleOCR
    import numpy as np
    if _paddle_ocr is None:
        _paddle_ocr = PaddleOCR(use_angle_cls=False, lang='es', show_log=False)
    result = _paddle_ocr.ocr(np.array(img.convert("RGB")), cls=False)
    if not result or not result[0]: return ""
    return "\n".join(line[1][0] for line in result[0] if line and line[1])

def ocr_rapidocr(img):
    from rapidocr_onnxruntime import RapidOCR
    import numpy as np
    engine = RapidOCR()
    result, _ = engine(np.array(img.convert("RGB")))
    if not result: return ""
    return "\n".join(r[1] for r in result)

def ocr_tesseract(img):
    import pytesseract
    import platform, shutil
    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    else:
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
    return pytesseract.image_to_string(img, lang="spa+eng", config="--psm 6 --oem 1")

_doctr_model = None
def ocr_doctr(img):
    global _doctr_model
    import numpy as np
    from doctr.models import ocr_predictor
    if _doctr_model is None:
        _doctr_model = ocr_predictor(pretrained=True)
    arr = np.array(img.convert("RGB"))
    result = _doctr_model([arr])
    lines = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                lines.append(" ".join(w.value for w in line.words))
    return "\n".join(lines)

_surya_models = None
def ocr_surya(img):
    global _surya_models
    from surya.ocr import run_ocr
    from surya.model.detection.model import load_model as load_det, load_processor as load_det_proc
    from surya.model.recognition.model import load_model as load_rec
    from surya.model.recognition.processor import load_processor as load_rec_proc
    if _surya_models is None:
        _surya_models = (load_det_proc(), load_det(), load_rec(), load_rec_proc())
    det_proc, det_model, rec_model, rec_proc = _surya_models
    results = run_ocr([img], [["es", "en"]], det_model, det_proc, rec_model, rec_proc)
    return "\n".join(line.text for page in results for line in page.text_lines)

_trocr_pipe = None
def ocr_trocr(img):
    global _trocr_pipe
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    if _trocr_pipe is None:
        _trocr_pipe = (
            TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed"),
            VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed"),
        )
    processor, model = _trocr_pipe
    pixel_values = processor(img.convert("RGB"), return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(pixel_values)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

# ── Engine registry ───────────────────────────────────────────────────────────
# thread_safe=True → 4 workers; False → 1 worker (engine no soporta concurrencia)

ENGINES = [
    {"name": "tesseract",  "fn": ocr_tesseract,  "thread_safe": True},
    {"name": "rapidocr",   "fn": ocr_rapidocr,   "thread_safe": True},
    {"name": "doctr",      "fn": ocr_doctr,       "thread_safe": False},
    {"name": "paddleocr",  "fn": ocr_paddleocr,  "thread_safe": False},
    {"name": "easyocr",    "fn": ocr_easyocr,    "thread_safe": False},
    {"name": "surya",      "fn": ocr_surya,       "thread_safe": False},
]

CSV_FIELDS = ["imagen", "CBU", "CUIT", "titular", "banco"]

# ── Runner ────────────────────────────────────────────────────────────────────

def run_engine(cfg, imagenes, out_dir, on_progress=None):
    name = cfg["name"]
    fn   = cfg["fn"]
    w    = WORKERS if cfg["thread_safe"] else 1

    _skip.clear()

    # warm-up / import check
    try:
        print(f"\n[{name}] Inicializando... (S = skip)", flush=True)
        if on_progress:
            on_progress({"done": 0, "n": 0, "avg_ms": 0, "eta": 0,
                         "imagen": f"cargando modelo {name}...",
                         "CBU": "", "CUIT": "", "titular": "", "banco": "", "_status": "init"})
        fn(Image.open(imagenes[0]))
        if on_progress:
            on_progress({"done": 0, "n": 0, "avg_ms": 0, "eta": 0,
                         "imagen": f"modelo listo, iniciando procesamiento...",
                         "CBU": "", "CUIT": "", "titular": "", "banco": "", "_status": "init"})
    except ImportError as e:
        print(f"[{name}] SKIP — no instalado: {e}")
        return None
    except Exception as e:
        print(f"[{name}] SKIP — error init: {e}")
        return None

    n = len(imagenes)
    csv_path = out_dir / f"{name}.csv"

    # estado compartido entre threads
    state = {"done": 0, "tiempos": [], "campos": {"CBU": 0, "CUIT": 0, "titular": 0, "banco": 0}, "exitosas": 0}
    lock = threading.Lock()

    def process_one(img_path):
        if _skip.is_set():
            return None
        try:
            img = Image.open(img_path)
            t0 = time.perf_counter()
            texto = fn(img)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            texto = f"ERROR: {e}"
            elapsed_ms = 0
        datos = extraer_datos(texto)
        return img_path.name, elapsed_ms, datos, texto

    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csvf.flush()

        with ThreadPoolExecutor(max_workers=w) as ex:
            futures = {ex.submit(process_one, p): p for p in imagenes}
            for fut in as_completed(futures):
                if _skip.is_set():
                    for f in futures:
                        f.cancel()
                    break
                res = fut.result()
                if res is None:
                    continue
                img_name, elapsed_ms, datos, texto = res

                with lock:
                    state["tiempos"].append(elapsed_ms)
                    for campo in state["campos"]:
                        if datos.get(campo):
                            state["campos"][campo] += 1
                    if all(datos.get(c) for c in ("CBU", "CUIT", "titular", "banco")):
                        state["exitosas"] += 1
                    state["done"] += 1
                    done = state["done"]
                    avg = sum(state["tiempos"]) / len(state["tiempos"])
                    pct = done / n
                    bar = "#" * int(pct * 25) + "-" * (25 - int(pct * 25))
                    eta = avg * (n - done) / 1000
                    print(
                        f"\r[{name}] {done:>3}/{n} [{bar}] {avg:>6.0f}ms/img  ETA {eta:>4.0f}s  [S=skip]  ",
                        end="", flush=True,
                    )
                    if on_progress:
                        on_progress({
                            "done": done, "n": n, "avg_ms": round(avg),
                            "eta": round(eta), "imagen": img_name,
                            "CBU": datos["CBU"], "CUIT": datos["CUIT"],
                            "titular": datos["titular"], "banco": datos["banco"],
                        })

                writer.writerow({
                    "imagen": img_name,
                    "CBU": datos["CBU"],
                    "CUIT": datos["CUIT"],
                    "titular": datos["titular"],
                    "banco": datos["banco"],
                })
                csvf.flush()

    done = state["done"]
    avg_ms = sum(state["tiempos"]) / len(state["tiempos"]) if state["tiempos"] else 0
    status = "SKIPPED" if _skip.is_set() else "OK"

    exitosas = state["exitosas"]
    vacias = done - exitosas
    pct_exito = (exitosas / done * 100) if done else 0
    stats_msg = (
        f"OCR completo — {done} imágenes | "
        f"Exitosas (4/4 campos): {exitosas} ({pct_exito:.1f}%) | "
        f"Filas con campos vacíos: {vacias}"
    )
    print(f"\n[{name}] {status} -- {done}/{n} imgs  avg {avg_ms:.0f}ms -> {csv_path.name}")
    print(f"[{name}] {stats_msg}")

    if on_progress:
        on_progress({
            "done": done, "n": done, "avg_ms": round(avg_ms), "eta": 0,
            "imagen": stats_msg,
            "CBU": "", "CUIT": "", "titular": "", "banco": "",
            "_status": "ocr_stats",
            "stats": {
                "total": done,
                "exitosas": exitosas,
                "vacias": vacias,
                "pct_exito": round(pct_exito, 1),
            },
        })

    return {
        "engine": name,
        "avg_ms": avg_ms,
        "n": done,
        **{f"{k}%": state["campos"][k] / done * 100 if done else 0 for k in state["campos"]},
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    batch_dir = Path(__file__).parent / "batch"
    out_dir   = Path(__file__).parent / "benchmark_results"
    out_dir.mkdir(exist_ok=True)

    imagenes = sorted(
        f for f in batch_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    )[:100]

    print(f"Imágenes: {len(imagenes)} en {batch_dir}")
    print("S = skip engine actual  |  Ctrl+C = salir\n")

    _start_key_listener()

    summary = []
    try:
        for cfg in ENGINES:
            result = run_engine(cfg, imagenes, out_dir)
            if result:
                summary.append(result)
    except KeyboardInterrupt:
        print("\n\nInterrumpido.")
    finally:
        _stop_listener.set()

    if not summary:
        return

    print("\n" + "=" * 75)
    print(f"{'Engine':<12} {'workers':>7} {'avg ms':>8} {'n':>5} {'CBU%':>7} {'CUIT%':>7} {'titular%':>10} {'banco%':>8}")
    print("-" * 75)
    for s in sorted(summary, key=lambda x: x["avg_ms"]):
        cfg = next(c for c in ENGINES if c["name"] == s["engine"])
        w = WORKERS if cfg["thread_safe"] else 1
        print(f"{s['engine']:<12} {w:>7} {s['avg_ms']:>8.0f} {s['n']:>5} "
              f"{s['CBU%']:>7.1f} {s['CUIT%']:>7.1f} {s['titular%']:>10.1f} {s['banco%']:>8.1f}")
    print("=" * 75)
    print(f"\nCSVs en: {out_dir}/")


if __name__ == "__main__":
    main()

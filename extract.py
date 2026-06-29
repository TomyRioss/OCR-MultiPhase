"""
Primera extracción ultra rápida — solo ULTRA + RÁPIDAS configs (skip FULL).
Salida: resultados/resultado{n}.csv

USO:
    python extract.py --carpeta ./imagenes
    python extract.py --carpeta ./imagenes --workers 8
"""

import argparse
import csv
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from extractor_transferencias import (
    SUPPORTED_EXTENSIONS,
    _beep_progreso,
    _ocr_intentos,
    _score,
    _CONFIGS_ULTRA,
    _CONFIGS_RAPIDAS,
    clave_dedup,
    listar_imagenes,
    resolver_salida,
)
import platform, shutil
import pytesseract
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
from PIL import Image, ImageEnhance, ImageOps
from extractor_transferencias import extraer_datos

DEFAULT_WORKERS = 6
log = logging.getLogger(__name__)


def _setup_logging():
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / f"extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(),
        ],
    )


def _ocr_rapido(img_original):
    """Solo ULTRA + RÁPIDAS. Sin FULL para máxima velocidad."""
    import pytesseract
    from PIL import ImageEnhance, ImageOps
    from extractor_transferencias import opencv_preprocess, opencv_preprocess_colored, opencv_preprocess_white_on_color, opencv_preprocess_soft

    mejor_datos: dict = {}
    mejor_score = -1

    # Intentos con preprocesamiento OpenCV (estándar + variantes para texto coloreado)
    for preprocess_fn in (opencv_preprocess, opencv_preprocess_colored, opencv_preprocess_white_on_color, opencv_preprocess_soft):
        img_cv = preprocess_fn(img_original)
        texto_cv = pytesseract.image_to_string(img_cv, lang="spa+eng", config="--psm 6 --oem 1")
        datos_cv = extraer_datos(texto_cv)
        s_cv = _score(datos_cv)
        # Si score igual pero este tiene banco y el actual no, preferir este
        if s_cv > mejor_score or (s_cv == mejor_score and datos_cv.get("banco") and not mejor_datos.get("banco")):
            mejor_score = s_cv
            mejor_datos = datos_cv
        if mejor_score == 4:
            return mejor_datos

    gris_base = img_original.convert("L")
    es_dark = (sum(gris_base.getdata()) / (gris_base.width * gris_base.height)) < 128
    img_base = ImageOps.invert(gris_base) if es_dark else gris_base

    for escala, umbral, psm, contraste, nitidez in list(_CONFIGS_ULTRA) + list(_CONFIGS_RAPIDAS):
        img = img_base.copy()
        if contraste != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contraste)
        if nitidez != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(nitidez)
        img = img.resize((img.width * escala, img.height * escala), Image.LANCZOS)
        if umbral is not None:
            img = img.point(lambda x: 0 if x < umbral else 255, "1")
        texto = pytesseract.image_to_string(img, lang="spa+eng", config=f"--psm {psm} --oem 1")
        datos = extraer_datos(texto)
        s = _score(datos)
        if s > mejor_score:
            mejor_score = s
            mejor_datos = datos
        if mejor_score == 4:
            break

    return mejor_datos


def procesar_imagen_rapido(ruta: Path) -> dict:
    try:
        img = Image.open(ruta)
        datos = _ocr_rapido(img)
        datos["_archivo"] = ruta.name
        datos["imagen"] = ruta.name
        return datos
    except Exception as e:
        return {
            "titular": f"ERROR: {e}",
            "banco": "", "CBU": "", "CUIT": "",
            "tipo_cuenta": "", "_archivo": ruta.name, "imagen": ruta.name,
        }


def procesar_todas(imagenes: list[Path], salida: Path, workers: int):
    campos = ["imagen", "CBU", "CUIT", "titular", "banco"]
    vistos: set[str] = set()
    errores = descartados = procesadas = 0
    total = len(imagenes)
    prev_pct = 0

    with open(salida, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()

        pool = ProcessPoolExecutor(max_workers=workers)
        try:
            futuros = {pool.submit(procesar_imagen_rapido, img): img for img in imagenes}
            for fut in tqdm(as_completed(futuros), total=len(futuros), desc="extract", unit="img"):
                res = fut.result()
                titular = res.get("titular", "")

                if titular.startswith("ERROR"):
                    errores += 1
                    log.error(f"FALLO: {res['_archivo']} — {titular}")
                    writer.writerow({k: res.get(k, "") for k in campos})
                    f.flush()
                else:
                    clave = clave_dedup(titular)
                    if clave and clave in vistos:
                        descartados += 1
                    else:
                        if clave:
                            vistos.add(clave)
                        procesadas += 1
                        writer.writerow({k: res.get(k, "") for k in campos})
                        f.flush()

                pct = int((procesadas + errores + descartados) / total * 100)
                for hito in (25, 50, 75):
                    if prev_pct < hito <= pct:
                        _beep_progreso(hito)
                prev_pct = pct
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    _beep_progreso(100)
    log.info(f"✓ {procesadas} procesadas | ✗ {errores} errores | ⊘ {descartados} duplicados")
    log.info(f"→ {salida.resolve()}")
    return {"procesadas": procesadas, "errores": errores, "descartados": descartados}


def main():
    _setup_logging()
    parser = argparse.ArgumentParser(description="Extracción rápida (sin FULL configs)")
    parser.add_argument("--carpeta", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    if not args.carpeta.exists():
        print(f"ERROR: carpeta '{args.carpeta}' no existe.")
        return

    imagenes = listar_imagenes(args.carpeta)
    if not imagenes:
        print("Sin imágenes.")
        return

    salida = resolver_salida(Path(__file__).parent / "resultados")
    log.info(f"Salida: {salida}")

    from repair import reparar_csv, postprocesar_csv

    print("\n=== FASE 1: Extracción Tesseract ===")
    inicio = datetime.now()
    stats = procesar_todas(imagenes, salida, args.workers)
    dur = (datetime.now() - inicio).total_seconds()
    log.info(f"Tiempo: {dur:.1f}s | {len(imagenes)/dur:.1f} img/s")

    # Leer CSV para contar campos vacíos
    import csv as _csv
    total_filas = exitosas = vacias = 0
    try:
        with open(salida, encoding="utf-8") as _f:
            for row in _csv.DictReader(_f):
                total_filas += 1
                if all(row.get(c, "").strip() for c in ("CBU", "CUIT", "titular", "banco")):
                    exitosas += 1
                else:
                    vacias += 1
        pct = exitosas / total_filas * 100 if total_filas else 0
        print(f"\n{'='*50}")
        print(f"  OCR STATS (post-tesseract, pre-reparación)")
        print(f"  Total filas:          {total_filas}")
        print(f"  Filas exitosas (4/4): {exitosas}  ({pct:.1f}%)")
        print(f"  Filas con vacíos:     {vacias}   ({100-pct:.1f}%)")
        print(f"{'='*50}")
    except Exception as e:
        log.warning(f"No se pudo leer stats del CSV: {e}")

    print("\n=== FASE 2: Reparación por campo ===")
    reparar_csv(salida, args.carpeta)

    print("\n=== FASE 3: Post-proceso (filtrar CVU / quitar columna imagen) ===")
    postprocesar_csv(salida)


if __name__ == "__main__":
    main()

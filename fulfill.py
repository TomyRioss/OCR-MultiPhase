"""
Post-procesamiento de calidad: reprocesa solo filas con campos vacíos usando todos los configs OCR.
Lee el CSV de extract.py, escribe resultado en resultados_finales/resultado_final{n}.csv

USO:
    python fulfill.py --csv resultados/resultado1.csv --carpeta ./imagenes
    python fulfill.py --csv resultados/resultado1.csv --carpeta ./imagenes --workers 4
"""

import argparse
import csv
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from extractor_transferencias import (
    _beep_progreso,
    _ocr_intentos,
    _score,
    extraer_datos,
)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image

DEFAULT_WORKERS = 4
CAMPOS = ["titular", "banco", "CBU", "CUIT", "name"]
log = logging.getLogger(__name__)


def _setup_logging():
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / f"fulfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(),
        ],
    )


def _es_incompleta(fila: dict) -> bool:
    for c in ("titular", "banco", "CBU", "CUIT"):
        v = fila.get(c, "").strip()
        if c == "CBU" and v == "N/A":
            continue  # CVU válido
        if not v or v.startswith("ERROR"):
            return True
    return False


def _merge(original: dict, nuevo: dict) -> dict:
    """Rellena campos vacíos del original con los del nuevo si mejoran el score."""
    resultado = dict(original)
    for c in ("titular", "banco", "CBU", "CUIT"):
        v_orig = original.get(c, "").strip()
        v_nuevo = nuevo.get(c, "").strip()
        orig_vacio = not v_orig or v_orig.startswith("ERROR")
        nuevo_lleno = v_nuevo and not v_nuevo.startswith("ERROR")
        if orig_vacio and nuevo_lleno:
            resultado[c] = v_nuevo
    return resultado


def reprocesar_imagen(ruta: Path) -> dict:
    try:
        img = Image.open(ruta)
        datos = _ocr_intentos(img)  # todos los configs (ULTRA+RAPIDAS+FULL)
        datos["_archivo"] = ruta.name
        datos["name"] = ruta.name
        return datos
    except Exception as e:
        return {
            "titular": f"ERROR: {e}",
            "banco": "", "CBU": "", "CUIT": "",
            "tipo_cuenta": "", "_archivo": ruta.name, "name": ruta.name,
        }


def resolver_salida_final(carpeta: Path) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidato = carpeta / f"resultado_final{n}.csv"
        if not candidato.exists():
            return candidato
        n += 1


def main():
    _setup_logging()
    parser = argparse.ArgumentParser(description="Reprocesa filas incompletas con máxima calidad OCR")
    parser.add_argument("--csv", type=Path, required=True, help="CSV de extract.py")
    parser.add_argument("--carpeta", type=Path, required=True, help="Carpeta de imágenes originales")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: '{args.csv}' no existe.")
        return
    if not args.carpeta.exists():
        print(f"ERROR: carpeta '{args.carpeta}' no existe.")
        return

    with open(args.csv, encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))

    incompletas = [(i, fila) for i, fila in enumerate(filas) if _es_incompleta(fila)]
    log.info(f"Total filas: {len(filas)} | Incompletas: {len(incompletas)}")

    if not incompletas:
        log.info("Nada que reprocesar.")
        return

    # Mapear nombre de archivo → índice en filas
    idx_por_nombre = {fila.get("name", ""): i for i, fila in incompletas}

    # Buscar imágenes en carpeta
    imagenes_disponibles = {p.name: p for p in args.carpeta.iterdir()
                            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}}

    pendientes: list[tuple[int, Path]] = []
    sin_imagen = []
    for nombre, idx in idx_por_nombre.items():
        if nombre in imagenes_disponibles:
            pendientes.append((idx, imagenes_disponibles[nombre]))
        else:
            sin_imagen.append(nombre)

    if sin_imagen:
        log.warning(f"{len(sin_imagen)} imágenes no encontradas en carpeta: {sin_imagen[:5]}{'...' if len(sin_imagen)>5 else ''}")

    log.info(f"Reprocesando {len(pendientes)} imágenes con todos los configs...")

    salida = resolver_salida_final(Path(__file__).parent / "resultados_finales")
    f_out = open(salida, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=CAMPOS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(filas)  # escribe todas las filas (completas e incompletas) al inicio
    f_out.flush()
    log.info(f"-> {salida.resolve()}")

    mejoras = 0
    total = len(pendientes)
    prev_pct = 0

    pool = ProcessPoolExecutor(max_workers=args.workers)
    try:
        futuros = {pool.submit(reprocesar_imagen, ruta): (idx, ruta) for idx, ruta in pendientes}
        for fut in tqdm(as_completed(futuros), total=len(futuros), desc="fulfill", unit="img"):
            idx, ruta = futuros[fut]
            nuevo = fut.result()
            original = filas[idx]
            score_antes = _score(original)
            merged = _merge(original, nuevo)
            score_despues = _score(merged)
            filas[idx] = merged
            if score_despues > score_antes:
                mejoras += 1
                log.info(f"Mejorado {ruta.name}: {score_antes}->{score_despues} campos")

            procesadas = list(futuros.values()).index((idx, ruta)) + 1
            pct = int(procesadas / total * 100)
            for hito in (25, 50, 75):
                if prev_pct < hito <= pct:
                    _beep_progreso(hito)
            prev_pct = pct
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    # reescribir con datos mejorados
    f_out.seek(0)
    f_out.truncate()
    writer.writeheader()
    writer.writerows(filas)
    f_out.close()

    _beep_progreso(100)
    log.info(f"OK {mejoras} filas mejoradas de {len(pendientes)} reprocesadas")


if __name__ == "__main__":
    main()

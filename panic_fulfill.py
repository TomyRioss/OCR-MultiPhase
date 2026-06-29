"""
Último recurso: usa GPT-4o vision para extraer campos faltantes.
Lee CSV de fulfill/extract, manda imágenes incompletas a la API, merge y guarda.

USO:
    python panic_fulfill.py --csv resultados_finales/resultado_final1.csv --carpeta ./batch2
    python panic_fulfill.py --csv resultados_finales/resultado_final1.csv --carpeta ./batch2 --workers 8

.env:
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini   # opcional, default: gpt-4o-mini
"""

import argparse
import base64
import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

CAMPOS = ["titular", "banco", "CBU", "CUIT", "name"]
DEFAULT_WORKERS = 8
DEFAULT_MODEL = "gpt-4o-mini"

log = logging.getLogger(__name__)

PROMPT = """Sos un extractor de datos de capturas de pantalla de transferencias bancarias argentinas.
Analizá la imagen y devolvé SOLO un JSON con estos campos:
{
  "titular": "Nombre Apellido",
  "banco": "Nombre del banco o billetera",
  "CBU": "CBU o CVU de 22 dígitos, o N/A si no hay",
  "CUIT": "CUIT con formato XX-XXXXXXXX-X"
}
Si un campo no es visible, ponelo como cadena vacía "".
No agregues texto extra, solo el JSON."""


def _setup_logging():
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / f"panic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _es_incompleta(fila: dict) -> bool:
    for c in ("titular", "banco", "CBU", "CUIT"):
        v = fila.get(c, "").strip()
        if c == "CBU" and v == "N/A":
            continue
        if not v or v.startswith("ERROR"):
            return True
    return False


def _merge(original: dict, nuevo: dict) -> dict:
    resultado = dict(original)
    for c in ("titular", "banco", "CBU", "CUIT"):
        v_orig = original.get(c, "").strip()
        v_nuevo = nuevo.get(c, "").strip()
        orig_vacio = not v_orig or v_orig.startswith("ERROR")
        nuevo_lleno = v_nuevo and not v_nuevo.startswith("ERROR") and v_nuevo != ""
        if orig_vacio and nuevo_lleno:
            resultado[c] = v_nuevo
    return resultado


def _score(d: dict) -> int:
    return sum(1 for k in ("titular", "banco", "CBU", "CUIT")
               if d.get(k) and not str(d[k]).startswith("ERROR") and d[k] not in ("", None))


def analizar_imagen(cliente: OpenAI, modelo: str, ruta: Path) -> dict:
    with open(ruta, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    ext = ruta.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/jpeg")

    resp = cliente.chat.completions.create(
        model=modelo,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}},
            ],
        }],
        max_tokens=300,
    )

    texto = resp.choices[0].message.content.strip()
    # limpiar markdown si GPT envuelve en ```json
    if texto.startswith("```"):
        texto = texto.split("```")[1]
        if texto.startswith("json"):
            texto = texto[4:]
    return json.loads(texto.strip())


def _worker(args_tuple):
    cliente, modelo, idx, ruta = args_tuple
    try:
        datos = analizar_imagen(cliente, modelo, ruta)
        datos["_archivo"] = ruta.name
        return idx, datos, None
    except Exception as e:
        return idx, {}, str(e)


def resolver_salida(carpeta: Path) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidato = carpeta / f"resultado_panic{n}.csv"
        if not candidato.exists():
            return candidato
        n += 1


def main():
    _setup_logging()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY no encontrada en .env")
        return

    modelo = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--carpeta", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    if not args.csv.exists():
        log.error(f"CSV no encontrado: {args.csv}")
        return
    if not args.carpeta.exists():
        log.error(f"Carpeta no encontrada: {args.carpeta}")
        return

    with open(args.csv, encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))

    incompletas = [(i, fila) for i, fila in enumerate(filas) if _es_incompleta(fila)]
    log.info(f"Total filas: {len(filas)} | Incompletas: {len(incompletas)} | Modelo: {modelo}")

    if not incompletas:
        log.info("Nada que reprocesar.")
        return

    extensiones = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    imagenes_disponibles = {p.name: p for p in args.carpeta.iterdir()
                            if p.is_file() and p.suffix.lower() in extensiones}

    pendientes = []
    sin_imagen = []
    for i, fila in incompletas:
        nombre = fila.get("name", "")
        if nombre in imagenes_disponibles:
            pendientes.append((i, imagenes_disponibles[nombre]))
        else:
            sin_imagen.append(nombre)

    if sin_imagen:
        log.warning(f"{len(sin_imagen)} imagenes no encontradas: {sin_imagen[:3]}{'...' if len(sin_imagen) > 3 else ''}")

    log.info(f"Enviando {len(pendientes)} imagenes a {modelo}...")

    cliente = OpenAI(api_key=api_key)
    mejoras = 0
    errores = 0

    tareas = [(cliente, modelo, idx, ruta) for idx, ruta in pendientes]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futuros = {pool.submit(_worker, t): t for t in tareas}
        for fut in tqdm(as_completed(futuros), total=len(futuros), desc="panic", unit="img"):
            idx, nuevo, error = fut.result()
            if error:
                errores += 1
                log.warning(f"Error en {futuros[fut][3].name}: {error}")
                continue
            original = filas[idx]
            score_antes = _score(original)
            merged = _merge(original, nuevo)
            score_despues = _score(merged)
            filas[idx] = merged
            if score_despues > score_antes:
                mejoras += 1
                log.info(f"Mejorado {futuros[fut][3].name}: {score_antes}->{score_despues}")

    salida = resolver_salida(Path(__file__).parent / "resultados_finales")
    with open(salida, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filas)

    log.info(f"OK {mejoras} mejoradas | {errores} errores | -> {salida.resolve()}")


if __name__ == "__main__":
    main()

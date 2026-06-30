"""
Extractor de datos bancarios desde imágenes de transferencia
Usa: Tesseract OCR local (sin API, sin costo)
Output: CSV con titular, banco, CBU, CUIT
        - Si la imagen muestra CVU en lugar de CBU → columna CBU = N/A
        - Un solo registro por titular (el primero encontrado); duplicados descartados

USO:
    # 1. Instalar Tesseract para Windows:
    #    https://github.com/UB-Mannheim/tesseract/wiki
    #    Instalar con idioma español (spa) incluido.
    #    Agregar al PATH: C:\\Program Files\\Tesseract-OCR

    # 2. Instalar dependencias Python:
    #    pip install pytesseract Pillow tqdm

    python extractor_transferencias.py --carpeta ./imagenes
    python extractor_transferencias.py --carpeta ./imagenes --workers 8
    python extractor_transferencias.py --carpeta ./imagenes --reanudar resultados/resultado1.csv
"""

import os
import csv
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import platform, shutil
import pytesseract
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
DEFAULT_WORKERS = 6

BANCOS_CONOCIDOS = [
    "Nación", "Galicia", "Santander", "BBVA", "Macro", "HSBC", "Citibank",
    "Provincia BA", "Provincia", "Patagonia", "Supervielle", "Comafi",
    "Credicoop", "Industrial", "Brubank", "Mercado Pago", "Uala", "Ualá",
    "NaranjaX", "Naranja X", "Nuevo Banco de Santa Fe", "Santa Fe",
    "Bind", "Wilobank", "Reba", "Lemon", "Personal Pay", "Modo", "Bapro",
    "Claro Pay", "YPF Virtual Wallet", "Bancor", "BPN",
    "Administradora San Juan SA", "Coopel", "Coopeplus",
]

# ──────────────────────────────────────────────
# LOGGING — se inicializa en main() para evitar que workers creen logs vacíos
# ──────────────────────────────────────────────
log = logging.getLogger(__name__)


def _setup_logging():
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / f"extractor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(),
        ]
    )


# ──────────────────────────────────────────────
# EXTRACCIÓN OCR
# ──────────────────────────────────────────────
def _solo_letras(linea: str) -> str:
    """Quita caracteres basura del inicio (íconos OCR como f, e, ►, @, «) dejando solo texto."""
    return re.sub(r'^[^a-zA-ZáéíóúÁÉÍÓÚñÑ]+', '', linea).strip()


def _es_nombre(s: str) -> bool:
    """True si la cadena parece un nombre propio: 2+ palabras, solo letras y coma/guion."""
    palabras = s.replace(",", " ").split()
    return (len(palabras) >= 2
            and all(re.match(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ'\-]+$", w) for w in palabras)
            and not re.search(r'\d', s))


def _limpiar_titular(raw: str) -> str:
    """Filtra solo palabras alfabéticas de 2+ letras y capitaliza."""
    palabras = [w for w in raw.replace(",", " ").split()
                if re.match(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ'\-]{2,}$", w)]
    return " ".join(w.capitalize() for w in palabras)


def _es_banco(linea: str) -> bool:
    limpia = _solo_letras(linea)
    return any(b.lower() in limpia.lower() for b in BANCOS_CONOCIDOS)


def extraer_datos(texto: str) -> dict:
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]

    # ── TITULAR ──────────────────────────────────────────────────────────────
    # Ancla primaria: "TULAR DE LA CUENTA" (cubre "TITULAR" y el error OCR "FITUTAR")
    titular = ""
    for i, linea in enumerate(lineas):
        if re.search(r'TU[LT]AR\s+DE\s+LA\s+CUENTA', linea, re.IGNORECASE):
            partes = []
            j = i + 1
            while j < len(lineas):
                sig = lineas[j]
                if re.match(r'CUIT\s*[:\-]', sig, re.IGNORECASE):
                    break
                if re.search(r'\d{10,}', sig):   # número largo = CBU/alias → stop
                    break
                if _es_banco(sig):                # línea de banco → stop
                    break
                partes.append(sig)
                j += 1
            titular = _limpiar_titular(" ".join(partes))
            break

    # Fallback: primera línea post "Nueva Transferencia" que parece nombre propio
    if not titular:
        for i, linea in enumerate(lineas):
            if "TRANSFERENCIA" in linea.upper():
                for j in range(i + 1, min(i + 5, len(lineas))):
                    if _es_nombre(lineas[j]):
                        titular = _limpiar_titular(lineas[j])
                        break
                break

    # ── CUIT ─────────────────────────────────────────────────────────────────
    # Exactamente XX-XXXXXXXX-X (2-8-1 dígitos). Corregir O→0 en dígito final.
    cuit = ""
    m = re.search(r'CUIT\s*[:\-]?\s*(\d{2})\s*[-–]\s*(\d{8})\s*[-–]\s*([\dO])', texto, re.IGNORECASE)
    if m:
        cuit = f"{m.group(1)}-{m.group(2)}-{m.group(3).replace('O', '0')}"

    # ── TIPO DE CUENTA ───────────────────────────────────────────────────────
    tipo_cuenta = ""
    if re.search(r'\bCVU\b', texto, re.IGNORECASE):
        tipo_cuenta = "CVU"
    elif re.search(r'\bCBU\b', texto, re.IGNORECASE):
        tipo_cuenta = "CBU"

    # ── CBU ──────────────────────────────────────────────────────────────────
    # Exactamente 22 dígitos numéricos (ni 21 ni 23).
    cbu = ""
    m = re.search(r'(?<!\d)(\d{22})(?!\d)', texto)
    if m:
        cbu = m.group(1) if tipo_cuenta != "CVU" else "N/A"
    elif tipo_cuenta == "CVU":
        cbu = "N/A"

    # ── BANCO ─────────────────────────────────────────────────────────────────
    # Buscar en TODAS las líneas (no solo tras CUIT) porque el layout varía.
    # Limpiar íconos OCR del inicio antes de comparar.
    banco = ""
    for linea in lineas:
        limpia = _solo_letras(linea)
        for b in BANCOS_CONOCIDOS:
            if b.lower() in limpia.lower():
                banco = b
                break
        if banco:
            break

    return {
        "titular": titular,
        "banco": banco,
        "CBU": cbu,
        "CUIT": cuit,
        "tipo_cuenta": tipo_cuenta,
    }


def opencv_preprocess(pil_img) -> Image.Image:
    """CLAHE + denoise + 2x upscale + adaptive threshold para maximizar legibilidad bancaria."""
    arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if gray.mean() < 128:
        gray = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    return Image.fromarray(gray)


def opencv_preprocess_colored(pil_img) -> Image.Image:
    """Min-channel trick: oscurece texto coloreado (verde/azul/naranja) sobre fondo blanco.
    Cubre logos bancarios como 'Provincia BA' en verde, 'Galicia' en azul, etc."""
    arr = np.array(pil_img.convert("RGB"))
    # min de canales R,G,B: texto de cualquier color saturado → oscuro; fondo blanco → 255
    gray = np.min(arr, axis=2).astype(np.uint8)
    if gray.mean() < 128:
        gray = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=25, C=8)
    return Image.fromarray(gray)


def opencv_preprocess_white_on_color(pil_img) -> Image.Image:
    """Max-channel invertido: oscurece texto blanco sobre fondo oscuro/coloreado."""
    arr = np.array(pil_img.convert("RGB"))
    # 255 - max: texto blanco → 0 (negro); fondo coloreado oscuro → más claro
    gray = (255 - np.max(arr, axis=2)).astype(np.uint8)
    if gray.mean() < 128:
        gray = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=25, C=8)
    return Image.fromarray(gray)


def opencv_preprocess_soft(pil_img) -> Image.Image:
    """Sin binarización agresiva: solo escala 2x + CLAHE suave.
    Preserva badges/pills con texto coloreado de mediano contraste que el adaptive threshold destruye."""
    arr = np.array(pil_img.convert("RGB"))
    # min-channel para texto coloreado sobre blanco
    gray = np.min(arr, axis=2).astype(np.uint8)
    if gray.mean() < 128:
        gray = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation=cv2.INTER_LANCZOS4)
    # threshold global simple — no destruye badges de contraste medio
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(gray)


def _score(datos: dict) -> int:
    """Cuenta campos no vacíos (CBU=N/A cuenta como lleno, es CVU válido)."""
    return sum(1 for k in ("titular", "banco", "CBU", "CUIT")
               if datos.get(k) and datos[k] not in ("", None))


def _beep_progreso(pct: int):
    pass


_CONFIGS_ULTRA = [
    (1, None, 6, 1.0, 1.0),  # tamaño original, sin binarizar
    (1, 140,  6, 1.0, 1.0),  # tamaño original, binarizado
]

_CONFIGS_RAPIDAS = [
    (2, 140,  6, 1.0, 1.0),  # 2x, binarizado
    (2, None, 6, 1.0, 1.0),  # 2x, sin binarizar
    (2, 140,  4, 1.0, 1.0),  # PSM columna
    (2, 140,  3, 1.0, 1.0),  # PSM auto
]

_CONFIGS_FULL = [
    (2, 120,  6, 1.0, 1.0),  # umbral bajo
    (2, None, 4, 1.0, 1.0),  # sin binarizar PSM 4
    (2, 140, 11, 1.0, 1.0),  # texto disperso
    (2, 140,  6, 2.0, 1.0),  # alto contraste
    (2, 140,  6, 1.0, 2.0),  # alta nitidez
    (2, 140,  6, 2.0, 2.0),  # contraste + nitidez
    (1, 140,  6, 2.0, 2.0),  # sin upscale + realce
    (3, 140,  6, 1.0, 1.0),  # 3x upscale
    (2, 160,  6, 1.0, 1.0),  # umbral alto
    (2, 140,  6, 1.5, 1.5),  # realce moderado
]


def _ocr_intentos(img_original) -> dict:
    """
    3 pasadas: ultra (original size) → rápida (2x) → full (11 configs).
    Sale en cuanto score==4. OEM 1 (LSTM only) en todas — ~25% más rápido que OEM 3.
    """
    mejor_datos: dict = {}
    mejor_score = -1

    # Intentos con preprocesamiento OpenCV (estándar + variantes para texto coloreado)
    for preprocess_fn in (opencv_preprocess, opencv_preprocess_colored, opencv_preprocess_white_on_color, opencv_preprocess_soft):
        img_cv = preprocess_fn(img_original)
        texto_cv = pytesseract.image_to_string(img_cv, lang="spa+eng", config="--psm 6 --oem 1")
        datos_cv = extraer_datos(texto_cv)
        s_cv = _score(datos_cv)
        if s_cv > mejor_score or (s_cv == mejor_score and datos_cv.get("banco") and not mejor_datos.get("banco")):
            mejor_score = s_cv
            mejor_datos = datos_cv
        if mejor_score == 4:
            return mejor_datos

    gris_base = img_original.convert("L")
    es_dark = (sum(gris_base.getdata()) / (gris_base.width * gris_base.height)) < 128
    img_base = ImageOps.invert(gris_base) if es_dark else gris_base

    def _probar(configs):
        nonlocal mejor_datos, mejor_score
        for escala, umbral, psm, contraste, nitidez in configs:
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
            if s > mejor_score or (s == mejor_score and datos.get("banco") and not mejor_datos.get("banco")):
                mejor_score = s
                mejor_datos = datos
            if mejor_score == 4:
                return

    _probar(_CONFIGS_ULTRA)
    if mejor_score < 4:
        _probar(_CONFIGS_RAPIDAS)
    if mejor_score < 4:
        _probar(_CONFIGS_FULL)

    return mejor_datos


def procesar_imagen(ruta: Path) -> dict:
    """Worker: OCR + extracción con reintentos por config. Ejecuta en proceso separado."""
    try:
        img_original = Image.open(ruta)
        datos = _ocr_intentos(img_original)
        datos["_archivo"] = ruta.name
        datos["imagen"] = ruta.name
        return datos
    except Exception as e:
        return {
            "titular": f"ERROR: {e}",
            "banco": "", "CBU": "", "CUIT": "",
            "tipo_cuenta": "", "_archivo": ruta.name, "imagen": ruta.name,
        }


# ──────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────
def listar_imagenes(carpeta: Path) -> list[Path]:
    imagenes = [
        f for f in sorted(carpeta.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    log.info(f"Encontradas {len(imagenes)} imágenes en {carpeta}")
    return imagenes


def normalizar_titular(titular: str | None) -> str:
    if not titular:
        return ""
    titular = titular.replace(",", " ")
    return " ".join(w.capitalize() for w in titular.split())


def clave_dedup(titular: str) -> str:
    return titular.strip().lower()


def cargar_progreso(ruta_csv: Path) -> tuple[set[str], set[str]]:
    """Lee CSV existente. Devuelve (archivos_ya_procesados, titulares_vistos)."""
    if not ruta_csv.exists():
        return set(), set()
    archivos = set()
    titulares = set()
    errores_previos = 0
    with open(ruta_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nombre = row.get("imagen", "") or row.get("name", "")
            titular = row.get("titular", "")
            if titular.startswith("ERROR"):
                errores_previos += 1
                continue  # se reintenta
            archivos.add(nombre)
            if titular:
                titulares.add(clave_dedup(titular))
    log.info(f"Resume: {len(archivos)} exitosas saltadas, {errores_previos} errores previos se reintentan")
    return archivos, titulares


def resolver_salida(carpeta_resultados: Path, nombre_base: str = "resultado") -> Path:
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidato = carpeta_resultados / f"{nombre_base}{n}.csv"
        if not candidato.exists():
            return candidato
        n += 1


# ──────────────────────────────────────────────
# ORQUESTADOR
# ──────────────────────────────────────────────
def procesar_todas(imagenes: list[Path], salida: Path, workers: int, reanudar: bool):
    ya_procesados, vistos = cargar_progreso(salida) if reanudar else (set(), set())

    imagenes_pendientes = [img for img in imagenes if img.name not in ya_procesados]
    log.info(f"Pendientes: {len(imagenes_pendientes)} imágenes | Workers: {workers}")

    campos = ["imagen", "CBU", "CUIT", "titular", "banco"]
    modo_escritura = "a" if reanudar and salida.exists() else "w"

    errores = 0
    descartados = 0
    procesadas = 0
    total_pendientes = len(imagenes_pendientes)
    prev_pct = 0

    with open(salida, modo_escritura, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        if modo_escritura == "w":
            writer.writeheader()

        pool = ProcessPoolExecutor(max_workers=workers)
        try:
            futuros = {pool.submit(procesar_imagen, img): img for img in imagenes_pendientes}
            for fut in tqdm(as_completed(futuros), total=len(futuros), desc="Procesando", unit="img"):
                resultado = fut.result()
                titular = resultado.get("titular", "")

                if titular.startswith("ERROR"):
                    errores += 1
                    log.error(f"FALLO: {resultado['_archivo']} — {titular}")
                    writer.writerow({k: resultado.get(k, "") for k in campos})
                    f.flush()
                else:
                    clave = clave_dedup(titular)
                    if clave and clave in vistos:
                        descartados += 1
                        log.info(f"Duplicado descartado: '{titular}' ({resultado['_archivo']})")
                    else:
                        if clave:
                            vistos.add(clave)
                        procesadas += 1
                        writer.writerow({k: resultado.get(k, "") for k in campos})
                        f.flush()

                if total_pendientes > 0:
                    pct_actual = int((procesadas + errores + descartados) / total_pendientes * 100)
                    for hito in (25, 50, 75):
                        if prev_pct < hito <= pct_actual:
                            _beep_progreso(hito)
                    prev_pct = pct_actual
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    _beep_progreso(100)
    total = procesadas + len(ya_procesados)
    log.info(f"\n{'='*50}")
    log.info(f"✓ Procesadas:   {procesadas} nuevas ({total} total)")
    log.info(f"✗ Errores:      {errores}")
    log.info(f"⊘ Duplicados:   {descartados}")
    log.info(f"→ CSV guardado: {salida.resolve()}")
    log.info(f"{'='*50}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
def main():
    _setup_logging()
    parser = argparse.ArgumentParser(
        description="Extrae datos bancarios de imágenes de transferencia usando Tesseract OCR"
    )
    parser.add_argument("--carpeta", type=Path, required=True,
                        help="Carpeta con las imágenes (.jpg, .png, .webp)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Procesos paralelos CPU (default: {DEFAULT_WORKERS})")
    parser.add_argument("--reanudar", type=Path, default=None,
                        help="Ruta al CSV de una corrida anterior para continuar desde donde quedó")
    args = parser.parse_args()

    if not args.carpeta.exists():
        print(f"ERROR: La carpeta '{args.carpeta}' no existe.")
        return

    imagenes = listar_imagenes(args.carpeta)
    if not imagenes:
        print(f"No se encontraron imágenes en {args.carpeta}")
        return

    if len(imagenes) > 500:
        resp = input(f"¿Proceder con {len(imagenes)} imágenes? [s/N]: ")
        if resp.lower() not in ("s", "si", "sí", "y", "yes"):
            print("Cancelado.")
            return

    carpeta_resultados = Path(__file__).parent / "resultados"

    if args.reanudar:
        salida = args.reanudar
        reanudar = True
    else:
        salida = resolver_salida(carpeta_resultados)
        reanudar = False

    log.info(f"Salida CSV: {salida}")

    from repair import reparar_csv, postprocesar_csv

    print("\n=== FASE 1: Extracción Tesseract ===")
    inicio = datetime.now()
    procesar_todas(imagenes, salida, args.workers, reanudar)
    duracion = (datetime.now() - inicio).total_seconds()
    log.info(f"Tiempo total: {duracion:.1f}s ({duracion/60:.1f} min) | {len(imagenes)/duracion:.1f} img/s")

    print("\n=== FASE 2: Reparación por campo ===")
    reparar_csv(salida, args.carpeta)

    print("\n=== FASE 3: Post-proceso (filtrar CVU / quitar columna imagen) ===")
    postprocesar_csv(salida)


if __name__ == "__main__":
    main()

"""
Fase 2: Reparación por campo.
Lee un CSV existente, detecta campos vacíos (CBU/CUIT/banco),
reprocesa cada fila con el motor específico del campo faltante.

Prioridad: CBU (Tesseract) → CUIT (RapidOCR) → Banco (EasyOCR)

USO:
    python repair.py --csv resultados/resultado1.csv --carpeta ./imagenes
"""

import csv
import argparse
from pathlib import Path
from PIL import Image

from extractor_transferencias import extraer_datos

CAMPOS_REPARAR = [
    ("CBU",   "tesseract"),
    ("CUIT",  "rapidocr"),
    ("banco", "tesseract_banco"),
]


def _campo_vacio(valor: str) -> bool:
    return not valor or valor.strip() == ""


def _ocr_banco_crop(img: Image.Image, debug_path: Path = None) -> str:
    import numpy as np, cv2, pytesseract, shutil, platform
    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    else:
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"
    w, h = img.size
    # zona amplia cubre badge con nombres de 1, 2 o 3 líneas
    crop = img.crop((0, int(h * 0.20), w, int(h * 0.45)))
    arr  = np.array(crop.convert("RGB"))
    gray = np.min(arr, axis=2).astype(np.uint8)
    if gray.mean() < 128:
        gray = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    gray  = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation=cv2.INTER_LANCZOS4)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    prepr = Image.fromarray(gray)
    if debug_path:
        prepr.save(str(debug_path))
    return pytesseract.image_to_string(prepr, lang="spa+eng", config="--psm 6 --oem 1")


def _ocr_con_motor(img, motor: str, debug_path: Path = None) -> str:
    if motor == "tesseract":
        from benchmark_ocr import ocr_tesseract
        return ocr_tesseract(img)
    elif motor == "tesseract_banco":
        return _ocr_banco_crop(img, debug_path=debug_path)
    elif motor == "rapidocr":
        from benchmark_ocr import ocr_rapidocr
        return ocr_rapidocr(img)
    elif motor == "easyocr":
        from benchmark_ocr import ocr_easyocr
        return ocr_easyocr(img)
    return ""


def reparar_csv(csv_path: Path, carpeta_imagenes: Path, on_log=None, stop_event=None, debug_dir: Path = None):
    def log(msg):
        print(msg)
        if on_log:
            on_log(msg)

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        filas = list(reader)

    def _necesita_reparacion(fila):
        for campo, _ in CAMPOS_REPARAR:
            if campo == "CBU" and fila.get("CBU") == "N/A":
                continue
            if _campo_vacio(fila.get(campo, "")):
                return True
        return False

    incompletas = [i for i, fila in enumerate(filas) if _necesita_reparacion(fila)]

    if not incompletas:
        log("[Reparación] Sin campos vacíos. Nada que reparar.")
        return

    log(f"[Reparación] {len(incompletas)} filas con campos vacíos")

    recuperados = 0
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for idx, i in enumerate(incompletas, 1):
        if stop_event and stop_event.is_set():
            log("[Reparación] detenido por el usuario")
            break
        fila = filas[i]
        nombre = fila.get("imagen", "") or fila.get("name", "")
        img_path = carpeta_imagenes / nombre

        if not img_path.exists():
            log(f"[{idx}/{len(incompletas)}] {nombre} — imagen no encontrada, skip")
            continue

        try:
            img = Image.open(img_path)
        except Exception as e:
            log(f"[{idx}/{len(incompletas)}] {nombre} — error al abrir imagen: {e}")
            continue

        for campo, motor in CAMPOS_REPARAR:
            if campo == "CBU" and fila.get("CBU") == "N/A":
                continue
            if not _campo_vacio(fila.get(campo, "")):
                continue

            try:
                debug_path = None
                if debug_dir and motor == "tesseract_banco":
                    stem = Path(nombre).stem
                    debug_path = debug_dir / f"{stem}_banco_crop.png"
                texto = _ocr_con_motor(img, motor, debug_path=debug_path)
                datos = extraer_datos(texto)
                valor = datos.get(campo, "")
                if valor:
                    fila[campo] = valor
                    recuperados += 1
                    log(f"[{idx}/{len(incompletas)}] {nombre} — {campo} → {motor} ✓ {valor}")
                else:
                    log(f"[{idx}/{len(incompletas)}] {nombre} — {campo} → {motor} ✗")
            except Exception as e:
                log(f"[{idx}/{len(incompletas)}] {nombre} — {campo} → {motor} error: {e}")

        filas[i] = fila

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(filas)

    log(f"[Reparación] completada: {recuperados} campos recuperados en {len(incompletas)} filas")


def postprocesar_csv(csv_path: Path, on_log=None):
    """
    1. Guarda backup con columna imagen como {stem}_full.csv
    2. Elimina filas con CBU = "N/A" (cuentas CVU)
    3. Elimina columna "imagen" del CSV final
    """
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        filas = list(reader)

    # backup con imagen
    backup_path = csv_path.with_name(csv_path.stem + "_full.csv")
    with open(backup_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filas)
    print(f"[Post-proceso] Backup guardado: {backup_path.name}")

    # filtrar filas inválidas: sin titular, sin CBU y sin banco → imagen mal colocada
    antes = len(filas)
    def _es_valida(r):
        titular = r.get("titular", "").strip()
        cbu     = r.get("CBU", "").strip()
        banco   = r.get("banco", "").strip()
        return bool(titular) or (bool(cbu) and cbu != "N/A") or bool(banco)
    filas = [r for r in filas if _es_valida(r)]
    invalidas = antes - len(filas)
    if invalidas:
        msg = f"[Post-proceso] {invalidas} filas inválidas eliminadas (sin titular/CBU/banco)"
        print(msg)
        if on_log: on_log(msg)

    # filtrar CVU (CBU = "N/A")
    antes = len(filas)
    filas = [r for r in filas if r.get("CBU", "").strip() != "N/A"]
    eliminadas = antes - len(filas)
    msg = f"[Post-proceso] {eliminadas} filas CVU (N/A) eliminadas"
    print(msg)
    if on_log: on_log(msg)

    # quitar columna imagen
    campos_finales = [c for c in fieldnames if c != "imagen"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos_finales, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filas)
    msg = f"[Post-proceso] CSV final: {len(filas)} filas → {csv_path.name}"
    print(msg)
    if on_log: on_log(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reparación de campos vacíos en CSV OCR")
    parser.add_argument("--csv", type=Path, required=True, help="CSV a reparar")
    parser.add_argument("--carpeta", type=Path, required=True, help="Carpeta con imágenes originales")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV '{args.csv}' no existe.")
        raise SystemExit(1)
    if not args.carpeta.exists():
        print(f"ERROR: carpeta '{args.carpeta}' no existe.")
        raise SystemExit(1)

    reparar_csv(args.csv, args.carpeta)

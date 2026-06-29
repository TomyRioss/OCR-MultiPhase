import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
from pathlib import Path
from benchmark_ocr import opencv_preprocess, extraer_datos

imagenes = sorted(Path("imagenes").glob("*"))[:30]
sin_banco = []
con_banco = {}

for p in imagenes:
    try:
        img = Image.open(p)
        proc = opencv_preprocess(img)
        texto = pytesseract.image_to_string(proc, lang="spa+eng", config="--psm 6 --oem 1")
        datos = extraer_datos(texto)
        b = datos.get("banco", "")
        if b:
            con_banco[b] = con_banco.get(b, 0) + 1
        else:
            sin_banco.append(p.name)
    except Exception as e:
        sin_banco.append(f"{p.name} ERROR:{e}")

print(f"Con banco ({len(con_banco)} distintos):")
for b, n in sorted(con_banco.items()):
    print(f"  {b}: {n}")
print(f"\nSin banco: {len(sin_banco)}")
for n in sin_banco:
    print(f"  {n}")

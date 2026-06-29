"""
Resumen de calidad del CSV + lista de filas con campos vacíos.

USO:
    python check.py resultados/resultado1.csv
"""

import csv
import sys
from pathlib import Path

CAMPOS = ["titular", "banco", "CBU", "CUIT"]


def analizar_csv(ruta):
    ruta = Path(ruta)
    with open(ruta, encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))

    total = len(filas)
    vacios   = {c: 0 for c in CAMPOS}
    na_count = {c: 0 for c in CAMPOS}
    ok_count = {c: 0 for c in CAMPOS}

    for fila in filas:
        for c in CAMPOS:
            v = fila.get(c, "").strip()
            if not v or v.startswith("ERROR"):
                vacios[c] += 1
            elif v.upper() == "N/A":
                na_count[c] += 1
            else:
                ok_count[c] += 1

    incompletas = []
    for fila in filas:
        faltantes = [
            c for c in CAMPOS
            if not fila.get(c, "").strip() or fila.get(c, "").startswith("ERROR")
        ]
        faltantes = [f for f in faltantes if not (f == "CBU" and fila.get("CBU", "") == "N/A")]
        if faltantes:
            incompletas.append({"imagen": fila.get("imagen", fila.get("name", "?")), "faltantes": faltantes})

    return {
        "nombre": ruta.name,
        "total": total,
        "campos": {c: {"vacios": vacios[c], "na": na_count[c], "ok": ok_count[c]} for c in CAMPOS},
        "incompletas": incompletas,
    }


def main():
    if len(sys.argv) < 2:
        print("USO: python check.py <ruta_csv>")
        sys.exit(1)

    ruta = Path(sys.argv[1])
    if not ruta.exists():
        print(f"ERROR: '{ruta}' no existe.")
        sys.exit(1)

    r = analizar_csv(ruta)
    total = r["total"]
    print(f"\nCSV: {ruta}  ({total} filas)\n")

    ancho = 10
    print(f"{'Campo':<12} {'Vacíos':>{ancho}} {'N/A':>{ancho}} {'OK':>{ancho}}")
    print("-" * (12 + ancho * 3 + 2))
    for c in CAMPOS:
        s = r["campos"][c]
        print(f"{c:<12} {s['vacios']:>{ancho}} {s['na']:>{ancho}} {s['ok']:>{ancho}}")

    print(f"\nFilas con campos faltantes: {len(r['incompletas'])} / {total}\n")


if __name__ == "__main__":
    main()

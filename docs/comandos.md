# Comandos

## Pipeline principal

```bash
# 1. Extracción rápida
python extract.py --carpeta ./imagenes

# 2. Revisar calidad
python check.py resultados/resultado1.csv

# 3. Rellenar incompletos (calidad máxima)
python fulfill.py --csv resultados/resultado1.csv --carpeta ./imagenes
```

Opciones:
- `--workers N` — procesos paralelos CPU (default: 6 en extract, 4 en fulfill)

## Salidas

| Script | Carpeta de salida |
|--------|-------------------|
| `extract.py` | `resultados/resultado{n}.csv` |
| `fulfill.py` | `resultados_finales/resultado_final{n}.csv` |

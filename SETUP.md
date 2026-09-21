# Guia de instalacion y migracion entre equipos

Proyecto de tesis: **Mecanismo de validacion automatizada para el contraste de
datos catastrales entre resoluciones PDF y base de datos en el Valle del Cauca**
(Maestria en IA y Ciencia de Datos, UAO).

Esta guia sirve para poner en marcha el proyecto en un equipo nuevo o para
migrar entre computadores.

---

## 1. Arquitectura de almacenamiento (importante)

El proyecto se divide en dos partes que se guardan en lugares distintos:

| Que | Donde | Como llega al PC nuevo |
|-----|-------|------------------------|
| **Codigo** (scripts, configuracion) | GitHub | `git clone` |
| **Datos** (PDFs, cortes catastrales, reportes) | Google Drive personal | Sincronizacion / descarga |

Motivo: los datos pesan ~2.7 GB y contienen informacion personal (nombres,
documentos, direcciones), por lo que **no** se versionan en git. Van en Google
Drive privado (15 GB gratis, sin limite de tamano por archivo).

---

## 2. Que trae git (automatico con `git clone`)

Repositorio: `https://github.com/j0rg3c45/tesis-ia-catastro`

- `main.py` — Etapa 1: extraccion OCR + regex por predio (por lotes) + telemetria
- `src/metrics.py` — modulo de auditoria/metricas (telemetria del pipeline)
- `scripts/consolidar_base_catastral.py` — consolida el corte catastral por NPN
- `scripts/comparar_pdf_vs_catastro.py` — compara tabulado PDF vs base catastral (+ metricas de cruce)
- `scripts/comparar_bases.py` — comparacion previa por NPN
- `README.md`, `requirements.txt`, `.gitignore`, este `SETUP.md`
- `data/contexto_catastral/general_destinacion_economica.txt` — diccionario de destinos (dependencia del pipeline)
- `data/structured/*.xlsx` — base oficial y tabulado de prueba

---

## 3. Que NO trae git (llevar manualmente por Google Drive)

Estas carpetas estan ignoradas por peso y datos personales. Hay que copiarlas
al PC nuevo (o sincronizarlas desde Drive):

| Carpeta | Contenido | Peso aprox. |
|---------|-----------|-------------|
| `data/raw_pdfs/` | ~2.568 PDFs de resoluciones | ~627 MB |
| `data/validation/` | Cortes catastrales (ZIP + 20260731, 20260831) | ~2 GB |
| `data/comparacion/` | Base consolidada (~94 MB) + reportes | ~100 MB |
| `data/contexto_catastral/` (PDFs/modelos IGAC) | Diccionario LADM, instructivos | ~18 MB |
| `data/contexto_tesis/main.tex` | Documento LaTeX de la tesis | minimo |
| `data/raw_text/` | Textos OCR (opcional, se regeneran) | variable |
| `data/reports/` | Metricas y reportes generados (se regeneran al correr) | variable |

**Recomendado:** mantener la carpeta `data/` completa en Google Drive y
sincronizarla con la app "Google Drive para escritorio" en ambos equipos.

---

## 4. Dependencias del sistema (instalar en el PC nuevo)

No vienen en git ni en pip; son programas del sistema:

### 4.1 Tesseract OCR (con idioma espanol)
- Instalar Tesseract OCR e incluir el paquete de idioma **spa**.
- Ruta actual en el equipo principal:
  `C:\Users\Jorge\AppData\Local\Programs\Tesseract-OCR\tesseract.exe` (v5.4.0)
- Si en el PC nuevo queda en otra ruta, indicarla con la variable de entorno
  `TESSERACT_CMD` (ver seccion 6). Si esta en el PATH, se detecta solo.

### 4.2 Poppler (necesario para pdf2image)
- Instalar Poppler y dejar su carpeta `bin` accesible.
- Ruta actual: `C:\Users\Jorge\AppData\Local\Programs\poppler-24.02.0\Library\bin`

---

## 5. Entorno de Python

Crear un entorno virtual limpio (con pip) e instalar dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notas:
- El proyecto se ha ejecutado con `pandas`, `openpyxl`, `pytesseract`,
  `pdf2image`, `opencv-python-headless`, `Pillow`, `tqdm` (y `spacy` opcional).
- Para leer los PDF de documentacion se uso `pypdf` (opcional, no requerido por
  el pipeline principal): `pip install pypdf`.

---

## 6. Variables de entorno (portabilidad)

`main.py` deriva sus rutas del proyecto y admite estas variables opcionales
(PowerShell). Ajustarlas si cambian las rutas en el PC nuevo:

```powershell
# Ruta a la carpeta de datos (util si los datos estan en Google Drive)
$env:DATA_DIR = "G:\Mi unidad\tesis-data"

# Ruta al ejecutable de Tesseract (si no esta en la ruta por defecto ni en PATH)
$env:TESSERACT_CMD = "C:\ruta\a\tesseract.exe"

# Nivel de log (DEBUG, INFO, ...). Por defecto INFO
$env:LOG_LEVEL = "INFO"

# Reprocesar todo ignorando el cache por hash
$env:FORCE_REPROCESS = "true"

# Tamano de lote de PDFs (guardado incremental). Por defecto 100
$env:BATCH_SIZE = "100"

# --- Parametros de OCR (afinados por experimento) ---
# DPI del OCR. Por defecto 300 (equilibrio calidad/velocidad). Subir a 400
# mejora un poco mas la lectura pero es mas lento.
$env:OCR_DPI = "300"

# Preprocesamiento agresivo (denoise+sharpen+Otsu). Por defecto false.
# El experimento mostro que degrada el OCR en estos documentos; dejar en false.
$env:OCR_AGGRESSIVE = "false"
```

**Nota sobre calidad de OCR:** un experimento controlado sobre documentos
reales determino que la mejor configuracion es **DPI 300 en escala de grises
simple (sin preprocesamiento agresivo)**, que elevo la lectura del Codigo
Homologado de ~31 % a ~86 %. Estos son los valores por defecto del sistema.

---

## 7. Flujo de trabajo (orden de ejecucion)

```powershell
# Etapa 1: extraer y tabular los PDFs (por lotes; reanudable via cache)
python main.py

# Consolidar el corte catastral (R1+R2) en 1 fila por NPN
python scripts/consolidar_base_catastral.py --corte 20260831

# Comparar el tabulado de PDFs contra la base catastral consolidada (cruce por NPN)
python scripts/comparar_pdf_vs_catastro.py
```

Salidas:
- `data/structured/TABULADO_RESULTADOS.xlsx` — tabulado de los PDFs (1 fila por predio)
- `data/comparacion/BASE_CATASTRAL_CONSOLIDADA_<corte>.csv` — base consolidada
- `data/comparacion/COMPARACION_PDF_VS_CATASTRO_<timestamp>.xlsx` — reporte en Excel (4 hojas)
- `data/comparacion/REPORTE_VALIDACION_<timestamp>.txt` — reporte de validacion en texto plano (Objetivo 4)

Metricas de auditoria (telemetria), generadas automaticamente en `data/reports/metricas/`:
- `metricas_proceso_<timestamp>.csv` — una fila por PDF (estado, n_paginas, n_predios,
  tiempo OCR, tiempo parsing, completitud, campos faltantes). La produce `main.py`.
- `resumen_metricas_tesis.json` y `resumen_metricas_tesis.txt` — consolidado de la
  extraccion (totales, tiempos, tasa de deteccion por variable). Los produce `main.py`.
- `metricas_cruce_<timestamp>.json` y `.txt` — tasa de cruce por NPN y matriz de
  concordancia por campo. Los produce `scripts/comparar_pdf_vs_catastro.py`.

Nota: la carpeta `data/reports/` esta ignorada por git (contiene datos), no se versiona.

### Nota para la corrida del lote completo (2.568 PDFs)

- Colocar todos los PDFs en `data/raw_pdfs/` (se admiten subcarpetas; `os.walk` los recorre).
- La Etapa 1 procesa por lotes de `BATCH_SIZE` y es **reanudable**: si se interrumpe,
  al volver a ejecutar `python main.py` el cache por hash salta los ya procesados.
- Con DPI 300 el proceso es mas lento que con 200, pero da mejor calidad. Se puede
  dejar corriendo por horas; el guardado por lote evita perder trabajo.
- Al terminar la Etapa 1, ejecutar la consolidacion (si cambio el corte) y la comparacion.

---

## 8. Checklist rapido al cambiar de PC

1. [ ] `git clone` del repositorio
2. [ ] Instalar Tesseract OCR (idioma spa) y Poppler
3. [ ] Crear `.venv` e instalar `requirements.txt`
4. [ ] Sincronizar/copiar la carpeta `data/` desde Google Drive
5. [ ] Ajustar `DATA_DIR` y `TESSERACT_CMD` si las rutas cambian
6. [ ] Ejecutar el flujo de la seccion 7

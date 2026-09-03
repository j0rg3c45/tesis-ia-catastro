# Tesis IA Catastro

## Sistema automatizado de validacion documental catastral

Proyecto de tesis para la Maestria en Inteligencia Artificial y Ciencia de Datos.

---

## Descripcion del proyecto

Este proyecto implementa un sistema automatizado que procesa resoluciones
catastrales en formato PDF (Gobernacion del Valle del Cauca), extrae la
informacion de cada predio mediante OCR y expresiones regulares, y la compara
contra la base catastral oficial para detectar inconsistencias.

El sistema permite:

1. Procesar de forma masiva archivos PDF catastrales.
2. Extraer texto mediante OCR (Tesseract) con preprocesamiento de imagen.
3. Estructurar los datos con una fila por predio y tabularlos a Excel.
4. Cruzar la informacion extraida contra la base oficial por Numero Predial
   Nacional (NPN).
5. Generar reportes de inconsistencias campo por campo.

---

## Objetivo general

Disenar e implementar una arquitectura modular y reproducible que permita la
validacion automatizada de la informacion contenida en documentos catastrales
utilizando tecnicas de Inteligencia Artificial y Ciencia de Datos.

---

## Arquitectura del sistema

El pipeline se organiza en tres etapas.

### Etapa 1 - Extraccion (main.py)

- Convierte cada PDF a imagenes con pdf2image (requiere Poppler).
- Preprocesa las imagenes con OpenCV (escala de grises y, en modo agresivo,
  reduccion de ruido, realce y binarizacion Otsu).
- Aplica OCR con Tesseract en espanol.
- Segmenta el texto en bloques, uno por predio (marcador "Numero de matricula
  inmobiliaria"), y extrae los campos dentro de cada bloque, de modo que todos
  los campos de un predio quedan alineados en la misma fila.
- Descarta bloques sin NPN valido de 30 digitos (fragmentos partidos por saltos
  de pagina o encabezados).
- Guarda el texto crudo en `data/raw_text/` y el tabulado en
  `data/structured/TABULADO_RESULTADOS.xlsx`.
- Usa un cache por hash MD5 para no reprocesar archivos ya completados.

### Etapa 2 - Comparacion (scripts/comparar_bases.py)

- Cruza `TABULADO_RESULTADOS.xlsx` contra la base oficial
  `1_REGISTRO_R1_CONTRALORIA_20260501.xlsx` por NPN.
- Aplica un cruce con tolerancia (match exacto, relleno con ceros y por
  prefijo mas secuencia) para compensar errores de OCR en los digitos del NPN.
- Compara campo por campo (texto, numeros y fechas) y detecta inconsistencias.
- Genera un reporte Excel en `reports/` con resumen, comparacion por predio,
  inconsistencias detalladas y NPN sin coincidencia.

### Etapa 3 - Dashboard (dashboard/app.py)

Pendiente de implementacion.

---

## Estructura del repositorio

```
tesis-ia-catastro/
├── main.py                     # Etapa 1: extraccion OCR + regex por predio
├── requirements.txt
├── scripts/
│   └── comparar_bases.py       # Etapa 2: cruce e inconsistencias
├── dashboard/
│   └── app.py                  # Etapa 3: dashboard (pendiente)
├── src/                        # Modulos previstos para refactor futuro
│   ├── ingestion/
│   ├── nlp/
│   ├── reporting/
│   └── utils/
├── data/
│   ├── raw_pdfs/               # PDFs de entrada (ignorado por git)
│   ├── raw_text/               # Texto OCR crudo (ignorado por git)
│   └── structured/             # Excel tabulado y base oficial (ignorado por git)
└── reports/                    # Reportes de comparacion (ignorado por git)
```

Nota: las carpetas de datos y los reportes estan en `.gitignore`, ya que son
insumos y artefactos generados, no codigo.

---

## Requisitos

Dependencias del sistema (no se instalan con pip):

- Tesseract OCR con el paquete de idioma espanol (`spa`).
- Poppler (necesario para pdf2image).

Dependencias de Python: ver `requirements.txt`. Instalacion:

```
pip install -r requirements.txt
```

Adicionalmente, el modelo de spaCy en espanol si se requiere en pasos futuros:

```
python -m spacy download es_core_news_sm
```

---

## Configuracion por variables de entorno

`main.py` deriva sus rutas del directorio del proyecto y admite estas variables
opcionales para ser reproducible en cualquier equipo:

| Variable         | Descripcion                                              | Valor por defecto             |
|------------------|----------------------------------------------------------|-------------------------------|
| `DATA_DIR`       | Raiz de la carpeta de datos.                             | `<proyecto>/data`             |
| `TESSERACT_CMD`  | Ruta al ejecutable de Tesseract.                         | Ruta local o `tesseract` PATH |
| `LOG_LEVEL`      | Nivel de logging (`DEBUG`, `INFO`, ...).                 | `INFO`                        |
| `FORCE_REPROCESS`| Si es `true`, ignora el cache y reprocesa todo.          | `false`                       |

Si `TESSERACT_CMD` no existe, se asume que `tesseract` esta disponible en el
PATH del sistema.

---

## Uso

### 1. Extraccion (Etapa 1)

Colocar los PDFs en `data/raw_pdfs/` (se admiten subcarpetas) y ejecutar:

```
python main.py
```

Genera `data/raw_text/*.txt` y `data/structured/TABULADO_RESULTADOS.xlsx` con
una fila por predio.

Para reprocesar todo ignorando el cache (PowerShell):

```
$env:FORCE_REPROCESS = "true"; python main.py
```

### 2. Comparacion (Etapa 2)

Ubicar la base oficial en `data/structured/` y ejecutar:

```
python scripts/comparar_bases.py
```

Genera un reporte con marca de tiempo en `reports/`.

---

## Campos extraidos por predio

Cada fila del tabulado contiene: Nombre del archivo, Resolucion, Fecha de
resolucion, Numero de matricula inmobiliaria, Numero predial, Numero Predial
Nacional, Codigo Homologado, Municipio, Propietario, Documento de
identificacion, Direccion, Area del predio, Area construida, Destinacion
economica, Avaluo, Fecha de inscripcion catastral y Vigencia fiscal.

---

## Notas y limitaciones

- La calidad de la extraccion depende de la calidad del OCR. Errores de un solo
  digito en el NPN pueden impedir el cruce exacto con la base oficial; el
  matching con tolerancia mitiga parte de estos casos.
- El procesamiento de lotes grandes (miles de PDFs) puede requerir ajustes de
  rendimiento en la orquestacion (paralelismo por procesos y escritura del
  Excel en modo streaming).

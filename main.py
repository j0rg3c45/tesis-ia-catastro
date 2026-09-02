"""
main.py
ENERO 2026 
Orquestador principal del sistema de validación documental catastral.
Este módulo coordina el flujo completo del pipeline:

1. Ingesta de PDFs
2. Extracción de texto
3. Procesamiento NLP
4. Validación contra base de datos
5. Generación de reporte

Autor: Proyecto de Tesis - Maestría en IA y Ciencia de Datos
"""


# ==============================================================================
# SCRIPT FINAL Y ROBUSTO PARA EXTRACCIÓN DE DATOS DE PDF USANDO Tesseract OCR
# ==============================================================================
#
# Cómo usar este código:
# 1. Instala las dependencias necesarias:
#    pip install pytesseract pdf2image openpyxl tqdm spacy opencv-python-headless Pillow
#    python -m spacy download es_core_news_sm
#
# 2. Asegúrate de tener Tesseract OCR instalado y Poppler para pdf2image.
#
# 3. Ejecuta este script. Los resultados se guardarán en el archivo Excel especificado.
#
# ==============================================================================

import os
import re
from pdf2image import convert_from_path
from openpyxl import Workbook
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import time
from functools import partial
import logging
import gc
import shutil
# --- IMPORTACIÓN DE Tesseract OCR ---
import pytesseract

# ==============================================================================
# CONFIGURACIÓN PRINCIPAL
# ==============================================================================
CONFIG = {
    "dpi": 200,
    "max_workers": multiprocessing.cpu_count() // 2,
    "batch_size": 1,
    "cache_enabled": True,
    
    # --- CONFIGURACIÓN DE Tesseract ---
    "tesseract_lang": 'spa',  # Español
    "tesseract_config": '--psm 6',  # Modo de segmentación de página
    
    # --- OPCIÓN DE DEPURACIÓN ---
    # True = usa el preprocesamiento agresivo (denoise, sharpen, otsu).
    # False = usa un preprocesamiento más simple (solo escala de grises).
    "use_aggressive_preprocessing": True,
}

# ==============================================================================
# RUTAS CONFIGURABLES
# ==============================================================================
# Directorio raíz del proyecto (carpeta que contiene este archivo main.py).
# Todas las rutas se derivan de aquí para que el proyecto sea reproducible
# sin importar dónde se clone el repositorio.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Permite sobreescribir la raíz de datos con la variable de entorno DATA_DIR.
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))

# Ruta de entrada: carpeta con los PDFs
BASE_FOLDER_PATH = os.path.join(DATA_DIR, "raw_pdfs")

# Ruta de salida: carpeta donde se guardarán los textos extraídos (raw)
OUTPUT_FOLDER_PATH = os.path.join(DATA_DIR, "raw_text")

# Ruta de salida: carpeta donde se guardará el Excel estructurado
STRUCTURED_FOLDER_PATH = os.path.join(DATA_DIR, "structured")

# Nombre del archivo Excel de salida
OUTPUT_EXCEL_NAME = "TABULADO_RESULTADOS.xlsx"

# --- CONFIGURACIÓN DE TESSERACT ---
# Ruta al ejecutable de Tesseract. Se puede sobreescribir con la variable de
# entorno TESSERACT_CMD. Si no se define y la ruta por defecto no existe, se
# asume que 'tesseract' está disponible en el PATH del sistema.
_TESSERACT_DEFAULT = r'C:\Users\Jorge\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
_TESSERACT_CMD = os.environ.get("TESSERACT_CMD", _TESSERACT_DEFAULT)
if os.path.exists(_TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

# Configurar logging para ver el progreso y errores.
# El nivel se puede ajustar con la variable de entorno LOG_LEVEL (DEBUG, INFO, ...).
# El archivo de log se escribe en la raíz del proyecto (ruta absoluta) para no
# depender del directorio de trabajo desde el que se ejecute el script.
_LOG_LEVEL = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "catastro_debug.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# FUNCIONES AUXILIARES (CACHÉ Y HASH)
# ==============================================================================

def setup_cache_folder(base_folder):
    cache_folder = os.path.join(base_folder, ".cache")
    os.makedirs(cache_folder, exist_ok=True)
    return cache_folder
def get_file_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
def is_file_processed(file_path, cache_folder):
    if not CONFIG["cache_enabled"]:
        return False
    file_hash = get_file_hash(file_path)
    cache_file = os.path.join(cache_folder, f"{file_hash}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
            return cache_data.get("completed", False)
    return False
def mark_file_completed(file_path, cache_folder, data):
    if not CONFIG["cache_enabled"]:
        return
    file_hash = get_file_hash(file_path)
    cache_file = os.path.join(cache_folder, f"{file_hash}.json")
    with open(cache_file, "w") as f:
        json.dump({
            "completed": True,
            "timestamp": time.time(),
            "data": data
        }, f)

# ==============================================================================
# FUNCIONES DE PROCESAMIENTO DE IMAGEN Y TEXTO
# ==============================================================================

def pdf_to_images(pdf_path, dpi=None):
    if dpi is None:
        dpi = CONFIG["dpi"]
    try:
        images = convert_from_path(pdf_path, dpi, thread_count=CONFIG["max_workers"])
        return images
    except Exception as e:
        logger.error(f"Error al convertir PDF {pdf_path}: {e}")
        return []
def preprocess_image(image):
    """
    Preprocesamiento con dos modos: agresivo y simple.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    if CONFIG["use_aggressive_preprocessing"]:
        # Modo agresivo (denoise, sharpen, otsu)
        denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    else:
        # Modo simple (solo escala de grises)
        return gray
def extract_text_from_image(image, page_num):
    """
    Extrae texto de la imagen usando Tesseract OCR.
    """
    processed_image = preprocess_image(image)
    
    if cv2.countNonZero(processed_image) == 0:
        logger.warning(f"Página {page_num} en blanco detectada, omitiendo OCR.")
        return ""

    try:
        # Convertir de numpy array a PIL Image para Tesseract
        pil_image = Image.fromarray(processed_image)
        
        # Extraer texto con Tesseract
        text = pytesseract.image_to_string(
            pil_image, 
            lang=CONFIG["tesseract_lang"],
            config=CONFIG["tesseract_config"]
        )
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error en Tesseract OCR en la página {page_num}: {e}")
        return ""

# ==============================================================================
# FUNCIONES DE EXTRACCIÓN Y LIMPIEZA DE DATOS
# ==============================================================================

# ==============================================================================
# EXTRACCIÓN POR BLOQUES/PREDIO
# ==============================================================================
# Las resoluciones catastrales contienen la ficha de uno o varios predios.
# Cada ficha se abre con "Número de matrícula inmobiliaria". Segmentamos el
# texto por ese marcador y extraemos los campos DENTRO de cada bloque, de modo
# que todos los campos de un predio quedan alineados en la misma fila.
# Resolución y fecha son únicas por documento y se repiten en cada fila.

NR = "NR"

# Campos que se extraen por predio, en el orden de salida del Excel.
CAMPOS_PREDIO = [
    "Número de matrícula inmobiliaria:",
    "Número predial:",
    "Número Predial Nacional:",
    "Código Homologado:",
    "Municipio:",
    "Propietario:",
    "Documento de identificación:",
    "Dirección:",
    "Área predio:",
    "Área construida:",
    "Destinación economica:",
    "Avalúo:",
    "Fecha de la inscripción Catastral:",
    "Vigencia Fiscal:",
]

# Marcador que abre la ficha de cada predio.
_MARCADOR_PREDIO = re.compile(r'NUMERO\s+DE\s+MATRICULA\s+INMOBILIARIA', re.IGNORECASE)


def limpiar_valor(valor):
    """Quita ruido de OCR al inicio/fin: guiones, puntos y comas, pipes, etc."""
    if not valor:
        return ""
    v = re.sub(r'\s+', ' ', valor).strip()
    v = re.sub(r'[\s\-;:|.,!¡]+$', '', v)
    v = re.sub(r'^[\s\-;:|.,!¡]+', '', v)
    return v.strip()


def limpiar_texto_nombre(valor):
    """
    Limpieza extra para texto libre (propietario, dirección, municipio):
    elimina tokens residuales de 1 carácter que suele dejar el OCR al final
    (ej. 'MUNICIPIO DE CANDELARIA M' -> 'MUNICIPIO DE CANDELARIA').
    """
    if not valor or valor == NR:
        return valor
    v = limpiar_valor(valor)
    v = re.sub(r'(\s+[\-|;:.,_\'"]+)+$', '', v).strip()
    for _ in range(3):
        nuevo = re.sub(r'\s+[A-Z0-9_]$', '', v).strip()
        nuevo = re.sub(r'[\s_\-]+$', '', nuevo).strip()
        if nuevo == v:
            break
        v = nuevo
    return v.strip()


def _match(patron, texto, grupo=1, flags=re.IGNORECASE):
    """Busca un patrón y devuelve el grupo limpio, o 'NR' si no hay match."""
    m = re.search(patron, texto, flags)
    if m:
        return limpiar_valor(m.group(grupo))
    return NR


def segmentar_predios(texto_norm):
    """
    Divide el texto normalizado en bloques, uno por predio.
    Cada bloque va desde 'NUMERO DE MATRICULA INMOBILIARIA' hasta el siguiente
    marcador (o el final). El encabezado/considerandos previos se descartan.
    """
    posiciones = [m.start() for m in _MARCADOR_PREDIO.finditer(texto_norm)]
    if not posiciones:
        return []
    bloques = []
    for i, ini in enumerate(posiciones):
        fin = posiciones[i + 1] if i + 1 < len(posiciones) else len(texto_norm)
        bloques.append(texto_norm[ini:fin].strip())
    return bloques


def extraer_datos_documento(texto_norm):
    """Extrae los campos únicos por documento: número de resolución y fecha."""
    m = re.search(
        r'RESOLUCION\s+NO\.?\s*([0-9][0-9\.,\-]*M[0-9]{2}\-[0-9]+)\s+DE\s+(\d{4})',
        texto_norm, re.IGNORECASE)
    if m:
        num = m.group(1).replace(',', '.')  # OCR a veces pone coma por punto
        resolucion = f"{num} DE {m.group(2)}"
    else:
        resolucion = NR

    fecha_res = _match(r'\(\s*(\d{1,2}\s+DE\s+[A-Z]+\s+DE\s+\d{4})\s*\)', texto_norm)
    if fecha_res == NR:
        fecha_res = _match(r'(\d{1,2}\s+DE\s+[A-Z]+\s+DE\s+\d{4})', texto_norm)
    return {"RESOLUCIÓN No.": resolucion, "FechaResolucion": fecha_res}


def _limpiar_npn(valor):
    """Deja sólo dígitos."""
    return re.sub(r'[^0-9]', '', valor)


def extraer_campos_predio(bloque):
    """Extrae los campos de un predio a partir de su bloque de texto."""
    campos = {}

    campos["Número de matrícula inmobiliaria:"] = _match(
        r'NUMERO\s+DE\s+MATRICULA\s+INMOBILIARIA\s*[:\-;]?\s*(\d{3}\s*[\-]\s*\d{4,7})', bloque)

    campos["Número predial:"] = _match(
        r'NUMERO\s+PREDIAL\s*[:\-;]?\s*(NO\s+REGISTRA|\d[\d\s\-]{6,})', bloque)

    # NPN: 30 dígitos tolerando espacios internos del OCR.
    npn = NR
    m = re.search(
        r'NUMERO\s+PREDIA[L1]\s+NACIONAL\s*[:\-;]?\s*([\d][\d\s]{27,45}\d)',
        bloque, re.IGNORECASE)
    if m:
        cand = _limpiar_npn(m.group(1))
        if len(cand) >= 30:
            npn = cand[:30]
        elif len(cand) >= 20:
            npn = cand.ljust(30, '0')  # el OCR perdió ceros finales
        else:
            npn = cand
    campos["Número Predial Nacional:"] = npn

    # Código Homologado: tolerante a confusión O/0 (alfanumérico de 9-13).
    campos["Código Homologado:"] = _match(
        r'CODIGO\s+HOMOLOGADO\s*[:\-;]?\s*([A-Z0-9]{9,13})', bloque)

    # Municipio: corta en PROPIETARIO/DEPARTAMENTO tolerando ruido intermedio.
    muni = _match(
        r'MUNICIPIO\s*[:\-;]?\s*([A-Z][A-Z\s]{2,25}?)\s*[|.,;\-]*\s*(?:PROPIETARIO|DEPARTAMENTO)',
        bloque)
    if muni == NR:
        muni = _match(r'MUNICIPIO\s*[:\-;]?\s*([A-Z][A-Z ]{2,25})', bloque)
    campos["Municipio:"] = limpiar_texto_nombre(muni)

    # Propietario: anclado a la ficha del predio, corta antes de DOCUMENTO.
    prop = _match(
        r'PROPIETARIO\s*[:\-;]?\s*(.+?)(?=\s+DOCUMENTO\s+DE\s+IDENTIFICACION)',
        bloque, flags=re.IGNORECASE | re.DOTALL)
    if prop != NR and len(prop) > 160:
        prop = prop[:160]
    campos["Propietario:"] = limpiar_texto_nombre(prop)

    # Documento de identificación: 'NO REGISTRA' o tipo+número, corta en DIRECCION.
    doc = _match(
        r'DOCUMENTO\s+DE\s+IDENTIFICACION\s*[:\-;]?\s*(NO\s+REGISTRA|(?:N|C|NIT|CC)?\s*\d[\d\.\-\s;C]*?)(?=\s*DIRECCION|\s*$)',
        bloque, flags=re.IGNORECASE | re.DOTALL)
    campos["Documento de identificación:"] = limpiar_valor(doc)

    # Dirección: corta antes de ÁREA.
    dire = _match(
        r'DIRECCION\s*[:\-;]?\s*(.+?)(?=\s+AREA\s+PREDIO|\s+AREA\s+DEL|\s+AREA\s+CONSTRUIDA|\s*$)',
        bloque, flags=re.IGNORECASE | re.DOTALL)
    campos["Dirección:"] = limpiar_texto_nombre(dire)

    # Área predio: sin límite rígido de dígitos; admite . o , decimal.
    campos["Área predio:"] = _match(
        r'AREA\s+(?:DEL\s+)?PREDIO\s*[:\-;]?\s*(\d[\d.,]*)\s*(?:M2|M\?|M\*|METROS|MTS)?', bloque)

    # Área construida.
    area_c = _match(
        r'AREA\s+CONSTRUIDA\s*[:\-;]?\s*(NO\s+REGISTRA|NO|\d[\d.,]*)\s*(?:M2|M\?|M\*)?', bloque)
    if area_c == "NO":
        area_c = "NO REGISTRA"
    campos["Área construida:"] = area_c

    # Destinación económica.
    dest = _match(
        r'DESTINACION\s*(?:ECONOMICA)?\s*[:\-;]?\s*([A-Z][A-Z\s]{4,45}?)\s*[.,;\-]*\s*(?=AVALUO|FECHA|NUMERO|VIGENCIA|$)',
        bloque)
    if dest == NR:
        dest = _match(r'DESTINACION\s*(?:ECONOMICA)?\s*[:\-;]?\s*([A-Z][A-Z ]{4,45})', bloque)
    campos["Destinación economica:"] = limpiar_texto_nombre(dest)

    # Avalúo: primer monto (el más reciente suele ir primero).
    campos["Avalúo:"] = _match(r'AVALUO\s*[:\-;]?\s*\$?\s*([\d.,]+)', bloque)

    campos["Fecha de la inscripción Catastral:"] = _match(
        r'FECHA\s+DE\s+LA\s+INSCRIPCION\s+CATASTRAL\s*[:\-;]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})', bloque)

    campos["Vigencia Fiscal:"] = _match(
        r'VIGENCIA\s+FISCAL\s*[:\-;]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})', bloque)

    return campos


def extraer_predios_de_texto(text):
    """
    Función principal de extracción: recibe el texto OCR de un documento y
    devuelve una lista de diccionarios, uno por predio, con todos los campos
    (documento + predio) alineados.

    Descarta bloques sin NPN válido de 30 dígitos: son fragmentos partidos por
    saltos de página/encabezados, no fichas completas aptas para el cruce.
    """
    logger.info(f" Iniciando extracción por predio. Texto: {len(text)} caracteres")
    texto_norm = normalize_text(text)
    doc = extraer_datos_documento(texto_norm)
    bloques = segmentar_predios(texto_norm)
    logger.info(f"   Bloques (fichas de predio) detectados: {len(bloques)}")

    predios = []
    for b in bloques:
        campos = extraer_campos_predio(b)
        npn = campos.get("Número Predial Nacional:", NR)
        npn_dig = re.sub(r'\D', '', npn) if npn != NR else ''
        if len(npn_dig) != 30:
            logger.debug(f"   Bloque descartado (NPN inválido: '{npn}')")
            continue
        predios.append({**doc, **campos})

    logger.info(f"   Predios válidos extraídos: {len(predios)}")
    return predios
def normalize_text(text):
    """
    Normaliza texto OCR de documentos catastrales colombianos.
    """
    if not text:
        return ""
    
    # Convertir a mayúsculas
    text = text.upper()
    
    # Reemplazar caracteres de codificación corrupta comunes en OCR español
    replacements = {
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U',
        'À': 'A', 'È': 'E', 'Ì': 'I', 'Ò': 'O', 'Ù': 'U',
        'Ñ': 'N', 'Ç': 'C',
        'Ã': 'A',  # Codificación corrupta típica
        ' ': ' ',   # Espacios irregulares
        'º': 'O',  # Símbolo de ordinal masculino -> O
        'ª': 'A',  # Símbolo de ordinal femenino -> A
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Normalizar espacios múltiples, tabs, saltos de línea
    text = re.sub(r'\s+', ' ', text)
    
    # Normalizar guiones y separadores
    text = text.replace('–', '-').replace('—', '-')
    
    return text.strip()


# ==============================================================================
# FUNCIONES DE ORQUESTACIÓN (PROCESAMIENTO PARALELO)
# ==============================================================================

logger.info("Verificando instalación de Tesseract OCR...")
try:
    # Verificar que Tesseract esté accesible
    version = pytesseract.get_tesseract_version()
    logger.info(f"Tesseract OCR versión {version} detectado y listo.")
except Exception as e:
    logger.error(f"Error: No se pudo acceder a Tesseract OCR. Verifica la ruta: {e}")
    raise
def process_pdf(file_path, cache_folder, base_folder, output_folder, force_reprocess=False):
    try:
        filename = os.path.basename(file_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESANDO: {filename}")
        logger.info(f"{'='*60}")
        
        if not force_reprocess and is_file_processed(file_path, cache_folder):
            logger.info(f"⚠️ Ya en caché, omitiendo")
            return None
        
        # OCR
        images = pdf_to_images(file_path)
        if not images:
            logger.error(f" No se extrajeron imágenes")
            return None
        
        logger.info(f" {len(images)} páginas")
        text = ""
        for i, image in enumerate(images):
            page_text = extract_text_from_image(image, i + 1)
            if page_text:
                text += page_text + "\n"
        
        # Guardar TXT
        output_file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Extraer datos: una fila por predio (enfoque por bloques)
        logger.info(f" Extrayendo datos estructurados (una fila por predio)...")
        predios = extraer_predios_de_texto(text)

        if not predios:
            logger.warning(f"No se extrajo ningún predio válido de {filename}")
            mark_file_completed(file_path, cache_folder, [])
            return []

        # Construir una fila por predio, en el orden de columnas del Excel.
        # Columnas: Nombre del archivo + RESOLUCIÓN No. + FechaResolucion + CAMPOS_PREDIO
        filas = []
        for predio in predios:
            fila = [filename, predio.get("RESOLUCIÓN No.", NR), predio.get("FechaResolucion", NR)]
            fila += [predio.get(campo, NR) for campo in CAMPOS_PREDIO]
            filas.append(fila)

        # Guardar en caché
        mark_file_completed(file_path, cache_folder, filas)

        logger.info(f"COMPLETADO: {filename} ({len(filas)} predios extraídos)")
        return filas

    except Exception as e:
        logger.error(f" ERROR en {filename}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
def process_pdfs_in_folder(base_folder_path, output_folder_path, structured_folder_path,
                            output_excel_name):
    # Crear carpetas de salida si no existen
    os.makedirs(output_folder_path, exist_ok=True)
    os.makedirs(structured_folder_path, exist_ok=True)
    
    # Ruta completa del archivo Excel
    output_excel = os.path.join(structured_folder_path, output_excel_name)

    # === LIMPIAR CACHÉ SOLO SI SE FUERZA EL REPROCESAMIENTO ===
    # Con FORCE_REPROCESS=true se elimina el caché para reprocesar todo desde cero.
    # De lo contrario, se conserva para saltar archivos ya procesados.
    force_reprocess = os.environ.get('FORCE_REPROCESS', 'false').lower() == 'true'
    cache_folder = os.path.join(base_folder_path, ".cache")
    if force_reprocess and os.path.exists(cache_folder):
        shutil.rmtree(cache_folder)
        logger.warning("MODO FORCE_REPROCESS: caché eliminado, se reprocesarán todos los archivos.")

    # === VERIFICAR SI EL EXCEL ESTÁ BLOQUEADO ===
    if os.path.exists(output_excel):
        try:
            # Intentar abrir en modo append para verificar permisos
            with open(output_excel, 'a'):
                pass
        except PermissionError:
            logger.error(f"ERROR: El archivo Excel está abierto en otro programa: {output_excel}")
            logger.error("   Cierra el archivo Excel y vuelve a ejecutar el script.")
            # Crear nombre alternativo
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_excel = os.path.join(structured_folder_path, f"TABULADO_RESULTADOS_{timestamp}.xlsx")
            logger.info(f"   Se creará archivo alternativo: {output_excel}")

    # Asegurar que la carpeta de caché exista
    cache_folder = setup_cache_folder(base_folder_path)

    workbook = Workbook()
    sheet = workbook.active
    # Nueva estructura: UNA FILA POR PREDIO.
    # Columnas: archivo + datos de documento (resolución/fecha) + campos del predio.
    header = ["Nombre del archivo", "RESOLUCIÓN No.", "FechaResolucion"] + list(CAMPOS_PREDIO)
    sheet.append(header)

    pdf_files = []
    for root, dirs, files in os.walk(base_folder_path):
        for filename in files:
            if filename.endswith(".pdf"):
                pdf_files.append(os.path.join(root, filename))
    
    logger.info(f"Se encontraron {len(pdf_files)} archivos PDF para procesar con {CONFIG['max_workers']} workers (hilos).")

    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        process_func = partial(process_pdf,
                              cache_folder=cache_folder,
                              base_folder=base_folder_path, output_folder=output_folder_path,
                              force_reprocess=force_reprocess)
        
        futures = {executor.submit(process_func, pdf_file): pdf_file for pdf_file in pdf_files}
        
        completed = 0
        errors = 0
        total_predios = 0
        for future in tqdm(as_completed(futures), total=len(pdf_files), desc="Procesando archivos PDF"):
            pdf_file = futures[future]
            try:
                # result es una lista de filas (una por predio); None = error, [] = sin predios/caché
                result = future.result()
                if result:
                    for fila in result:
                        sheet.append(fila)
                        total_predios += 1
                    completed += 1
                    # Guardar cada 5 archivos para no perder datos
                    if completed % 5 == 0:
                        try:
                            workbook.save(output_excel)
                            logger.info(f"Guardado parcial: {completed} archivos, {total_predios} predios.")
                        except PermissionError:
                            logger.error(f"No se pudo guardar - archivo Excel bloqueado. Continuando...")
                            errors += 1
                elif result is None:
                    errors += 1
                else:
                    # Lista vacía: archivo sin predios válidos o ya en caché
                    completed += 1
            except Exception as e:
                logger.error(f"Error procesando {pdf_file}: {e}")
                errors += 1
    
    # === INTENTAR GUARDAR FINAL CON RETRY ===
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            workbook.save(output_excel)
            logger.info(f"Proceso completado. Datos guardados en {output_excel}")
            logger.info(f"Resumen: {completed} archivos procesados, {total_predios} predios tabulados, {errors} con errores.")
            logger.info(f"Textos extraídos guardados en: {output_folder_path}")
            break
        except PermissionError as e:
            if attempt < max_attempts - 1:
                logger.warning(f" Intento {attempt + 1}/{max_attempts}: Excel bloqueado. Esperando 3 segundos...")
                time.sleep(3)
            else:
                # Guardar con nombre alternativo
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                alt_path = os.path.join(structured_folder_path, f"TABULADO_RESULTADOS_{timestamp}.xlsx")
                workbook.save(alt_path)
                logger.error(f" No se pudo guardar en ubicación original después de {max_attempts} intentos.")
                logger.info(f" Archivo guardado en ubicación alternativa: {alt_path}")
        except Exception as e:
            logger.error(f"Error inesperado guardando Excel: {e}")
            break


def clear_cache_for_file(file_path, cache_folder):
    """Elimina la entrada de caché para un archivo específico"""
    file_hash = get_file_hash(file_path)
    cache_file = os.path.join(cache_folder, f"{file_hash}.json")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        logger.info(f"Caché eliminado para: {os.path.basename(file_path)}")
        return True
    return False

# ==============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    # Usar las rutas configuradas al inicio del archivo
    base_folder_path = BASE_FOLDER_PATH
    output_folder_path = OUTPUT_FOLDER_PATH
    structured_folder_path = STRUCTURED_FOLDER_PATH
    output_excel_name = OUTPUT_EXCEL_NAME

    process_pdfs_in_folder(base_folder_path, output_folder_path, structured_folder_path,
                           output_excel_name)
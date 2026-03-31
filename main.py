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
import spacy
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
# Ruta de entrada: carpeta con los PDFs
BASE_FOLDER_PATH = r"C:\Users\Jorge\Desktop\repositorio_Git_Hub\tesis-ia-catastro\data\raw_pdfs"

# Ruta de salida: carpeta donde se guardarán los textos extraídos (raw)
OUTPUT_FOLDER_PATH = r"C:\Users\Jorge\Desktop\repositorio_Git_Hub\tesis-ia-catastro\data\raw_text"

# Ruta de salida: carpeta donde se guardará el Excel estructurado
STRUCTURED_FOLDER_PATH = r"C:\Users\Jorge\Desktop\repositorio_Git_Hub\tesis-ia-catastro\data\structured"

# Nombre del archivo Excel de salida
OUTPUT_EXCEL_NAME = "TABULADO_RESULTADOS.xlsx"

# --- CONFIGURACIÓN DE TESSERACT ---
# Especificar la ruta completa de Tesseract en tu PC
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Jorge\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Configurar logging para ver el progreso y errores
logging.basicConfig(
    level=logging.DEBUG,  # Cambiar a INFO después de depurar
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("catastro_debug.log", encoding='utf-8'),
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

def extract_sections(text, section_starts, section_ends):
    start_pattern = '|'.join(section_starts)
    end_patterns = [re.escape(end) for end in section_ends]
    end_pattern = '|'.join(end_patterns)
    pattern = re.compile(f'(?s)({start_pattern})(.*?)(?=({end_pattern})|$)', re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    sections = [match[1].strip() for match in matches]
    if sections:
        result = sections[0]
        for end_pattern in end_patterns:
            end_index = result.lower().find(end_pattern.lower())
            if end_index != -1:
                result = result[:end_index].strip()
                break
        return result
    return ""
def extract_info_from_text(text, keywords, char_counts, nlp):
    """
    Extrae información de resoluciones catastrales con múltiples predios.
    Diseñado para texto OCR de la Gobernación del Valle del Cauca.
    """
    logger.info(f" Iniciando extracción. Texto: {len(text)} caracteres")
    extracted_texts = []
    
    # Normalizar texto
    normalized_text = normalize_text(text)
    
    # Log de diagnóstico (primeros 1000 caracteres)
    logger.debug(f"Texto normalizado:\n{normalized_text[:1000]}...")
    
    for idx, (keyword, char_count) in enumerate(zip(keywords, char_counts)):
        logger.debug(f"\n[{idx}] Procesando: '{keyword}'")
        found_values = []
        
        # === RESOLUCIÓN (única por documento) ===
        if keyword == "RESOLUCIÓN No.":
            # Buscar: RESOLUCION No. 1.120.50.03.01.M02-00342 DE 2025
            pattern = re.compile(r'RESOLUCION\s+NO\.?\s*([0-9][0-9\.\-]*M[0-9]{2}\-[0-9]+)\s+DE\s+(\d{4})')
            match = pattern.search(normalized_text)
            if match:
                valor = f"{match.group(1)} DE {match.group(2)}"
                found_values.append(valor[:char_count])
                logger.debug(f"   Resolución: {valor}")
            else:
                # Fallback más simple
                pattern2 = re.compile(r'RESOLUCION\s+NO\.?\s*([^,\n]{5,30})')
                match2 = pattern2.search(normalized_text)
                found_values.append(match2.group(1).strip()[:char_count] if match2 else "NR")
        
        elif keyword == "FechaResolucion":
            # Buscar: 16 de Diciembre de 2025 (varios formatos)
            pattern = re.compile(r'(\d{1,2})\s+DE\s+([A-Z]+)\s+DE\s+(\d{4})')
            match = pattern.search(normalized_text)
            if match:
                valor = f"{match.group(1)} DE {match.group(2)} DE {match.group(3)}"
                found_values.append(valor[:char_count])
                logger.debug(f"   Fecha: {valor}")
            else:
                found_values.append("NR")

        # === CAMPOS QUE SE REPITEN POR PREDIO (múltiples valores) ===
        elif keyword == "Número de matrícula inmobiliaria:":
            # Buscar todos: 378-281007, 378-281008, etc.
            pattern = re.compile(r'NUMERO\s+DE\s+MATRICULA\s+INMOBILIARIA\s*[:\-]?\s*(\d{3}\-\d{6})')
            matches = pattern.findall(normalized_text)
            found_values = [m[:char_count] for m in matches] if matches else ["NR"]
            logger.debug(f"   Matrículas: {len(matches)} encontradas")

        elif keyword == "Número predial:":
            # Puede ser "No registra" o un número
            pattern = re.compile(r'NUMERO\s+PREDIAL\s*[:\-]?\s*(NO\s+REGISTRA|\d[\d\-]*)')
            matches = pattern.findall(normalized_text)
            found_values = [m[:char_count] if m else "NR" for m in matches] if matches else ["NR"]
            logger.debug(f"   Números prediales: {len(matches)} encontrados")

        elif keyword == "Número Predial Nacional:":
            # 20-24 dígitos típicamente
            pattern = re.compile(r'NUMERO\s+PREDIAL\s+NACIONAL\s*[:\-]?\s*(\d{20,24})')
            matches = pattern.findall(normalized_text)
            found_values = [m[:char_count] for m in matches] if matches else ["NR"]
            logger.debug(f"   NPNs: {len(matches)} encontrados")

        elif keyword == "Código Homologado:":
            # Formato: CCK000SUHZC, CCKOOOSUJAF (varía por OCR)
            pattern = re.compile(r'CODIGO\s+HOMOLOGADO\s*[:\-]?\s*([A-Z]{2,3}\d{3,}[A-Z]{2,4})')
            matches = pattern.findall(normalized_text)
            found_values = [m[:char_count] for m in matches] if matches else ["NR"]
            logger.debug(f"   Códigos: {len(matches)} encontrados")

        elif keyword == "Municipio:":
            pattern = re.compile(r'MUNICIPIO\s*[:\-]?\s*([A-Z]{3,20})(?=\s+PROPIETARIO|\n|$)')
            matches = pattern.findall(normalized_text)
            # Eliminar duplicados manteniendo orden
            seen = set()
            unique = []
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    unique.append(m)
            found_values = [m[:char_count] for m in unique] if unique else ["NR"]
            logger.debug(f"   Municipios: {len(unique)} únicos")

        elif keyword == "Propietario:":
            # DOS TIPOS de propietarios en tu texto:
            # 1. MUNICIPIO DE CANDELARIA (simple)
            # 2. FIDUCIARIA DAVIVIENDA S.A. COMO VOCERA Y ADMINISTRADORA DEL FIDEICOMISO BELORIZONTE (complejo)
            pattern = re.compile(r'PROPIETARIO\s*[:\-]?\s*(.*?)(?=\s+DOCUMENTO|\s+MUNICIPIO|\n\s*DOCUMENTO)', re.DOTALL)
            matches = pattern.findall(normalized_text)
            if matches:
                for m in matches:
                    # Limpiar: eliminar "Documento de identificación" si se coló
                    limpio = re.sub(r'DOCUMENTO.*', '', m, flags=re.IGNORECASE)
                    limpio = re.sub(r'\s+', ' ', limpio).strip()
                    if limpio and len(limpio) > 3:
                        found_values.append(limpio[:char_count])
            if not found_values:
                found_values = ["NR"]
            logger.debug(f"   Propietarios: {len(found_values)} encontrados")

        elif keyword == "Documento de identificación:":
            # Puede ser "No registra" o "N 8300537006"
            pattern = re.compile(r'DOCUMENTO\s+DE\s+IDENTIFICACION\s*[:\-]?\s*(NO\s+REGISTRA|N?\s*\d[\d\.\-]*)')
            matches = pattern.findall(normalized_text)
            found_values = []
            for m in matches:
                valor = m[0] if isinstance(m, tuple) else m
                valor = re.sub(r'\s+', ' ', str(valor)).strip()
                if valor:
                    found_values.append(valor[:char_count])
            if not found_values:
                found_values = ["NR"]
            logger.debug(f"   Documentos: {len(found_values)} encontrados")

        elif keyword == "Dirección:":
            # UR BELORIZONTE C 21A, UR BELORIZONTE Mz F1, etc.
            pattern = re.compile(r'DIRECCION\s*[:\-]?\s*(UR\s+[^\n]+?)(?=\s+AREA|\s+ÁREA|\n\s*AREA)', re.IGNORECASE | re.DOTALL)
            matches = pattern.findall(normalized_text)
            found_values = []
            for m in matches:
                limpio = re.sub(r'\s+', ' ', str(m)).strip()
                if len(limpio) > 5:
                    found_values.append(limpio[:char_count])
            if not found_values:
                found_values = ["NR"]
            logger.debug(f"   Direcciones: {len(found_values)} encontradas")

        elif keyword == "Área predio:":
            # 1845.84 m2, 1168.72 m2 - capturar número con decimales
            pattern = re.compile(r'AREA\s+(?:DEL\s+)?(?:PREDIO|TERRENO)\s*[:\-]?\s*(\d{1,4}(?:[.,]\d{1,2})?)\s*(?:M2|M²|METROS|MTS)?', re.IGNORECASE)
            matches = pattern.findall(normalized_text)
            found_values = [m[0] if isinstance(m, tuple) else m for m in matches] if matches else ["NR"]
            logger.debug(f"   Áreas predio: {len(matches)} encontradas")

        elif keyword == "Área construida:":
            pattern = re.compile(r'AREA\s+CONSTRUIDA\s*[:\-]?\s*(NO\s+REGISTRA|\d{1,4}(?:[.,]\d{1,2})?)\s*(?:M2|M²)?', re.IGNORECASE)
            matches = pattern.findall(normalized_text)
            found_values = []
            for m in matches:
                valor = m[0] if isinstance(m, tuple) else m
                if valor:
                    found_values.append(str(valor)[:char_count])
            if not found_values:
                found_values = ["NR"]
            logger.debug(f"   Áreas construidas: {len(matches)} encontradas")

        elif keyword == "Destinación economica:":
            # "Lote urbanizado no construido"
            pattern = re.compile(r'DESTINACION\s*[:\-]?\s*([A-Z\s]{10,40}?)(?=\n|AVALUO|$)', re.IGNORECASE)
            matches = pattern.findall(normalized_text)
            found_values = []
            seen = set()
            for m in matches:
                limpio = re.sub(r'\s+', ' ', str(m)).strip()
                if limpio and limpio not in seen and len(limpio) > 5:
                    seen.add(limpio)
                    found_values.append(limpio[:char_count])
            if not found_values:
                found_values = ["NR"]
            logger.debug(f"   Destinaciones: {len(found_values)} encontradas")

        elif keyword == "Avalúo:":
            # $ 356,372,000 Vigencia... - capturar solo el monto inicial
            pattern = re.compile(r'AVALUO\s*[:\-]?\s*\$?\s*([\d\.,]+)\s*(?:VIGENCIA|FECHA|$)', re.IGNORECASE)
            matches = pattern.findall(normalized_text)
            found_values = [m[:char_count] for m in matches] if matches else ["NR"]
            logger.debug(f"   Avalúos: {len(matches)} encontrados")

        elif keyword == "Fecha de la inscripción Catastral:":
            # 18/06/2024 o 18/08/2024
            pattern = re.compile(r'FECHA\s+DE\s+LA\s+INSCRIPCION\s+CATASTRAL\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})')
            matches = pattern.findall(normalized_text)
            found_values = [m[:char_count] for m in matches] if matches else ["NR"]
            logger.debug(f"  Fechas inscripción: {len(matches)} encontradas")
        
        elif keyword == "Vigencia Fiscal:":
            # 01/01/2026
            pattern = re.compile(r'VIGENCIA\s+FISCAL\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})')
            matches = pattern.findall(normalized_text)
            found_values = [m[:char_count] for m in matches] if matches else ["NR"]
            logger.debug(f"  Vigencias: {len(matches)} encontradas")

        else:
            # Fallback genérico
            normalized_kw = normalize_text(keyword).rstrip(':')
            pattern = re.compile(re.escape(normalized_kw) + r'\s*[:\-]?\s*(.*?)(?=\n|$)', re.DOTALL)
            matches = pattern.findall(normalized_text)
            found_values = [m.strip()[:char_count] for m in matches] if matches else ["NR"]
        
        # Unir valores múltiples con separador
        if found_values and found_values != ["NR"]:
            resultado = ' | '.join(str(v) for v in found_values if v and v != "NR")
        else:
            resultado = "NR"
        
        extracted_texts.append(resultado)
        logger.info(f"[{idx}] {keyword}: {resultado[:60]}{'...' if len(resultado) > 60 else ''}")
    
    return extracted_texts
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
def count_NPNs(npn_text):
    if npn_text and npn_text != "NR":
        unique_npns = set(npn_text.split('|'))
        return len(unique_npns)
    return 0

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
def process_pdf(file_path, keywords, char_counts, nlp, cache_folder, base_folder, output_folder, force_reprocess=False):
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
        
        # Extraer datos
        logger.info(f" Extrayendo datos estructurados...")
        final_extracted_texts = extract_info_from_text(text, keywords, char_counts, nlp)
        
        # Verificar extracción
        campos_ok = sum(1 for f in final_extracted_texts if f != "NR")
        logger.info(f"Campos extraídos: {campos_ok}/{len(keywords)}")
        
        # Calcular NPNs (contar cuántos hay separados por |)
        try:
            npn_index = keywords.index("Número Predial Nacional:")
            npn_text = final_extracted_texts[npn_index]
            count_NPN = len([x for x in npn_text.split('|') if x.strip() and x.strip() != "NR"])
        except Exception as e:
            logger.error(f"Error contando NPNs: {e}")
            count_NPN = 0
        
        result = [filename] + final_extracted_texts + [count_NPN]
        
        # Guardar en caché
        mark_file_completed(file_path, cache_folder, result)
        
        logger.info(f"COMPLETADO: {filename} ({count_NPN} NPNs encontrados)")
        return result
        
    except Exception as e:
        logger.error(f" ERROR en {filename}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
def process_pdfs_in_folder(base_folder_path, output_folder_path, structured_folder_path, 
                            output_excel_name, keywords, char_counts):
    # Crear carpetas de salida si no existen
    os.makedirs(output_folder_path, exist_ok=True)
    os.makedirs(structured_folder_path, exist_ok=True)
    
    # Ruta completa del archivo Excel
    output_excel = os.path.join(structured_folder_path, output_excel_name)
    
    # Agrega esto temporalmente al inicio de process_pdfs_in_folder
    cache_folder = os.path.join(base_folder_path, ".cache")
    if os.path.exists(cache_folder):
        shutil.rmtree(cache_folder)
        os.makedirs(cache_folder, exist_ok=True)
        logger.info("Caché eliminado")

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
    
    cache_folder = setup_cache_folder(base_folder_path)
    
    workbook = Workbook()
    sheet = workbook.active
    header = ["Nombre del archivo"] + list(keywords) + ["Conteo_NPN"]
    sheet.append(header)
    
    nlp = spacy.blank("es")
    
    pdf_files = []
    for root, dirs, files in os.walk(base_folder_path):
        for filename in files:
            if filename.endswith(".pdf"):
                pdf_files.append(os.path.join(root, filename))
    
    logger.info(f"Se encontraron {len(pdf_files)} archivos PDF para procesar con {CONFIG['max_workers']} workers (hilos).")
    
    # === LIMPIAR CACHÉ SI SE REQUIERE PROCESAMIENTO FORZADO ===
    force_reprocess = os.environ.get('FORCE_REPROCESS', 'false').lower() == 'true'
    if force_reprocess:
        logger.warning("MODO FORCE_REPROCESS: Se ignorará el caché y se reprocesarán todos los archivos.")
    
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        process_func = partial(process_pdf, keywords=keywords, char_counts=char_counts, nlp=nlp, 
                              cache_folder=cache_folder, 
                              base_folder=base_folder_path, output_folder=output_folder_path,
                              force_reprocess=force_reprocess)
        
        futures = {executor.submit(process_func, pdf_file): pdf_file for pdf_file in pdf_files}
        
        completed = 0
        errors = 0
        for future in tqdm(as_completed(futures), total=len(pdf_files), desc="Procesando archivos PDF"):
            pdf_file = futures[future]
            try:
                result = future.result()
                if result:
                    sheet.append(result)
                    completed += 1
                    # Guardar cada 5 archivos en lugar de 10 para no perder datos
                    if completed % 5 == 0:
                        try:
                            workbook.save(output_excel)
                            logger.info(f"Guardado parcial: {completed} archivos procesados.")
                        except PermissionError:
                            logger.error(f"No se pudo guardar - archivo Excel bloqueado. Continuando...")
                            errors += 1
                else:
                    errors += 1
            except Exception as e:
                logger.error(f"Error procesando {pdf_file}: {e}")
                errors += 1
    
    # === INTENTAR GUARDAR FINAL CON RETRY ===
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            workbook.save(output_excel)
            logger.info(f"Proceso completado. Datos guardados en {output_excel}")
            logger.info(f"Resumen: {completed} archivos procesados exitosamente, {errors} con errores.")
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
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        process_func = partial(
            process_pdf, 
            keywords=keywords, 
            char_counts=char_counts, 
            nlp=nlp, 
            cache_folder=cache_folder, 
            base_folder=base_folder_path, 
            output_folder=output_folder_path,
            force_reprocess=force_reprocess  #
        )
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
    
    keywords = ["RESOLUCIÓN No.", "FechaResolucion", "Número de matrícula inmobiliaria:", "Número predial:", 
                "Número Predial Nacional:", "Código Homologado:", "Municipio:", "Propietario:", 
                "Documento de identificación:", "Dirección:", "Área predio:", "Área construida:", 
                "Destinación economica:", "Avalúo:", "Fecha de la inscripción Catastral:", "Vigencia Fiscal:"]
    char_counts = [36, 24, 11, 21, 32, 13, 12, 850, 850, 260, 10, 10, 30, 14, 12, 12] 
    
    process_pdfs_in_folder(base_folder_path, output_folder_path, structured_folder_path, 
                          output_excel_name, keywords, char_counts)
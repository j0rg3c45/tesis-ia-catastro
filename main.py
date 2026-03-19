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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    extracted_texts = []
    doc = nlp(text)
    normalized_text = normalize_text(text)
    
    for keyword, char_count in zip(keywords, char_counts):
        found_values = []
        
        if keyword == "RESOLUCIÓN No.":
            pattern = re.compile(r'RESOLUCI[oÓ]N\s*No\.\s*([1-9]\d*(?:\.\d+)*\.?(?:[MZ][A-Z0-9-]{8})?)(?:-[A-Z0-9-]*)?', re.IGNORECASE)
            match = pattern.search(normalized_text)
            found_values.append(match.group(1).strip()[:char_count] if match else "NR")
        
        elif keyword == "FechaResolucion":
            pattern = re.compile(r'(\d{1,2}\s+DE\s+[A-Z]+?\s+DE\s+\d{4})', re.IGNORECASE)
            match = pattern.search(normalized_text)
            found_values.append(match.group(1).strip()[:char_count] if match else "NR")

        elif keyword == "Número de matrícula inmobiliaria:":
            pattern = re.compile(r'Número\s+de\s+matr[ií]cula\s+inmobiliaria\s*[:\-–]?\s*([0-9\-]+)', re.IGNORECASE)
            matches = pattern.findall(normalized_text)
            if matches:
                found_values.extend([match.strip()[:char_count] for match in matches])
            else:
                found_values.append("NR")

        elif keyword == "Propietario:":
            pattern = re.compile(r'(Propietarios?):\s*(.*?)\s*(Documentos? de identificación:|Dirección:|Número de matrícula inmobiliaria:|$)', re.DOTALL | re.IGNORECASE)
            matches = pattern.findall(normalized_text)
            if matches:
                propietarios = [match[1].replace('\n', ' ').strip()[:char_count] for match in matches]
                found_values.extend(propietarios)
            else:
                found_values.append("NR")

        elif keyword == "Documento de identificación:":
            pattern = re.compile(r'(Documentos? de identificación:)\s*(.*?)\s*(Dirección:|Número de matrícula inmobiliaria:|$)', re.DOTALL | re.IGNORECASE)
            matches = pattern.findall(normalized_text)
            if matches:
                identificaciones = [match[1].replace('\n', ' ').strip()[:char_count] for match in matches]
                found_values.extend(identificaciones)
            else:
                found_values.append("NR")
        
        elif keyword in ["Área predio:", "Área construida:"]:
            area_pattern = re.compile(r'Área\s+(del\s+)?(Predio|Terreno):', re.IGNORECASE) if keyword == "Área predio:" else re.compile(r'Área\s+Construida:', re.IGNORECASE)
            matches = area_pattern.finditer(normalized_text)
            areas = []
            for match in matches:
                area_match = re.search(r'\s*(\d+[\d\s,.]*)', normalized_text[match.end():match.end() + 20])
                if area_match:
                    areas.append(area_match.group(1).strip().replace(',', '').replace(' ', '')[:char_count])
            if areas:
                found_values.extend(areas)
            else:
                found_values.append("NR")

        elif keyword == "Fecha de la inscripción Catastral:":
            pattern = re.compile(r'Fecha\s+de\s+la\s+inscripci[oóÓ]n\s+Catastral:\s*(\d{1,2}/\d{1,2}/\d{4})', re.IGNORECASE)
            matches = pattern.findall(normalized_text)
            if matches:
                found_values.extend([match.strip()[:char_count] for match in matches])
            else:
                found_values.append("NR")
        
        elif keyword == "Vigencia Fiscal:":
            pattern = re.compile(r'Vigencia\s+Fiscal:\s*(\d{1,2}/\d{1,2}/\d{4})', re.IGNORECASE)
            matches = pattern.findall(normalized_text)
            if matches:
                found_values.extend([match.strip()[:char_count] for match in matches])
            else:
                found_values.append("NR")

        elif keyword == "Destinación economica:":
            pattern = re.compile(r'(Destinaci[oóÓ]n\s+econ[oóÓ]mica|Destino|Uso\s+econ[oóÓ]mico|Clasificaci[oóÓ]n\s+econ[oóÓ]mica):\s*(.*?)(?=\n|$)', re.IGNORECASE)
            matches = pattern.findall(normalized_text)
            if matches:
                destinos = [match[1].strip()[:char_count] for match in matches]
                found_values.extend(destinos)
            else:
                found_values.append("NR")

        else:
            keyword_pattern = re.compile(re.escape(keyword) + r'\s*(.*?)(?=\n|$)', re.DOTALL | re.IGNORECASE)
            matches = keyword_pattern.findall(normalized_text)
            if matches:
                found_values.extend([match.strip()[:char_count] for match in matches])
            else:
                found_values.append("NR")
        
        extracted_texts.append(' | '.join(found_values))
    
    return extracted_texts

def normalize_text(text):
    text = text.upper()
    replacements = {'Ã': 'A', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'É': 'E', 'Á': 'A', 'Ñ': 'N'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'\s+', ' ', text)
    return text

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

def process_pdf(file_path, keywords, char_counts, nlp, cache_folder, base_folder, output_folder):
    try:
        filename = os.path.basename(file_path)
        logger.info(f"Iniciando procesamiento del archivo: {filename}")
        
        if is_file_processed(file_path, cache_folder):
            logger.info(f"Archivo {filename} ya procesado, omitiendo...")
            return None
        
        images = pdf_to_images(file_path)
        if not images:
            logger.error(f"No se pudieron extraer imágenes del PDF {filename}")
            return None
        
        logger.info(f"Se extrajeron {len(images)} imágenes de {filename}. Iniciando OCR...")
        text = ""
        
        for i, image in enumerate(images):
            try:
                page_text = extract_text_from_image(image, i + 1)
                if page_text:
                    text += page_text + "\n"
                if (i + 1) % 5 == 0:
                    logger.info(f"Procesadas {i+1}/{len(images)} páginas de {filename}...")
                    gc.collect()
            except Exception as e:
                logger.error(f"Error extrayendo texto de la página {i+1} de {filename}: {e}")
        
        # --- Guardar texto extraído en la carpeta de salida (raw_text) ---
        # Crear nombre de archivo de salida (mismo nombre pero extensión .txt)
        output_filename = os.path.splitext(filename)[0] + ".txt"
        output_file_path = os.path.join(output_folder, output_filename)
        
        # Guardar el texto extraído
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Texto extraído guardado en: {output_file_path}")

        if not text.strip():
            logger.error(f"El texto extraído para {filename} está completamente vacío.")
            return None

        logger.info(f"OCR completado para {filename}. Extrayendo secciones...")
        
        header_keywords = ["RESOLUCIÓN No.", "FechaResolucion"]
        header_char_counts = [char_counts[keywords.index(k)] for k in header_keywords]
        header_info = extract_info_from_text(text, header_keywords, header_char_counts, nlp)

        resuelve_pattern = r"R\s*[E3][S5]\s*UE\s*LV\s*[E3]:?"
        section_starts = [resuelve_pattern]
        section_ends = ["NOTIFÍQUESE Y CÚMPLASE", "COMUNÍQUESE Y CÚMPLASE", "NOTIFÍQUESE", "COMUNÍQUESE"]
        
        full_resuelve_block = extract_sections(text, section_starts, section_ends)
        
        section_to_process = None
        if not full_resuelve_block:
            logger.warning(f"No se encontró la sección 'RESUELVE' en {filename}. Usando texto completo.")
            section_to_process = text
        else:
            section_to_process = full_resuelve_block
        
        if not section_to_process:
            logger.warning(f"No se pudo extraer ninguna sección procesable de {filename}")
            return None
        
        property_keywords = [k for k in keywords if k not in header_keywords]
        property_char_counts = [char_counts[keywords.index(k)] for k in property_keywords]
        property_info = extract_info_from_text(section_to_process, property_keywords, property_char_counts, nlp)
        
        final_extracted_texts = header_info + property_info
        
        if all(field == "NR" for field in final_extracted_texts):
            logger.warning(f"No se encontraron datos relevantes en {filename}")
            return None
        
        npn_text = final_extracted_texts[keywords.index("Número Predial Nacional:")]
        count_NPN = count_NPNs(npn_text)
        
        result = [filename] + final_extracted_texts + [count_NPN]
        mark_file_completed(file_path, cache_folder, result)
        
        logger.info(f"Procesamiento completado para {filename}.")
        return result
    
    except Exception as e:
        logger.error(f"Error procesando el archivo {file_path}: {e}")
        return None

def process_pdfs_in_folder(base_folder_path, output_folder_path, structured_folder_path, 
                            output_excel_name, keywords, char_counts):
    # Crear carpetas de salida si no existen
    os.makedirs(output_folder_path, exist_ok=True)
    os.makedirs(structured_folder_path, exist_ok=True)
    
    # Ruta completa del archivo Excel
    output_excel = os.path.join(structured_folder_path, output_excel_name)
    
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
    
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        process_func = partial(process_pdf, keywords=keywords, char_counts=char_counts, nlp=nlp, 
                              cache_folder=cache_folder, 
                              base_folder=base_folder_path, output_folder=output_folder_path)
        
        futures = {executor.submit(process_func, pdf_file): pdf_file for pdf_file in pdf_files}
        
        completed = 0
        for future in tqdm(as_completed(futures), total=len(pdf_files), desc="Procesando archivos PDF"):
            pdf_file = futures[future]
            try:
                result = future.result()
                if result:
                    sheet.append(result)
                    completed += 1
                    if completed % 10 == 0:
                        workbook.save(output_excel)
                        logger.info(f"Guardado parcial: {completed} archivos procesados.")
            except Exception as e:
                logger.error(f"Error procesando {pdf_file}: {e}")
    
    workbook.save(output_excel)
    logger.info(f"Proceso completado. Datos guardados en {output_excel}")
    logger.info(f"Textos extraídos guardados en: {output_folder_path}")

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
"""
comparar_bases.py
MAYO 2026
Etapa 2: Comparación entre resultados extraídos (TABULADO_RESULTADOS.xlsx)
y la base catastral oficial (1_REGISTRO_R1_CONTRALORIA_20260501.xlsx).

El cruce se realiza por el campo Número Predial Nacional (NPN).
Se detectan inconsistencias campo por campo entre ambas fuentes.

Autor: Proyecto de Tesis - Maestría en IA y Ciencia de Datos
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import re

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

# Rutas de archivos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "structured")

ARCHIVO_RESULTADOS = os.path.join(DATA_DIR, "TABULADO_RESULTADOS.xlsx")
ARCHIVO_CATASTRAL = os.path.join(DATA_DIR, "1_REGISTRO_R1_CONTRALORIA_20260501.xlsx")

# Carpeta de reportes
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "comparacion.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# MAPEO DE CAMPOS ENTRE AMBAS FUENTES
# ==============================================================================

# ==============================================================================
# EQUIVALENCIAS DE CAMPOS ENTRE AMBAS FUENTES
# ==============================================================================
# Campo de cruce: numero_predial (base) ↔ Número Predial Nacional: (tabulado)
#
# Campos comparables: (campo_tabulado, campo_catastral, nombre_legible)
CAMPOS_COMPARABLES = [
    ("Municipio:", "municipio", "Municipio"),
    ("Número predial:", "numero_predial_anterior", "Número Predial Anterior"),
    ("Propietario:", "nombre", "Propietario/Nombre"),
    ("Documento de identificación:", "numero_documento", "Documento Identificación"),
    ("Dirección:", "direccion", "Dirección"),
    ("Área predio:", "area_terreno", "Área Terreno"),
    ("Área construida:", "area_construida", "Área Construida"),
    ("Destinación economica:", "destino_economico", "Destinación Económica"),
    ("Avalúo:", "avaluo", "Avalúo"),
]


# ==============================================================================
# FUNCIONES DE CARGA DE DATOS
# ==============================================================================

def cargar_tabulado(ruta):
    """
    Carga el archivo TABULADO_RESULTADOS.xlsx y expande los NPNs múltiples
    (separados por |) en filas individuales para facilitar el cruce.
    """
    logger.info(f"Cargando resultados extraídos: {ruta}")
    df = pd.read_excel(ruta)
    logger.info(f"  Filas originales: {len(df)}, Columnas: {len(df.columns)}")
    logger.info(f"  Columnas: {list(df.columns)}")

    # La columna NPN puede tener múltiples valores separados por |
    col_npn = "Número Predial Nacional:"
    
    if col_npn not in df.columns:
        logger.error(f"No se encontró la columna '{col_npn}' en el tabulado.")
        return pd.DataFrame()

    # Expandir NPNs múltiples en filas individuales
    filas_expandidas = []
    for _, row in df.iterrows():
        npn_raw = str(row[col_npn]).strip()
        if npn_raw in ("NR", "nan", "", "None"):
            continue
        
        # Separar por | y limpiar
        npns = [n.strip() for n in npn_raw.split("|") if n.strip() and n.strip() != "NR"]
        
        for npn in npns:
            nueva_fila = row.copy()
            nueva_fila[col_npn] = npn
            filas_expandidas.append(nueva_fila)

    df_expandido = pd.DataFrame(filas_expandidas)
    logger.info(f"  Filas después de expandir NPNs: {len(df_expandido)}")
    
    # Normalizar NPN: solo dígitos
    df_expandido["NPN_NORMALIZADO"] = df_expandido[col_npn].apply(normalizar_npn)
    
    return df_expandido


def cargar_base_catastral(ruta):
    """
    Carga la base catastral oficial.
    El campo de cruce es 'numero_predial' (equivale a NPN del tabulado).
    """
    logger.info(f"Cargando base catastral: {ruta}")
    logger.info("  (Esto puede tardar unos segundos por el tamaño del archivo...)")
    
    df = pd.read_excel(ruta)
    logger.info(f"  Filas: {len(df)}, Columnas: {len(df.columns)}")
    logger.info(f"  Columnas: {list(df.columns)}")
    
    # El campo de cruce es numero_predial (equivale al NPN del tabulado)
    col_cruce = "numero_predial"
    if col_cruce not in df.columns:
        logger.error(f"No se encontró la columna '{col_cruce}' en la base catastral.")
        return pd.DataFrame()
    
    df["NPN_NORMALIZADO"] = df[col_cruce].apply(normalizar_npn)
    
    return df


# ==============================================================================
# FUNCIONES DE NORMALIZACIÓN
# ==============================================================================

def normalizar_npn(valor):
    """
    Normaliza un NPN: elimina espacios, guiones y deja solo dígitos.
    """
    if pd.isna(valor):
        return ""
    valor_str = str(valor).strip()
    # Solo dígitos
    return re.sub(r'[^0-9]', '', valor_str)


def normalizar_texto(valor):
    """Normaliza texto para comparación: mayúsculas, sin espacios extra."""
    if pd.isna(valor) or str(valor).strip() in ("NR", "nan", "", "None", "N/A"):
        return ""
    return re.sub(r'\s+', ' ', str(valor).upper().strip())


def normalizar_numero(valor):
    """Normaliza valores numéricos: elimina separadores de miles, puntos, etc."""
    if pd.isna(valor) or str(valor).strip() in ("NR", "nan", "", "None", "N/A", "NO REGISTRA"):
        return None
    valor_str = str(valor).strip()
    # Eliminar $, espacios, puntos de miles
    valor_str = re.sub(r'[$\s]', '', valor_str)
    valor_str = valor_str.replace(',', '')  # Quitar comas de miles
    # Si tiene punto como separador de miles (ej: 1.094.188.000)
    if valor_str.count('.') > 1:
        valor_str = valor_str.replace('.', '')
    try:
        return float(valor_str)
    except ValueError:
        return None


def normalizar_fecha(valor):
    """Normaliza fechas a formato comparable."""
    if pd.isna(valor) or str(valor).strip() in ("NR", "nan", "", "None"):
        return ""
    valor_str = str(valor).strip()
    
    # Formato dd/mm/yyyy
    match = re.match(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', valor_str)
    if match:
        return f"{match.group(3)}-{match.group(2).zfill(2)}-{match.group(1).zfill(2)}"
    
    # Formato yyyy-mm-dd (ya normalizado)
    match = re.match(r'(\d{4})-(\d{2})-(\d{2})', valor_str)
    if match:
        return valor_str
    
    return valor_str


# ==============================================================================
# FUNCIÓN PRINCIPAL DE COMPARACIÓN
# ==============================================================================

def extraer_secuencia_predio(npn):
    """
    Extrae la secuencia significativa del NPN para matching flexible.
    El NPN tiene estructura: DDMMMSSSSSSSSSSPPPPPPPPPPPPPPPP
    donde muchos dígitos son ceros de relleno.
    
    Estrategia: extraer los primeros 15 caracteres (depto+muni+sector) como prefijo,
    y la parte significativa del predio (dígitos no-cero centrales).
    """
    if not npn or len(npn) < 15:
        return "", ""
    prefijo = npn[:15]  # Depto + Municipio + parte sector
    # La parte del predio es lo que queda después del prefijo
    resto = npn[15:]
    return prefijo, resto


def buscar_match_flexible(npn_ocr, npns_catastral_set, npns_catastral_dict):
    """
    Busca coincidencia flexible para un NPN extraído por OCR que puede
    tener dígitos faltantes o errores.
    
    Estrategias (en orden de confianza):
    1. Match exacto (si el OCR capturó bien los 30 dígitos)
    2. Match por padding a 30 dígitos (si solo faltan ceros al final)
    3. Match por prefijo + secuencia significativa del predio
    """
    if not npn_ocr:
        return None, "sin_npn"
    
    # Estrategia 1: Match exacto
    if npn_ocr in npns_catastral_set:
        return npn_ocr, "exacto"
    
    # Estrategia 2: Padding con ceros a la derecha hasta 30 dígitos
    if len(npn_ocr) < 30:
        npn_padded = npn_ocr.ljust(30, '0')
        if npn_padded in npns_catastral_set:
            return npn_padded, "padding_derecha"
    
    # Estrategia 3: Buscar por prefijo (primeros 15) + secuencia significativa
    # El OCR puede perder un dígito intermedio, así que buscamos NPNs de la base
    # que compartan prefijo y contengan la secuencia del predio
    if len(npn_ocr) >= 15:
        prefijo_ocr = npn_ocr[:15]
        # Extraer la parte significativa (no-cero) después del prefijo
        resto_ocr = npn_ocr[15:].lstrip('0')
        
        if resto_ocr and len(resto_ocr) >= 4:
            # Buscar en la base NPNs con mismo prefijo que contengan la secuencia
            candidatos = []
            for npn_base in npns_catastral_dict.get(prefijo_ocr, []):
                resto_base = npn_base[15:]
                if resto_ocr[:5] in resto_base:
                    candidatos.append(npn_base)
            
            if len(candidatos) == 1:
                return candidatos[0], "secuencia_unica"
            elif len(candidatos) > 1:
                # Si hay múltiples, intentar con más dígitos de la secuencia
                for npn_base in candidatos:
                    resto_base = npn_base[15:]
                    if resto_ocr in resto_base:
                        return npn_base, "secuencia_completa"
                # Si aún hay ambigüedad, tomar el más similar
                return candidatos[0], "secuencia_ambigua"
    
    return None, "sin_match"


def realizar_cruce(df_tabulado, df_catastral):
    """
    Realiza el cruce entre ambas bases usando NPN con matching flexible
    para compensar errores de OCR en la extracción de dígitos.
    """
    logger.info("=" * 60)
    logger.info("REALIZANDO CRUCE POR NPN (matching flexible)")
    logger.info("=" * 60)
    
    # Preparar estructuras para búsqueda eficiente
    npns_catastral_set = set(df_catastral["NPN_NORMALIZADO"].unique())
    
    # Diccionario por prefijo (primeros 15 dígitos) para búsqueda rápida
    npns_catastral_dict = {}
    for npn in npns_catastral_set:
        if len(npn) >= 15:
            prefijo = npn[:15]
            if prefijo not in npns_catastral_dict:
                npns_catastral_dict[prefijo] = []
            npns_catastral_dict[prefijo].append(npn)
    
    # Buscar match para cada NPN del tabulado
    matches = []
    sin_match = []
    estadisticas_match = {"exacto": 0, "padding_derecha": 0, "secuencia_unica": 0,
                          "secuencia_completa": 0, "secuencia_ambigua": 0, "sin_match": 0, "sin_npn": 0}
    
    npns_tabulado = df_tabulado["NPN_NORMALIZADO"].unique()
    logger.info(f"  NPNs en tabulado (extraídos): {len(npns_tabulado)}")
    logger.info(f"  NPNs en base catastral: {len(npns_catastral_set)}")
    
    mapeo_npn = {}  # OCR -> Base
    for npn_ocr in npns_tabulado:
        npn_base, tipo_match = buscar_match_flexible(npn_ocr, npns_catastral_set, npns_catastral_dict)
        estadisticas_match[tipo_match] += 1
        
        if npn_base:
            mapeo_npn[npn_ocr] = npn_base
            matches.append(npn_ocr)
            logger.info(f"    MATCH ({tipo_match}): {npn_ocr} -> {npn_base}")
        else:
            sin_match.append(npn_ocr)
            logger.warning(f"    SIN MATCH: {npn_ocr}")
    
    logger.info(f"\n  Resultados del matching:")
    for tipo, count in estadisticas_match.items():
        if count > 0:
            logger.info(f"    - {tipo}: {count}")
    logger.info(f"  Total con match: {len(matches)}")
    logger.info(f"  Total sin match: {len(sin_match)}")
    
    # Aplicar mapeo al tabulado
    df_tabulado_mapeado = df_tabulado.copy()
    df_tabulado_mapeado["NPN_MATCH"] = df_tabulado_mapeado["NPN_NORMALIZADO"].map(mapeo_npn)
    
    # Filtrar solo los que tienen match
    df_tabulado_con_match = df_tabulado_mapeado[df_tabulado_mapeado["NPN_MATCH"].notna()].copy()
    
    if df_tabulado_con_match.empty:
        logger.warning("No se encontraron coincidencias.")
        return pd.DataFrame(), set(matches), set(sin_match)
    
    # Realizar merge usando el NPN mapeado
    df_cruce = pd.merge(
        df_tabulado_con_match,
        df_catastral,
        left_on="NPN_MATCH",
        right_on="NPN_NORMALIZADO",
        how="inner",
        suffixes=("_extraido", "_catastral")
    )
    
    logger.info(f"  Registros cruzados (merge): {len(df_cruce)}")
    
    return df_cruce, set(matches), set(sin_match)


def comparar_campos(df_cruce):
    """
    Compara campo por campo los registros cruzados y detecta inconsistencias.
    Retorna un DataFrame con las inconsistencias encontradas.
    """
    logger.info("=" * 60)
    logger.info("COMPARANDO CAMPOS")
    logger.info("=" * 60)
    
    inconsistencias = []
    
    for _, row in df_cruce.iterrows():
        npn = row.get("NPN_MATCH", row.get("NPN_NORMALIZADO_extraido", row.get("NPN_NORMALIZADO", "")))
        archivo = row.get("Nombre del archivo", "N/A")
        
        for col_tabulado, col_catastral, nombre_campo in CAMPOS_COMPARABLES:
            # Obtener valores
            # Después del merge, las columnas pueden tener sufijos si hay conflicto
            val_extraido = row.get(col_tabulado, row.get(f"{col_tabulado}_extraido", ""))
            val_catastral = row.get(col_catastral, row.get(f"{col_catastral}_catastral", ""))
            
            # Comparar según tipo de campo
            es_inconsistente = False
            detalle = ""
            
            if nombre_campo in ("Área Terreno", "Área Construida", "Avalúo"):
                # Comparación numérica
                num_extraido = normalizar_numero(val_extraido)
                num_catastral = normalizar_numero(val_catastral)
                
                if num_extraido is not None and num_catastral is not None:
                    if num_extraido != num_catastral:
                        es_inconsistente = True
                        diferencia = abs(num_extraido - num_catastral)
                        detalle = f"Diferencia: {diferencia:,.2f}"

            elif nombre_campo == "Municipio":
                # El tabulado tiene nombre (ALCALA), la base tiene código (020)
                # Solo reportar para revisión manual, no como inconsistencia automática
                texto_extraido = normalizar_texto(val_extraido)
                texto_catastral = normalizar_texto(val_catastral)
                if texto_extraido and texto_catastral:
                    # Si la base tiene un código numérico y el tabulado un nombre, no es inconsistencia
                    if texto_catastral.isdigit():
                        detalle = f"Código: {texto_catastral} vs Nombre: {texto_extraido}"
                        # No marcar como inconsistente, es diferencia de formato
                    elif texto_extraido != texto_catastral:
                        es_inconsistente = True
                        detalle = "Valores diferentes"
                        
            else:
                # Comparación de texto
                texto_extraido = normalizar_texto(val_extraido)
                texto_catastral = normalizar_texto(val_catastral)
                
                if texto_extraido and texto_catastral:
                    # Verificar si uno contiene al otro (por variaciones OCR)
                    if texto_extraido != texto_catastral:
                        if texto_extraido in texto_catastral or texto_catastral in texto_extraido:
                            detalle = "Coincidencia parcial"
                        else:
                            es_inconsistente = True
                            detalle = "Valores diferentes"
            
            if es_inconsistente:
                inconsistencias.append({
                    "NPN": npn,
                    "Archivo_Origen": archivo,
                    "Campo": nombre_campo,
                    "Valor_Extraido": str(val_extraido)[:200],
                    "Valor_Base_Catastral": str(val_catastral)[:200],
                    "Detalle": detalle
                })
    
    df_inconsistencias = pd.DataFrame(inconsistencias)
    
    # Resumen
    logger.info(f"\n  Total inconsistencias encontradas: {len(df_inconsistencias)}")
    if not df_inconsistencias.empty:
        resumen = df_inconsistencias.groupby("Campo").size().sort_values(ascending=False)
        logger.info("  Inconsistencias por campo:")
        for campo, count in resumen.items():
            logger.info(f"    - {campo}: {count}")
    
    return df_inconsistencias


# ==============================================================================
# GENERACIÓN DE REPORTE
# ==============================================================================

def generar_reporte_por_predio(df_cruce, df_inconsistencias):
    """
    Genera un reporte detallado POR PREDIO donde cada fila es un predio
    y se muestran los valores de ambas fuentes lado a lado con el resultado
    de la comparación.
    """
    filas_reporte = []
    
    for _, row in df_cruce.iterrows():
        npn = row.get("NPN_MATCH", row.get("NPN_NORMALIZADO_extraido", ""))
        archivo = row.get("Nombre del archivo", "N/A")
        
        fila = {
            "NPN_Cruce": npn,
            "Archivo_PDF": archivo,
        }
        
        # Para cada campo comparable, mostrar ambos valores y si coincide o no
        for col_tabulado, col_catastral, nombre_campo in CAMPOS_COMPARABLES:
            val_extraido = row.get(col_tabulado, row.get(f"{col_tabulado}_extraido", ""))
            val_catastral = row.get(col_catastral, row.get(f"{col_catastral}_catastral", ""))
            
            # Determinar resultado de comparación
            resultado = "—"
            
            if nombre_campo in ("Área Terreno", "Área Construida", "Avalúo"):
                num_ext = normalizar_numero(val_extraido)
                num_cat = normalizar_numero(val_catastral)
                if num_ext is not None and num_cat is not None:
                    if num_ext == num_cat:
                        resultado = "✓ COINCIDE"
                    else:
                        resultado = f"✗ DIFERENCIA ({abs(num_ext - num_cat):,.2f})"
                elif num_ext is None and num_cat is None:
                    resultado = "— Sin datos"
                else:
                    resultado = "⚠ Dato faltante"
                    
            elif nombre_campo == "Municipio":
                texto_ext = normalizar_texto(val_extraido)
                texto_cat = normalizar_texto(val_catastral)
                if texto_cat and texto_cat.isdigit():
                    resultado = f"Código: {texto_cat} / Nombre: {texto_ext}"
                elif texto_ext and texto_cat:
                    resultado = "✓ COINCIDE" if texto_ext == texto_cat else "✗ DIFERENTE"
                else:
                    resultado = "— Sin datos"
            else:
                texto_ext = normalizar_texto(val_extraido)
                texto_cat = normalizar_texto(val_catastral)
                if texto_ext and texto_cat:
                    if texto_ext == texto_cat:
                        resultado = "✓ COINCIDE"
                    elif texto_ext in texto_cat or texto_cat in texto_ext:
                        resultado = "≈ PARCIAL"
                    else:
                        resultado = "✗ DIFERENTE"
                elif not texto_ext and not texto_cat:
                    resultado = "— Sin datos"
                else:
                    resultado = "⚠ Dato faltante"
            
            fila[f"{nombre_campo}_Extraido"] = str(val_extraido)[:150] if pd.notna(val_extraido) else ""
            fila[f"{nombre_campo}_BaseCatastral"] = str(val_catastral)[:150] if pd.notna(val_catastral) else ""
            fila[f"{nombre_campo}_Resultado"] = resultado
        
        # Contar inconsistencias de este predio
        if not df_inconsistencias.empty:
            inc_predio = df_inconsistencias[df_inconsistencias["NPN"] == npn]
            fila["Total_Inconsistencias"] = len(inc_predio)
            fila["Campos_Inconsistentes"] = ", ".join(inc_predio["Campo"].tolist()) if len(inc_predio) > 0 else "Ninguno"
        else:
            fila["Total_Inconsistencias"] = 0
            fila["Campos_Inconsistentes"] = "Ninguno"
        
        filas_reporte.append(fila)
    
    return pd.DataFrame(filas_reporte)


def generar_reporte(df_cruce, df_inconsistencias, npns_coinciden, npns_solo_tabulado):
    """
    Genera un reporte Excel con los resultados de la comparación.
    Incluye reporte por predio con comparación campo a campo.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_reporte = os.path.join(REPORTS_DIR, f"reporte_comparacion_{timestamp}.xlsx")
    
    logger.info(f"\nGenerando reporte: {ruta_reporte}")
    
    # Generar reporte por predio
    df_por_predio = generar_reporte_por_predio(df_cruce, df_inconsistencias)
    
    with pd.ExcelWriter(ruta_reporte, engine='openpyxl') as writer:
        # Hoja 1: Resumen general
        resumen = pd.DataFrame({
            "Métrica": [
                "Total NPNs en tabulado (extraídos de PDFs)",
                "Total NPNs en base catastral",
                "NPNs con coincidencia (cruce exitoso)",
                "NPNs sin coincidencia en base catastral",
                "Total registros cruzados",
                "Total inconsistencias detectadas",
                "Predios sin inconsistencias",
                "Predios con inconsistencias",
                "Fecha de análisis"
            ],
            "Valor": [
                len(npns_coinciden) + len(npns_solo_tabulado),
                "576,063",
                len(npns_coinciden),
                len(npns_solo_tabulado),
                len(df_cruce),
                len(df_inconsistencias),
                len(df_por_predio[df_por_predio["Total_Inconsistencias"] == 0]),
                len(df_por_predio[df_por_predio["Total_Inconsistencias"] > 0]),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        })
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        
        # Hoja 2: REPORTE POR PREDIO (principal)
        df_por_predio.to_excel(writer, sheet_name="Reporte_Por_Predio", index=False)
        
        # Hoja 3: Inconsistencias detalladas
        if not df_inconsistencias.empty:
            df_inconsistencias.to_excel(writer, sheet_name="Inconsistencias", index=False)
        else:
            pd.DataFrame({"Mensaje": ["No se encontraron inconsistencias"]}).to_excel(
                writer, sheet_name="Inconsistencias", index=False
            )
        
        # Hoja 4: NPNs sin coincidencia
        if npns_solo_tabulado:
            df_sin_match = pd.DataFrame({
                "NPN_Sin_Coincidencia": list(npns_solo_tabulado),
                "Posible_Causa": ["Error OCR en dígitos del NPN"] * len(npns_solo_tabulado)
            })
            df_sin_match.to_excel(writer, sheet_name="NPNs_Sin_Match", index=False)
    
    logger.info(f"Reporte guardado exitosamente: {ruta_reporte}")
    logger.info(f"  - Hoja 'Reporte_Por_Predio': {len(df_por_predio)} predios comparados")
    return ruta_reporte


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================

def main():
    logger.info("=" * 60)
    logger.info("ETAPA 2: COMPARACIÓN DE BASES CATASTRALES")
    logger.info("=" * 60)
    logger.info(f"Archivo resultados: {ARCHIVO_RESULTADOS}")
    logger.info(f"Archivo catastral:  {ARCHIVO_CATASTRAL}")
    logger.info("")
    
    # 1. Cargar datos
    df_tabulado = cargar_tabulado(ARCHIVO_RESULTADOS)
    if df_tabulado.empty:
        logger.error("No se pudieron cargar los resultados. Abortando.")
        return
    
    df_catastral = cargar_base_catastral(ARCHIVO_CATASTRAL)
    if df_catastral.empty:
        logger.error("No se pudo cargar la base catastral. Abortando.")
        return
    
    # 2. Realizar cruce por NPN
    df_cruce, npns_coinciden, npns_solo_tabulado = realizar_cruce(df_tabulado, df_catastral)
    
    if df_cruce.empty:
        logger.warning("No se encontraron coincidencias entre ambas bases.")
        logger.warning("Verifica que los NPNs tengan el mismo formato en ambos archivos.")
        return
    
    # 3. Comparar campos
    df_inconsistencias = comparar_campos(df_cruce)
    
    # 4. Generar reporte
    ruta_reporte = generar_reporte(df_cruce, df_inconsistencias, npns_coinciden, npns_solo_tabulado)
    
    # 5. Resumen final
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESUMEN FINAL")
    logger.info("=" * 60)
    logger.info(f"  Cruces exitosos: {len(npns_coinciden)} NPNs")
    logger.info(f"  Inconsistencias: {len(df_inconsistencias)} campos con diferencias")
    logger.info(f"  Reporte: {ruta_reporte}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

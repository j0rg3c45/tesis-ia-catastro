"""
comparar_pdf_vs_catastro.py

Compara el tabulado extraido de los PDFs (una fila por predio) contra la base
catastral consolidada (una fila por NPN), cruzando por Numero Predial Nacional.

Entradas:
- Tabulado PDFs:  data/_prueba_100/structured/TABULADO_RESULTADOS.xlsx  (o el que se pase)
- Base consolidada: data/comparacion/BASE_CATASTRAL_CONSOLIDADA_<corte>.csv

Salida:
- data/comparacion/COMPARACION_PDF_VS_CATASTRO_<timestamp>.xlsx con 4 hojas:
    1. Resumen          -> metricas de cruce y % coincidencia por campo
    2. Detalle_Por_Predio -> valores PDF vs base lado a lado + resultado
    3. Inconsistencias  -> solo los campos que difieren
    4. NPN_Sin_Match    -> NPN de los PDFs que no estan en la base

Cruce por NPN (llave base de todos los cruces). 'destino' es el termino oficial.

Autor: Proyecto de Tesis - Maestria en IA y Ciencia de Datos
"""

import os
import re
import argparse
import logging
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP_DIR = os.path.join(BASE_DIR, "data", "comparacion")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Mapeo de campos a comparar: (nombre_legible, col_pdf, col_base, tipo)
# tipo: 'texto' | 'numero' | 'fecha' | 'destino'
CAMPOS = [
    ("Codigo Homologado", "Código Homologado:", "codigo_homologado", "texto"),
    ("Matricula Inmobiliaria", "Número de matrícula inmobiliaria:", "matricula_inmobiliaria", "texto"),
    ("Numero Predial Anterior", "Número predial:", "numero_predial_anterior", "texto"),
    ("Propietario", "Propietario:", "nombre", "propietario"),
    ("Documento", "Documento de identificación:", "documento_completo", "texto"),
    ("Direccion", "Dirección:", "direccion", "texto"),
    ("Area Terreno", "Área predio:", "area_terreno", "numero"),
    ("Area Construida", "Área construida:", "area_construida", "numero"),
    ("Avaluo", "Avalúo:", "avaluo", "numero"),
    ("Destino", "Destinación economica:", "destino", "destino"),
    ("Vigencia", "Vigencia Fiscal:", "vigencia", "fecha"),
]

VACIOS = {"", "NR", "NAN", "NONE", "N/A", "NO REGISTRA"}


# ------------------------------------------------------------------ utilidades
def solo_digitos(v):
    if pd.isna(v):
        return ""
    return re.sub(r"\D", "", str(v))


def norm_texto(v):
    if pd.isna(v):
        return ""
    s = str(v).upper().strip()
    if s in VACIOS:
        return ""
    # quitar tildes
    for a, b in [("Á", "A"), ("É", "E"), ("Í", "I"), ("Ó", "O"), ("Ú", "U"), ("Ñ", "N")]:
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def norm_numero(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s.upper() in VACIOS:
        return None
    s = re.sub(r"[$\s]", "", s)
    # separador de miles con punto (1.094.188) o coma; decimal el ultimo
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")  # coma decimal
    else:
        s = s.replace(",", "")
        if s.count(".") > 1:
            s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def norm_fecha(v):
    """Devuelve YYYY-MM-DD si puede, si no el texto normalizado."""
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if s.upper() in VACIOS:
        return ""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.match(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    return norm_texto(s)


def comparar_valor(tipo, val_pdf, val_base):
    """
    Devuelve (resultado, coincide) donde resultado es una etiqueta legible.
    coincide: True/False/None (None = sin datos para comparar).
    """
    if tipo == "numero":
        a, b = norm_numero(val_pdf), norm_numero(val_base)
        if a is None and b is None:
            return "— sin datos", None
        if a is None or b is None:
            return "⚠ dato faltante", None
        if abs(a - b) < 0.01:
            return "✓ coincide", True
        return f"✗ difiere (Δ {abs(a-b):,.2f})", False

    if tipo == "fecha":
        a, b = norm_fecha(val_pdf), norm_fecha(val_base)
        if not a and not b:
            return "— sin datos", None
        if not a or not b:
            return "⚠ dato faltante", None
        return ("✓ coincide", True) if a == b else ("✗ difiere", False)

    if tipo == "propietario":
        # Comparacion insensible al orden y TOLERANTE a ruido de OCR.
        # Cada propietario se descompone en un conjunto de tokens significativos
        # (palabras de >=3 letras, se descartan tokens de 1-2 chars que suelen ser
        # ruido de OCR como 'E', 'EA', '_'). Dos nombres se consideran "el mismo"
        # si comparten una fraccion alta de tokens (>=70% del mas corto).
        STOP = {"DE", "LA", "EL", "LOS", "LAS", "DEL", "Y"}

        def nombre_a_tokens(nombre):
            n = norm_texto(nombre)
            n = re.sub(r"[-_.,:;/]", " ", n)
            n = re.sub(r"\s+", " ", n).strip()
            toks = set()
            for t in n.split():
                if len(t) >= 3 and t not in STOP:  # descartar ruido corto y conectores
                    toks.add(t)
            return toks

        def lista_personas(v):
            partes = re.split(r"[;]", str(v)) if pd.notna(v) else []
            personas = []
            for p in partes:
                toks = nombre_a_tokens(p)
                if toks:
                    personas.append(toks)
            return personas

        def mismo_nombre(t1, t2):
            if not t1 or not t2:
                return False
            inter = t1 & t2
            menor = min(len(t1), len(t2))
            return len(inter) / menor >= 0.7  # 70% de tokens compartidos

        A = lista_personas(val_pdf)
        B = lista_personas(val_base)
        if not A and not B:
            return "— sin datos", None
        if not A or not B:
            return "⚠ dato faltante", None

        # Emparejar cada persona de A con alguna de B (match difuso por tokens)
        emparejados = 0
        usados = set()
        for ta in A:
            for j, tb in enumerate(B):
                if j in usados:
                    continue
                if mismo_nombre(ta, tb):
                    emparejados += 1
                    usados.add(j)
                    break

        total = max(len(A), len(B))
        if emparejados == total:
            return "✓ coincide", True
        if emparejados >= 1:
            # Coincidencia parcial: comparten al menos un propietario
            if emparejados == min(len(A), len(B)):
                return "✓ coincide (subset)", True
            return f"≈ parcial ({emparejados}/{total})", True
        return "✗ difiere", False

    # texto y destino
    a, b = norm_texto(val_pdf), norm_texto(val_base)
    if not a and not b:
        return "— sin datos", None
    if not a or not b:
        return "⚠ dato faltante", None
    if a == b:
        return "✓ coincide", True
    # coincidencia parcial: uno contenido en el otro (util para nombres/direcciones)
    if a in b or b in a:
        return "≈ parcial", True
    return "✗ difiere", False


# ------------------------------------------------------------------ carga
def cargar_pdf(ruta):
    logger.info(f"Cargando tabulado PDFs: {ruta}")
    df = pd.read_excel(ruta, dtype=str)
    col_npn = "Número Predial Nacional:"
    # Expandir NPN multiples separados por | (por si acaso) y normalizar
    filas = []
    for _, row in df.iterrows():
        raw = str(row.get(col_npn, "")).strip()
        for npn in re.split(r"[|]", raw):
            npn_n = solo_digitos(npn)
            if len(npn_n) == 30:
                nueva = row.copy()
                nueva["_NPN"] = npn_n
                filas.append(nueva)
    out = pd.DataFrame(filas)
    logger.info(f"  Predios PDF con NPN valido (30 dig): {len(out)}")
    return out


def cargar_base(ruta):
    logger.info(f"Cargando base consolidada: {ruta}")
    df = pd.read_csv(ruta, dtype=str)
    df["_NPN"] = df["numero_predial_nacional"].apply(solo_digitos)
    df = df[df["_NPN"].str.len() == 30].drop_duplicates(subset=["_NPN"], keep="first")
    logger.info(f"  Predios base con NPN valido: {len(df)}")
    return df


# ------------------------------------------------------------------ comparacion
def comparar(df_pdf, df_base):
    base_idx = df_base.set_index("_NPN")
    detalle = []
    inconsistencias = []
    sin_match = []
    # contadores por campo
    stats = {c[0]: {"coincide": 0, "difiere": 0, "sin_datos": 0, "faltante": 0} for c in CAMPOS}

    npn_base_set = set(base_idx.index)

    for _, row in df_pdf.iterrows():
        npn = row["_NPN"]
        archivo = row.get("Nombre del archivo", "")
        if npn not in npn_base_set:
            sin_match.append({"NPN": npn, "Archivo_PDF": archivo})
            continue

        brow = base_idx.loc[npn]
        if isinstance(brow, pd.DataFrame):
            brow = brow.iloc[0]

        fila = {"NPN": npn, "Archivo_PDF": archivo}
        for legible, col_pdf, col_base, tipo in CAMPOS:
            vp = row.get(col_pdf, "")
            vb = brow.get(col_base, "")
            resultado, coincide = comparar_valor(tipo, vp, vb)
            fila[f"{legible} (PDF)"] = "" if pd.isna(vp) else str(vp)
            fila[f"{legible} (Base)"] = "" if pd.isna(vb) else str(vb)
            fila[f"{legible} [=]"] = resultado

            if coincide is True:
                stats[legible]["coincide"] += 1
            elif coincide is False:
                stats[legible]["difiere"] += 1
                inconsistencias.append({
                    "NPN": npn, "Archivo_PDF": archivo, "Campo": legible,
                    "Valor_PDF": str(vp), "Valor_Base": str(vb), "Resultado": resultado,
                })
            elif "faltante" in resultado:
                stats[legible]["faltante"] += 1
            else:
                stats[legible]["sin_datos"] += 1

        detalle.append(fila)

    return pd.DataFrame(detalle), pd.DataFrame(inconsistencias), pd.DataFrame(sin_match), stats


def construir_resumen(df_pdf, df_base, df_detalle, df_incons, df_sinmatch, stats):
    total_pdf = df_pdf["_NPN"].nunique()
    con_match = len(df_detalle)
    resumen = [
        ("Predios PDF (NPN unicos)", total_pdf),
        ("Predios en base consolidada", df_base["_NPN"].nunique()),
        ("NPN de PDF con match en base", con_match),
        ("NPN de PDF SIN match", len(df_sinmatch)),
        ("% cruce", f"{100*con_match/total_pdf:.1f}%" if total_pdf else "0%"),
        ("Total inconsistencias (campos)", len(df_incons)),
        ("", ""),
        ("--- Coincidencia por campo (sobre NPN con match) ---", ""),
    ]
    for legible, _, _, _ in CAMPOS:
        s = stats[legible]
        comparables = s["coincide"] + s["difiere"]
        pct = f"{100*s['coincide']/comparables:.0f}%" if comparables else "n/a"
        resumen.append((f"{legible}: coincide/comparables",
                        f"{s['coincide']}/{comparables} ({pct}) | difiere={s['difiere']} | faltante={s['faltante']} | sin_datos={s['sin_datos']}"))
    return pd.DataFrame(resumen, columns=["Metrica", "Valor"])


def main():
    parser = argparse.ArgumentParser(description="Compara tabulado PDFs vs base catastral consolidada por NPN.")
    parser.add_argument("--pdf", default=os.path.join(BASE_DIR, "data", "_prueba_100", "structured", "TABULADO_RESULTADOS.xlsx"))
    parser.add_argument("--base", default=os.path.join(COMP_DIR, "BASE_CATASTRAL_CONSOLIDADA_20260831.csv"))
    args = parser.parse_args()

    df_pdf = cargar_pdf(args.pdf)
    df_base = cargar_base(args.base)

    logger.info("Comparando por NPN...")
    df_detalle, df_incons, df_sinmatch, stats = comparar(df_pdf, df_base)
    df_resumen = construir_resumen(df_pdf, df_base, df_detalle, df_incons, df_sinmatch, stats)

    os.makedirs(COMP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(COMP_DIR, f"COMPARACION_PDF_VS_CATASTRO_{ts}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df_resumen.to_excel(w, sheet_name="Resumen", index=False)
        df_detalle.to_excel(w, sheet_name="Detalle_Por_Predio", index=False)
        (df_incons if not df_incons.empty else pd.DataFrame({"Mensaje": ["Sin inconsistencias"]})
         ).to_excel(w, sheet_name="Inconsistencias", index=False)
        (df_sinmatch if not df_sinmatch.empty else pd.DataFrame({"Mensaje": ["Todos los NPN cruzaron"]})
         ).to_excel(w, sheet_name="NPN_Sin_Match", index=False)

    logger.info("=" * 60)
    logger.info(f"Comparacion completada.")
    logger.info(f"  Predios PDF: {df_pdf['_NPN'].nunique()} | con match: {len(df_detalle)} | sin match: {len(df_sinmatch)}")
    logger.info(f"  Inconsistencias: {len(df_incons)}")
    logger.info(f"  Reporte: {out}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

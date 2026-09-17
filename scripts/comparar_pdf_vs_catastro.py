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
import json
import argparse
import logging
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP_DIR = os.path.join(BASE_DIR, "data", "comparacion")
METRICAS_DIR = os.path.join(BASE_DIR, "data", "reports", "metricas")

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
            return "[-] sin datos", None
        if a is None or b is None:
            return "[!] dato faltante", None
        if abs(a - b) < 0.01:
            return "[OK] coincide", True
        return f"[X] difiere (dif {abs(a-b):,.2f})", False

    if tipo == "fecha":
        a, b = norm_fecha(val_pdf), norm_fecha(val_base)
        if not a and not b:
            return "[-] sin datos", None
        if not a or not b:
            return "[!] dato faltante", None
        return ("[OK] coincide", True) if a == b else ("[X] difiere", False)

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
            return "[-] sin datos", None
        if not A or not B:
            return "[!] dato faltante", None

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
            return "[OK] coincide", True
        if emparejados >= 1:
            # Coincidencia parcial: comparten al menos un propietario
            if emparejados == min(len(A), len(B)):
                return "[OK] coincide (subset)", True
            return f"[~] parcial ({emparejados}/{total})", True
        return "[X] difiere", False

    # texto y destino
    a, b = norm_texto(val_pdf), norm_texto(val_base)
    if not a and not b:
        return "[-] sin datos", None
    if not a or not b:
        return "[!] dato faltante", None
    if a == b:
        return "[OK] coincide", True
    # coincidencia parcial: uno contenido en el otro (util para nombres/direcciones)
    if a in b or b in a:
        return "[~] parcial", True
    return "[X] difiere", False


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


def generar_reporte_txt(ruta_txt, df_pdf, df_base, df_detalle, df_incons,
                        df_sinmatch, stats, corte):
    """
    Genera el REPORTE DE VALIDACION en formato texto plano (.txt), conforme al
    Objetivo Especifico 4: organiza los campos identificados, documenta las
    inconsistencias detectadas y aporta informacion contextual para su
    verificacion y correccion.

    Estructura del archivo:
      1. Encabezado y metadatos de la ejecucion
      2. Resumen global del cruce
      3. Concordancia por campo
      4. Detalle de inconsistencias por predio (agrupado por NPN)
      5. Predios sin correspondencia en la base catastral
    """
    total_pdf = df_pdf["_NPN"].nunique()
    con_match = len(df_detalle)
    sin_match = len(df_sinmatch)
    W = 78  # ancho de linea

    def linea(c="="):
        return c * W

    lineas = []
    ap = lineas.append

    # 1. ENCABEZADO
    ap(linea("="))
    ap("REPORTE DE VALIDACION DOCUMENTAL CATASTRAL".center(W))
    ap("Contraste: resoluciones (PDF) vs base de datos catastral".center(W))
    ap(linea("="))
    ap(f"Fecha de generacion : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ap(f"Corte base catastral: {corte}")
    ap(f"Llave de cruce      : Numero Predial Nacional (NPN, 30 digitos)")
    ap("")

    # 2. RESUMEN GLOBAL
    ap(linea("-"))
    ap("1. RESUMEN GLOBAL")
    ap(linea("-"))
    pct_cruce = f"{100*con_match/total_pdf:.1f}%" if total_pdf else "0%"
    ap(f"  Predios extraidos de PDF (NPN unicos) : {total_pdf}")
    ap(f"  Predios en base catastral consolidada : {df_base['_NPN'].nunique()}")
    ap(f"  Predios con correspondencia (cruce)   : {con_match} ({pct_cruce})")
    ap(f"  Predios SIN correspondencia en base   : {sin_match}")
    ap(f"  Total de campos inconsistentes        : {len(df_incons)}")
    ap("")

    # 3. CONCORDANCIA POR CAMPO
    ap(linea("-"))
    ap("2. CONCORDANCIA POR CAMPO (sobre predios con cruce)")
    ap(linea("-"))
    ap(f"  {'Campo':<26}{'Coincide':>10}{'Compara':>9}{'%':>6}{'Difiere':>9}{'Faltan':>8}")
    for legible, _, _, _ in CAMPOS:
        s = stats[legible]
        comparables = s["coincide"] + s["difiere"]
        pct = f"{100*s['coincide']/comparables:.0f}%" if comparables else "n/a"
        ap(f"  {legible:<26}{s['coincide']:>10}{comparables:>9}{pct:>6}{s['difiere']:>9}{s['faltante']:>8}")
    ap("")

    # 4. DETALLE DE INCONSISTENCIAS POR PREDIO
    ap(linea("-"))
    ap("3. INCONSISTENCIAS DETECTADAS (por predio)")
    ap(linea("-"))
    if df_incons.empty:
        ap("  No se detectaron inconsistencias en los predios cruzados.")
    else:
        # Agrupar por NPN para dar contexto por predio
        for npn, grupo in df_incons.groupby("NPN", sort=False):
            archivo = grupo.iloc[0]["Archivo_PDF"]
            ap(f"  NPN: {npn}")
            ap(f"  Archivo PDF: {archivo}")
            ap(f"  Campos inconsistentes: {len(grupo)}")
            for _, r in grupo.iterrows():
                ap(f"    - {r['Campo']}:")
                ap(f"        PDF  : {str(r['Valor_PDF'])[:120]}")
                ap(f"        Base : {str(r['Valor_Base'])[:120]}")
                ap(f"        Estado: {r['Resultado']}")
            ap("  " + linea("-")[:W-2])
    ap("")

    # 5. PREDIOS SIN MATCH
    ap(linea("-"))
    ap("4. PREDIOS SIN CORRESPONDENCIA EN LA BASE CATASTRAL")
    ap(linea("-"))
    if df_sinmatch.empty:
        ap("  Todos los predios extraidos cruzaron con la base.")
    else:
        ap("  (Posibles causas: error de OCR en digitos del NPN, o predio de")
        ap("   un municipio/corte distinto al de la base consolidada.)")
        ap("")
        ap(f"  {'NPN':<34} Archivo PDF")
        for _, r in df_sinmatch.iterrows():
            ap(f"  {str(r['NPN']):<34} {r['Archivo_PDF']}")
    ap("")
    ap(linea("="))
    ap("FIN DEL REPORTE".center(W))
    ap(linea("="))

    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    return ruta_txt


def generar_metricas_cruce(df_pdf, df_base, df_detalle, df_sinmatch, stats, corte, ts):
    """
    Genera la SECCION SEPARADA de metricas del cruce (etapa de contraste):
    - tasa de cruce por NPN
    - matriz de concordancia porcentual campo por campo frente a la base

    Se escribe como archivo independiente (JSON + TXT) en data/reports/metricas/,
    sin mezclarse con el resumen de metricas de extraccion.
    """
    os.makedirs(METRICAS_DIR, exist_ok=True)
    total_pdf = int(df_pdf["_NPN"].nunique())
    con_match = int(len(df_detalle))
    tasa = round(con_match / total_pdf, 4) if total_pdf else 0.0

    concordancia = {}
    for legible, _, _, _ in CAMPOS:
        s = stats[legible]
        comparables = s["coincide"] + s["difiere"]
        concordancia[legible] = {
            "coincide": s["coincide"],
            "compara": comparables,
            "pct_concordancia": round(s["coincide"] / comparables, 4) if comparables else None,
            "difiere": s["difiere"],
            "faltante": s["faltante"],
            "sin_datos": s["sin_datos"],
        }

    resumen = {
        "etapa": "contraste_pdf_vs_catastro",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "corte_base_catastral": corte,
        "cruce_npn": {
            "npn_pdf_unicos": total_pdf,
            "npn_base_unicos": int(df_base["_NPN"].nunique()),
            "npn_con_match": con_match,
            "npn_sin_match": int(len(df_sinmatch)),
            "tasa_cruce": tasa,
        },
        "matriz_concordancia_por_campo": concordancia,
    }

    json_path = os.path.join(METRICAS_DIR, f"metricas_cruce_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    # Version .txt legible
    W = 70
    L = ["=" * W, "METRICAS DEL CRUCE PDF vs BASE CATASTRAL".center(W), "=" * W]
    L.append(f"Fecha              : {resumen['timestamp']}")
    L.append(f"Corte catastral    : {corte}")
    c = resumen["cruce_npn"]
    L.append(f"NPN de PDF (unicos): {c['npn_pdf_unicos']}")
    L.append(f"NPN con match      : {c['npn_con_match']}")
    L.append(f"NPN sin match      : {c['npn_sin_match']}")
    L.append(f"Tasa de cruce      : {c['tasa_cruce']*100:.1f}%")
    L.append("")
    L.append("-" * W)
    L.append("MATRIZ DE CONCORDANCIA POR CAMPO")
    L.append("-" * W)
    L.append(f"  {'Campo':<26}{'Coincide':>9}{'Compara':>9}{'%':>7}")
    for campo, m in concordancia.items():
        pct = f"{m['pct_concordancia']*100:.0f}%" if m["pct_concordancia"] is not None else "n/a"
        L.append(f"  {campo:<26}{m['coincide']:>9}{m['compara']:>9}{pct:>7}")
    L.append("=" * W)
    txt_path = os.path.join(METRICAS_DIR, f"metricas_cruce_{ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return json_path, txt_path


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

    # Reporte en TEXTO PLANO (.txt) - Objetivo Especifico 4
    m_corte = re.search(r"(\d{8})", os.path.basename(args.base))
    corte = m_corte.group(1) if m_corte else "N/D"
    out_txt = os.path.join(COMP_DIR, f"REPORTE_VALIDACION_{ts}.txt")
    generar_reporte_txt(out_txt, df_pdf, df_base, df_detalle, df_incons,
                        df_sinmatch, stats, corte)

    # Metricas del cruce como SECCION SEPARADA (matriz de concordancia + tasa cruce)
    m_json, m_txt = generar_metricas_cruce(df_pdf, df_base, df_detalle,
                                           df_sinmatch, stats, corte, ts)

    logger.info("=" * 60)
    logger.info(f"Comparacion completada.")
    logger.info(f"  Predios PDF: {df_pdf['_NPN'].nunique()} | con match: {len(df_detalle)} | sin match: {len(df_sinmatch)}")
    logger.info(f"  Inconsistencias: {len(df_incons)}")
    logger.info(f"  Reporte Excel : {out}")
    logger.info(f"  Reporte TXT   : {out_txt}")
    logger.info(f"  Metricas cruce: {m_json}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

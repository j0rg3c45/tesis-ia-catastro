"""
consolidar_base_catastral.py

Consolida los 3 registros (R1, R2, R3) de un corte de la base catastral en
formato Contraloria (delimitado por '|') en UN solo archivo con una fila por
Numero Predial Nacional (NPN), alineado a las columnas del tabulado extraido
de los PDFs, para poder compararlos por NPN.

Corte por defecto: 20260831 (el mas actualizado).

Diseno:
- R1 es la base (propietario, documento, direccion, area, avaluo, destino,
  vigencia, codigo homologado, numero predial anterior).
- Un NPN puede tener varias filas en R1 (predios multi-propietario): los
  campos de persona (nombre, documento) se CONCATENAN con '; '. Los campos del
  predio (area, avaluo, destino, direccion, vigencia, etc.) son iguales entre
  las filas del mismo NPN, se toma el primero.
- R2 aporta la matricula_inmobiliaria (unica columna que no esta en R1).
- R3 (caracteristicas fisicas) no tiene equivalente en el tabulado de PDFs; se
  deja fuera de la consolidacion para el cruce.

Salida: CSV en data/comparacion/BASE_CATASTRAL_CONSOLIDADA_<corte>.csv

Autor: Proyecto de Tesis - Maestria en IA y Ciencia de Datos
"""

import os
import argparse
import logging
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def cargar_diccionario_destino():
    """
    Carga el diccionario de destinacion economica (codigo -> descripcion) desde
    data/contexto_catastral/general_destinacion_economica.txt. Devuelve un dict
    {codigo: descripcion}. Si el archivo no existe, devuelve dict vacio.
    """
    ruta = os.path.join(BASE_DIR, "data", "contexto_catastral", "general_destinacion_economica.txt")
    if not os.path.exists(ruta):
        logger.warning("No se encontro el diccionario de destinacion economica.")
        return {}
    # Formato con comillas simples y columnas: id|codigo|descripcion|destino|...
    # OJO: pese a los nombres, la columna 'destino' contiene el CODIGO de letra
    # (A, D, S, ...) que aparece en R1.destino_economico, y la columna 'codigo'
    # contiene el NOMBRE legible (Habitacional, Agropecuario, ...). La columna
    # 'descripcion' trae notas de fuente (LAMD, CATASTRO REGISTRO) o vacio.
    # El archivo esta guardado en UTF-8 pero con mojibake si se lee como latin1.
    # Se lee como latin1 y luego se re-decodifica para recuperar tildes/enes.
    df = pd.read_csv(ruta, sep='|', dtype=str, encoding='latin1', quotechar="'")
    df.columns = [c.strip().strip("'") for c in df.columns]

    def _fix_mojibake(s):
        if not isinstance(s, str):
            return s
        try:
            return s.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s

    mapeo = {}
    for _, row in df.iterrows():
        cod = str(row.get("destino", "")).strip().strip("'")
        nombre = _fix_mojibake(str(row.get("codigo", "")).strip().strip("'"))
        if cod and nombre and nombre.lower() != "nan":
            mapeo[cod] = nombre
    logger.info(f"Diccionario de destinacion economica: {len(mapeo)} codigos.")
    return mapeo


def cargar_r1(ruta):
    logger.info(f"Cargando R1: {ruta}")
    df = pd.read_csv(ruta, sep='|', dtype=str, encoding='latin1', on_bad_lines='skip')
    logger.info(f"  R1 filas: {len(df)}")
    return df


def cargar_r2(ruta):
    logger.info(f"Cargando R2: {ruta}")
    df = pd.read_csv(ruta, sep='|', dtype=str, encoding='latin1', on_bad_lines='skip')
    logger.info(f"  R2 filas: {len(df)}")
    return df


def _agg_concat_unicos(df, col_npn, columna):
    """
    Agregacion vectorizada: por cada NPN concatena con '; ' los valores no
    vacios y unicos de 'columna', preservando el orden de aparicion.
    Mucho mas rapido que groupby con funcion Python sobre cientos de miles de filas.
    """
    tmp = df[[col_npn, columna]].copy()
    tmp[columna] = tmp[columna].fillna("").astype(str).str.strip()
    tmp = tmp[tmp[columna] != ""]
    tmp = tmp.drop_duplicates(subset=[col_npn, columna])  # unicos por NPN
    out = tmp.groupby(col_npn)[columna].agg("; ".join)
    return out


def consolidar(r1, r2, mapeo_destino=None):
    """Consolida R1 (+ matricula de R2) en una fila por NPN (vectorizado)."""
    col_npn = "numero_predial_nacional"
    r1 = r1.copy()

    # Documento completo: tipo + numero (ej. 'N 890114335')
    r1["documento_completo"] = (r1.get("tipo_documento", pd.Series("", index=r1.index)).fillna("").str.strip()
                                + " "
                                + r1.get("numero_documento", pd.Series("", index=r1.index)).fillna("").str.strip()
                                ).str.strip()

    logger.info("Agrupando R1 por NPN (una fila por predio, vectorizado)...")
    # Base: primer registro por NPN (los campos del predio son iguales entre filas del mismo NPN)
    base = r1.drop_duplicates(subset=[col_npn], keep="first").set_index(col_npn)

    # Campos de persona: concatenar multi-propietario / multi-documento
    nombres = _agg_concat_unicos(r1, col_npn, "nombre")
    docs = _agg_concat_unicos(r1, col_npn, "documento_completo")
    base["nombre"] = nombres
    base["documento_completo"] = docs

    base = base.reset_index()
    logger.info(f"  NPN unicos en R1: {len(base)}")

    # Matricula desde R2 (una por NPN, concatenada si hay varias)
    if not r2.empty and "matricula_inmobiliaria" in r2.columns and col_npn in r2.columns:
        logger.info("Agrupando matricula de R2 por NPN...")
        mat = _agg_concat_unicos(r2, col_npn, "matricula_inmobiliaria").rename("matricula_inmobiliaria")
        base = base.merge(mat, on=col_npn, how="left")
        logger.info(f"  NPN con matricula en R2: {base['matricula_inmobiliaria'].notna().sum()}")
    else:
        base["matricula_inmobiliaria"] = ""

    # Destino: la columna 'destino_economico' del R1 trae la LETRA (codigo).
    # Se renombra a 'destino_codigo' y se agrega 'destino' con el NOMBRE legible
    # (Habitacional, Agropecuario, ...). 'destino' es el termino oficial del
    # proyecto para el destino de los predios.
    if "destino_economico" in base.columns:
        mapeo_destino = mapeo_destino or {}
        base = base.rename(columns={"destino_economico": "destino_codigo"})
        base["destino"] = base["destino_codigo"].map(mapeo_destino).fillna("")

    return base


def alinear_a_tabulado(base):
    """
    Renombra/ordena las columnas del consolidado para que sean claras y
    faciles de mapear contra el tabulado de PDFs. Se conservan los nombres
    de la base (snake_case) mas la matricula.
    """
    orden = [
        "numero_predial_nacional",
        "codigo_homologado",
        "numero_predial_anterior",
        "matricula_inmobiliaria",
        "departamento",
        "municipio",
        "nombre",
        "documento_completo",
        "direccion",
        "destino_codigo",
        "destino",
        "area_terreno",
        "area_construida",
        "avaluo",
        "vigencia",
    ]
    cols = [c for c in orden if c in base.columns]
    resto = [c for c in base.columns if c not in cols]
    return base[cols + resto]


def main():
    parser = argparse.ArgumentParser(description="Consolida la base catastral (R1+R2) por NPN.")
    parser.add_argument("--corte", default="20260831", help="Nombre de la carpeta del corte en data/validation")
    parser.add_argument("--xlsx", action="store_true", help="Exportar tambien a .xlsx (mas lento)")
    args = parser.parse_args()

    corte = args.corte
    carpeta = os.path.join(BASE_DIR, "data", "validation", corte)
    r1_path = os.path.join(carpeta, f"1_REGISTRO_CONTRALORIA_R1_{corte}.txt")
    r2_path = os.path.join(carpeta, f"2_REGISTRO_CONTRALORIA_R2_{corte}.txt")

    if not os.path.exists(r1_path):
        logger.error(f"No se encontro R1: {r1_path}")
        return
    if not os.path.exists(r2_path):
        logger.warning(f"No se encontro R2: {r2_path} (se continuara sin matricula)")

    mapeo_destino = cargar_diccionario_destino()
    r1 = cargar_r1(r1_path)
    r2 = cargar_r2(r2_path) if os.path.exists(r2_path) else pd.DataFrame()

    base = consolidar(r1, r2, mapeo_destino)
    base = alinear_a_tabulado(base)

    out_dir = os.path.join(BASE_DIR, "data", "comparacion")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"BASE_CATASTRAL_CONSOLIDADA_{corte}.csv")
    base.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"CSV guardado: {out_csv}")

    # Tambien exportar a xlsx (opcional; puede tardar por el volumen)
    if args.xlsx:
        out_xlsx = os.path.join(out_dir, f"BASE_CATASTRAL_CONSOLIDADA_{corte}.xlsx")
        base.to_excel(out_xlsx, index=False)
        logger.info(f"XLSX guardado: {out_xlsx}")

    logger.info("=" * 60)
    logger.info(f"Consolidacion completada: {len(base)} predios (1 fila por NPN)")
    logger.info(f"Columnas: {list(base.columns)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

"""
metrics.py

Modulo de auditoria y captura de metricas (telemetria) del pipeline de
extraccion de resoluciones catastrales.

Diseno pensado para ThreadPoolExecutor: cada documento produce un
RegistroMetrica (dataclass) de forma aislada dentro de su hilo; el hilo
principal los acumula en un MetricasTracker (que no se toca desde los hilos
worker, por lo que es seguro por diseno).

Salidas (en data/reports/metricas/):
  - metricas_proceso_<timestamp>.csv : una fila por PDF procesado
  - resumen_metricas_tesis.json      : consolidado global de la corrida
  - resumen_metricas_tesis.txt       : version legible del consolidado

Autor: Proyecto de Tesis - Maestria en IA y Ciencia de Datos
"""

import os
import csv
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional


# Estados posibles del procesamiento de un documento
ESTADO_OK = "OK"                 # se extrajo al menos un predio valido
ESTADO_SIN_PREDIOS = "SIN_PREDIOS"  # OCR ok pero 0 predios validos
ESTADO_ERROR = "ERROR"           # excepcion durante el procesamiento
ESTADO_CACHE = "CACHE"           # ya estaba procesado (saltado)


@dataclass
class RegistroMetrica:
    """Metricas de auditoria de UN documento PDF."""
    archivo: str
    estado: str = ESTADO_OK
    n_paginas: int = 0
    n_predios: int = 0
    tiempo_ocr_s: float = 0.0
    tiempo_parsing_s: float = 0.0
    tiempo_total_s: float = 0.0
    campos_esperados: int = 0
    campos_extraidos_prom: float = 0.0     # promedio de campos no-NR por predio
    completitud_prom: float = 0.0          # campos_extraidos_prom / campos_esperados (0-1)
    campos_faltantes: List[str] = field(default_factory=list)  # NR en >50% de predios
    error_msg: str = ""
    timestamp: str = ""

    def to_csv_row(self) -> Dict:
        d = asdict(self)
        # La lista de campos faltantes se serializa como texto separado por ';'
        d["campos_faltantes"] = ";".join(self.campos_faltantes)
        return d


# Valor que marca un campo no detectado en la extraccion
_NR = "NR"


def construir_registro_extraccion(archivo, campos_predio, campos_documento,
                                  predios, n_paginas, tiempo_ocr_s,
                                  tiempo_parsing_s, umbral_faltante=0.5):
    """
    Construye un RegistroMetrica con estado OK/SIN_PREDIOS a partir del
    resultado de la extraccion de un PDF.

    - campos_predio: lista de nombres de campos por predio (CAMPOS_PREDIO).
    - campos_documento: lista de campos de nivel documento (p.ej. Resolucion, Fecha).
    - predios: lista de dicts (uno por predio) tal como los devuelve la extraccion.
    - umbral_faltante: fraccion de predios con NR para marcar un campo como faltante.
    """
    todos_los_campos = list(campos_documento) + list(campos_predio)
    n_esperados = len(todos_los_campos)
    tiempo_total = tiempo_ocr_s + tiempo_parsing_s

    if not predios:
        return RegistroMetrica(
            archivo=archivo, estado=ESTADO_SIN_PREDIOS,
            n_paginas=n_paginas, n_predios=0,
            tiempo_ocr_s=round(tiempo_ocr_s, 3),
            tiempo_parsing_s=round(tiempo_parsing_s, 3),
            tiempo_total_s=round(tiempo_total, 3),
            campos_esperados=n_esperados,
            campos_faltantes=list(todos_los_campos),
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

    n_predios = len(predios)
    # Conteo de valores no-NR por campo (para completitud y campos faltantes)
    presentes_por_campo = {c: 0 for c in todos_los_campos}
    suma_campos_por_predio = 0
    for p in predios:
        no_nr = 0
        for c in todos_los_campos:
            val = str(p.get(c, _NR)).strip()
            if val and val != _NR:
                presentes_por_campo[c] += 1
                no_nr += 1
        suma_campos_por_predio += no_nr

    campos_extraidos_prom = suma_campos_por_predio / n_predios
    completitud = campos_extraidos_prom / n_esperados if n_esperados else 0.0

    # Campo faltante: presente en menos de (1 - umbral) de los predios
    faltantes = [c for c in todos_los_campos
                 if presentes_por_campo[c] / n_predios < (1 - umbral_faltante)]

    return RegistroMetrica(
        archivo=archivo, estado=ESTADO_OK,
        n_paginas=n_paginas, n_predios=n_predios,
        tiempo_ocr_s=round(tiempo_ocr_s, 3),
        tiempo_parsing_s=round(tiempo_parsing_s, 3),
        tiempo_total_s=round(tiempo_total, 3),
        campos_esperados=n_esperados,
        campos_extraidos_prom=round(campos_extraidos_prom, 2),
        completitud_prom=round(completitud, 4),
        campos_faltantes=faltantes,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )


class MetricasTracker:
    """
    Acumula los RegistroMetrica de todos los documentos y genera las salidas
    consolidadas. Se usa desde el hilo principal (no desde los workers).
    """

    def __init__(self, campos_predio, campos_documento, out_dir,
                 dpi=None, workers=None):
        self.campos_predio = list(campos_predio)
        self.campos_documento = list(campos_documento)
        self.todos_los_campos = list(campos_documento) + list(campos_predio)
        self.out_dir = out_dir
        self.dpi = dpi
        self.workers = workers
        self.registros: List[RegistroMetrica] = []
        self._t0 = time.perf_counter()
        os.makedirs(out_dir, exist_ok=True)

    def agregar(self, registro: Optional[RegistroMetrica]):
        if registro is not None:
            self.registros.append(registro)

    # -------------------------------------------------- salidas
    def escribir_csv(self, ts):
        ruta = os.path.join(self.out_dir, f"metricas_proceso_{ts}.csv")
        campos = list(RegistroMetrica("").to_csv_row().keys())
        with open(ruta, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            for r in self.registros:
                w.writerow(r.to_csv_row())
        return ruta

    def _consolidar(self):
        total = len(self.registros)
        exitosos = sum(1 for r in self.registros if r.estado == ESTADO_OK)
        sin_predios = sum(1 for r in self.registros if r.estado == ESTADO_SIN_PREDIOS)
        con_error = sum(1 for r in self.registros if r.estado == ESTADO_ERROR)
        en_cache = sum(1 for r in self.registros if r.estado == ESTADO_CACHE)

        ok = [r for r in self.registros if r.estado == ESTADO_OK]
        total_predios = sum(r.n_predios for r in ok)
        t_ocr = sum(r.tiempo_ocr_s for r in self.registros)
        t_pars = sum(r.tiempo_parsing_s for r in self.registros)
        t_prom = (sum(r.tiempo_total_s for r in self.registros) / total) if total else 0.0

        # Completitud promedio por variable (sobre docs OK, ponderada por predios)
        completitud_por_variable = {}
        total_predios_ok = total_predios if total_predios else 0
        if total_predios_ok:
            presentes = {c: 0 for c in self.todos_los_campos}
            # Reconstruir desde registros no es posible (no guardan por-campo);
            # se aproxima con completitud global. Para exactitud por campo se
            # recomienda el reporte de comparacion. Aqui se deja la completitud
            # global promedio y, si se requiere granularidad, se calcula aparte.
        completitud_global = (sum(r.completitud_prom * r.n_predios for r in ok) /
                              total_predios_ok) if total_predios_ok else 0.0

        return {
            "corrida": {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "n_pdfs": total,
                "duracion_total_s": round(time.perf_counter() - self._t0, 2),
                "dpi": self.dpi,
                "workers": self.workers,
            },
            "procesamiento": {
                "total_procesados": total,
                "exitosos": exitosos,
                "sin_predios": sin_predios,
                "con_error": con_error,
                "en_cache": en_cache,
                "total_predios_extraidos": total_predios,
                "tiempo_ocr_total_s": round(t_ocr, 2),
                "tiempo_parsing_total_s": round(t_pars, 2),
                "tiempo_promedio_por_pdf_s": round(t_prom, 3),
                "completitud_global_promedio": round(completitud_global, 4),
            },
            "completitud_por_variable": self._completitud_por_variable(ok),
        }

    def _completitud_por_variable(self, registros_ok):
        """
        Tasa de deteccion por campo: fraccion de documentos OK en los que el
        campo NO aparece en la lista de campos_faltantes (es decir, se detecto
        en la mayoria de sus predios).
        """
        n = len(registros_ok)
        if not n:
            return {c: 0.0 for c in self.todos_los_campos}
        detectados = {c: 0 for c in self.todos_los_campos}
        for r in registros_ok:
            faltan = set(r.campos_faltantes)
            for c in self.todos_los_campos:
                if c not in faltan:
                    detectados[c] += 1
        return {c: round(detectados[c] / n, 4) for c in self.todos_los_campos}

    def escribir_json(self, ts, nombre="resumen_metricas_tesis.json"):
        resumen = self._consolidar()
        ruta = os.path.join(self.out_dir, nombre)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(resumen, f, ensure_ascii=False, indent=2)
        return ruta, resumen

    def escribir_txt(self, resumen, nombre="resumen_metricas_tesis.txt"):
        ruta = os.path.join(self.out_dir, nombre)
        W = 70
        L = []
        L.append("=" * W)
        L.append("RESUMEN DE METRICAS DEL PROCESO DE EXTRACCION".center(W))
        L.append("=" * W)
        c = resumen["corrida"]
        L.append(f"Fecha        : {c['timestamp']}")
        L.append(f"PDFs         : {c['n_pdfs']}")
        L.append(f"Duracion (s) : {c['duracion_total_s']}")
        L.append(f"DPI OCR      : {c['dpi']}    Workers: {c['workers']}")
        L.append("")
        p = resumen["procesamiento"]
        L.append("-" * W)
        L.append("PROCESAMIENTO")
        L.append("-" * W)
        L.append(f"  Total procesados        : {p['total_procesados']}")
        L.append(f"  Exitosos (OK)           : {p['exitosos']}")
        L.append(f"  Sin predios             : {p['sin_predios']}")
        L.append(f"  Con error               : {p['con_error']}")
        L.append(f"  En cache (saltados)     : {p['en_cache']}")
        L.append(f"  Predios extraidos       : {p['total_predios_extraidos']}")
        L.append(f"  Tiempo OCR total (s)    : {p['tiempo_ocr_total_s']}")
        L.append(f"  Tiempo parsing total (s): {p['tiempo_parsing_total_s']}")
        L.append(f"  Tiempo prom por PDF (s) : {p['tiempo_promedio_por_pdf_s']}")
        L.append(f"  Completitud global prom : {p['completitud_global_promedio']*100:.1f}%")
        L.append("")
        L.append("-" * W)
        L.append("TASA DE DETECCION POR VARIABLE (sobre docs OK)")
        L.append("-" * W)
        for campo, tasa in resumen["completitud_por_variable"].items():
            L.append(f"  {campo:<38} {tasa*100:6.1f}%")
        L.append("")
        L.append("=" * W)
        with open(ruta, "w", encoding="utf-8") as f:
            f.write("\n".join(L))
        return ruta

    def finalizar(self, ts=None):
        """Escribe CSV, JSON y TXT. Devuelve las rutas generadas."""
        ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.escribir_csv(ts)
        json_path, resumen = self.escribir_json(ts)
        txt_path = self.escribir_txt(resumen)
        return {"csv": csv_path, "json": json_path, "txt": txt_path, "resumen": resumen}

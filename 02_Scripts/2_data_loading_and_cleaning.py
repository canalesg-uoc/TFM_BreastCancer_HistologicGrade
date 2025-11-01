#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocesamiento de GSE4922 desde family SOFT:
- Lee el SOFT (GSE4922_family.soft.gz)
- Construye la matriz de expresión (probes x muestras) usando 'VALUE'
- Mapea probe -> símbolo de gen desde la GPL
- Resuelve múltiples probes por gen eligiendo la de mayor varianza
- Extrae la variable objetivo (Grado histológico: GH-1/2/3)
- Alinea X y Y por muestra
- Guarda CSVs finales en 01_Data/processed/
"""

import os
import sys
import numpy as np
import pandas as pd
import GEOparse

# --- Configuración de Archivos y GEO ID ---
GEO_ID = 'GSE4922'
RAW_DATA_PATH = os.path.join('01_Data', 'raw')
PROCESSED_DATA_PATH = os.path.join('01_Data', 'processed')
OUTPUT_FILE_X = os.path.join(PROCESSED_DATA_PATH, f'{GEO_ID}_expression_matrix.csv')   # (muestras x genes)
OUTPUT_FILE_Y = os.path.join(PROCESSED_DATA_PATH, f'{GEO_ID}_target_variable.csv')     # etiquetas GH
OUTPUT_FILE_MAP = os.path.join(PROCESSED_DATA_PATH, f'{GEO_ID}_probe_to_gene_map.csv') # trazabilidad

def _first_symbol(v: str) -> str:
    """Extrae el primer símbolo de gen de una cadena separada por delimitadores comunes en GEO."""
    if pd.isna(v) or not isinstance(v, str):
        return np.nan
    s = v.strip()
    for sep in [' /// ', ' // ', ' ; ', ';', ',', ' | ', '|', '///', '//', ' ']:
        if sep in s:
            return s.split(sep)[0].strip()
    return s

def _identify_gene_col(gpl_table: pd.DataFrame) -> str:
    """
    Identifica la columna que contiene los símbolos de genes en la tabla GPL.
    Búsqueda case-insensitive y tolerante a alias.
    """
    if gpl_table is None or gpl_table.empty:
        return None
    cols_ci = {c.lower(): c for c in gpl_table.columns}
    candidates_ci = [
        'gene symbol', 'gene symbols', 'symbol', 'gene_symbol',
        'genesymbol', 'gene title', 'gene', 'ilmn_gene', 'unigene_symbol'
    ]
    for cand in candidates_ci:
        if cand in cols_ci:
            return cols_ci[cand]
    # fallback: cualquier columna que contenga 'symbol'
    for lc, orig in cols_ci.items():
        if 'symbol' in lc:
            return orig
    return None

def main():
    print(f"Iniciando carga y limpieza de datos para el dataset {GEO_ID}...")

    # 1) Cargar archivo SOFT (contiene metadatos + tablas GSM con VALUE)
    gse_file_path = os.path.join(RAW_DATA_PATH, f'{GEO_ID}_family.soft.gz')
    if not os.path.exists(gse_file_path):
        print(f"❌ Archivo no encontrado: {gse_file_path}\n   Descarga el family SOFT desde GEO y colócalo en esa ruta.", file=sys.stderr)
        sys.exit(1)

    try:
        gse = GEOparse.get_GEO(filepath=gse_file_path, silent=True)
    except Exception as e:
        print(f"❌ No se pudo leer el SOFT: {e}", file=sys.stderr)
        sys.exit(1)

    print("✅ Dataset cargado correctamente (family SOFT).")

    # 2) Validar que hay GSMs y que existe la columna VALUE
    if not gse.gsms:
        print("❌ No se encontraron muestras (GSM) en el SOFT. Revisa el archivo.", file=sys.stderr)
        sys.exit(1)

    first_gsm = next(iter(gse.gsms.values()))
    if first_gsm.table is None or 'VALUE' not in first_gsm.table.columns:
        print("❌ Las GSM no contienen la columna 'VALUE'.\n   Alternativas: usar Series Matrix o procesar CEL con RMA.", file=sys.stderr)
        sys.exit(1)

    # 3) Matriz de expresión por pivot (probes x muestras)
    try:
        X = gse.pivot_samples(values='VALUE')  # filas=ID_REF/probe, columnas=GSM
    except Exception as e:
        print(f"❌ Error al pivotar muestras: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Dimensiones iniciales de X (Sondas x Muestras): {X.shape}")

    # 4) Cargar GPL y detectar columna de símbolo de gen
    if not gse.gpls:
        print("❌ No se encontraron plataformas (GPL) en el SOFT. Revise el archivo.", file=sys.stderr)
        sys.exit(1)

    gpl_id = list(gse.gpls.keys())[0]
    gpl = gse.gpls[gpl_id]

    gene_col = _identify_gene_col(gpl.table)
    if not gene_col:
        print("❌ No se pudo identificar la columna de símbolo de gen en la GPL.", file=sys.stderr)
        print(f"   Columnas disponibles (primeras 15): {list(gpl.table.columns)[:15]}", file=sys.stderr)
        sys.exit(1)

    # 5) Construir mapa probe -> gene con limpieza
    gpl_tbl = gpl.table[['ID', gene_col]].copy()
    gpl_tbl[gene_col] = gpl_tbl[gene_col].astype(str)

    NA_VALUES = {'nan', 'NAN', '---', '', ' ', 'none', 'NONE', 'na', 'NA'}
    gpl_tbl[gene_col] = gpl_tbl[gene_col].apply(lambda s: np.nan if s.strip() in NA_VALUES else s)
    gpl_tbl[gene_col] = gpl_tbl[gene_col].apply(_first_symbol)

    probe_to_gene_map = gpl_tbl.set_index('ID')[gene_col]

    # Guardar mapa para trazabilidad (opcional pero útil)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    gpl_tbl.rename(columns={gene_col: 'Gene_Symbol'}).to_csv(OUTPUT_FILE_MAP, index=False)

    # 6) Añadir Gene_Symbol a X y filtrar probes sin gen
    if not X.index.isin(probe_to_gene_map.index).any():
        print("⚠️ Aviso: El índice de X no parece coincidir con IDs de sonda de la GPL. Se intentará continuar.", file=sys.stderr)

    X['Gene_Symbol'] = X.index.map(probe_to_gene_map)

    initial_probes = len(X)
    X = X.dropna(subset=['Gene_Symbol']).copy()
    print(f"Sondas descartadas (sin gen mapeado): {initial_probes - len(X)}")

    # 7) Índice jerárquico (Gene, Probe) y elección de probe por máxima varianza
    X['Probe_ID'] = X.index
    X = X.set_index(['Gene_Symbol', 'Probe_ID'])

    # Asegurar numérico
    X_numeric = X.apply(pd.to_numeric, errors='coerce')
    # varianza por fila (probe)
    row_var = X_numeric.var(axis=1, numeric_only=True)

    # idx de la probe con mayor varianza por gen (nivel 0 del MultiIndex)
    try:
        idx_best = row_var.groupby(level=0).idxmax()
    except Exception as e:
        print(f"❌ Error al seleccionar la probe de mayor varianza por gen: {e}", file=sys.stderr)
        sys.exit(1)

    X_final = X_numeric.loc[idx_best].copy()
    X_final.index = X_final.index.droplevel(1)  # índice = Gene
    print(f"Dimensiones finales de X (Gen x Muestras): {X_final.shape}")

    # 8) Extraer variable objetivo Y (grado histológico)
    y_df = gse.phenotype_data.copy()
    initial_samples = len(y_df)

    # Buscar columna de grado (tolerante)
    candidate_cols = [c for c in y_df.columns if any(k in c.lower() for k in ['histologic', 'grade', 'elston', 'nottingham'])]
    if not candidate_cols:
        print("❌ No se encontró una columna de grado histológico en los metadatos.", file=sys.stderr)
        print(f"   Columnas disponibles (primeras 20): {list(y_df.columns)[:20]}", file=sys.stderr)
        sys.exit(1)

    col_grade = candidate_cols[0]
    y_raw = y_df[col_grade].astype(str).str.lower()

    # Extraer dígito 1/2/3
    y_num = y_raw.str.extract(r'(\d+)')[0].dropna()
    perdidas_y = initial_samples - len(y_num)
    if perdidas_y > 0:
        print(f"⚠️ Muestras descartadas (grado no parseable): {perdidas_y}")

    y_num = y_num.astype(int)
    y = pd.DataFrame({'Histological_Grade': 'GH-' + y_num.astype(str)}, index=y_num.index)

    print(f"Muestras con Grado Histológico definido: {len(y)}")
    print("Distribución de clases (Y):")
    print(y['Histological_Grade'].value_counts().sort_index())

    # 9) Alinear X y Y por muestra
    common_samples = X_final.columns.intersection(y.index)
    if len(common_samples) == 0:
        print("❌ No hay intersección entre muestras de X y Y. Revisa los IDs de muestra.", file=sys.stderr)
        print(f"   Ejemplo IDs X: {list(X_final.columns)[:5]}")
        print(f"   Ejemplo IDs Y: {list(y.index)[:5]}")
        sys.exit(1)

    X_final = X_final.loc[:, common_samples]
    y = y.loc[common_samples]
    print(f"Tras alinear: X={X_final.shape} | Y={y.shape}")

    # 10) Guardar resultados
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    # X final como (muestras x genes)
    X_final.T.to_csv(OUTPUT_FILE_X)
    y.to_csv(OUTPUT_FILE_Y)

    print("\n✅ Datos preprocesados guardados exitosamente.")
    print(f"   X: {X_final.T.shape[0]} muestras x {X_final.T.shape[1]} genes -> {OUTPUT_FILE_X}")
    print(f"   Y: {y.shape[0]} muestras -> {OUTPUT_FILE_Y}")
    print(f"   Mapa probe→gene (trazabilidad): {OUTPUT_FILE_MAP}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error crítico durante el preprocesamiento: {e}", file=sys.stderr)
        sys.exit(1)
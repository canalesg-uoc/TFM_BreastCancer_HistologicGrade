#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# -----------------------------------------------------------------------------
# TFM_BreastCancer_HistologicGrade
# Script 1: 1_data_preparation_full.py
# -----------------------------------------------------------------------------
#
# Descripción:
# Consolidación del preprocesamiento inicial.
# 1. Descarga los datasets GSE4922 (principal) y GSE1456 (validación) de GEO.
# 2. Carga el SOFT de ambos, construye la matriz de expresión, mapea a genes,
#    resuelve probes duplicadas (por máxima varianza) y extrae el target.
# 3. Guarda los archivos CSV limpios para ambos IDs en '01_Data/processed'.
# -----------------------------------------------------------------------------
"""

import os
import sys
import numpy as np
import pandas as pd
import GEOparse
from typing import List

# --- Configuración Global ---
# Modificado para incluir GSE1456 como dataset de validación
GEO_IDS = ['GSE4922', 'GSE1456'] 
MAIN_GEO_ID = GEO_IDS[0] # GSE4922 es el dataset principal (ENTRENAMIENTO)
VAL_GEO_ID = GEO_IDS[1]  # GSE1456 es el dataset de validación

RAW_DATA_PATH = os.path.join('01_Data', 'raw') 
PROCESSED_DATA_PATH = os.path.join('01_Data', 'processed')

# =============================================================================
# FUNCIONES AUXILIARES 
# =============================================================================

def _first_symbol(v: str) -> str:
    """Extrae el primer símbolo de gen de una cadena separada por delimitadores."""
    if pd.isna(v) or not isinstance(v, str):
        return np.nan
    s = v.strip()
    for sep in [' /// ', ' // ', ' ; ', ';', ',', ' | ', '|', '///', '//', ' ']:
        if sep in s:
            return s.split(sep)[0].strip()
    return s

def _identify_gene_col(gpl_table: pd.DataFrame) -> str:
    """Identifica la columna que contiene los símbolos de genes en la tabla GPL."""
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

# =============================================================================
# FASE 1: DESCARGA DE DATOS
# =============================================================================

def download_datasets(geo_ids: List[str], raw_data_path: str):
    """Descarga los datasets de GEO y los guarda en la ruta especificada."""
    print(f"\n--- 1. DESCARGA: Iniciando de los datasets: {', '.join(geo_ids)} de GEO ---")
    
    # Crear el directorio '01_Data/raw' si no existe
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
        print(f"Directorio creado: {raw_data_path}")

    # Iterar y descargar cada dataset
    for GEO_ID in geo_ids:
        print(f"\n--- Iniciando descarga de {GEO_ID} ---")
        try:
            # GEOparse gestiona la descarga del archivo family SOFT (.soft.gz) y lo guarda
            gse = GEOparse.get_GEO(geo=GEO_ID, destdir=raw_data_path, silent=True)
            print(f"✅ Descarga de {GEO_ID} completada. Archivos guardados en: {raw_data_path}")
            
        except Exception as e:
            print(f"\n❌ Error al descargar o parsear {GEO_ID}: {e}", file=sys.stderr)
            
    print("\n🎉 ¡Proceso de descarga finalizado!")


# =============================================================================
# FASE 2: CARGA Y LIMPIEZA DE UN SOLO DATASET (Adaptada)
# =============================================================================

def process_single_dataset(geo_id: str):
    """Carga, limpia y procesa un dataset GEO específico, guardando sus salidas."""
    print(f"\n--- 2. PREPROCESAMIENTO: Iniciando para el dataset {geo_id} ---")

    # Rutas de salida específicas para el GEO_ID actual
    output_file_x = os.path.join(PROCESSED_DATA_PATH, f'{geo_id}_expression_matrix.csv')
    output_file_y = os.path.join(PROCESSED_DATA_PATH, f'{geo_id}_target_variable.csv')
    output_file_map = os.path.join(PROCESSED_DATA_PATH, f'{geo_id}_probe_to_gene_map.csv')

    # 1) Cargar archivo SOFT
    gse_file_path = os.path.join(RAW_DATA_PATH, f'{geo_id}_family.soft.gz')
    if not os.path.exists(gse_file_path):
        print(f"❌ Archivo SOFT no encontrado: {gse_file_path}", file=sys.stderr)
        print("   Asegúrate de haber ejecutado la descarga y que el archivo .soft.gz exista.", file=sys.stderr)
        return

    try:
        gse = GEOparse.get_GEO(filepath=gse_file_path, silent=True)
    except Exception as e:
        print(f"❌ No se pudo leer el SOFT: {e}", file=sys.stderr)
        return

    print("✅ Dataset cargado correctamente (family SOFT).")

    # 2) Matriz de expresión X (Probes x Muestras)
    try:
        X = gse.pivot_samples(values='VALUE')
    except Exception as e:
        print(f"❌ Error al pivotar muestras: {e}", file=sys.stderr)
        return
    print(f"Dimensiones iniciales de X (Sondas x Muestras): {X.shape}")
    
    # 3) Mapeo Probe -> Gen (GPL)
    if not gse.gpls:
        print("❌ No se encontraron plataformas (GPL).", file=sys.stderr)
        return
    
    gpl_id = list(gse.gpls.keys())[0]
    gpl = gse.gpls[gpl_id]
    gene_col = _identify_gene_col(gpl.table)

    if not gene_col:
        print("❌ No se pudo identificar la columna de símbolo de gen en la GPL.", file=sys.stderr)
        return

    # Construir mapa probe -> gene con limpieza
    gpl_tbl = gpl.table[['ID', gene_col]].copy()
    gpl_tbl[gene_col] = gpl_tbl[gene_col].astype(str)
    NA_VALUES = {'nan', 'NAN', '---', '', ' ', 'none', 'NONE', 'na', 'NA'}
    gpl_tbl[gene_col] = gpl_tbl[gene_col].apply(lambda s: np.nan if s.strip().lower() in NA_VALUES else s)
    gpl_tbl[gene_col] = gpl_tbl[gene_col].apply(_first_symbol)

    probe_to_gene_map = gpl_tbl.set_index('ID')[gene_col]

    # Guardar mapa para trazabilidad
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    gpl_tbl.rename(columns={gene_col: 'Gene_Symbol'}).to_csv(output_file_map, index=False)

    # 4) Filtrado y Selección de Probe por Máxima Varianza
    X['Gene_Symbol'] = X.index.map(probe_to_gene_map)
    initial_probes = len(X)
    X = X.dropna(subset=['Gene_Symbol']).copy()
    print(f"Sondas descartadas (sin gen mapeado): {initial_probes - len(X)}")

    X['Probe_ID'] = X.index
    X = X.set_index(['Gene_Symbol', 'Probe_ID'])
    X_numeric = X.apply(pd.to_numeric, errors='coerce')
    row_var = X_numeric.var(axis=1, numeric_only=True)
    
    # Seleccionar la probe con mayor varianza por gen
    try:
        idx_best = row_var.groupby(level=0).idxmax()
        X_final = X_numeric.loc[idx_best].copy()
        X_final.index = X_final.index.droplevel(1)  # Índice = Gene
        print(f"Dimensiones finales de X (Gen x Muestras): {X_final.shape}")
    except Exception as e:
        print(f"❌ Error al seleccionar la probe de mayor varianza por gen: {e}", file=sys.stderr)
        return

    # 5) Extracción y Limpieza de Variable Objetivo Y (Grado Histológico)
    y_df = gse.phenotype_data.copy()
    
    candidate_cols = [c for c in y_df.columns if any(k in c.lower() for k in ['histologic', 'grade', 'elston', 'nottingham'])]
    if not candidate_cols:
        print("❌ No se encontró una columna de grado histológico en los metadatos.", file=sys.stderr)
        return

    col_grade = candidate_cols[0]
    y_raw = y_df[col_grade].astype(str).str.lower()
    y_num = y_raw.str.extract(r'(\d+)')[0].dropna()
    y_num = y_num.astype(int)
    y = pd.DataFrame({'Histological_Grade': 'GH-' + y_num.astype(str)}, index=y_num.index)

    # 6) Alinear X y Y y Guardar
    common_samples = X_final.columns.intersection(y.index)
    if len(common_samples) == 0:
        print("❌ No hay intersección entre muestras de X y Y.", file=sys.stderr)
        return

    X_final = X_final.loc[:, common_samples]
    y = y.loc[common_samples]
    
    # Guardar resultados
    y.index.name = "Sample"
    # Guardamos como Muestras x Genes (transpuesto)
    X_final.T.to_csv(output_file_x) 
    y.to_csv(output_file_y)

    print("\n✅ Datos preprocesados guardados exitosamente.")
    print(f"   X ({geo_id}): {X_final.T.shape} -> {output_file_x}")
    print(f"   Y ({geo_id}): {y.shape} -> {output_file_y}")


# =============================================================================
# FUNCIÓN PRINCIPAL (MODIFICADA para procesar ambos IDs)
# =============================================================================

def main():
    try:
        # FASE 1: Descarga de datasets (GSE4922 y GSE1456)
        download_datasets(GEO_IDS, RAW_DATA_PATH)
        
        # FASE 2: Procesamiento del dataset principal (GSE4922)
        print(f"\n--- Procesando Dataset Principal: {MAIN_GEO_ID} ---")
        process_single_dataset(MAIN_GEO_ID) 
        
        # FASE 3: Procesamiento del dataset de validación (GSE1456)
        print(f"\n--- Procesando Dataset de Validación: {VAL_GEO_ID} ---")
        process_single_dataset(VAL_GEO_ID)
        
        print("\n\n🎉 ¡SCRIPT 1 (PREPARACIÓN COMPLETA) FINALIZADO!")
        
    except Exception as e:
        print(f"\n❌ Error crítico en el pipeline principal: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

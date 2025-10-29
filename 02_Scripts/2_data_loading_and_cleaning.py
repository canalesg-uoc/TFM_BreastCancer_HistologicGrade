# -----------------------------------------------------------------------------
# TFM_BreastCancer_HistologicGrade
# Script 2: 2_data_loading_and_cleaning.py
# -----------------------------------------------------------------------------

import GEOparse
import pandas as pd
import numpy as np
import os
import sys

# --- Configuración ---
GEO_ID = 'GSE4922'
RAW_DATA_PATH = os.path.join('01_Data', 'raw')
PROCESSED_DATA_PATH = os.path.join('01_Data', 'processed')
OUTPUT_FILE_X = os.path.join(PROCESSED_DATA_PATH, f'{GEO_ID}_expression_matrix.csv')
OUTPUT_FILE_Y = os.path.join(PROCESSED_DATA_PATH, f'{GEO_ID}_target_variable.csv')

print(f"Iniciando carga y limpieza de datos para el dataset {GEO_ID}...")

try:
    # 1) Cargar archivo descargado
    gse_file_path = os.path.join(RAW_DATA_PATH, f'{GEO_ID}_family.soft.gz')
    if not os.path.exists(gse_file_path):
        print(f"❌ Archivo no encontrado: {gse_file_path}. Asegúrate de que el archivo fue descargado.", file=sys.stderr)
        sys.exit(1)

    gse = GEOparse.get_GEO(filepath=gse_file_path, silent=True)
    print("✅ Dataset cargado correctamente.")

    # 2) Matriz de expresión (Sondas x Muestras)
    X = gse.pivot_samples(values='VALUE')
    print(f"Dimensiones iniciales de X (Sondas x Muestras): {X.shape}")

    # 3) Mapeo de IDs de sonda a símbolo de gen
    gpl_id = list(gse.gpls.keys())[0]
    gpl = gse.gpls[gpl_id]

    # Columna con símbolo de gen (varía según plataforma)
    if 'Gene symbol' in gpl.table.columns:
        gene_col = 'Gene symbol'
    elif 'GB_ACC' in gpl.table.columns:
        gene_col = 'GB_ACC'
    else:
        gene_col = 'Gene Title'

    # Nota: algunas plataformas usan separadores múltiples entre símbolos (///, //, ;, ,)
    def _first_symbol(v):
        if pd.isna(v):
            return np.nan
        s = str(v)
        for sep in [' /// ', ' // ', ' ; ', ';', ',', ' | ', '|', '///', '//']:
            if sep in s:
                return s.split(sep)[0].strip()
        return s.strip()

    gpl_tbl = gpl.table[['ID', gene_col]].copy()
    gpl_tbl[gene_col] = gpl_tbl[gene_col].apply(_first_symbol)
    probe_to_gene_map = gpl_tbl.set_index('ID')[gene_col]

    # Asignar símbolo de gen una única vez
    X['Gene_Symbol'] = X.index.map(probe_to_gene_map)

    # Limpiar: quitar sondas sin gen mapeado
    X = X.dropna(subset=['Gene_Symbol']).copy()

    # Guardar el Probe_ID para tener un índice único por sonda
    X['Probe_ID'] = X.index

    # Índice jerárquico (Gene, Probe) para resolver sondas duplicadas por gen
    X = X.set_index(['Gene_Symbol', 'Probe_ID'])
    print(f"Tras mapeo y limpieza: {X.shape} (índice = [Gene, Probe])")

    # 4) Resolver múltiples sondas por gen usando máxima varianza
    # Asegurar numérico
    X_numeric = X.apply(pd.to_numeric, errors='coerce')

    # Varianza por fila (cada fila es una sonda)
    row_var = X_numeric.var(axis=1, numeric_only=True)

    # Para cada gen (nivel 0 del índice), quedarnos con la sonda (probe) de mayor varianza
    idx_best = row_var.groupby(level=0).idxmax()

    # Construir matriz final con una sola sonda por gen
    X_final = X_numeric.loc[idx_best].copy()

    # Opcional: eliminar el nivel de probe y quedarnos con índice = Gene
    X_final.index = X_final.index.droplevel(1)
    print(f"Dimensiones finales de X (Gen x Muestras): {X_final.shape}")

    # 5) Variable objetivo (Y: Grado histológico)
    y_df = gse.phenotype_data

    # Buscar columna de grado (nombres varían)
    candidate_cols = [c for c in y_df.columns if 'histologic' in c.lower() or 'grade' in c.lower()]
    if not candidate_cols:
        print("\n❌ ERROR CRÍTICO: No se encontró una columna de grado histológico en los metadatos.", file=sys.stderr)
        sys.exit(1)

    # Tomamos la primera candidata y extraemos el número 1/2/3
    y_raw = y_df[candidate_cols[0]].astype(str).str.lower()
    y_num = y_raw.str.extract(r'(\d+)')[0].dropna()
    y_num = y_num.astype(int)

    # Construimos DataFrame Y con prefijo
    y = pd.DataFrame({'Histological_Grade': 'GH-' + y_num.astype(str)})
    print(f"Muestras con Grado Histológico definido: {len(y)}")
    print("Distribución de clases (Y):")
    print(y['Histological_Grade'].value_counts())

    # 6) Alinear X y Y por muestra
    common_samples = X_final.columns.intersection(y.index)
    if len(common_samples) == 0:
        print("\n❌ No hay intersección entre muestras de X y Y. Revisa los IDs de muestra.", file=sys.stderr)
        sys.exit(1)

    X_final = X_final.loc[:, common_samples]
    y = y.loc[common_samples]
    print(f"Tras alinear: X={X_final.shape}  Y={y.shape}")

    # 7) Guardar
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    # Formato final: X como (Muestras x Genes)
    X_final.T.to_csv(OUTPUT_FILE_X)
    y.to_csv(OUTPUT_FILE_Y)

    print(f"\n✅ Datos preprocesados guardados en: {PROCESSED_DATA_PATH}")
    print(f"   X: {X_final.T.shape[0]} muestras x {X_final.T.shape[1]} genes -> {OUTPUT_FILE_X}")
    print(f"   Y: {y.shape[0]} muestras -> {OUTPUT_FILE_Y}")

except Exception as e:
    print(f"\n❌ Error durante el preprocesamiento: {e}", file=sys.stderr)
    sys.exit(1)
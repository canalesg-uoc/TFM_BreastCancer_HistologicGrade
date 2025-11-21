#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# -----------------------------------------------------------------------------
# TFM_BreastCancer_HistologicGrade
# Script 7: 7_validation_and_final_metrics.py
# -----------------------------------------------------------------------------
#
# Descripción: VALIDACIÓN EXTERNA EN GSE1456
# 1. Carga el modelo final de SVM-RFE (entrenado en GSE4922).
# 2. Carga la firma de genes de consenso (Top 50 más estables).
# 3. Carga y prepara el dataset GSE1456 (X_val, y_val).
# 4. **FIX CRÍTICO:** Reindexa el dataset de validación para que tenga el mismo 
#    orden y número de columnas (genes) que el de entrenamiento.
# 5. Evalúa el rendimiento del modelo en el conjunto de validación.
# -----------------------------------------------------------------------------
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
import sys

# =========================
# 1) CONFIGURACIÓN Y RUTAS
# =========================
OUTPUT_DIR = Path('results') # Directorio donde se guardaron los resultados del Script 2
PROCESSED_DATA_DIR = Path('01_Data/processed')

# Archivos de entrada
# NOTA: Usamos el modelo del outer_fold_1 como proxy del mejor modelo final entrenado.
MODEL_PATH = OUTPUT_DIR / 'best_models/svmrfe_outer_1.joblib' 
GENE_CONSENSUS_PATH = OUTPUT_DIR / 'frequency_SVMRFE.csv'      
X_VAL_PATH = PROCESSED_DATA_DIR / 'GSE1456_expression_matrix.csv'
Y_VAL_PATH = PROCESSED_DATA_DIR / 'GSE1456_target_variable.csv'
X_TRAIN_REF_PATH = PROCESSED_DATA_DIR / 'X_train.csv' # Para obtener el orden correcto de genes

# Archivos de salida
REPORT_PATH = OUTPUT_DIR / '7_external_validation_report_GSE1456.txt'
CONF_MATRIX_PATH = OUTPUT_DIR / '7_external_validation_conf_matrix_GSE1456.csv'
TOP_K_CONSENSUS = 50 

# =========================
# 2) FUNCIÓN PRINCIPAL
# =========================
def run_external_validation():
    print("--- Iniciando Validación Externa con GSE1456 ---")

    # A. Cargar modelo, genes y datos
    try:
        final_model = joblib.load(MODEL_PATH) 
        X_val = pd.read_csv(X_VAL_PATH, index_col=0) # Muestras x Genes
        y_val_df = pd.read_csv(Y_VAL_PATH, index_col=0)
        
        # Cargamos la lista de genes de consenso (Top K)
        consensus_df = pd.read_csv(GENE_CONSENSUS_PATH)
        consensus_genes = consensus_df['gene'].head(TOP_K_CONSENSUS).tolist()
        
    except FileNotFoundError as e:
        print(f"❌ Error: Archivo de entrada no encontrado: {e}.", file=sys.stderr)
        print("Asegúrate de haber ejecutado los scripts 1, 2 y el preprocesamiento de GSE1456.")
        sys.exit(1)

    # B. Preparar el Target (y)
    # Mapear las clases (GH-1, GH-2, GH-3) a enteros (1, 2, 3)
    y_val_df['Histological_Grade'] = y_val_df['Histological_Grade'].str.replace('GH-', '').astype(int)
    y_val = y_val_df['Histological_Grade'].values
    unique_classes = np.unique(y_val)
    
    # C. Alinear y Filtrar la Matriz de Expresión (X) - FIX CRÍTICO
    print(f"Dataset de Validación GSE1456 cargado. Muestras: {X_val.shape[0]}, Genes: {X_val.shape[1]}")

    # 1. Obtener la lista COMPLETA y ORDENADA de genes del set de entrenamiento (GSE4922)
    try:
        X_train_ref = pd.read_csv(X_TRAIN_REF_PATH, index_col=0)
        # Obtenemos la lista completa de genes que vio el modelo durante el fit
        reference_genes_order = X_train_ref.columns.tolist() 
        print(f"Genes en referencia (GSE4922): {len(reference_genes_order)}")
    except FileNotFoundError:
        print(f"❌ Error: Archivo de referencia X_train.csv no encontrado en {X_TRAIN_REF_PATH}.", file=sys.stderr)
        sys.exit(1)
    
    # 2. Reindexar la matriz de validación (X_val)
    # Esto asegura que X_val tenga TODAS las columnas de GSE4922, en el orden correcto.
    # Los genes faltantes en GSE1456 (que no estaban en su GPL) se rellenan con NaN (el Imputer los manejará).
    X_val_reindexed = X_val.reindex(columns=reference_genes_order)

    # 3. Verificación de NaNs críticos: 
    nan_count = X_val_reindexed.isnull().values.sum()
    if nan_count > 0:
        print(f"⚠️ Advertencia: Se generaron {nan_count} valores NaN tras reindexar (genes faltantes en la GPL). El Imputer del Pipeline los manejará.")

    X_val_final = X_val_reindexed
    
    # D. Predicción
    # El modelo espera el orden de GSE4922, por eso le pasamos la matriz reindexada.
    # Usamos .values para pasar el array de NumPy, simplificando la compatibilidad con scikit-learn.
    X_val_array = X_val_final.values 

    print(f"Validación final con {X_val_final.shape[1]} genes (Alineación con el orden de entrenamiento).")

    try:
        y_pred = final_model.predict(X_val_array)
    except Exception as e:
        print(f"❌ Error en la predicción: {e}", file=sys.stderr)
        print("Si el error es por 'Feature names', la reindexación no fue completa. Verifica los datos.")
        sys.exit(1)
    
    # E. Métricas
    bacc = balanced_accuracy_score(y_val, y_pred)
    # El reporte de clasificación se ajusta a las etiquetas únicas presentes en el target (y_val)
    report = classification_report(y_val, y_pred, digits=4, labels=unique_classes, target_names=[f'GH-{c}' for c in unique_classes])
    conf_matrix = confusion_matrix(y_val, y_pred, labels=unique_classes)

    print("\n--- RESULTADOS DE VALIDACIÓN EXTERNA (GSE1456) ---")
    print(f"✅ Balanced Accuracy (GSE1456): {bacc:.4f}")
    print("\nReporte de Clasificación:\n", report)
    print("\nMatriz de Confusión:\n", conf_matrix)

    # F. Guardar resultados
    with open(REPORT_PATH, 'w') as f:
        f.write(f"Balanced Accuracy (GSE1456): {bacc:.4f}\n\n")
        f.write("Reporte de Clasificación:\n")
        f.write(report)
        f.write("\nMatriz de Confusión (etiquetas: 1, 2, 3):\n")
        f.write(str(conf_matrix))
    
    # Guardar matriz de confusión con etiquetas
    cm_df = pd.DataFrame(conf_matrix, 
                         columns=[f'Pred_GH-{c}' for c in unique_classes], 
                         index=[f'True_GH-{c}' for c in unique_classes])
    cm_df.to_csv(CONF_MATRIX_PATH)

    print(f"\n✅ Resultados guardados en: {REPORT_PATH}")
    print(f"✅ Matriz de confusión guardada en: {CONF_MATRIX_PATH}")


if __name__ == '__main__':
    run_external_validation()

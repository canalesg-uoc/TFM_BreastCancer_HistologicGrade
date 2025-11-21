#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# -----------------------------------------------------------------------------
# TFM_BreastCancer_HistologicGrade
# Script 6: 6_biological_enrichment.py
# -----------------------------------------------------------------------------
#
# Descripción: 
# Realiza el análisis de Enriquecimiento Funcional (GO/KEGG) de la lista de 
# genes clave más estables seleccionados por SVM-RFE (Lista de Consenso).
# Este paso da el significado biológico a los resultados de Machine Learning.
# 
# NOTA IMPORTANTE: Se requiere la librería 'gseapy' (pip install gseapy) y 
# una conexión a Internet activa para acceder a las bases de datos externas.
# -----------------------------------------------------------------------------
"""

import os
import pandas as pd
import gseapy as gp
from pathlib import Path
import sys

# =========================
# 1. CONFIGURACIÓN Y RUTAS
# =========================

OUT_DIR = Path("results")
# Ruta del archivo de frecuencia/consenso generado por el Script 3
STABILITY_FREQ_FILE = OUT_DIR / "frequency_SVMRFE.csv" 
ENRICHMENT_OUTPUT_DIR = OUT_DIR / "enrichment_analysis"
ENRICHMENT_OUTPUT_DIR.mkdir(exist_ok=True)

# Parámetros de Enriquecimiento
# Se utiliza la lista de los genes más estables (ej. top 50 por frecuencia de selección)
TOP_K_CONSENSUS = 50 

# Bases de Datos de Enriquecimiento (GO: Gene Ontology; KEGG: Vías Metabólicas)
GENE_SETS = ['KEGG_2021_Human', 'GO_Biological_Process_2021', 'GO_Molecular_Function_2021'] 

# =========================
# 2. FUNCIÓN PRINCIPAL DE ENRIQUECIMIENTO
# =========================

def run_enrichment_analysis():
    
    print(f"Leyendo lista de genes de consenso desde: {STABILITY_FREQ_FILE}")
    
    # 1. Cargar la lista de consenso (frecuencia de selección de SVM-RFE)
    try:
        freq_df = pd.read_csv(STABILITY_FREQ_FILE)
    except FileNotFoundError:
        print(f"❌ Error: Archivo de frecuencia no encontrado en {STABILITY_FREQ_FILE}.")
        print("Asegúrate de que el script 3_feature_list_extraction.py se ejecutó.")
        sys.exit(1)

    # 2. Obtener la lista de los TOP_K genes más estables
    # Asumimos que freq_df está ordenado por 'freq' o 'pct', como lo genera el Script 3
    consensus_genes = freq_df['gene'].head(TOP_K_CONSENSUS).tolist()
    
    if not consensus_genes:
        print("❌ Error: La lista de genes de consenso está vacía. No se puede realizar el enriquecimiento.")
        sys.exit(1)

    print(f"--- 1. Iniciando Análisis de Enriquecimiento para los Top {len(consensus_genes)} Genes ---")
    
    # 3. Ejecutar el análisis de enriquecimiento (Enrichr via gseapy)
    try:
        enr = gp.enrichr(
            gene_list=consensus_genes,
            gene_sets=GENE_SETS,
            organism='human',
            outdir=str(ENRICHMENT_OUTPUT_DIR), # Guarda los resultados en CSV y gráficos
            cutoff=0.05, # Umbral de p-value ajustado (FDR)
            verbose=False
        )
        
        # 4. Procesar y guardar resultados
        print("\n✅ Análisis de Enriquecimiento Completado.")
        
        # Guardamos el resumen de los resultados principales
        enr_df = enr.results.copy()
        
        # Filtrar por p-valor ajustado significativo (para la memoria)
        significant_df = enr_df[enr_df['Adjusted P-value'] < 0.05].sort_values(by='Adjusted P-value')
        
        significant_df.to_csv(ENRICHMENT_OUTPUT_DIR / "summary_enrichr_results.csv", index=False)

        print(f"Resultados significativos guardados en: {ENRICHMENT_OUTPUT_DIR.resolve()}/summary_enrichr_results.csv")
        
        # 5. Mostrar resultados clave para el usuario
        print("\n--- Principales Vías/Términos Enriquecidos (Top 5 en total, p<0.05) ---")
        
        if significant_df.empty:
            print("No se encontraron vías biológicas enriquecidas de forma significativa (Adj P-value < 0.05).")
        else:
            for index, row in significant_df.head(5).iterrows():
                print(f"  - {row['Term']} ({row['Gene_set'].split('_')[0]}) (Adj P: {row['Adjusted P-value']:.4e})")

    except Exception as e:
        print(f"\n❌ Error durante el análisis de enriquecimiento (Requiere Internet y 'gseapy'): {e}", file=sys.stderr)
        print("Intente ejecutar 'pip install gseapy' y verifique su conexión a internet.")


# =========================
# 3. MAIN
# =========================

if __name__ == "__main__":
    print("\n--- INICIANDO FASE DE INTERPRETACIÓN BIOLÓGICA (SCRIPT 6) ---")
    run_enrichment_analysis()

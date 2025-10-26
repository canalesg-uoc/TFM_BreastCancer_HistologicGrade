# -----------------------------------------------------------------------------
# TFM_BreastCancer_HistologicGrade
# Script 1: 1_data_preprocessing.py
# -----------------------------------------------------------------------------
#
# Descripción: Descarga los datasets GSE4922 (principal) y GSE2990 (validación) 
# de GEO y los guarda en la carpeta raw.
#
# -----------------------------------------------------------------------------

import GEOparse
import os
import sys

# --- Configuración ---
# Lista de IDs de GEO a descargar (GSE4922 es el principal)
GEO_IDS = ['GSE4922', 'GSE2990'] 
# Ruta de destino para guardar los archivos descargados
RAW_DATA_PATH = os.path.join('01_Data', 'raw') 

print(f"Iniciando descarga de los datasets: {', '.join(GEO_IDS)} de GEO...")

try:
    # 1. Crear el directorio '01_Data/raw' si no existe
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
        print(f"Directorio creado: {RAW_DATA_PATH}")

    # 2. Iterar y descargar cada dataset
    for GEO_ID in GEO_IDS:
        print(f"\n--- Iniciando descarga de {GEO_ID} ---")
        
        try:
            # Descargar la serie de GEO. GEOparse gestionará la descarga 
            # del archivo Series Matrix (.txt.gz) y lo guardará en la carpeta.
            gse = GEOparse.get_GEO(geo=GEO_ID,
                                   destdir=RAW_DATA_PATH,
                                   silent=False)

            print(f"✅ Descarga de {GEO_ID} completada. Archivos guardados en: {RAW_DATA_PATH}")

            # Opcional: Mostrar información básica sobre el dataset
            print("--- Información General ---")
            print(f"Título: {gse.metadata['title'][0]}")
            
            # Intenta obtener el título de la plataforma (GPL)
            gpl_keys = list(gse.gpls.keys())
            if gpl_keys:
                 print(f"Plataforma (Array): {gse.gpls[gpl_keys[0]].metadata['title'][0]}")
            
            print(f"Muestras totales: {len(gse.gsms)}")

        except Exception as e:
            print(f"\n❌ Error al descargar o parsear {GEO_ID}: {e}", file=sys.stderr)
            
    print("\n\n🎉 ¡Proceso de descarga de todos los datasets finalizado!")


except Exception as e:
    print(f"\n❌ Error grave en la configuración del script: {e}", file=sys.stderr)

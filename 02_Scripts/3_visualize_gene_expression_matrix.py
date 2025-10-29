# -----------------------------------------------------------------------------
# Exploración de la matriz de expresión GSE4922
# -----------------------------------------------------------------------------
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Configuración ---
GEO_ID = 'GSE4922'
PROCESSED_DATA_PATH = os.path.join('01_Data', 'processed')
EXPR_FILE = os.path.join(PROCESSED_DATA_PATH, f'{GEO_ID}_expression_matrix.csv')
TARGET_FILE = os.path.join(PROCESSED_DATA_PATH, f'{GEO_ID}_target_variable.csv')

# --- Cargar los datos procesados ---
X = pd.read_csv(EXPR_FILE, index_col=0)   # Muestras x Genes
y = pd.read_csv(TARGET_FILE, index_col=0) # Variable objetivo (Grado)

print("✅ Archivos cargados correctamente.")
print(f"Dimensiones de X: {X.shape} (Muestras x Genes)")
print(f"Dimensiones de y: {y.shape}\n")

# --- Vista general ---
print("🧬 Primeras filas de la matriz de expresión:")
print(X.head(10))  # Imprime tabla en consola

# --- Genes con mayor varianza ---
gene_var = X.var(axis=0).sort_values(ascending=False)
print("\n🔝 Genes con mayor varianza:")
print(gene_var.head(10))

# --- Distribución de clases ---
print("\n📊 Distribución de clases:")
print(y['Histological_Grade'].value_counts())

# --- Verificar coherencia ---
print("\n🔎 Coincidencia de muestras:", all(X.index == y.index))

# --- Visualización simple ---
# Ejemplo: histograma de una muestra (solo si tienes matplotlib instalado)
X.iloc[0].hist(bins=50)
plt.title("Distribución de niveles de expresión (ejemplo muestra 1)")
plt.xlabel("Nivel de expresión")
plt.ylabel("Frecuencia")
plt.show()
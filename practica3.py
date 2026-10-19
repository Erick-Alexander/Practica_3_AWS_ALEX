"""
PRÁCTICA 3: LIMPIEZA DE DATOS AVANZADA E IMPUTACIÓN MULTIVARIADA (KNN) - ora si para descomponer datos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer

# En esta sección se carga el Dataset dañado por el ruido, es decir el original.
data = {
    'edad': [25, 30, np.nan, 45, 22, 120, 35, 28, np.nan, 40],
    'ingreso_mensual': [15000, 22000, 18000, np.nan, 12000, 950000, 25000, 21000, 19000, np.nan],
    'scoring_credito': [650, 700, 680, 710, 590, 800, 670, 640, 620, 730]
}
df_dirty = pd.DataFrame(data)

print("=== 1. DATASET RUIDOSO ORIGINAL ===")
print(df_dirty)
print("\n")

# Aquí es donde por medio de los Cuartiles 1 y 3 se realiza el tratamiento de Outliers e Inconsistencias por IQR.
Q1 = df_dirty['ingreso_mensual'].quantile(0.25)
Q3 = df_dirty['ingreso_mensual'].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR

print("=== 2. RESULTADOS ESTADÍSTICOS DE CUARTILES (IQR) ===")
print(f"Primer Cuartil (Q1 - 25%): ${Q1:,.2f} MXN")
print(f"Tercer Cuartil (Q3 - 75%): ${Q3:,.2f} MXN")
print(f"Rango Intercuartílico (IQR): ${IQR:,.2f} MXN")
print(f"Límite Superior para Outliers: ${limite_superior:,.2f} MXN")

# Aquí es donde se reemplazan los valores del outliers que estan dañados que equivalen a NAN.
df_preprocesado = df_dirty.copy()
df_preprocesado.loc[df_preprocesado['ingreso_mensual'] > limite_superior, 'ingreso_mensual'] = np.nan
df_preprocesado.loc[df_preprocesado['edad'] > 100, 'edad'] = np.nan

print("\n=== DATASET CON OUTLIERS Y ERRORES CONVERTIDOS A NaN ===")
print(df_preprocesado)
print("\n")

# Por medio del KNN se realiza la imputación avanzada con KNNImputer (k=2)
imputer_knn = KNNImputer(n_neighbors=2)
df_clean_knn = pd.DataFrame(imputer_knn.fit_transform(df_preprocesado), columns=df_dirty.columns)

print("=== 3. DATASET LIMPIO CON KNN (k=2) ===")
print(df_clean_knn)
print("\n")

# Aquí es donde se realiza la comparación entre la media y la mediana.
imputer_media = SimpleImputer(strategy='mean')
imputer_mediana = SimpleImputer(strategy='median')

df_media = pd.DataFrame(imputer_media.fit_transform(df_preprocesado), columns=df_dirty.columns)
df_mediana = pd.DataFrame(imputer_mediana.fit_transform(df_preprocesado), columns=df_dirty.columns)

print("=== 4. COMPARATIVA DE VALORES IMPUTADOS EN REGISTROS CON FALTANTES ===")
comparativa = pd.DataFrame({
    'Edad_Original': df_dirty['edad'],
    'Edad_KNN_k2': df_clean_knn['edad'],
    'Edad_Media': df_media['edad'],
    'Ingreso_Original': df_dirty['ingreso_mensual'],
    'Ingreso_KNN_k2': df_clean_knn['ingreso_mensual'],
    'Ingreso_Media': df_media['ingreso_mensual']
})
print(comparativa)

# Aquí es donde gracias a aquella librería de plt se generan las imagenes para realizar la comparativa de forma gráfica.
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df_dirty['ingreso_mensual'], color='salmon')
plt.title('Ingreso Mensual Original (Con Outlier 950k)')
plt.ylabel('Monto (MXN)')

plt.subplot(1, 2, 2)
sns.boxplot(y=df_clean_knn['ingreso_mensual'], color='lightgreen')
plt.title('Ingreso Mensual Imputado (KNN k=2)')
plt.ylabel('Monto (MXN)')

plt.tight_layout()
plt.savefig('comparativa_outliers.png')
print("\n[OK] Gráfico guardado exitosamente como 'comparativa_outliers.png'")

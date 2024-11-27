import pandas as pd
import numpy as np

# Cargar el dataset desde el archivo CSV
df = pd.read_csv("envios.csv", sep=";")

# --- Ajustes Dinámicos ---
factor_distancia = df['distance'].mean()
factor_volumen = df['product_volume'].mean()
factor_flete = df['freight_value'].mean()

# --- Clase 1: `on_time_delivery_class` ---
# Calcular días estimados ajustados dinámicamente
estimated_days = 1 + (df['distance'] / factor_distancia) * 0.7 + (df['product_volume'] / factor_volumen) * 0.3
delivery_time_threshold = np.percentile(estimated_days, 60)  # Percentil dinámico
df['on_time_delivery_class'] = np.where(
    estimated_days <= delivery_time_threshold, 'On Time', 'Late'
)

# --- Clase 2: `freight_efficiency_class` ---
# Calcular eficiencia del envío
freight_efficiency_score = (
    df['freight_value'] / (df['product_volume'].replace(0, 1) + 1) +
    df['freight_value'] / (df['distance'] + 1)
)
freight_efficiency_threshold = np.percentile(freight_efficiency_score, 50)  # Mediana
df['freight_efficiency_class'] = np.where(
    freight_efficiency_score <= freight_efficiency_threshold, 'Efficient', 'Inefficient'
)

# --- Clase 3: `logistic_complexity_class` ---
# Calcular factores dinámicos basados en categorías
category_volume_median = df.groupby('product_category')['product_volume'].median().to_dict()
category_price_median = df.groupby('product_category')['product_price'].median().to_dict()

# Calcular un nuevo factor dinámico basado en volumen y precio
df['category_complexity_factor'] = df['product_category'].apply(
    lambda x: (category_volume_median.get(x, factor_volumen) / factor_volumen) +
              (category_price_median.get(x, df['product_price'].mean()) / df['product_price'].mean())
)

# Calcular score ajustado para complejidad logística
logistic_complexity_score = (
    (df['distance'] / factor_distancia) * 0.6 +
    (df['product_volume'] / factor_volumen) * 0.3 +
    df['category_complexity_factor'].fillna(5) * 0.1
)
logistic_complexity_threshold = np.percentile(logistic_complexity_score, 75)
df['logistic_complexity_class'] = np.where(
    logistic_complexity_score > logistic_complexity_threshold, 'High Complexity', 'Low Complexity'
)

# --- Guardar el Dataset Etiquetado ---
df.to_csv("envios_etiquetados.csv", sep=";", index=False)

# Mostrar Conteo de Clases
print("Conteo de Clases de Entrega a Tiempo:")
print(df['on_time_delivery_class'].value_counts())
print("\nConteo de Clases de Eficiencia del Envío:")
print(df['freight_efficiency_class'].value_counts())
print("\nConteo de Clases de Complejidad Logística:")
print(df['logistic_complexity_class'].value_counts())

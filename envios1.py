import pandas as pd
import numpy as np

# Cargar el dataset
df = pd.read_csv("envios.csv", sep=";")

# --- Ajustes Dinámicos ---
factor_distancia = df['distance'].mean()
factor_volumen = df['product_volume'].mean()

# --- Clase 1: `on_time_delivery_class` ---
# Calcular días estimados
estimated_days = 1 + df['distance'] / factor_distancia + df['product_volume'] / factor_volumen
# Usar la mediana como threshold
delivery_time_threshold = np.median(estimated_days)
# Etiquetar
df['on_time_delivery_class'] = np.where(
    estimated_days <= delivery_time_threshold, 'On Time', 'Late'
)

# --- Clase 2: `freight_efficiency_class` ---
# Calcular el score de eficiencia
freight_cost_score = (
    df['freight_value'] / df['product_price'].replace(0, 1) +
    df['distance'] / factor_distancia +
    df['product_volume'] / factor_volumen
)
# Usar la mediana como threshold
freight_efficiency_threshold = np.median(freight_cost_score)
# Etiquetar
df['freight_efficiency_class'] = np.where(
    freight_cost_score <= freight_efficiency_threshold, 'Efficient', 'Inefficient'
)

# --- Clase 3: `logistic_complexity_class` ---
# Listas de categorías por nivel de complejidad
high_complexity_categories = [
    'furniture_decor', 'electronics', 'home_appliances', 'musical_instruments',
    'furniture_living_room', 'furniture_bedroom', 'construction_tools_lights'
]
low_complexity_categories = [
    'health_beauty', 'books_technical', 'toys', 'watches_gifts', 'stationery'
]

# Asignar factores
df['category_complexity_factor'] = df['product_category'].apply(
    lambda x: 10 if x in high_complexity_categories else 
              2 if x in low_complexity_categories else 
              5  # Complejidad media (default)
)

# Calcular Score de Complejidad Logística
logistic_complexity_score = (
    (df['distance'] / factor_distancia) +
    (df['product_volume'] / factor_volumen) +
    df['category_complexity_factor']
)
# Umbral Dinámico
logistic_complexity_threshold = np.median(logistic_complexity_score)
# Etiquetar Complejidad Logística
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

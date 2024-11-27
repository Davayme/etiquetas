import pandas as pd
import numpy as np

# Cargar el dataset
df = pd.read_csv("envios.csv", sep=";")

# --- Calcular Derivados ---
df['total_order_cost'] = df['product_price'] + df['freight_value']

# --- Clase 1: High Freight Value Class ---
freight_value_threshold = np.percentile(df['freight_value'], 75)
df['high_freight_value_class'] = np.where(
    df['freight_value'] > freight_value_threshold, 'High Freight', 'Low Freight'
)

# --- Clase 2: Long Distance Delivery Class ---
distance_mean = df['distance'].mean()
df['long_distance_class'] = np.where(
    df['distance'] > distance_mean, 'Long Distance', 'Short Distance'
)

# --- Clase 3: High Revenue Order Class ---
total_order_cost_threshold = np.percentile(df['total_order_cost'], 75)
df['high_revenue_class'] = np.where(
    df['total_order_cost'] > total_order_cost_threshold, 'High Revenue', 'Low Revenue'
)

# --- Guardar el Dataset Etiquetado ---
df.to_csv("envios_etiquetados_nuevo.csv", sep=";", index=False)

# Mostrar Conteo de Clases
print("Conteo de Clases de High Freight Value:")
print(df['high_freight_value_class'].value_counts())
print("\nConteo de Clases de Long Distance Delivery:")
print(df['long_distance_class'].value_counts())
print("\nConteo de Clases de High Revenue Order:")
print(df['high_revenue_class'].value_counts())

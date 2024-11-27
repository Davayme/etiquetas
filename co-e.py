import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset etiquetado
df = pd.read_csv("envios_etiquetados_nuevo.csv", sep=";")

# --- Codificar Variables Categóricas ---
# Convertir las clases en valores numéricos para calcular correlaciones
class_map = {
    'High Freight': 1, 'Low Freight': 0,
    'Long Distance': 1, 'Short Distance': 0,
    'High Revenue': 1, 'Low Revenue': 0
}
df['high_freight_value_class'] = df['high_freight_value_class'].map(class_map)
df['long_distance_class'] = df['long_distance_class'].map(class_map)
df['high_revenue_class'] = df['high_revenue_class'].map(class_map)

# Codificar otras variables categóricas como numéricas (si aplica)
df['product_category'] = df['product_category'].astype('category').cat.codes
df['customer_city_name'] = df['customer_city_name'].astype('category').cat.codes
df['seller_city_name'] = df['seller_city_name'].astype('category').cat.codes

# --- Seleccionar Features Relevantes ---
features = [
    'distance', 'product_volume', 'freight_value', 'product_price', 
    'total_order_cost', 'product_category', 'customer_city_name', 'seller_city_name'
]
target_classes = ['high_freight_value_class', 'long_distance_class', 'high_revenue_class']

# --- Matriz de Correlación entre Variables ---
# Calcular la matriz de correlación
corr_matrix_features = df[features].corr()

# Visualizar la matriz de correlación entre variables
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix_features, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Matriz de Correlación Entre Features")
plt.show()

# --- Matriz de Correlación entre Features y Clases ---
# Calcular la matriz de correlación entre features y clases
corr_matrix_classes = df[features + target_classes].corr()

# Visualizar la correlación entre features y clases
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix_classes[target_classes], annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Matriz de Correlación Entre Features y Clases")
plt.show()

# --- Imprimir Correlaciones Fuertes ---
correlation_threshold = 0.5
strong_corr_features_classes = corr_matrix_classes.loc[features, target_classes][
    (corr_matrix_classes.loc[features, target_classes] > correlation_threshold) |
    (corr_matrix_classes.loc[features, target_classes] < -correlation_threshold)
]

print("Correlaciones fuertes entre features y clases:")
print(strong_corr_features_classes)

# Guardar correlaciones fuertes en CSV
strong_corr_features_classes.reset_index().to_csv("correlaciones_fuertes_features_clases.csv", index=False)

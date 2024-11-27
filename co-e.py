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
    'distance', 'product_volume', 'product_price', 'total_order_cost', 
    'product_category', 'customer_city_name', 'seller_city_name'
]
target_classes = ['high_freight_value_class', 'long_distance_class', 'high_revenue_class']

# --- Matriz de Correlación entre Variables ---
# Calcular la matriz de correlación
corr_matrix_features = df[features].corr()

# Visualizar la matriz de correlación entre variables
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix_features, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Matriz de Correlación Entre Variables", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# --- Matriz de Correlación entre Features y Clases ---
# Calcular la matriz de correlación entre features y clases
corr_matrix_classes = df[features + target_classes].corr()

# Visualizar la correlación entre features y clases
plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix_classes[target_classes], annot=True, cmap="coolwarm", fmt=".2f",
    cbar=True, linewidths=0.5, square=True, annot_kws={"size": 10}
)
plt.title("Matriz de Correlación Entre Variables y Clases", fontsize=14, weight='bold')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# --- Graficar Distribuciones de Variables Numéricas ---
variables_to_plot = ['distance', 'product_volume', 'freight_value', 'product_price', 'total_order_cost']

plt.figure(figsize=(16, 12))
for i, var in enumerate(variables_to_plot, start=1):
    plt.subplot(3, 2, i)  # Organizar los gráficos en una cuadrícula
    sns.histplot(df[var], kde=True, bins=20, color="blue", alpha=0.7)
    plt.title(f"Distribución de '{var}'", fontsize=14, weight='bold')
    plt.ylabel("Frecuencia", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ajustar el diseño y agregar un título general
plt.suptitle("Distribución de Variables Numéricas del Dataset", fontsize=18, weight='bold', y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar espacio para el título general
plt.show()

# --- Imprimir Correlaciones Fuertes ---
correlation_threshold = 0.5
strong_corr_features_classes = corr_matrix_classes.loc[features, target_classes][
    (corr_matrix_classes.loc[features, target_classes] > correlation_threshold) |
    (corr_matrix_classes.loc[features, target_classes] < -correlation_threshold)
]

print("Correlaciones fuertes entre features y clases:")
print(strong_corr_features_classes)



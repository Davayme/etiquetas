import pandas as pd
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

# --- Imprimir Correlaciones Fuertes ---
correlation_threshold = 0.5
strong_corr_features_classes = corr_matrix_classes.loc[features, target_classes][
    (corr_matrix_classes.loc[features, target_classes] > correlation_threshold) |
    (corr_matrix_classes.loc[features, target_classes] < -correlation_threshold)
]

print("Correlaciones fuertes entre features y clases:")
print(strong_corr_features_classes)

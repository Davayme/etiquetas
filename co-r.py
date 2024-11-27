import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
data = pd.read_csv("reviews_binomial.csv", sep=';')

# Codificar variables categóricas para análisis de correlación
# Las convertimos a numéricas para Spearman/Pearson según corresponda
categorical_features = ['product_category', 'product_brand', 'customer_city_name', 'seller_city_name', 'order_month', 'order_day']
data_encoded = data.copy()

# Mapear categóricos a números para simplificación
for col in categorical_features:
    data_encoded[col] = data_encoded[col].astype("category").cat.codes

# También convertir las clases binomiales a valores numéricos
class_features = ['product_return_class', 'satisfaction_class_binomial']
class_map = {
    'Baja Probabilidad': 0, 'Alta Probabilidad': 1,
    'Satisfecho': 1, 'No Satisfecho': 0
}
for cls in class_features:
    data_encoded[cls] = data_encoded[cls].map(class_map)

# Selección de columnas relevantes
features = ['review_score', 'freight_value', 'order_price', 
            'product_category', 'product_brand', 'customer_city_name', 
            'seller_city_name', 'order_month', 'order_year', 'order_day']

# Matriz de correlación entre features
corr_features = data_encoded[features].corr(method='pearson')

# Matriz de correlación entre features y clases
corr_with_classes = data_encoded[features + class_features].corr(method='pearson')

# Visualizar matriz de correlación entre features
plt.figure(figsize=(12, 8))
sns.heatmap(corr_features, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación entre Features")
plt.show()

# Visualizar matriz de correlación entre features y clases
plt.figure(figsize=(12, 8))
sns.heatmap(corr_with_classes, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación entre Features y Clases")
plt.show()

# Identificar correlaciones fuertes entre features
high_corr_features = corr_features[
    (corr_features > 0.7) | (corr_features < -0.7)
].stack().reset_index()
high_corr_features.columns = ['Feature 1', 'Feature 2', 'Correlation']
high_corr_features = high_corr_features[high_corr_features['Feature 1'] != high_corr_features['Feature 2']]
print("Correlaciones fuertes entre features:")
print(high_corr_features)

# Identificar correlaciones fuertes entre features y clases
high_corr_with_classes = corr_with_classes[class_features][
    (corr_with_classes[class_features] > 0.7) | (corr_with_classes[class_features] < -0.7)
].dropna(how="all")
print("Correlaciones fuertes entre features y clases:")
print(high_corr_with_classes)

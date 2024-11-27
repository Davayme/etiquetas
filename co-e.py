import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset etiquetado
df = pd.read_csv("envios_etiquetados2.csv", sep=";")

# --- Codificar Variables Categóricas ---
# Convertir las clases en valores numéricos para calcular correlaciones
class_map = {
    'On Time': 0, 'Late': 1,
    'Efficient': 0, 'Inefficient': 1,
    'Low Complexity': 0, 'High Complexity': 1
}
df['on_time_delivery_class'] = df['on_time_delivery_class'].map(class_map)
df['freight_efficiency_class'] = df['freight_efficiency_class'].map(class_map)
df['logistic_complexity_class'] = df['logistic_complexity_class'].map(class_map)

# Codificar otras variables categóricas como numéricas (si aplica)
df['product_category'] = df['product_category'].astype('category').cat.codes
df['customer_city_name'] = df['customer_city_name'].astype('category').cat.codes
df['seller_city_name'] = df['seller_city_name'].astype('category').cat.codes

# --- Seleccionar Features y Clases ---
features = [
    'distance', 'product_volume', 'freight_value', 'product_price', 
    'product_category', 'customer_city_name', 'seller_city_name'
]
classes = ['on_time_delivery_class', 'freight_efficiency_class', 'logistic_complexity_class']

# --- Matriz de Correlación de Features ---
# Calcular la matriz de correlación entre variables
corr_features = df[features].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_features, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Matriz de Correlación Entre Features")
plt.show()

# --- Matriz de Correlación entre Features y Clases ---
# Calcular la matriz de correlación entre variables y clases
corr_features_classes = df[features + classes].corr()

# Mostrar sólo las correlaciones entre features y clases
corr_with_classes = corr_features_classes.loc[features, classes]

plt.figure(figsize=(10, 6))
sns.heatmap(corr_with_classes, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Matriz de Correlación Entre Features y Clases")
plt.show()

# --- Identificar Correlaciones Fuertes ---
correlation_threshold = 0.7
# Correlaciones fuertes entre features
strong_corr_features = corr_features.stack().reset_index()
strong_corr_features.columns = ['Feature 1', 'Feature 2', 'Correlation']
strong_corr_features = strong_corr_features[
    (strong_corr_features['Correlation'] > correlation_threshold) &
    (strong_corr_features['Feature 1'] != strong_corr_features['Feature 2'])
]

# Correlaciones fuertes entre features y clases
strong_corr_classes = corr_with_classes.stack().reset_index()
strong_corr_classes.columns = ['Feature', 'Class', 'Correlation']
strong_corr_classes = strong_corr_classes[
    (strong_corr_classes['Correlation'] > correlation_threshold) |
    (strong_corr_classes['Correlation'] < -correlation_threshold)
]

# Imprimir Resultados
print("\nCorrelaciones fuertes entre features:")
print(strong_corr_features)

print("\nCorrelaciones fuertes entre features y clases:")
print(strong_corr_classes)



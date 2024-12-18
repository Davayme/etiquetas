import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv("synthetic_dataset.csv", sep=";")

# 1. Estadísticas descriptivas
print("Estadísticas descriptivas:")
print(df.describe(include='all'))

# 2. Revisión de valores nulos
print("\nValores nulos en cada columna:")
print(df.isnull().sum())

# 3. Análisis de la distribución de variables numéricas
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True, bins=50, color='skyblue')
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()

# 4. Correlación entre variables numéricas
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de correlación entre variables numéricas')
plt.show()

# 5. Análisis de la distribución de variables categóricas
categorical_columns = df.select_dtypes(include=[np.object, 'category']).columns.tolist()

for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=df, palette='Set2')
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.show()

# 6. Análisis de correlaciones específicas
# Correlación entre "review_score" y otras variables
print("\nCorrelaciones específicas con 'review_score':")
review_correlations = df.corr()['review_score'].sort_values(ascending=False)
print(review_correlations)

# 7. Boxplots para ver posibles valores atípicos (outliers)
for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[col], color='lightcoral')
    plt.title(f'Boxplot de {col}')
    plt.xlabel(col)
    plt.show()

# 8. Análisis de la distribución de las categorías más frecuentes
# Ver las categorías más comunes para 'product_category' y 'customer_region'
print("\nFrecuencia de categorías más comunes en 'product_category':")
print(df['product_category'].value_counts().head(10))

print("\nFrecuencia de regiones más comunes en 'customer_region':")
print(df['customer_region'].value_counts().head(10))

# 9. Comprobación de sesgos en el precio y en las puntuaciones de reseñas
plt.figure(figsize=(10, 6))
sns.histplot(df['order_price'], kde=True, bins=50, color='orange')
plt.title('Distribución de Order Price')
plt.xlabel('Order Price')
plt.ylabel('Frecuencia')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x=df['review_score'], data=df, palette='muted')
plt.title('Distribución de Review Score')
plt.xlabel('Review Score')
plt.ylabel('Frecuencia')
plt.show()

# 10. Análisis de patrones y relaciones en las variables de ventas
# Analizar la relación entre el precio y la puntuación de la reseña
plt.figure(figsize=(10, 6))
sns.scatterplot(x='order_price', y='review_score', data=df, color='green')
plt.title('Relación entre el precio y la puntuación de la reseña')
plt.xlabel('Order Price')
plt.ylabel('Review Score')
plt.show()

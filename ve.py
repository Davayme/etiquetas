import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def verificar_dataset(filepath):
    # Cargar datos
    df = pd.read_csv(filepath, sep=';')
    
    # 1. Estadísticas básicas
    print("\n=== Estadísticas Básicas ===")
    print(df.describe())
    print("\nValores faltantes:")
    print(df.isnull().sum())
    
    # 2. Distribución de categorías
    plt.figure(figsize=(15, 6))
    df['product_category'].value_counts().plot(kind='bar')
    plt.title('Distribución de Categorías')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 3. Matriz de correlación
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación')
    plt.tight_layout()
    plt.show()
    
    # 4. Detección de outliers
    print("\n=== Detección de Outliers ===")
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)][col]
        print(f"\n{col}: {len(outliers)} outliers ({(len(outliers)/len(df)*100):.2f}%)")
    
    # 5. Verificar balance de datos
    plt.figure(figsize=(10, 6))
    df['review_score'].value_counts().plot(kind='bar')
    plt.title('Distribución de Review Scores')
    plt.show()
    
    # 6. Distribución geográfica
    print("\n=== Distribución Geográfica ===")
    print("\nVendedores por región:")
    print(df['seller_region'].value_counts(normalize=True))
    print("\nCompradores por región:")
    print(df['customer_region'].value_counts(normalize=True))
    
    # 7. Verificar relaciones precio-categoría
    plt.figure(figsize=(15, 6))
    sns.boxplot(x='product_category', y='order_price', data=df)
    plt.xticks(rotation=45)
    plt.title('Distribución de Precios por Categoría')
    plt.tight_layout()
    plt.show()
    
    # 8. Análisis temporal
    plt.figure(figsize=(12, 6))
    df.groupby('order_month')['order_price'].mean().plot(kind='line')
    plt.title('Precio Promedio por Mes')
    plt.show()
    
    # 9. Test de normalidad
    print("\n=== Test de Normalidad (Shapiro-Wilk) ===")
    for col in numerical_cols[:5]:  # Limitamos a 5 columnas
        stat, p = stats.shapiro(df[col].sample(min(1000, len(df))))
        print(f"{col}: p-value = {p:.10f}")

if __name__ == "__main__":
    verificar_dataset("synthetic_dataset.csv")
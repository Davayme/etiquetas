import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv('labeled_dataset.csv', delimiter=';')

# Definir las características numéricas y categóricas
caracteristicas_numericas = [
    'product_weight', 'shipping_time_days', 'freight_value', 
    'order_price', 'product_discount', 'review_score', 
    'inventory_stock_level', 'seller_response_time', 'customer_complaints'
]

caracteristicas_categoricas = [
    'order_year', 'order_month', 'order_day_of_week', 
    'customer_region', 'seller_region', 'product_category', 
    'customer_gender'
]

# Separar las características de entrada (X) y la variable objetivo (y)
X = df[caracteristicas_numericas + caracteristicas_categoricas]
y = df['customer_satisfaction']  # La variable objetivo es 'customer_satisfaction'

# Preprocesamiento de las características: normalización de numéricas y One-Hot Encoding de categóricas
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Normalización de características numéricas
escalador = MinMaxScaler()
X_numericas = escalador.fit_transform(X[caracteristicas_numericas])

# One-Hot Encoding de características categóricas
codificador = OneHotEncoder(sparse_output=False)
X_categoricas = codificador.fit_transform(X[caracteristicas_categoricas])

# Unir ambas partes
X_preprocesado = np.hstack([X_numericas, X_categoricas])

# Aplicar t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_preprocesado)

# Visualizar los datos proyectados en 2D
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.colorbar(label='Satisfacción del cliente')
plt.title('Distribución de datos con t-SNE')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.show()

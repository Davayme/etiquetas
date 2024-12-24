import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Cargar y preparar los datos
df = pd.read_csv('labeled_dataset.csv', sep=';')

# 2. Seleccionar features relevantes para t-SNE
features = ['review_score', 'shipping_time_days', 'product_discount', 
            'customer_complaints', 'seller_response_time', 'order_price',
            'freight_value', 'product_weight', 'inventory_stock_level']

# 3. Preparar los datos
X = df[features].values
y = df['customer_satisfaction'].values

# 4. Tomar una muestra aleatoria de 40,000 registros
np.random.seed(42)
sample_size = 40000
indices = np.random.choice(len(X), sample_size, replace=False)
X_sample = X[indices]
y_sample = y[indices]

# 5. Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# 6. Reducir dimensionalidad primero con PCA
n_components = min(9, len(features))  # Ajustar a número válido de componentes
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# 7. Aplicar t-SNE con parámetros optimizados
tsne = TSNE(n_components=2,
            perplexity=30,
            early_exaggeration=12,
            learning_rate=200,
            n_iter=1000,
            random_state=42)

X_tsne = tsne.fit_transform(X_pca)

# 8. Crear visualización mejorada
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                     c=y_sample,
                     cmap='viridis',
                     alpha=0.6,
                     s=50)

plt.colorbar(scatter)
plt.title('Proyección t-SNE - Mejor Separación de Clases')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')

# Añadir leyenda
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=plt.cm.viridis(i/2), 
                            label=f'Clase {i}', markersize=10)
                  for i in range(3)]
plt.legend(handles=legend_elements, title='Clases', loc='upper right')

plt.tight_layout()
plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Imprimir estadísticas de la distribución de clases
print("\nDistribución de clases en la muestra:")
unique, counts = np.unique(y_sample, return_counts=True)
for clase, count in zip(unique, counts):
    percentage = (count/len(y_sample))*100
    print(f"Clase {clase}: {count} muestras ({percentage:.2f}%)")
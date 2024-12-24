import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Paso 1: Crear un dataset sintético con 3 clases
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=3, random_state=42)

# Paso 2: Normalizar los datos (opcional pero recomendable para t-SNE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 3: Aplicar t-SNE para reducir las dimensiones a 2D
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Paso 4: Visualizar la distribución de las clases
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=20)
plt.title('Distribución de Clases con t-SNE')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.colorbar(scatter, label='Clase')
plt.show()

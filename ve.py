import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generar datos sintéticos
n_samples = 1000
X, y = make_blobs(
    n_samples=n_samples,
    centers=3,
    n_features=20,
    random_state=42,
    cluster_std=2.0
)

# Aplicar t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    random_state=42
)
X_tsne = tsne.fit_transform(X)

# Visualizar resultados
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y,
    cmap='viridis'
)

plt.title('Proyección t-SNE - Datos Sintéticos')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.colorbar(scatter)
plt.legend(*scatter.legend_elements(), title="Clases")
plt.show()
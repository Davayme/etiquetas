import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('etiquetado_envios.csv', sep=';')

# Definir características
numeric_features = ['product_price',
    'product_volume',
    'freight_value',
    'distance']
categorical_features = ['customer_region', 'seller_region']

# Preprocesar datos numéricos
scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[numeric_features])

# Preprocesar datos categóricos
categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_categorical = categorical_encoder.fit_transform(df[categorical_features])

# Combinar características
X = np.hstack([X_numeric, X_categorical])

# Las etiquetas ya son numéricas (0, 1, 2), no necesitas mapeo
y = df['shipping_label'].values  # Elimina el .map(label_map)

# Muestrear 50000 datos
np.random.seed(42)
indices = np.random.choice(len(X), 50000, replace=False)
X_sample = X[indices]
y_sample = y[indices]

# Aplicar t-SNE
print("Iniciando TSNE...")
tsne = TSNE(
    n_components=2,
    perplexity=30,        # Aumentado de 5 a 30 para mejor balance local-global
    early_exaggeration=12,
    learning_rate=200,    # Valor específico en lugar de 'auto'
    random_state=42,
    n_iter=2000,         # Aumentado para mejor convergencia
    init='pca'           # Mejor inicialización usando PCA
)
X_tsne = tsne.fit_transform(X_sample)
print("TSNE completado!")

# Crear visualización
plt.figure(figsize=(12, 8))

# Definir colores para las clases
colors = ['#FF9999', '#66B2FF', '#99FF99']  # Rojo claro, Azul claro, Verde claro
cmap = plt.cm.colors.ListedColormap(colors)

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y_sample,
    cmap=cmap,
    alpha=0.6,
    s=30
)

plt.title('Distribución de Rentabilidad de Envíos (t-SNE)\n50,000 muestras', fontsize=14, pad=20)
plt.xlabel('Componente 1', fontsize=12)
plt.ylabel('Componente 2', fontsize=12)

# Crear leyenda
labels = ['Baja Rentabilidad', 'Media Rentabilidad', 'Alta Rentabilidad']
legend_elements = [
    plt.Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor=colors[i],
        label=labels[i],
        markersize=10
    ) for i in range(len(labels))
]

plt.legend(
    handles=legend_elements,
    title='Niveles de Rentabilidad',
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

# Añadir información sobre las características utilizadas
plt.figtext(
    1.15, 0.5,
    f'Características utilizadas:\n\nNuméricas:\n' + 
    '\n'.join([f'- {f}' for f in numeric_features]) +
    '\n\nCategorías:\n' +
    '\n'.join([f'- {f}' for f in categorical_features]),
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8)
)

plt.tight_layout()
plt.savefig('tsne_rentabilidad_50k.png', dpi=300, bbox_inches='tight')
plt.close()

# Imprimir estadísticas de la distribución de clases
print("\nDistribución de clases en la visualización:")
unique, counts = np.unique(y_sample, return_counts=True)
for label, count in zip(labels, counts):
    percentage = (count / len(y_sample)) * 100
    print(f"{label}: {count} muestras ({percentage:.1f}%)")
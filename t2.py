import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar y verificar datos
df = pd.read_csv('labeled_dataset.csv', sep=';')
print("Datos faltantes:\n", df.isnull().sum())

# 2. Preparar features
numeric_features = [
   'review_score', 
   'customer_complaints',
   'shipping_time_days', 
   'freight_value',
   'order_price'
]

categorical_features = [
   'customer_region',
   'seller_region'
]

# 3. Codificar variables categóricas con OneHotEncoder
categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_categorical = categorical_encoder.fit_transform(df[categorical_features])

# 4. Normalizar datos numéricos
scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[numeric_features])

# 5. Combinar features
X = np.hstack([X_numeric, X_categorical])
y = df['customer_satisfaction'].values

# 6. Tomar muestra estratificada
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=len(df)-40000, random_state=42)
for train_idx, _ in sss.split(X, y):
   X_sample = X[train_idx]
   y_sample = y[train_idx]

# 7. Reducir dimensionalidad con PCA
pca = PCA(n_components=0.95)  # Mantener 95% de la varianza
X_pca = pca.fit_transform(X_sample)
print(f"Componentes PCA utilizados: {pca.n_components_}")

# 8. Cross-validation para diferentes perplexities
perplexities = [5, 30, 50]
kf = KFold(n_splits=3, shuffle=True, random_state=42)

for perplexity in perplexities:
   tsne = TSNE(
       n_components=2,
       perplexity=perplexity,
       early_exaggeration=12,
       learning_rate='auto',
       n_iter=1000,
       random_state=42
   )
   
   X_tsne = tsne.fit_transform(X_pca)
   
      # Visualización
   plt.figure(figsize=(12, 8))

   # Definir colores fijos que coincidan con el gráfico
   colors = ['#3F007D', '#35B779', '#FDE725']  # Morado, Verde, Amarillo
   cmap = plt.cm.colors.ListedColormap(colors)

   scatter = plt.scatter(
      X_tsne[:, 0], 
      X_tsne[:, 1],
      c=y_sample,
      cmap=cmap,
      alpha=0.6,
      s=50
   )

   plt.title('Distribución de Satisfacción del Cliente (t-SNE)', fontsize=14)
   plt.xlabel('Componente 1', fontsize=12)
   plt.ylabel('Componente 2', fontsize=12)

   # Leyenda con colores correctos
   labels = ['Insatisfecho', 'Neutral', 'Satisfecho']
   legend_elements = [plt.Line2D([0], [0], 
                              marker='o', 
                              color='w',
                              markerfacecolor=colors[i],
                              label=labels[i],
                              markersize=10)
                  for i in range(len(labels))]

   plt.legend(handles=legend_elements, 
            title='Nivel de Satisfacción',
            bbox_to_anchor=(1.05, 1),
            loc='upper left')

   plt.tight_layout()
   plt.savefig(f'tsne_viz_perplexity_{perplexity}.png', dpi=300, bbox_inches='tight')
   plt.show()

# 9. Métricas de calidad
print("\nDistribución de clases:")
print(pd.Series(y_sample).value_counts(normalize=True))

# 10. Guardar resultados procesados
np.save('X_tsne_processed.npy', X_tsne)
np.save('y_processed.npy', y_sample)
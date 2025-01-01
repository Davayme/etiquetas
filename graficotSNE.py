import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('labeled_dataset.csv', sep=';')

# Features
numeric_features = ['review_score', 'customer_complaints', 'shipping_time_days', 'freight_value', 'order_price']
categorical_features = ['customer_region', 'seller_region']

# Preprocesar datos
categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_categorical = categorical_encoder.fit_transform(df[categorical_features])
X_numeric = StandardScaler().fit_transform(df[numeric_features])
X = np.hstack([X_numeric, X_categorical])
y = df['customer_satisfaction'].values

# Muestrear 50000 datos
np.random.seed(42)
indices = np.random.choice(len(X), 50000, replace=False)
X_sample = X[indices]
y_sample = y[indices]

# t-SNE
tsne = TSNE(n_components=2, perplexity=5, early_exaggeration=12, learning_rate='auto', random_state=42)
X_tsne = tsne.fit_transform(X_sample)

# Visualización
plt.figure(figsize=(12, 8))
colors = ['#3F007D', '#35B779', '#FDE725']
cmap = plt.cm.colors.ListedColormap(colors)

scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap=cmap, alpha=0.6, s=50)

plt.title('Distribución de Satisfacción del Cliente (t-SNE)', fontsize=14)
plt.xlabel('Componente 1', fontsize=12)
plt.ylabel('Componente 2', fontsize=12)

labels = ['Insatisfecho', 'Neutral', 'Satisfecho']
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                           label=labels[i], markersize=10) for i in range(len(labels))]

plt.legend(handles=legend_elements, title='Nivel de Satisfacción', 
         bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
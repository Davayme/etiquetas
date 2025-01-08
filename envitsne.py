import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Leer el CSV
df = pd.read_csv('etiquetado_envios.csv', sep=';')

# Tomar 50,000 muestras aleatorias
n_samples = min(50000, len(df))
df_sample = df.sample(n=n_samples, random_state=42)

# Hacer one-hot encoding de las variables categóricas
df_encoded = pd.get_dummies(
    df_sample,
    columns=['customer_region', 'seller_region'],
    prefix=['customer', 'seller']
)

# Seleccionar las características numéricas
numeric_features = [
    'product_price',
    'product_volume',
    'freight_value',
    'distance'
]

# Combinar características numéricas con one-hot encoding
feature_columns = numeric_features + [col for col in df_encoded.columns 
                                    if col.startswith(('customer_', 'seller_'))]

# Preparar los datos
X = df_encoded[feature_columns].values
y = df_encoded['shipping_label'].values

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=5,
    random_state=42,
    early_exaggeration=12, learning_rate='auto',
    n_jobs=-1
)
X_tsne = tsne.fit_transform(X_scaled)

# Crear DataFrame con resultados
df_tsne = pd.DataFrame(
    X_tsne,
    columns=['t-SNE 1', 't-SNE 2']
)
df_tsne['Label'] = y

# Configurar visualización
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    data=df_tsne,
    x='t-SNE 1',
    y='t-SNE 2',
    hue='Label',
    palette='deep',
    alpha=0.6
)

plt.title('Visualización t-SNE de Datos de Envío')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

# Añadir leyenda
plt.legend(title='Clase de Envío')

# Mostrar el gráfico
plt.show()

# Imprimir información sobre la distribución de clases
print("\nDistribución de clases:")
print(df_sample['shipping_label'].value_counts(normalize=True).multiply(100).round(2))

# Imprimir información sobre las dimensiones después del one-hot encoding
print("\nDimensiones de los datos después del one-hot encoding:")
print(f"Número de características: {X.shape[1]}")
print("\nCaracterísticas incluidas:")
for feat in feature_columns:
    print(f"- {feat}")
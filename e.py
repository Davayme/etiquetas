import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv("reviews_binomial.csv", sep=";")

# Eliminar espacios en los nombres de las columnas
df.columns = df.columns.str.strip()

# Preprocesamiento: Convertir columnas categóricas a numéricas
# Usamos Label Encoding para columnas categóricas que no son ordinales

from sklearn.preprocessing import LabelEncoder

# Columnas categóricas a convertir
categorical_columns = ['product_category', 'product_brand', 'comment_type', 'order_month', 
                       'order_day', 'order_trimester', 'order_semester', 'product_return_class', 
                       'satisfaction_class_binomial']

# Inicializamos el encoder
label_encoder = LabelEncoder()

# Aplicamos el LabelEncoder a las columnas categóricas
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Seleccionamos las columnas numéricas para calcular la correlación
numeric_columns = ['review_score', 'freight_value', 'order_price', 'product_price', 
                   'product_volume', 'product_category', 'product_brand']

# Cálculo de la matriz de correlación entre las variables numéricas
correlation_matrix = df[numeric_columns].corr()

# Visualización de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Matriz de Correlación entre Variables')
plt.show()

# Si quieres ver la correlación entre las clases (product_return_class y satisfaction_class_binomial)
# Mostramos la correlación entre las clases y las variables numéricas
correlation_classes = df[['product_return_class', 'satisfaction_class_binomial'] + numeric_columns].corr()

# Visualización de la correlación con las clases
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_classes, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Matriz de Correlación con las Clases')
plt.show()

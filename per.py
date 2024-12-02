import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset con el delimitador correcto (punto y coma)
df = pd.read_csv('reviews_binomial.csv', sep=';')

# Mapear los meses a valores numéricos (enero=1, febrero=2, ..., diciembre=12)
month_mapping = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
    'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}

# Convertir las columnas de texto a valores numéricos
df['order_month'] = df['order_month'].map(month_mapping)
df['order_year'] = pd.to_numeric(df['order_year'], errors='coerce')
df['order_day'] = df['order_day'].map({
    'lunes': 1, 'martes': 2, 'miércoles': 3, 'jueves': 4, 'viernes': 5, 'sábado': 6, 'domingo': 7
})

# Aplicar One-Hot Encoding a las columnas categóricas
df_encoded = pd.get_dummies(df, columns=['product_category', 'product_brand'])

# Convertir las columnas booleanas a enteros
df_encoded = df_encoded.astype({col: 'int' for col in df_encoded.columns if df_encoded[col].dtype == 'bool'})

# Filtrar las clases 'Satisfecho' y 'No Satisfecho'
df_satisfecho = df_encoded[df_encoded['satisfaction_class_binomial'] == 'Satisfecho']
df_no_satisfecho = df_encoded[df_encoded['satisfaction_class_binomial'] == 'No Satisfecho']

# Tomar 2000 ejemplos aleatorios de cada clase
df_satisfecho_sampled = df_satisfecho.sample(n=2000, random_state=42)
df_no_satisfecho_sampled = df_no_satisfecho.sample(n=2000, random_state=42)

# Concatenar las dos clases para obtener el dataset balanceado
df_balanced = pd.concat([df_satisfecho_sampled, df_no_satisfecho_sampled])
df_balanced = df_balanced.reset_index(drop=True)

# Seleccionar las características y la etiqueta
X = df_balanced[['review_score', 'order_price'] + 
                [col for col in df_encoded.columns if 'product_category_' in col or 'product_brand_' in col]].values
y = np.where(df_balanced['satisfaction_class_binomial'] == 'Satisfecho', 1, -1)

# Inicialización de los pesos y el sesgo
w = np.zeros(X.shape[1])
b = 0
alpha = 0.01  # Tasa de aprendizaje
epochs = 100  # Reducir las épocas para acelerar la ejecución

# Función de activación vectorizada
def sign_activation(z):
    return np.where(z >= 0, 1, -1)

# Entrenamiento del Perceptrón (Vectorizado)
for epoch in range(epochs):
    # Calcular la salida (z) y la predicción (y_hat)
    z = np.dot(X, w) + b
    y_hat = sign_activation(z)

    # Encontrar los índices donde la predicción es incorrecta
    errors = y != y_hat
    if not np.any(errors):  # Si no hay errores, detener el entrenamiento
        break

    # Actualizar los pesos y el sesgo para las muestras mal clasificadas
    w += alpha * np.dot(errors * y, X)
    b += alpha * np.sum(errors * y)

# Resultados del entrenamiento
print("Pesos finales:", w)
print("Sesgo final:", b)

# Evaluación
z_test = np.dot(X, w) + b
y_test_hat = sign_activation(z_test)
accuracy = np.mean(y_test_hat == y) * 100
print(f'Precisión del modelo: {accuracy:.2f}%')

# Graficar la línea de decisión
plt.figure(figsize=(8, 6))

# Mostrar las predicciones y las etiquetas
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='o', label='Datos Reales', alpha=0.5)

# Graficar la línea de decisión
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = sign_activation(np.dot(np.c_[xx.ravel(), yy.ravel()], w[:2]) + b)  # Usando solo las dos primeras características

# Cambiar la forma de Z para que coincida con la forma de los datos en el meshgrid
Z = Z.reshape(xx.shape)

# Graficar la frontera de decisión
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.title('Frontera de Decisión del Perceptrón')
plt.xlabel('Review Score')
plt.ylabel('Order Price')

plt.show()

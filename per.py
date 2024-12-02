import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Leer el CSV
df = pd.read_csv('reviews_binomial.csv', sep=';')

# Seleccionar una muestra de 5000 registros aleatorios
df_sampled = df.sample(n=5000, random_state=42)

# Seleccionar las columnas relevantes para el modelo (todas las 4 características para el entrenamiento)
X = df_sampled[['review_score', 'order_price', 'product_price', 'freight_value']].values
y = df_sampled['satisfaction_class_binomial'].map({'Satisfecho': 1, 'No Satisfecho': 0}).values

# Dividir el conjunto de datos en entrenamiento y test (80% entrenamiento, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicialización de pesos (empezamos en 0)
weights = np.zeros(X_train.shape[1])
bias = 0

# Definir el umbral (hardlim) para la activación
hardlim = 0.5

# Función de activación
def activation_function(x):
    return 1 if x >= hardlim else 0

# Entrenamiento del perceptrón con vectorización
learning_rate = 0.1
epochs = 10  # Reducir las épocas para acelerar la prueba

for epoch in range(epochs):
    # Calculamos la salida del perceptrón para todo el conjunto de entrenamiento
    net_input = np.dot(X_train, weights) + bias
    predictions = np.where(net_input >= hardlim, 1, 0)
    
    # Calculamos el error y actualizamos los pesos y sesgo
    error = y_train - predictions
    weights += learning_rate * np.dot(X_train.T, error)  # Actualización vectorizada
    bias += learning_rate * np.sum(error)  # Actualización sesgo vectorizada

# Calcular el Error de Entrenamiento
net_input_train = np.dot(X_train, weights) + bias
predictions_train = np.where(net_input_train >= hardlim, 1, 0)
training_error = np.mean(predictions_train != y_train) * 100  # Error de entrenamiento en porcentaje

# Evaluar el modelo con los pesos entrenados para test
net_input_test = np.dot(X_test, weights) + bias
predictions_test = np.where(net_input_test >= hardlim, 1, 0)
test_error = np.mean(predictions_test != y_test) * 100  # Error de test en porcentaje

# Mostrar precisión y errores
print(f'Error de Entrenamiento: {training_error:.2f}%')
print(f'Error de Testeo: {test_error:.2f}%')
print(f'Pesos finales: {weights}')
print(f'Sesgo final: {bias}')

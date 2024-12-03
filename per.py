import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Leer el CSV
df = pd.read_csv('reviews_binomial.csv', sep=';')

# Seleccionar las columnas relevantes para el modelo
X = df[['review_score', 'freight_value', 'customer_city_name', 'order_month']]

# One-Hot Encoding de las columnas categóricas
encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(X[['customer_city_name', 'order_month']])

# Convertir la matriz dispersa a una matriz densa
encoded_columns = encoded_columns.toarray()

# Combinar las variables numéricas con las codificadas
X = np.hstack((X[['review_score', 'freight_value']].values, encoded_columns))

# Etiquetas
y = df['satisfaction_class_binomial'].map({'Satisfecho': 1, 'No Satisfecho': 0}).values

# Inicialización de los parámetros
learning_rate = 0.1
epochs = 1000
hardlim = 0.5

# Inicializar la validación cruzada estratificada con 5 folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Almacenaremos los errores de cada fold
train_errors = []
test_errors = []

# Entrenamiento y validación cruzada
for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    print(f"\n** Iniciando Fold {fold} **")
    
    # Dividir los datos en entrenamiento y test
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Inicialización de pesos y sesgo para cada fold
    weights = np.zeros(X_train.shape[1])
    bias = 0

    # Entrenamiento del perceptrón
    for epoch in range(epochs):
        if epoch % 100 == 0:  # Imprimir cada 100 épocas
            print(f"Epoch {epoch} - Predicciones: {np.where(np.dot(X_train, weights) + bias >= hardlim, 1, 0)[:5]} | Pesos: {weights[:5]} | Sesgo: {bias:.2f}")
        
        # Calculamos la salida del perceptrón para todo el conjunto de entrenamiento
        net_input = np.dot(X_train, weights) + bias
        predictions = (net_input >= hardlim).astype(int)
        
        # Calculamos el error y actualizamos los pesos y sesgo
        error = y_train - predictions
        weights += learning_rate * np.dot(X_train.T, error)
        bias += learning_rate * np.sum(error)
    
    # Calcular el Error de Entrenamiento para este fold
    net_input_train = np.dot(X_train, weights) + bias
    predictions_train = (net_input_train >= hardlim).astype(int)
    train_error = np.mean(predictions_train != y_train) * 100
    train_errors.append(train_error)

    # Evaluar el modelo con los datos de test
    net_input_test = np.dot(X_test, weights) + bias
    predictions_test = (net_input_test >= hardlim).astype(int)
    test_error = np.mean(predictions_test != y_test) * 100
    test_errors.append(test_error)
    
    print(f"** Fin del Fold {fold} **")
    print(f"Error de Entrenamiento: {train_error:.2f}% | Error de Testeo: {test_error:.2f}%")

# Promediar los errores de los folds
average_train_error = np.mean(train_errors)
average_test_error = np.mean(test_errors)

# Mostrar los resultados
print(f"\nError de Entrenamiento promedio: {average_train_error:.2f}%")
print(f"Error de Testeo promedio: {average_test_error:.2f}%")

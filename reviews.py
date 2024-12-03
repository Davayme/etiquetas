import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Sección 1: Carga y Preprocesamiento de Datos
df = pd.read_csv('reviews_binomial.csv', sep=';')

# Selección de las columnas relevantes para el modelo
X = df[['review_score', 'freight_value', 'customer_city_name', 'order_month']]

# Codificación de las columnas categóricas (como el nombre de la ciudad y el mes del pedido)
encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(X[['customer_city_name', 'order_month']])
encoded_columns = encoded_columns.toarray()

# Combinar las variables numéricas con las codificadas
X = np.hstack((X[['review_score', 'freight_value']].values, encoded_columns))

# Normalización de las características numéricas
scaler = MinMaxScaler()
X_numeric = X[:, :2]  # Las primeras dos columnas son las características numéricas
X_numeric_scaled = scaler.fit_transform(X_numeric)
X[:, :2] = X_numeric_scaled  # Reemplazar las columnas numéricas escaladas

# Etiquetas: Convertir 'Satisfecho' y 'No Satisfecho' a 1 y 0
y = df['satisfaction_class_binomial'].map({'Satisfecho': 1, 'No Satisfecho': 0}).values

# Reducción de dimensionalidad con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X[:, :2])  # Usamos solo las dos primeras características para PCA

# Sección 2: Inicialización de Parámetros
learning_rate = 0.001  # Ajuste de la tasa de aprendizaje
epochs = 1000  # Número de iteraciones
hardlim = 0.5  # Función de activación (umbral)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Validación cruzada estratificada

train_errors = []  # Lista para los errores de entrenamiento
test_errors = []  # Lista para los errores de testeo

# Sección 3: Entrenamiento y Evaluación
for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    print(f"\n** Iniciando Fold {fold} **")
    
    # División de los datos en entrenamiento y test
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Inicialización de pesos y sesgo
    weights = np.zeros(X_train.shape[1])
    bias = 0
    
    # Sección 3.1: Entrenamiento del Perceptrón
    for epoch in range(epochs):
        if epoch % 100 == 0:  # Imprimir cada 100 épocas
            predictions = np.where(np.dot(X_train, weights) + bias >= hardlim, 1, 0)
            print(f"Epoch {epoch} - Predicciones: {predictions[:5]} | Pesos: {weights[:5]} | Sesgo: {bias:.2f}")
        
        # Cálculo de la salida del perceptrón
        net_input = np.dot(X_train, weights) + bias
        predictions = (net_input >= hardlim).astype(int)
        
        # Cálculo del error y actualización de los pesos y sesgo
        error = y_train - predictions
        weights += learning_rate * np.dot(X_train.T, error)
        bias += learning_rate * np.sum(error)
    
    # Sección 3.2: Evaluación del Modelo
    # Calcular error de entrenamiento
    net_input_train = np.dot(X_train, weights) + bias
    predictions_train = (net_input_train >= hardlim).astype(int)
    train_error = np.mean(predictions_train != y_train) * 100
    train_errors.append(train_error)
    
    # Evaluar el modelo con datos de test
    net_input_test = np.dot(X_test, weights) + bias
    predictions_test = (net_input_test >= hardlim).astype(int)
    test_error = np.mean(predictions_test != y_test) * 100
    test_errors.append(test_error)
    
    print(f"** Fin del Fold {fold} **")
    print(f"Error de Entrenamiento: {train_error:.2f}% | Error de Testeo: {test_error:.2f}%")

    # Sección 3.3: Mostrar Pesos Finales Resumidos
    print("\n** Pesos Finales (Top 5 más importantes) **")
    important_weights = np.argsort(np.abs(weights))[-5:]  # Top 5 pesos más importantes
    print(f"Pesos Finales: {weights[important_weights]}")

# Promediar los errores de los 5 folds
average_train_error = np.mean(train_errors)
average_test_error = np.mean(test_errors)

# Sección 4: Resultados Finales
print(f"\nError de Entrenamiento promedio: {average_train_error:.2f}%")
print(f"Error de Testeo promedio: {average_test_error:.2f}%")

# Sección 5: Graficar la frontera de decisión
def graficar_frontera_decision(X, y, weights, pca):
    """Genera un gráfico de la frontera de decisión utilizando PCA"""
    # Reducir las características a 2 componentes principales
    X_reducido_pca = pca.transform(X[:, :2])

    # Crear una malla para visualizar la frontera
    h = .02  # Resolución de la malla
    x_min, x_max = X_reducido_pca[:, 0].min() - 1, X_reducido_pca[:, 0].max() + 1
    y_min, y_max = X_reducido_pca[:, 1].min() - 1, X_reducido_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predecir las clases en la malla (usamos los mejores pesos)
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], weights[:2]) + weights[2]  # Incluyendo el sesgo
    Z = (Z >= hardlim).astype(int)
    Z = Z.reshape(xx.shape)

    # Graficar la frontera de decisión
    plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.RdYlBu)

    # Graficar los puntos de los datos reales
    plt.scatter(X_reducido_pca[:, 0], X_reducido_pca[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k")
    plt.title("Frontera de Decisión del Perceptrón (PCA)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.colorbar(label="Clase")
    plt.show()

# Graficar la frontera de decisión con los pesos obtenidos en el último fold
graficar_frontera_decision(X, y, weights, pca)

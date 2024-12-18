import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Cargar el dataset
df = pd.read_csv('labeled_dataset.csv', delimiter=';')

# Definir las características numéricas y categóricas
caracteristicas_numericas = [
    'product_weight', 'shipping_time_days', 'freight_value', 
    'order_price', 'product_discount', 'review_score', 
    'inventory_stock_level', 'seller_response_time', 'customer_complaints'
]

caracteristicas_categoricas = [
    'order_year', 'order_month', 'order_day_of_week', 
    'customer_region', 'seller_region', 'product_category', 
    'customer_gender'
]

# Separar las características de entrada (X) y la variable objetivo (y)
X = df[caracteristicas_numericas + caracteristicas_categoricas]
y = df['customer_satisfaction']  # La variable objetivo es 'customer_satisfaction'

# Preprocesamiento de las características: normalización de numéricas y One-Hot Encoding de categóricas
escalador = MinMaxScaler()
X_numericas = escalador.fit_transform(X[caracteristicas_numericas])

# One-Hot Encoding de características categóricas
codificador = OneHotEncoder(sparse_output=False)
X_categoricas = codificador.fit_transform(X[caracteristicas_categoricas])

# Unir ambas partes
X_preprocesado = np.hstack([X_numericas, X_categoricas])

# Convertir las etiquetas 'customer_satisfaction' en formato adecuado
y = y.values

# Inicialización de pesos y sesgos
def inicializar_pesos(n_entradas, n_clases):
    # Inicialización según distribución normal con media 0 y desviación estándar 1
    pesos = np.random.uniform(-1, 1, (n_entradas, n_clases)) * 0.01
    sesgos = np.zeros((1, n_clases))
    return pesos, sesgos

# Función softmax
def softmax(z):
    exps = np.exp(z - np.max(z, axis=1, keepdims=True))  # Resta el valor máximo para estabilidad numérica
    return exps / np.sum(exps, axis=1, keepdims=True)

# Cálculo de la entropía cruzada (log loss) con regularización L2
def entropia_cruzada(y, y_pred, pesos, lambda_reg=0.01):
    m = y.shape[0]
    y = np.argmax(y, axis=1)  # Convertir y de one-hot a índices
    log_loss = -np.sum(np.log(y_pred[np.arange(m), y])) / m
    # Regularización L2: penalización por los pesos grandes
    regularizacion = (lambda_reg / (2 * m)) * np.sum(np.square(pesos))
    return log_loss + regularizacion

# Actualización de pesos utilizando gradiente descendente con regularización L2
def actualizar_pesos(X, y, y_pred, pesos, sesgos, tasa_aprendizaje=0.01, lambda_reg=0.01):
    m = X.shape[0]
    
    # Gradiente de la entropía cruzada
    grad_pesos = np.dot(X.T, (y_pred - y)) / m
    grad_sesgos = np.sum(y_pred - y, axis=0, keepdims=True) / m
    
    # Regularización L2: calcular el gradiente de la penalización
    grad_pesos += (lambda_reg / m) * pesos
    
    # Actualización de los pesos y sesgos
    pesos -= tasa_aprendizaje * grad_pesos
    sesgos -= tasa_aprendizaje * grad_sesgos
    return pesos, sesgos

# Entrenamiento con gradiente descendente
def entrenar(X, y, n_clases, n_iter=1000, tasa_aprendizaje=0.001, lambda_reg=0.01):
    n_entradas = X.shape[1]
    pesos, sesgos = inicializar_pesos(n_entradas, n_clases)
    
    for i in range(n_iter):
        # Calcular z = X * W + b
        z = np.dot(X, pesos) + sesgos
        
        # Aplicar la función softmax
        y_pred = softmax(z)
        
        # Calcular la entropía cruzada con regularización
        perdida = entropia_cruzada(y, y_pred, pesos, lambda_reg)
        
        # Actualizar pesos y sesgos
        pesos, sesgos = actualizar_pesos(X, y, y_pred, pesos, sesgos, tasa_aprendizaje, lambda_reg)
        
        if i % 100 == 0:
            print(f"Iteración {i}, Pérdida: {perdida:.4f}")
    
    return pesos, sesgos

# Función de predicción
def predecir(X, pesos, sesgos):
    z = np.dot(X, pesos) + sesgos
    y_pred = softmax(z)
    return np.argmax(y_pred, axis=1)

# Validación cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
errores_entrenamiento = []
errores_prueba = []

# Entrenamiento y evaluación con validación cruzada
for fold, (train_idx, test_idx) in enumerate(cv.split(X_preprocesado, y), 1):
    print(f"\n--- Fold {fold} ---")
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test = X_preprocesado[train_idx], X_preprocesado[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Convertir y_train y y_test a formato one-hot
    y_train_onehot = np.eye(3)[y_train]  # Asumimos 3 clases: 0, 1, 2
    y_test_onehot = np.eye(3)[y_test]
    
    # Entrenamiento
    pesos, sesgos = entrenar(X_train, y_train_onehot, n_clases=3, n_iter=1000, tasa_aprendizaje=0.50, lambda_reg=0.01)
    
    # Predicciones
    y_train_pred = predecir(X_train, pesos, sesgos)
    y_test_pred = predecir(X_test, pesos, sesgos)
    
    # Calcular precisión y error
    precision_entrenamiento = accuracy_score(y_train, y_train_pred)
    precision_prueba = accuracy_score(y_test, y_test_pred)
    
    error_entrenamiento = 1 - precision_entrenamiento
    error_prueba = 1 - precision_prueba
    
    errores_entrenamiento.append(error_entrenamiento)
    errores_prueba.append(error_prueba)
    
    print(f"Precisión entrenamiento: {precision_entrenamiento:.4f}, Error entrenamiento: {error_entrenamiento:.4f}")
    print(f"Precisión prueba: {precision_prueba:.4f}, Error prueba: {error_prueba:.4f}")

# Promedio de errores
print(f"\nError promedio de entrenamiento: {np.mean(errores_entrenamiento):.4f}")
print(f"Error promedio de prueba: {np.mean(errores_prueba):.4f}")

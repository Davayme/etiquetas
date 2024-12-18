import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

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
    pesos = np.random.uniform(-1, 1, (n_entradas, n_clases)) * 0.01
    sesgos = np.zeros((1, n_clases))
    return pesos, sesgos

# Función softmax
def softmax(z):
    exps = np.exp(z - np.max(z, axis=1, keepdims=True))  # Resta el valor máximo para estabilidad numérica
    return exps / np.sum(exps, axis=1, keepdims=True)

# Cálculo de la entropía cruzada (log loss) con regularización L2
def entropia_cruzada(y, y_pred):
    m = y.shape[0]
    y = np.argmax(y, axis=1)  # Convertir y de one-hot a índices
    log_loss = -np.sum(np.log(y_pred[np.arange(m), y])) / m
    return log_loss

# Actualización de pesos utilizando gradiente descendente
def actualizar_pesos(X, y, y_pred, pesos, sesgos, tasa_aprendizaje=0.01):
    m = X.shape[0]
    grad_pesos = np.dot(X.T, (y_pred - y)) / m
    grad_sesgos = np.sum(y_pred - y, axis=0, keepdims=True) / m
    pesos -= tasa_aprendizaje * grad_pesos
    sesgos -= tasa_aprendizaje * grad_sesgos
    return pesos, sesgos

# Entrenamiento con gradiente descendente
def entrenar(X, y, n_clases, n_iter=1000, tasa_aprendizaje=0.001):
    n_entradas = X.shape[1]
    pesos, sesgos = inicializar_pesos(n_entradas, n_clases)
    
    for i in range(n_iter):
        z = np.dot(X, pesos) + sesgos
        y_pred = softmax(z)
        perdida = entropia_cruzada(y, y_pred)
        pesos, sesgos = actualizar_pesos(X, y, y_pred, pesos, sesgos, tasa_aprendizaje)
        
        if i % 100 == 0:
            print(f"Iteración {i}, Pérdida: {perdida:.4f}")
    
    return pesos, sesgos

# Función de predicción
def predecir(X, pesos, sesgos):
    z = np.dot(X, pesos) + sesgos
    y_pred = softmax(z)
    return np.argmax(y_pred, axis=1)

# Agregar visualización de t-SNE
def graficar_tsne(X, y, titulo="Proyección t-SNE"):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title(titulo)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.legend(*scatter.legend_elements(), title="Clases")
    plt.colorbar(scatter)
    plt.show()


# Validación cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
errores_entrenamiento = []
errores_prueba = []
confusion_matrices = []
mejor_error_prueba = float('inf')
mejor_pesos = None
mejor_sesgos = None
mejor_fold = -1

mejor_X_train = None
mejor_y_train_pred = None

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
    pesos, sesgos = entrenar(X_train, y_train_onehot, n_clases=3, n_iter=1000, tasa_aprendizaje=0.50)
    
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
    
    # Guardar el mejor fold (con menor error de prueba)
    if error_prueba < mejor_error_prueba:
        mejor_error_prueba = error_prueba
        mejor_pesos = pesos
        mejor_sesgos = sesgos
        mejor_fold = fold
        mejor_X_train = X_train
        mejor_y_train_pred = y_train_pred
    
    # Calcular la matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    confusion_matrices.append(cm)

# Promedio de errores
print(f"\nError promedio de entrenamiento: {np.mean(errores_entrenamiento):.4f}")
print(f"Error promedio de prueba: {np.mean(errores_prueba):.4f}")

# Mostrar la matriz de confusión del mejor fold
print(f"\nMatriz de confusión del mejor fold (Fold {mejor_fold}):")
cm_mejor_fold = confusion_matrices[mejor_fold - 1]
print(cm_mejor_fold)

# Mostrar la matriz de confusión del mejor fold
plt.figure(figsize=(6,6))
sns.heatmap(cm_mejor_fold, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Clase 0', 'Clase 1', 'Clase 2'], yticklabels=['Clase 0', 'Clase 1', 'Clase 2'])
plt.title(f'Matriz de Confusión - Mejor Fold (Fold {mejor_fold})')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

graficar_tsne(mejor_X_train, mejor_y_train_pred, titulo=f"Proyección t-SNE - Mejor Fold (Fold {mejor_fold})")
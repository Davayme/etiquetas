# Importaciones necesarias para el modelo
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Carga y división de características
datos = pd.read_csv('labeled_dataset.csv', delimiter=';')

# Variables que influyen en la satisfacción del cliente
variables_numericas = [
   'product_weight', 'shipping_time_days', 'freight_value', 
   'order_price', 'product_discount', 'review_score', 'seller_response_time', 'customer_complaints'
]

variables_categoricas = [
   'order_year', 'order_month', 'order_day_of_week', 
   'customer_region', 'seller_region', 'product_category', 
   'customer_gender'
]

# Preparación de datos para el modelo
caracteristicas = datos[variables_numericas + variables_categoricas]
variable_objetivo = datos['customer_satisfaction']

# Normalización de variables numéricas
normalizador = MinMaxScaler()
datos_numericos_normalizados = normalizador.fit_transform(caracteristicas[variables_numericas])

# Codificación de variables categóricas
codificador_categorico = OneHotEncoder(sparse_output=False)
datos_categoricos_codificados = codificador_categorico.fit_transform(caracteristicas[variables_categoricas])

# Unión de datos preprocesados
datos_preprocesados = np.hstack([datos_numericos_normalizados, datos_categoricos_codificados])
etiquetas = variable_objetivo.values

def inicializar_parametros(numero_entradas, numero_clases):
   """Inicialización de pesos y sesgos del modelo"""
   pesos_iniciales = np.random.uniform(-1, 1, (numero_entradas, numero_clases)) * 0.01
   sesgos_iniciales = np.zeros((1, numero_clases))
   return pesos_iniciales, sesgos_iniciales

def calcular_softmax(valores):
   """Función de activación softmax con estabilidad numérica"""
   exponenciales = np.exp(valores - np.max(valores, axis=1, keepdims=True))
   return exponenciales / np.sum(exponenciales, axis=1, keepdims=True)

def calcular_perdida(etiquetas_reales, predicciones):
   """Cálculo de la función de pérdida (entropía cruzada)"""
   numero_muestras = etiquetas_reales.shape[0]
   indices_reales = np.argmax(etiquetas_reales, axis=1)
   return -np.sum(np.log(predicciones[np.arange(numero_muestras), indices_reales])) / numero_muestras

def actualizar_parametros(datos, etiquetas_reales, predicciones, pesos, sesgos, tasa_aprendizaje=0.01):
   """Actualización de parámetros usando gradiente descendente"""
   numero_muestras = datos.shape[0]
   gradiente_pesos = np.dot(datos.T, (predicciones - etiquetas_reales)) / numero_muestras
   gradiente_sesgos = np.sum(predicciones - etiquetas_reales, axis=0, keepdims=True) / numero_muestras
   
   pesos_actualizados = pesos - tasa_aprendizaje * gradiente_pesos
   sesgos_actualizados = sesgos - tasa_aprendizaje * gradiente_sesgos
   return pesos_actualizados, sesgos_actualizados

def entrenar_modelo(datos, etiquetas, numero_clases, iteraciones=1000, tasa_aprendizaje=0.001):
   """Entrenamiento completo del modelo"""
   numero_entradas = datos.shape[1]
   pesos, sesgos = inicializar_parametros(numero_entradas, numero_clases)
   
   for iteracion in range(iteraciones):
       salida_lineal = np.dot(datos, pesos) + sesgos
       predicciones = calcular_softmax(salida_lineal)
       perdida = calcular_perdida(etiquetas, predicciones)
       pesos, sesgos = actualizar_parametros(datos, etiquetas, predicciones, pesos, sesgos, tasa_aprendizaje)
       
       if iteracion % 100 == 0:
           print(f"Iteración {iteracion}, Pérdida: {perdida:.4f}")
   
   return pesos, sesgos

def realizar_prediccion(datos, pesos, sesgos):
   """Realiza predicciones usando el modelo entrenado"""
   salida_lineal = np.dot(datos, pesos) + sesgos
   predicciones = calcular_softmax(salida_lineal)
   return np.argmax(predicciones, axis=1)

# Configuración de la validación cruzada
validacion_cruzada = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
errores_entrenamiento = []
errores_prueba = []
matrices_confusion = []
mejor_error_prueba = float('inf')
mejores_parametros = {'pesos': None, 'sesgos': None, 'fold': -1}
mejores_datos = {'X_train': None, 'y_train_pred': None}

# Entrenamiento y evaluación con validación cruzada
for fold, (indices_entrenamiento, indices_prueba) in enumerate(validacion_cruzada.split(datos_preprocesados, etiquetas), 1):
   print(f"\n--- Iteración {fold} de Validación Cruzada ---")
   
   # División de datos en entrenamiento y prueba
   X_entrenamiento = datos_preprocesados[indices_entrenamiento]
   X_prueba = datos_preprocesados[indices_prueba]
   y_entrenamiento = etiquetas[indices_entrenamiento]
   y_prueba = etiquetas[indices_prueba]
   
   # Conversión a formato one-hot
   y_entrenamiento_onehot = np.eye(3)[y_entrenamiento]
   y_prueba_onehot = np.eye(3)[y_prueba]
   
   # Entrenamiento del modelo
   pesos, sesgos = entrenar_modelo(X_entrenamiento, y_entrenamiento_onehot, 
                                 numero_clases=3, iteraciones=1000, tasa_aprendizaje=0.85)
   
   # Realizar predicciones
   predicciones_entrenamiento = realizar_prediccion(X_entrenamiento, pesos, sesgos)
   predicciones_prueba = realizar_prediccion(X_prueba, pesos, sesgos)
   
   # Cálculo de métricas
   precision_entrenamiento = accuracy_score(y_entrenamiento, predicciones_entrenamiento)
   precision_prueba = accuracy_score(y_prueba, predicciones_prueba)
   
   error_entrenamiento = 1 - precision_entrenamiento
   error_prueba = 1 - precision_prueba
   
   # Almacenamiento de métricas
   errores_entrenamiento.append(error_entrenamiento)
   errores_prueba.append(error_prueba)
   
   print(f"Precisión en entrenamiento: {precision_entrenamiento:.4f}, Error: {error_entrenamiento:.4f}")
   print(f"Precisión en prueba: {precision_prueba:.4f}, Error: {error_prueba:.4f}")
   
   # Actualización del mejor modelo
   if error_prueba < mejor_error_prueba:
       mejor_error_prueba = error_prueba
       mejores_parametros['pesos'] = pesos
       mejores_parametros['sesgos'] = sesgos
       mejores_parametros['fold'] = fold
       mejores_datos['X_train'] = X_entrenamiento
       mejores_datos['y_train_pred'] = predicciones_entrenamiento
   
   # Cálculo de matriz de confusión
   matrices_confusion.append(confusion_matrix(y_prueba, predicciones_prueba))

# Resumen de resultados
print(f"\nError promedio en entrenamiento: {np.mean(errores_entrenamiento):.4f}")
print(f"Error promedio en prueba: {np.mean(errores_prueba):.4f}")

# Visualización de la matriz de confusión del mejor modelo
print(f"\nMatriz de confusión del mejor modelo (Iteración {mejores_parametros['fold']}):")
mejor_matriz_confusion = matrices_confusion[mejores_parametros['fold'] - 1]
print(mejor_matriz_confusion)

# Visualización gráfica de la matriz de confusión
plt.figure(figsize=(6,6))
sns.heatmap(mejor_matriz_confusion, annot=True, fmt='.2f', cmap='Blues',
           xticklabels=['Insatisfecho', 'Neutral', 'Satisfecho'],
           yticklabels=['Insatisfecho', 'Neutral', 'Satisfecho'])
plt.title(f'Matriz de Confusión - Mejor Modelo (Iteración {mejores_parametros["fold"]})')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()

              
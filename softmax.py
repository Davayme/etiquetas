"""
Clasificador Softmax para Predicción de Satisfacción del Cliente
=============================================================
Este módulo implementa un clasificador Softmax para predecir la satisfacción 
del cliente (Insatisfecho, Neutral, Satisfecho) basado en características del pedido.

Flujo del algoritmo:
1. Preprocesamiento de datos (normalización y codificación)
2. Propagación hacia adelante: Z = XW + b, luego softmax(Z)
3. Propagación hacia atrás: cálculo de gradientes
4. Actualización de parámetros
5. Evaluación con validación cruzada
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ClasificadorSoftmax:
    def __init__(self):
        # 1. Definición de características
        self.variables_numericas = [
            'review_score',         # Puntuación de la reseña
            'customer_complaints',   # Número de quejas
            'shipping_time_days',    # Tiempo de envío
            'freight_value',         # Valor del flete
            'order_price'           # Precio del pedido
        ]
        
        self.variables_categoricas = [
            'customer_region',       # Región del cliente
            'seller_region'          # Región del vendedor
        ]
        
        # Preprocesadores
        self.normalizador = MinMaxScaler()
        self.codificador = OneHotEncoder(sparse_output=False)
        
    def preprocesar_datos(self, archivo):
        """
        Paso 1: Preprocesamiento de datos
        - Normaliza variables numéricas a [0,1]
        - Codifica variables categóricas como one-hot
        """
        # Cargar datos
        datos = pd.read_csv(archivo, delimiter=';')
        
        # Preprocesar características
        X_num = self.normalizador.fit_transform(datos[self.variables_numericas])
        X_cat = self.codificador.fit_transform(datos[self.variables_categoricas])
        
        # Combinar características
        X = np.hstack([X_num, X_cat])
        y = datos['customer_satisfaction'].values
        
        return X, y
    
    def propagacion_adelante(self, X, pesos, sesgos):
        """
        Paso 2: Propagación hacia adelante
        - Calcula Z = XW + b
        - Aplica softmax(Z) para obtener probabilidades
        """
        # Función softmax con estabilidad numérica
        Z = np.dot(X, pesos) + sesgos
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def propagacion_atras(self, X, y_real, y_pred):
        """
        Paso 3: Propagación hacia atrás
        - Calcula gradientes para pesos y sesgos usando la función de pérdida cross-entropy
        """
        num_muestras = X.shape[0]
        error = y_pred - y_real  # Error de predicción
        grad_pesos = np.dot(X.T, error) / num_muestras
        grad_sesgos = np.sum(error, axis=0, keepdims=True) / num_muestras
        
        return grad_pesos, grad_sesgos
    
    def entrenar(self, X, y, num_clases=3, epocas=1000, tasa_aprendizaje=0.85):
        """
        Paso 4: Entrenamiento
        - Inicializa parámetros
        - Realiza propagación hacia adelante y atrás
        - Actualiza parámetros
        """
        # Inicialización
        num_caracteristicas = X.shape[1]
        pesos = np.random.uniform(-1, 1, (num_caracteristicas, num_clases)) * 0.01
        sesgos = np.zeros((1, num_clases))
        y_onehot = np.eye(num_clases)[y]
        
        # Entrenamiento
        for epoca in range(epocas):
            # Propagación hacia adelante
            y_pred = self.propagacion_adelante(X, pesos, sesgos)
            
            # Propagación hacia atrás
            grad_pesos, grad_sesgos = self.propagacion_atras(X, y_onehot, y_pred)
            
            # Actualizar parámetros
            pesos -= tasa_aprendizaje * grad_pesos
            sesgos -= tasa_aprendizaje * grad_sesgos
            
            # Mostrar progreso cada 100 épocas
            if epoca % 100 == 0:
                perdida = -np.mean(np.sum(y_onehot * np.log(y_pred + 1e-15), axis=1))
                print(f"Iteración {epoca}, Pérdida: {perdida:.4f}")
        
        return pesos, sesgos
    
    def evaluar(self, X, y, n_particiones=5):
        """
        Paso 5: Evaluación con validación cruzada
        """
        kfold = StratifiedKFold(n_splits=n_particiones, shuffle=True, random_state=42)
        resultados = {
            'error_entrenamiento': [], 
            'error_prueba': [],
            'matrices_confusion': [], 
            'mejor_modelo': None,
            'mejor_error': float('inf')
        }
        
        for particion, (idx_train, idx_test) in enumerate(kfold.split(X, y), 1):
            print(f"\n--- Fold {particion} ---")
            
            # Dividir datos
            X_train, X_test = X[idx_train], X[idx_test]
            y_train, y_test = y[idx_train], y[idx_test]
            
            # Entrenar modelo
            pesos, sesgos = self.entrenar(X_train, y_train)
            
            # Obtener predicciones
            y_train_pred = np.argmax(self.propagacion_adelante(X_train, pesos, sesgos), axis=1)
            y_test_pred = np.argmax(self.propagacion_adelante(X_test, pesos, sesgos), axis=1)
            
            # Calcular métricas
            precision_train = accuracy_score(y_train, y_train_pred)
            precision_test = accuracy_score(y_test, y_test_pred)
            
            error_train = 1 - precision_train
            error_test = 1 - precision_test
            
            # Mostrar resultados como en la imagen original
            print(f"Precisión entrenamiento: {precision_train:.4f}, Error entrenamiento: {error_train:.4f}")
            print(f"Precisión prueba: {precision_test:.4f}, Error prueba: {error_test:.4f}")
            
            # Guardar resultados
            resultados['error_entrenamiento'].append(error_train)
            resultados['error_prueba'].append(error_test)
            resultados['matrices_confusion'].append(confusion_matrix(y_test, y_test_pred))
            
            # Actualizar mejor modelo
            if error_test < resultados['mejor_error']:
                resultados['mejor_error'] = error_test
                resultados['mejor_modelo'] = {
                    'pesos': pesos, 
                    'sesgos': sesgos, 
                    'particion': particion
                }
        
        return resultados
    
    def mostrar_resultados(self, resultados):
        """
        Visualización de resultados
        """
        print("\nResultados Finales:")
        print(f"Error promedio entrenamiento: {np.mean(resultados['error_entrenamiento']):.4f}")
        print(f"Error promedio prueba: {np.mean(resultados['error_prueba']):.4f}")
        
        mejor_particion = resultados['mejor_modelo']['particion']
        mejor_matriz = resultados['matrices_confusion'][mejor_particion - 1]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            mejor_matriz,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=['Insatisfecho', 'Neutral', 'Satisfecho'],
            yticklabels=['Insatisfecho', 'Neutral', 'Satisfecho']
        )
        plt.title(f'Matriz de Confusión - Mejor Modelo (Partición {mejor_particion})')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.show()

def main():
    # Crear y ejecutar el clasificador
    modelo = ClasificadorSoftmax()
    
    # 1. Preprocesar datos
    X, y = modelo.preprocesar_datos('labeled_dataset_fuzzy.csv')
    
    # 2. Evaluar modelo
    resultados = modelo.evaluar(X, y)
    
    # 3. Mostrar resultados
    modelo.mostrar_resultados(resultados)

if __name__ == "__main__":
    main()
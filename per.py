import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class PreprocesamientoDatos:
    def __init__(self, archivo_csv):
        """
        Clase encargada de cargar y preprocesar los datos.
        :param archivo_csv: Ruta al archivo CSV con los datos.
        """
        self.df = pd.read_csv(archivo_csv, sep=';')
        self.X = None
        self.y = None

    def preprocesar(self):
        """
        Preprocesa los datos: selecciona características, codifica variables categóricas, 
        normaliza y convierte las etiquetas.
        """
        # Selección de columnas relevantes
        self.X = self.df[['review_score', 'freight_value', 'customer_city_name', 'order_month']]

        # Codificación de las columnas categóricas
        codificador = OneHotEncoder()
        columnas_codificadas = codificador.fit_transform(self.X[['customer_city_name', 'order_month']])
        columnas_codificadas = columnas_codificadas.toarray()

        # Combina las variables numéricas con las codificadas
        self.X = np.hstack((self.X[['review_score', 'freight_value']].values, columnas_codificadas))

        # Normalización de las características numéricas
        normalizador = MinMaxScaler()
        X_numerico = self.X[:, :2]  # Las primeras dos columnas son las características numéricas
        X_numerico_escalado = normalizador.fit_transform(X_numerico)
        self.X[:, :2] = X_numerico_escalado  # Reemplaza las columnas numéricas escaladas

        # Convertir 'Satisfecho' y 'No Satisfecho' a 1 y 0
        self.y = self.df['satisfaction_class_binomial'].map({'Satisfecho': 1, 'No Satisfecho': 0}).values

        return self.X, self.y

class Perceptron:
    def __init__(self, tasa_aprendizaje=0.001, epocas=1000, umbral=0.5):
        """
        Inicializa el perceptrón con los parámetros necesarios.

        :param tasa_aprendizaje: Tasa de aprendizaje para actualizar los pesos.
        :param epocas: Número de iteraciones para el entrenamiento.
        :param umbral: Umbral de activación.
        """
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        self.umbral = umbral
        self.pesos = None
        self.sesgo = None

    def entrenar(self, X, y):
        """
        Entrena el perceptrón utilizando los datos proporcionados.

        :param X: Características de entrenamiento.
        :param y: Etiquetas de entrenamiento.
        """
        # Inicialización de pesos y sesgo
        self.pesos = np.zeros(X.shape[1])
        self.sesgo = 0

        # Entrenamiento
        for epoca in range(self.epocas):
            if epoca % 100 == 0:  # Imprimir cada 100 épocas
                print(f"Época {epoca} - Pesos: {self.pesos[:5]} | Sesgo: {self.sesgo:.2f}")
            
            # Cálculo de la salida del perceptrón
            entrada_neta = np.dot(X, self.pesos) + self.sesgo
            predicciones = (entrada_neta >= self.umbral).astype(int)

            # Cálculo del error y actualización de los pesos y el sesgo
            error = y - predicciones
            self.pesos += self.tasa_aprendizaje * np.dot(X.T, error)
            self.sesgo += self.tasa_aprendizaje * np.sum(error)

    def predecir(self, X):
        """
        Realiza predicciones utilizando el modelo entrenado.

        :param X: Características a predecir.
        :return: Predicciones del modelo.
        """
        entrada_neta = np.dot(X, self.pesos) + self.sesgo
        return (entrada_neta >= self.umbral).astype(int)

class EvaluacionModelo:
    def __init__(self, modelo, X, y):
        """
        Clase encargada de evaluar el modelo utilizando validación cruzada.

        :param modelo: El modelo que se va a evaluar.
        :param X: Características de los datos.
        :param y: Etiquetas de los datos.
        """
        self.modelo = modelo
        self.X = X
        self.y = y
        self.errores_entrenamiento = []
        self.errores_test = []
        self.kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.pesos_mejor_fold = None
        self.sesgo_mejor_fold = None
        self.error_mejor_fold = float('inf')

    def evaluar(self):
        """
        Realiza la evaluación del modelo utilizando validación cruzada estratificada.
        """
        for fold, (indice_entrenamiento, indice_prueba) in enumerate(self.kf.split(self.X, self.y), 1):
            print(f"\n** Iniciando el Fold {fold} **")
            
            # División de los datos en entrenamiento y prueba
            X_train, X_test = self.X[indice_entrenamiento], self.X[indice_prueba]
            y_train, y_test = self.y[indice_entrenamiento], self.y[indice_prueba]
            
            # Entrenar el modelo
            self.modelo.entrenar(X_train, y_train)
            
            # Evaluar en datos de entrenamiento
            predicciones_train = self.modelo.predecir(X_train)
            error_train = np.mean(predicciones_train != y_train) * 100
            self.errores_entrenamiento.append(error_train)
            
            # Evaluar en datos de prueba
            predicciones_test = self.modelo.predecir(X_test)
            error_test = np.mean(predicciones_test != y_test) * 100
            self.errores_test.append(error_test)
            
            print(f"** Fin del Fold {fold} **")
            print(f"Error de Entrenamiento: {error_train:.2f}% | Error de Testeo: {error_test:.2f}%")

            # Guardar los pesos y sesgo del mejor fold basado en el error de testeo
            if error_test < self.error_mejor_fold:
                self.error_mejor_fold = error_test
                self.pesos_mejor_fold = self.modelo.pesos.copy()
                self.sesgo_mejor_fold = self.modelo.sesgo

        # Promediar los errores de los 5 folds
        error_train_promedio = np.mean(self.errores_entrenamiento)
        error_test_promedio = np.mean(self.errores_test)

        print(f"\nError de Entrenamiento promedio: {error_train_promedio:.2f}%")
        print(f"Error de Testeo promedio: {error_test_promedio:.2f}%")

        # Imprimir los pesos del mejor fold
        print("\n** Pesos y Sesgo del mejor Fold (con el menor error de testeo) **")
        print(f"Pesos: {self.pesos_mejor_fold}")
        print(f"Sesgo: {self.sesgo_mejor_fold}")

class GeneradorDatosSinteticos:
    def __init__(self, n_muestras=3000):
        """
        Inicializa el generador de datos sintéticos.
        """
        self.n_muestras = n_muestras
        self.X = None
        self.y = None

    def generar_datos(self):
        """
        Genera datos sintéticos normalizados con algo de ruido para simular datos reales.
        """
        # Generar datos con cierta estructura para simular patrones reales
        review_scores = np.concatenate([
            np.random.normal(0.8, 0.15, self.n_muestras // 2),  # Cluster de alta satisfacción
            np.random.normal(0.3, 0.15, self.n_muestras // 2)   # Cluster de baja satisfacción
        ])
        
        freight_values = np.concatenate([
            np.random.normal(0.4, 0.2, self.n_muestras // 2),   # Valores de envío para alta satisfacción
            np.random.normal(0.6, 0.2, self.n_muestras // 2)    # Valores de envío para baja satisfacción
        ])
        
        # Clipear valores para mantenerlos entre 0 y 1
        review_scores = np.clip(review_scores, 0, 1)
        freight_values = np.clip(freight_values, 0, 1)
        
        # Agregar ruido aleatorio adicional
        ruido = np.random.normal(0, 0.1, (self.n_muestras, 2))
        
        # Combinar en matriz X
        self.X = np.column_stack((review_scores, freight_values)) + ruido
        self.X = np.clip(self.X, 0, 1)  # Asegurar que los valores finales estén entre 0 y 1
        
        return self.X

    def clasificar_datos(self, pesos, sesgo, umbral=0.5):
        """
        Clasifica los datos usando los pesos y sesgo proporcionados.
        """
        pesos_reducidos = pesos[:2]
        entrada_neta = np.dot(self.X, pesos_reducidos) + sesgo
        return (entrada_neta >= umbral).astype(int)

    def visualizar_clasificacion(self, y_pred, pesos, sesgo, titulo="Clasificación de Datos Sintéticos"):
        """
        Visualiza la clasificación de los datos y la frontera de decisión.
        """
        plt.figure(figsize=(12, 8))
        
        # Graficar puntos clasificados
        puntos_satisfechos = self.X[y_pred == 1]
        puntos_no_satisfechos = self.X[y_pred == 0]
        
        plt.scatter(puntos_satisfechos[:, 0], puntos_satisfechos[:, 1], 
                   c='green', label='Satisfecho', alpha=0.5)
        plt.scatter(puntos_no_satisfechos[:, 0], puntos_no_satisfechos[:, 1], 
                   c='red', label='No Satisfecho', alpha=0.5)
        
        # Dibujar la frontera de decisión
        x_min, x_max = self.X[:, 0].min() - 0.1, self.X[:, 0].max() + 0.1
        y_min, y_max = self.X[:, 1].min() - 0.1, self.X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Obtener los pesos para las dos primeras características
        w1, w2 = pesos[:2]
        
        # La ecuación de la frontera de decisión es: w1*x + w2*y + sesgo = 0
        # Reorganizando para y: y = -(w1*x + sesgo)/w2
        Z = -(w1 * xx + sesgo) / w2
        
        plt.plot(xx[0], Z[0], 'b-', label='Frontera de decisión')
        
        plt.xlabel('Review Score (Normalizado)')
        plt.ylabel('Freight Value (Normalizado)')
        plt.title(titulo)
        plt.legend()
        plt.grid(True)
        
        # Ajustar los límites del gráfico
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        plt.show()

# Código Principal
if __name__ == "__main__":
    # Paso 1: Preprocesamiento de los datos
    preprocesador = PreprocesamientoDatos('reviews_binomial.csv')
    X, y = preprocesador.preprocesar()

    # Paso 2: Crear el modelo Perceptrón
    perceptron = Perceptron(tasa_aprendizaje=0.001, epocas=1000, umbral=0.5)

    # Paso 3: Evaluar el modelo
    evaluador = EvaluacionModelo(perceptron, X, y)
    evaluador.evaluar()

    # Paso 4: Generar y visualizar datos sintéticos
    print("\n** Generando y visualizando datos sintéticos **")
    generador = GeneradorDatosSinteticos(n_muestras=3000)
    X_sintetico = generador.generar_datos()
    y_pred = generador.clasificar_datos(evaluador.pesos_mejor_fold, evaluador.sesgo_mejor_fold)
    generador.visualizar_clasificacion(y_pred, evaluador.pesos_mejor_fold, evaluador.sesgo_mejor_fold)
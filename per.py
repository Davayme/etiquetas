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

            # Guardar los pesos del mejor fold basado en el error de testeo
            if error_test < self.error_mejor_fold:
                self.error_mejor_fold = error_test
                self.pesos_mejor_fold = self.modelo.pesos.copy()

        # Promediar los errores de los 5 folds
        error_train_promedio = np.mean(self.errores_entrenamiento)
        error_test_promedio = np.mean(self.errores_test)

        print(f"\nError de Entrenamiento promedio: {error_train_promedio:.2f}%")
        print(f"Error de Testeo promedio: {error_test_promedio:.2f}%")

        # Imprimir los pesos del mejor fold
        print("\n** Pesos del mejor Fold (con el menor error de testeo) **")
        print(f"Pesos: {self.pesos_mejor_fold}")

    def graficar_frontera_decision(self):
        """
        Función vacía para graficar la frontera de decisión. Completar con la lógica según sea necesario.
        """
        pass

    def graficar_otro(self):
        """
        Función vacía para graficar otro tipo de gráfico. Completar con la lógica según sea necesario.
        """
        pass

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

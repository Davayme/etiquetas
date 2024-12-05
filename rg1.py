import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import log_loss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class RegresionLogistica:
    
    def __init__(self, datos):
        self.datos = datos
        self.caracteristicas = None
        self.objetivo = None

    def preprocesar(self):
        # Seleccionar características relevantes
        self.caracteristicas = self.datos[['review_score', 'freight_value', 'customer_city_name', 'order_month']]

        # Codificación one-hot para variables categóricas
        codificador = OneHotEncoder()
        columnas_categoricas = codificador.fit_transform(self.caracteristicas[['customer_city_name', 'order_month']])
        columnas_categoricas = columnas_categoricas.toarray()

        # Combinar variables numéricas y categóricas
        self.caracteristicas = np.hstack((
            self.caracteristicas[['review_score', 'freight_value']].values, 
            columnas_categoricas
        ))

        # Normalización de variables numéricas
        normalizador = MinMaxScaler()
        datos_numericos = self.caracteristicas[:, :2]
        datos_numericos_normalizados = normalizador.fit_transform(datos_numericos)
        self.caracteristicas[:, :2] = datos_numericos_normalizados

        # Preparar variable objetivo (1: Satisfecho, 0: No Satisfecho)
        self.objetivo = self.datos['satisfaction_class_binomial'].map({
            'Satisfecho': 1, 
            'No Satisfecho': 0
        }).values

        return self.caracteristicas, self.objetivo

    def regresion_logistica(self, tasa_aprendizaje=0.1, num_iteraciones=100):

        X, y = self.preprocesar()
        validacion_cruzada = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        errores_entrenamiento = []
        errores_prueba = []

        def sigmoide(z):
            """Función de activación sigmoide"""
            return 1 / (1 + np.exp(-z))

        for num_fold, (idx_entrenamiento, idx_prueba) in enumerate(validacion_cruzada.split(X, y)):
            # Separar datos en entrenamiento y prueba
            X_entrenamiento, X_prueba = X[idx_entrenamiento], X[idx_prueba]
            y_entrenamiento, y_prueba = y[idx_entrenamiento], y[idx_prueba]

            # Inicializar pesos y bias
            num_caracteristicas = X_entrenamiento.shape[1]
            pesos = np.random.uniform(-0.01, 0.01, size=num_caracteristicas)
            bias = 1

            # Entrenamiento con descenso por gradiente
            for _ in range(num_iteraciones):
                # Predicción
                modelo_lineal = np.dot(X_entrenamiento, pesos) + bias
                y_pred = sigmoide(modelo_lineal)

                # Cálculo de gradientes
                error = y_pred - y_entrenamiento
                gradiente_pesos = np.dot(X_entrenamiento.T, error) / len(y_entrenamiento)
                gradiente_bias = np.sum(error) / len(y_entrenamiento)

                # Actualización de pesos y bias usando la tasa de aprendizaje
                pesos -= tasa_aprendizaje * gradiente_pesos
                bias -= tasa_aprendizaje * gradiente_bias

            # Calcular error de entrenamiento
            pred_entrenamiento = sigmoide(np.dot(X_entrenamiento, pesos) + bias)
            error_entrenamiento = log_loss(y_entrenamiento, pred_entrenamiento)
            errores_entrenamiento.append(error_entrenamiento)

            # Calcular error de prueba
            pred_prueba = sigmoide(np.dot(X_prueba, pesos) + bias)
            error_prueba = log_loss(y_prueba, pred_prueba)
            errores_prueba.append(error_prueba)

            print(f"\nFold {num_fold + 1}:")
            print(f"Error de entrenamiento: {error_entrenamiento:.4f}")
            print(f"Error de prueba: {error_prueba:.4f}")

        # Calcular y mostrar promedios finales
        error_entrenamiento_promedio = np.mean(errores_entrenamiento)
        error_prueba_promedio = np.mean(errores_prueba)
        
        print("\nResultados finales:")
        print(f"Error promedio de entrenamiento: {error_entrenamiento_promedio:.4f}")
        print(f"Error promedio de prueba: {error_prueba_promedio:.4f}")

class LogisticRegressionViz(RegresionLogistica):
    def __init__(self, datos, sample_size=200000):
        super().__init__(datos)
        self.sample_size = min(sample_size, len(datos))
        
    def visualizar_frontera(self):
        # Preparar los datos
        X, y = self.preprocesar()
        
        # Tomar una muestra aleatoria
        indices = np.random.choice(len(X), size=self.sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
        
        # Aplicar t-SNE
        print("Aplicando t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_sample)
        
        # Crear el gráfico
        plt.figure(figsize=(12, 8))
        
        # Graficar puntos por clase
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                            c=y_sample, 
                            cmap='coolwarm', 
                            alpha=0.6)
        
        plt.colorbar(scatter)
        plt.title('Visualización t-SNE de la Regresión Logística')
        plt.xlabel('t-SNE Componente 1')
        plt.ylabel('t-SNE Componente 2')
        
        # Mostrar el gráfico
        plt.show()
        
        # Calcular y mostrar estadísticas
        print("\nEstadísticas de la muestra:")
        print(f"Total de puntos: {len(y_sample)}")
        print(f"Satisfechos: {sum(y_sample == 1)} ({sum(y_sample == 1)/len(y_sample)*100:.2f}%)")
        print(f"No Satisfechos: {sum(y_sample == 0)} ({sum(y_sample == 0)/len(y_sample)*100:.2f}%)")

# Leer los datos
datos = pd.read_csv('reviews_binomial.csv', sep=';')

# Crear el modelo y entrenarlo
modelo = LogisticRegressionViz(datos)
modelo.regresion_logistica(tasa_aprendizaje=0.75)

# Visualizar la frontera
modelo.visualizar_frontera()

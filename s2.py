import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# 1. Cargar el dataset
archivo_datos = "labeled_dataset.csv"
df = pd.read_csv(archivo_datos, delimiter=";")

# 2. Definir las columnas numéricas y categóricas
# Aquí definimos qué columnas son numéricas y cuáles categóricas.
caracteristicas_numericas = ['product_weight', 'shipping_time_days', 'freight_value', 'order_price', 
                              'product_discount', 'review_score', 'inventory_stock_level', 
                              'seller_response_time', 'customer_complaints']

caracteristicas_categoricas = ['order_year', 'order_month', 'order_day_of_week', 'customer_region', 
                               'seller_region', 'product_category', 'customer_gender']

# 3. Preprocesamiento de los datos
# Aplicamos transformación a las características:
# - Normalizamos las características numéricas.
# - Aplicamos One-Hot Encoding a las características categóricas.
preprocesador = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), caracteristicas_numericas),  # Normaliza las características numéricas.
        ('cat', OneHotEncoder(), caracteristicas_categoricas)  # One-Hot Encoding para las categóricas.
    ])

# 4. Preprocesar los datos de entrada (X) y las etiquetas (y)
# 'X' contiene las características de entrada (todo menos la columna de 'customer_satisfaction').
# 'y' contiene la etiqueta objetivo ('customer_satisfaction').
X = df.drop(['customer_satisfaction'], axis=1)  # Eliminar la columna 'customer_satisfaction' de las características.
y = df['customer_satisfaction']  # Etiqueta 'customer_satisfaction'

# Transformamos los datos con el preprocesador
X = preprocesador.fit_transform(X)

# 5. **Validación Cruzada Estratificada**
# Usamos la validación cruzada estratificada para asegurarnos de que cada fold tenga una distribución representativa de las clases.
# Esto ayuda a evitar que un fold quede sesgado por una clase sobre-representada.

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Dividimos los datos en 5 partes (folds).
# Lista para almacenar las precisiones por cada fold.
precisiones_por_fold = []

# 6. **Iterar sobre cada fold para entrenar y evaluar el modelo**
for fold, (indice_entrenamiento, indice_prueba) in enumerate(k_fold.split(X, y)):
    print(f'Fold {fold + 1}')

    # Dividir los datos en entrenamiento y prueba usando los índices generados por StratifiedKFold.
    X_entrenamiento, X_prueba = X[indice_entrenamiento], X[indice_prueba]
    y_entrenamiento, y_prueba = y[indice_entrenamiento], y[indice_prueba]

    # 7. **Entrenamiento del modelo**
    # En este caso, utilizamos un modelo de regresión logística como ejemplo para clasificación.
    modelo = LogisticRegression(max_iter=500, random_state=42)  # Modelo de regresión logística.
    modelo.fit(X_entrenamiento, y_entrenamiento)  # Entrenamos el modelo con los datos de entrenamiento.

    # 8. **Evaluación del modelo**
    # Realizamos predicciones sobre los datos de prueba.
    y_pred = modelo.predict(X_prueba)
    
    # Calculamos la precisión del modelo, que es la proporción de predicciones correctas.
    precision = accuracy_score(y_prueba, y_pred)
    precisiones_por_fold.append(precision)

    print(f'Precisión para el Fold {fold + 1}: {precision:.4f}\n')

# 9. **Resultados finales**
# Calculamos la precisión promedio y la desviación estándar de las precisiones obtenidas en los 5 folds.
precision_promedio = np.mean(precisiones_por_fold)
desviacion_estandar = np.std(precisiones_por_fold)

print(f'Precisión promedio: {precision_promedio:.4f}')
print(f'Desviación estándar de las precisiones: {desviacion_estandar:.4f}')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el dataset
df = pd.read_csv("reviews_binomial.csv", sep=";")

# Codificar la variable target (product_return_class) a valores numéricos
class_map = {'Baja Probabilidad': 0, 'Alta Probabilidad': 1}
df['product_return_class'] = df['product_return_class'].map(class_map)

# Seleccionar las features y la clase target
features = ['review_score', 'freight_value', 'order_price', 'product_category', 'product_brand']
target = 'product_return_class'

# Codificar variables categóricas
df['product_category'] = df['product_category'].astype("category").cat.codes
df['product_brand'] = df['product_brand'].astype("category").cat.codes

# Dividir los datos en train y test
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Entrenar un modelo simple (Árbol de decisión)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

# Cargar el dataset
df = pd.read_csv('labeled_dataset.csv', delimiter=';')

# Definir las características numéricas y categóricas
numerical_features = [
    'product_weight', 'shipping_time_days', 'freight_value', 
    'order_price', 'product_discount', 'review_score', 
    'inventory_stock_level', 'seller_response_time', 'customer_complaints'
]

categorical_features = [
    'order_year', 'order_month', 'order_day_of_week', 
    'customer_region', 'seller_region', 'product_category', 
    'customer_gender'
]

# Separar las características de entrada (X) y la variable objetivo (y)
X = df[numerical_features + categorical_features]
y = df['customer_satisfaction']

# Preprocesamiento de las características: normalización de numéricas y One-Hot Encoding de categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Crear un pipeline con un modelo de regresión logística multinomial (softmax)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, 
                                      warm_start=True))  # warm_start=True para actualizar el modelo sin reiniciar
])

# Estratificación de la validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Variables para calcular los resultados globales
accuracy_scores = []
train_errors = []
test_errors = []

# Realizar validación cruzada con seguimiento por fold
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    print(f"\n--- Fold {fold} ---")

    # Dividir los datos en entrenamiento y test
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Entrenamiento del modelo
    model.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcular precisión y error de entrenamiento y test
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_error = 1 - train_accuracy
    test_error = 1 - test_accuracy

    # Guardar resultados de cada fold
    accuracy_scores.append(test_accuracy)
    train_errors.append(train_error)
    test_errors.append(test_error)

    # Mostrar resultados por fold
    print(f"Train accuracy: {train_accuracy:.4f}, Train error: {train_error:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}, Test error: {test_error:.4f}")

# Mostrar los resultados finales
print("\n--- Resumen Final ---")
print(f"Mean Test Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Mean Train Error: {np.mean(train_errors):.4f}")
print(f"Mean Test Error: {np.mean(test_errors):.4f}")

# Calcular log loss (entropía cruzada) para la evaluación
log_loss_value = log_loss(y, model.predict_proba(X))
print(f"Log Loss (Cross-Entropy Loss): {log_loss_value:.4f}")

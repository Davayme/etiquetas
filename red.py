# 1. Importar librerías necesarias
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# 2. Cargar datos
df = pd.read_csv('labeled_dataset_fuzzy.csv', sep=';')

# 3. Definir características
numeric_features = ['review_score', 'customer_complaints', 'shipping_time_days', 
                   'freight_value', 'order_price']
categorical_features = ['customer_region', 'seller_region']

# 4. Preprocesamiento
# Normalizar variables numéricas
scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[numeric_features])

# Codificar variables categóricas
encoder = OneHotEncoder(sparse_output=False)
X_categorical = encoder.fit_transform(df[categorical_features])

# Combinar features
X = np.hstack([X_numeric, X_categorical])
y = tf.keras.utils.to_categorical(df['customer_satisfaction'])

# 5. División train-test estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=df['customer_satisfaction'],
    random_state=42
)

# 6. Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# 7. Compilar el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 8. Callbacks para el entrenamiento
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=3
    )
]

# 9. Entrenamiento
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=512,
    callbacks=callbacks,
    verbose=1  # Muestra la barra de progreso
)

# 10. Evaluación
print("\nEvaluación del modelo:")
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f'Exactitud de entrenamiento: {train_acc:.4f}')
print(f'Exactitud de prueba: {test_acc:.4f}')

# 11. Visualización del entrenamiento
plt.figure(figsize=(12, 4))

# Gráfico de exactitud
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Exactitud del Modelo')
plt.xlabel('Época')
plt.ylabel('Exactitud')
plt.legend()

# Gráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()

# 12. Guardar el modelo (opcional)
model.save('modelo_satisfaccion_cliente.h5')
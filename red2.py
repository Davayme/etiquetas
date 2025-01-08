import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# 1. Cargar datos
df = pd.read_csv('envios_etiquetado_fuzzy1.csv', sep=';')

# 2. Definir características
numeric_features = ['product_price', 'product_volume', 'freight_value', 'distance']
categorical_features = ['customer_region', 'seller_region']

# 3. Preprocesamiento
# Normalizar variables numéricas
scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[numeric_features])

# Codificar variables categóricas
encoder = OneHotEncoder(sparse_output=False)
X_categorical = encoder.fit_transform(df[categorical_features])

# Combinar features
X = np.hstack([X_numeric, X_categorical])
y = tf.keras.utils.to_categorical(df['shipping_label'])

# 4. División train-test estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=df['shipping_label'],
    random_state=42
)

# 5. Construir modelo
model = tf.keras.Sequential([
    # Capa de entrada
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    # Primera capa oculta
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    # Segunda capa oculta
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    
    # Tercera capa oculta
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    
    # Capa de salida
    tf.keras.layers.Dense(3, activation='softmax')
])

# 6. Compilar modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True,
        monitor='val_loss'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=5,
        min_lr=0.0001
    )
]

# 8. Entrenamiento
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# 9. Evaluación
print("\nEvaluación del modelo:")
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f'Exactitud de entrenamiento: {train_acc:.4f}')
print(f'Exactitud de prueba: {test_acc:.4f}')

# 10. Visualización
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Exactitud del Modelo')
plt.xlabel('Época')
plt.ylabel('Exactitud')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()
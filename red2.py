import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# 1. Cargar datos
df = pd.read_csv('envios_etiquetado_fuzzy2.csv', sep=';')

# 2. Definir características
numeric_features = ['product_price', 'product_volume', 'freight_value', 'distance']
categorical_features = ['customer_region', 'seller_region']

# 3. Preprocesamiento
scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[numeric_features])

encoder = OneHotEncoder(sparse_output=False)
X_categorical = encoder.fit_transform(df[categorical_features])

X = np.hstack([X_numeric, X_categorical])
y = tf.keras.utils.to_categorical(df['shipping_label'])

# 4. División train-test estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=df['shipping_label'],
    random_state=42
)


model = tf.keras.Sequential([
    # Capa de entrada más pequeña
    tf.keras.layers.Dense(48, activation='relu', input_shape=(X.shape[1],),
                         kernel_regularizer=tf.keras.regularizers.l2(0.002)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.35),
    
    # Primera capa oculta
    tf.keras.layers.Dense(32, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.002)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.35),  # Mismo dropout para consistencia
    
    # Capa de salida
    tf.keras.layers.Dense(3, activation='softmax')
])

# 6. Compilar modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)  # Learning rate aún más bajo
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Callbacks mejorados para estabilidad
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=8,
        restore_best_weights=True,
        monitor='val_loss',
        min_delta=0.001
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.2,
        patience=4,
        min_lr=0.00001,
        monitor='val_loss',
        verbose=1
    )
]

# 8. Entrenamiento
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=128,  # Batch size más grande para más estabilidad
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
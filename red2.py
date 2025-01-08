import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# 1. Cargar datos y preprocesamiento (igual que antes)
df = pd.read_csv('envios_etiquetado_fuzzy2.csv', sep=';')

numeric_features = ['product_price', 'product_volume', 'freight_value', 'distance']
categorical_features = ['customer_region', 'seller_region']

scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[numeric_features])

encoder = OneHotEncoder(sparse_output=False)
X_categorical = encoder.fit_transform(df[categorical_features])

X = np.hstack([X_numeric, X_categorical])
y = tf.keras.utils.to_categorical(df['shipping_label'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=df['shipping_label'],
    random_state=42
)

# 2. Modelo más profundo
model = tf.keras.Sequential([
    # Capa de entrada
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],),
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    # Primera capa oculta
    tf.keras.layers.Dense(96, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    # Segunda capa oculta
    tf.keras.layers.Dense(64, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    
    # Tercera capa oculta
    tf.keras.layers.Dense(48, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    
    # Cuarta capa oculta
    tf.keras.layers.Dense(32, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    
    # Quinta capa oculta
    tf.keras.layers.Dense(24, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    
    # Capa de salida
    tf.keras.layers.Dense(3, activation='softmax')
])

# 3. Compilación con learning rate bajo
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Callbacks ajustados
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=12,
        restore_best_weights=True,
        monitor='val_loss',
        min_delta=0.001
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.2,
        patience=6,
        min_lr=0.00001,
        monitor='val_loss',
        verbose=1
    )
]

# 5. Entrenamiento
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# 6. Evaluación
print("\nEvaluación del modelo:")
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f'Exactitud de entrenamiento: {train_acc:.4f}')
print(f'Exactitud de prueba: {test_acc:.4f}')

# 7. Visualización
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
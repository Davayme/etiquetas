# 1. Importar librerías necesarias
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

# 2. Cargar datos
df = pd.read_csv('envios_etiquetado_fuzzy2.csv', sep=';')

# 3. Definir características
numeric_features = ['product_price', 'product_volume', 'freight_value', 'distance']
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
y = tf.keras.utils.to_categorical(df['shipping_label'])

# Calcular class weights
y_original = df['shipping_label']
class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(y_original), 
                                   y=y_original)
class_weight_dict = dict(enumerate(class_weights))

# 5. Configurar la validación cruzada
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Variables para almacenar resultados
histories = []
test_scores = []
train_errors = []
test_errors = []
best_val_acc = 0
best_model_weights = None
best_fold = 0
best_fold_history = None
best_X_val = None
best_y_val = None

# 6. Función para crear el modelo
def create_model(input_shape):
    model = tf.keras.Sequential([
        # Capa de entrada
        tf.keras.layers.Dense(96, activation='relu', input_shape=(input_shape,),
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Primera capa oculta
        tf.keras.layers.Dense(64, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Segunda capa oculta
        tf.keras.layers.Dense(48, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        # Tercera capa oculta
        tf.keras.layers.Dense(32, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        # Capa de salida
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 7. Entrenamiento con validación cruzada
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_original)):
    print(f'\nFold {fold + 1}/{n_splits}')
    
    # Split the data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Crear modelo
    model = create_model(X.shape[1])
    
    # Callbacks
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
    
    # Entrenar el modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=64,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluar el modelo
    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_val, y_val, verbose=0)
    test_scores.append(test_score)
    histories.append(history.history)
    train_errors.append(1 - train_score[1])
    test_errors.append(1 - test_score[1])
    
    print(f'\nResultados del Fold {fold + 1}:')
    print(f'Error de entrenamiento: {1 - train_score[1]:.4f}')
    print(f'Error de testeo: {1 - test_score[1]:.4f}')
    print(f'Pérdida de entrenamiento: {train_score[0]:.4f}')
    print(f'Pérdida de testeo: {test_score[0]:.4f}')
    
    # Guardar el mejor modelo
    if test_score[1] > best_val_acc:
        best_val_acc = test_score[1]
        best_model_weights = model.get_weights()
        best_fold = fold
        best_fold_history = history.history
        best_X_val = X_val
        best_y_val = y_val
        print(f'*** Nuevo mejor modelo encontrado en Fold {fold + 1} ***')

# 8. Calcular y mostrar resultados promedio
mean_test_loss = np.mean([score[0] for score in test_scores])
mean_test_acc = np.mean([score[1] for score in test_scores])
std_test_acc = np.std([score[1] for score in test_scores])

print('\nResultados de validación cruzada:')
print(f'Exactitud media: {mean_test_acc:.4f} ± {std_test_acc:.4f}')
print(f'Pérdida media: {mean_test_loss:.4f}')
print(f'\nMejor fold: {best_fold + 1}')
print(f'Mejor exactitud de validación: {best_val_acc:.4f}')

# 9. Visualizar resultados
plt.figure(figsize=(15, 10))

# Encontrar la longitud mínima de los historiales
min_length = min([len(h['accuracy']) for h in histories])

# Recortar todos los historiales a la misma longitud
histories_aligned = []
for h in histories:
    history_aligned = {
        'accuracy': h['accuracy'][:min_length],
        'val_accuracy': h['val_accuracy'][:min_length],
        'loss': h['loss'][:min_length],
        'val_loss': h['val_loss'][:min_length]
    }
    histories_aligned.append(history_aligned)

# Gráfico de exactitud promedio
plt.subplot(2, 2, 1)
mean_acc = np.mean([h['accuracy'] for h in histories_aligned], axis=0)
mean_val_acc = np.mean([h['val_accuracy'] for h in histories_aligned], axis=0)
plt.plot(mean_acc, label='Entrenamiento')
plt.plot(mean_val_acc, label='Validación')
plt.title('Exactitud Media (Todos los Folds)')
plt.xlabel('Época')
plt.ylabel('Exactitud')
plt.legend()

# Gráfico de pérdida promedio
plt.subplot(2, 2, 2)
mean_loss = np.mean([h['loss'] for h in histories_aligned], axis=0)
mean_val_loss = np.mean([h['val_loss'] for h in histories_aligned], axis=0)
plt.plot(mean_loss, label='Entrenamiento')
plt.plot(mean_val_loss, label='Validación')
plt.title('Pérdida Media (Todos los Folds)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Gráfico de exactitud del mejor fold
plt.subplot(2, 2, 3)
plt.plot(best_fold_history['accuracy'], label='Entrenamiento')
plt.plot(best_fold_history['val_accuracy'], label='Validación')
plt.title(f'Exactitud del Mejor Fold ({best_fold + 1})')
plt.xlabel('Época')
plt.ylabel('Exactitud')
plt.legend()

# Gráfico de pérdida del mejor fold
plt.subplot(2, 2, 4)
plt.plot(best_fold_history['loss'], label='Entrenamiento')
plt.plot(best_fold_history['val_loss'], label='Validación')
plt.title(f'Pérdida del Mejor Fold ({best_fold + 1})')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()

# 10. Crear y configurar el mejor modelo para las predicciones finales
best_model = create_model(X.shape[1])
best_model.set_weights(best_model_weights)

# Obtener predicciones del mejor modelo
best_predictions = best_model.predict(best_X_val)
y_pred_classes = np.argmax(best_predictions, axis=1)
y_val_classes = np.argmax(best_y_val, axis=1)

print('\nReporte de clasificación del mejor modelo:')
print(classification_report(y_val_classes, y_pred_classes))

# Mostrar errores promedio finales
print(f'\nError promedio de entrenamiento: {np.mean(train_errors):.4f}')
print(f'Error promedio de prueba: {np.mean(test_errors):.4f}')
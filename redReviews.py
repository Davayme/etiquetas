# 1. Importación de bibliotecas necesarias
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

#---------------------------- PREPARACIÓN DE DATOS ----------------------------#

# 2. Carga del conjunto de datos
# El archivo contiene información sobre satisfacción de clientes y características asociadas
datos_crudos = pd.read_csv('dtreviews.csv', sep=';')

# 3. Definición de características (variables de entrada)
# Variables numéricas que influyen en la satisfacción del cliente
caracteristicas_numericas = [
    'review_score',           # Puntuación de la reseña
    'customer_complaints',    # Número de quejas del cliente
    'shipping_time_days',     # Tiempo de envío en días
    'freight_value',         # Valor del flete
    'order_price'            # Precio del pedido
]

# Variables categóricas que pueden afectar la satisfacción
caracteristicas_categoricas = [
    'customer_region',        # Región del cliente
    'seller_region'          # Región del vendedor
]

# 4. Preprocesamiento de datos
# Normalización de variables numéricas (media 0, desviación estándar 1)
normalizador = StandardScaler()
X_normalizado = normalizador.fit_transform(datos_crudos[caracteristicas_numericas])

# Codificación one-hot de variables categóricas
codificador = OneHotEncoder(sparse_output=False)
X_categorico = codificador.fit_transform(datos_crudos[caracteristicas_categoricas])

# Combinación de todas las características
X_combinado = np.hstack([X_normalizado, X_categorico])

# Preparación de variable objetivo (satisfacción del cliente)
y_codificado = tf.keras.utils.to_categorical(datos_crudos['customer_satisfaction'])

# Cálculo de pesos para balancear las clases
y_original = datos_crudos['customer_satisfaction']
pesos_clases = compute_class_weight('balanced', 
                                  classes=np.unique(y_original), 
                                  y=y_original)
diccionario_pesos = dict(enumerate(pesos_clases))

#---------------------------- CONFIGURACIÓN DEL MODELO ----------------------------#

# 5. Configuración de la validación cruzada
numero_particiones = 5
validacion_cruzada = StratifiedKFold(n_splits=numero_particiones, 
                                    shuffle=True, 
                                    random_state=42)

# Variables para almacenar resultados del entrenamiento
historiales = []
puntajes_prueba = []
errores_entrenamiento = []
errores_prueba = []
mejor_precision = 0
mejores_pesos = None
mejor_particion = 0
historial_mejor_particion = None
mejor_X_val = None
mejor_y_val = None

# 6. Definición de la arquitectura de la red neuronal
def crear_modelo(forma_entrada):

    modelo = tf.keras.Sequential([
        # Primera capa densa (recibe los datos de entrada con forma_entrada características)
        tf.keras.layers.Dense(24, activation='relu', input_shape=(forma_entrada,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        # Segunda capa densa
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        # Capa de salida (3 neuronas para clasificación multiclase)
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    # Configuración del proceso de entrenamiento
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',  # Función de pérdida para clasificación multiclase
        metrics=['accuracy']              # Métrica principal: precisión
    )
    return modelo

#---------------------------- ENTRENAMIENTO DEL MODELO ----------------------------#

# 7. Entrenamiento usando validación cruzada
for indice_particion, (indices_entrenamiento, indices_validacion) in enumerate(
    validacion_cruzada.split(X_combinado, y_original)):
    
    print(f'\nPartición {indice_particion + 1}/{numero_particiones}')
    
    # Separación de datos para esta partición
    X_entrenamiento = X_combinado[indices_entrenamiento]
    X_validacion = X_combinado[indices_validacion]
    y_entrenamiento = y_codificado[indices_entrenamiento]
    y_validacion = y_codificado[indices_validacion]
    
    # Creación del modelo para esta partición
    modelo = crear_modelo(X_combinado.shape[1])
    
    # Configuración de callbacks para optimizar el entrenamiento
    callbacks = [
        # Detiene el entrenamiento si no hay mejora
        tf.keras.callbacks.EarlyStopping(
            patience=8,                    # Épocas a esperar antes de detener
            restore_best_weights=True,     # Restaura los mejores pesos encontrados
            monitor='val_loss',            # Monitorea la pérdida en validación
            min_delta=0.001                # Cambio mínimo considerado como mejora
        ),
        # Reduce la tasa de aprendizaje cuando el modelo se estanca
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.2,                    # Factor de reducción del learning rate
            patience=4,                    # Épocas a esperar antes de reducir
            min_lr=0.0001                  # Learning rate mínimo
        )
    ]
    
    # Entrenamiento del modelo
    historial = modelo.fit(
        X_entrenamiento, y_entrenamiento,
        validation_data=(X_validacion, y_validacion),
        epochs=100,                        # Número máximo de épocas
        batch_size=256,                    # Tamaño del lote
        callbacks=callbacks,
        class_weight=diccionario_pesos,    # Pesos para balancear clases
        verbose=1
    )
    
    # Evaluación del modelo
    puntuacion_entrenamiento = modelo.evaluate(X_entrenamiento, y_entrenamiento, verbose=0)
    puntuacion_prueba = modelo.evaluate(X_validacion, y_validacion, verbose=0)
    
    # Almacenamiento de resultados
    puntajes_prueba.append(puntuacion_prueba)
    historiales.append(historial.history)
    errores_entrenamiento.append(1 - puntuacion_entrenamiento[1])
    errores_prueba.append(1 - puntuacion_prueba[1])
    
    # Impresión de resultados de la partición
    print(f'\nResultados de la Partición {indice_particion + 1}:')
    print(f'Error de entrenamiento: {1 - puntuacion_entrenamiento[1]:.4f}')
    print(f'Error de prueba: {1 - puntuacion_prueba[1]:.4f}')
    print(f'Pérdida en entrenamiento: {puntuacion_entrenamiento[0]:.4f}')
    print(f'Pérdida en prueba: {puntuacion_prueba[0]:.4f}')
    
    # Actualización del mejor modelo si corresponde
    if puntuacion_prueba[1] > mejor_precision:
        mejor_precision = puntuacion_prueba[1]
        mejores_pesos = modelo.get_weights()
        mejor_particion = indice_particion
        historial_mejor_particion = historial.history
        mejor_X_val = X_validacion
        mejor_y_val = y_validacion
        print(f'*** Nuevo mejor modelo encontrado en Partición {indice_particion + 1} ***')

#---------------------------- EVALUACIÓN DE RESULTADOS ----------------------------#

# 8. Cálculo de métricas finales
perdida_media = np.mean([score[0] for score in puntajes_prueba])
precision_media = np.mean([score[1] for score in puntajes_prueba])
desviacion_precision = np.std([score[1] for score in puntajes_prueba])

print('\nResultados de la Validación Cruzada:')
print(f'Precisión media: {precision_media:.4f} ± {desviacion_precision:.4f}')
print(f'Pérdida media: {perdida_media:.4f}')
print(f'\nMejor partición: {mejor_particion + 1}')
print(f'Mejor precisión de validación: {mejor_precision:.4f}')

# 9. Visualización de resultados
plt.figure(figsize=(15, 10))

# Procesamiento de historiales para visualización
longitud_minima = min([len(h['accuracy']) for h in historiales])
historiales_alineados = []
for h in historiales:
    historial_alineado = {
        'accuracy': h['accuracy'][:longitud_minima],
        'val_accuracy': h['val_accuracy'][:longitud_minima],
        'loss': h['loss'][:longitud_minima],
        'val_loss': h['val_loss'][:longitud_minima]
    }
    historiales_alineados.append(historial_alineado)

# Gráficas de rendimiento
# Superior izquierda: Precisión media
plt.subplot(2, 2, 1)
precision_media = np.mean([h['accuracy'] for h in historiales_alineados], axis=0)
precision_val_media = np.mean([h['val_accuracy'] for h in historiales_alineados], axis=0)
plt.plot(precision_media, label='Entrenamiento')
plt.plot(precision_val_media, label='Validación')
plt.title('Precisión Media (Todas las Particiones)')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Superior derecha: Pérdida media
plt.subplot(2, 2, 2)
perdida_media = np.mean([h['loss'] for h in historiales_alineados], axis=0)
perdida_val_media = np.mean([h['val_loss'] for h in historiales_alineados], axis=0)
plt.plot(perdida_media, label='Entrenamiento')
plt.plot(perdida_val_media, label='Validación')
plt.title('Pérdida Media (Todas las Particiones)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Inferior izquierda: Precisión de la mejor partición
plt.subplot(2, 2, 3)
plt.plot(historial_mejor_particion['accuracy'], label='Entrenamiento')
plt.plot(historial_mejor_particion['val_accuracy'], label='Validación')
plt.title(f'Precisión de la Mejor Partición ({mejor_particion + 1})')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Inferior derecha: Pérdida de la mejor partición
plt.subplot(2, 2, 4)
plt.plot(historial_mejor_particion['loss'], label='Entrenamiento')
plt.plot(historial_mejor_particion['val_loss'], label='Validación')
plt.title(f'Pérdida de la Mejor Partición ({mejor_particion + 1})')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()

# 10. Evaluación final del mejor modelo
mejor_modelo = crear_modelo(X_combinado.shape[1])
mejor_modelo.set_weights(mejores_pesos)

# Predicciones con el mejor modelo
mejores_predicciones = mejor_modelo.predict(mejor_X_val)
clases_predichas = np.argmax(mejores_predicciones, axis=1)
clases_reales = np.argmax(mejor_y_val, axis=1)

print('\nReporte de clasificación del mejor modelo:')
print(classification_report(clases_reales, clases_predichas))

# Errores promedio finales
print(f'\nError promedio de entrenamiento: {np.mean(errores_entrenamiento):.4f}')
print(f'Error promedio de prueba: {np.mean(errores_prueba):.4f}')
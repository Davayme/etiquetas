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

# 2. Carga del conjunto de datos de envíos
datos_envios = pd.read_csv('envios_etiquetado_fuzzy2.csv', sep=';')

# 3. Definición de características para el análisis
# Variables numéricas que afectan la rentabilidad del envío
caracteristicas_numericas = [
    'product_price',      # Precio del producto
    'product_volume',     # Volumen del producto
    'freight_value',      # Valor del flete
    'distance'           # Distancia de envío
]

# Variables categóricas que influyen en la logística
caracteristicas_categoricas = [
    'customer_region',    # Región del cliente
    'seller_region'      # Región del vendedor
]

# 4. Preprocesamiento de datos
# Normalización de variables numéricas para escalar todos los valores
normalizador = StandardScaler()
X_normalizado = normalizador.fit_transform(datos_envios[caracteristicas_numericas])

# Codificación de variables categóricas en formato one-hot
codificador = OneHotEncoder(sparse_output=False)
X_categorico = codificador.fit_transform(datos_envios[caracteristicas_categoricas])

# Combinación de todas las características procesadas
X_combinado = np.hstack([X_normalizado, X_categorico])
y_codificado = tf.keras.utils.to_categorical(datos_envios['shipping_label'])

# Cálculo de pesos para manejar el desbalance de clases
y_original = datos_envios['shipping_label']
pesos_clases = compute_class_weight('balanced', 
                                  classes=np.unique(y_original), 
                                  y=y_original)
diccionario_pesos = dict(enumerate(pesos_clases))

#---------------------------- CONFIGURACIÓN DEL MODELO ----------------------------#

# 5. Configuración de la validación cruzada para evaluación robusta
numero_particiones = 5
validacion_cruzada = StratifiedKFold(n_splits=numero_particiones, shuffle=True, random_state=42)

# Variables para seguimiento del entrenamiento
historiales = []
puntajes_prueba = []
errores_entrenamiento = []
errores_prueba = []
mejor_precision = 0
mejores_pesos = None
mejor_particion = 0
historial_mejor_particion = None
mejor_X_validacion = None
mejor_y_validacion = None

# 6. Definición de la arquitectura de la red neuronal
def crear_modelo(forma_entrada):

    modelo = tf.keras.Sequential([
         # Primera capa densa con 96 neuronas
        tf.keras.layers.Dense(96, activation='relu', input_shape=(forma_entrada,),
                            kernel_regularizer=tf.keras.regularizers.l2(0.002)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
         # Segunda capa densa
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.002)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Tercera capa densa
        tf.keras.layers.Dense(48, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.002)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        # Cuarta capa densa
        tf.keras.layers.Dense(32, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.002)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        # Capa de salida para clasificación multiclase
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    # Configuración del optimizador con parámetros para estabilidad
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005,    # Tasa de aprendizaje conservadora
            beta_1=0.9,             # Momento exponencial primer orden
            beta_2=0.999,           # Momento exponencial segundo orden
            epsilon=1e-07,          # Factor de estabilidad numérica
            amsgrad=True            # Variante AMSGrad para mejor convergencia
        ),
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
        # Detención temprana si no hay mejora
        tf.keras.callbacks.EarlyStopping(
            patience=10,                    # Épocas a esperar antes de detener
            restore_best_weights=True,     # Restaura los mejores pesos encontrados
            monitor='val_loss',            # Monitorea la pérdida en validación
            min_delta=0.001,              # Cambio mínimo considerado como mejora
            mode='min'                    # Buscamos minimizar la pérdida
        ),
        # Reducción adaptativa de la tasa de aprendizaje
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.2,                    # Factor de reducción
            patience=5,                    # Épocas antes de reducir
            min_lr=0.00001,               # Tasa de aprendizaje mínima
            monitor='val_loss',           # Monitoreo de pérdida en validación
            mode='min',                   # Buscamos minimizar
            verbose=1                     # Mostrar mensajes de reducción
        )
    ]
    
    # Entrenamiento del modelo
    historial = modelo.fit(
        X_entrenamiento, y_entrenamiento,
        validation_data=(X_validacion, y_validacion),
        epochs=100,                      # Número máximo de épocas
        batch_size=256,                  # Tamaño del lote para estabilidad
        callbacks=callbacks,
        class_weight=diccionario_pesos,  # Pesos para balance de clases
        verbose=1,
        shuffle=True                     # Mezclar datos en cada época
    )
    
    # Evaluación del modelo en esta partición
    puntuacion_entrenamiento = modelo.evaluate(X_entrenamiento, y_entrenamiento, verbose=0)
    puntuacion_validacion = modelo.evaluate(X_validacion, y_validacion, verbose=0)
    
    # Almacenamiento de resultados
    puntajes_prueba.append(puntuacion_validacion)
    historiales.append(historial.history)
    errores_entrenamiento.append(1 - puntuacion_entrenamiento[1])
    errores_prueba.append(1 - puntuacion_validacion[1])
    
    # Impresión de resultados de la partición
    print(f'\nResultados de la Partición {indice_particion + 1}:')
    print(f'Error de entrenamiento: {1 - puntuacion_entrenamiento[1]:.4f}')
    print(f'Error de validación: {1 - puntuacion_validacion[1]:.4f}')
    print(f'Pérdida en entrenamiento: {puntuacion_entrenamiento[0]:.4f}')
    print(f'Pérdida en validación: {puntuacion_validacion[0]:.4f}')
    
    # Actualización del mejor modelo si corresponde
    if puntuacion_validacion[1] > mejor_precision:
        mejor_precision = puntuacion_validacion[1]
        mejores_pesos = modelo.get_weights()
        mejor_particion = indice_particion
        historial_mejor_particion = historial.history
        mejor_X_validacion = X_validacion
        mejor_y_validacion = y_validacion
        print(f'*** Nuevo mejor modelo encontrado en Partición {indice_particion + 1} ***')

#---------------------------- EVALUACIÓN Y VISUALIZACIÓN ----------------------------#

# 8. Cálculo de métricas finales
perdida_media = np.mean([score[0] for score in puntajes_prueba])
precision_media = np.mean([score[1] for score in puntajes_prueba])
desviacion_precision = np.std([score[1] for score in puntajes_prueba])

print('\nResultados Finales de la Validación Cruzada:')
print(f'Precisión media: {precision_media:.4f} ± {desviacion_precision:.4f}')
print(f'Pérdida media: {perdida_media:.4f}')
print(f'\nMejor partición: {mejor_particion + 1}')
print(f'Mejor precisión de validación: {mejor_precision:.4f}')

# 9. Visualización de resultados del entrenamiento
plt.figure(figsize=(15, 10))

# Alineación de historiales para visualización
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

# Generación de gráficas de rendimiento
plt.subplot(2, 2, 1)
precision_media = np.mean([h['accuracy'] for h in historiales_alineados], axis=0)
precision_val_media = np.mean([h['val_accuracy'] for h in historiales_alineados], axis=0)
plt.plot(precision_media, label='Entrenamiento')
plt.plot(precision_val_media, label='Validación')
plt.title('Precisión Media (Todas las Particiones)')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(2, 2, 2)
perdida_media = np.mean([h['loss'] for h in historiales_alineados], axis=0)
perdida_val_media = np.mean([h['val_loss'] for h in historiales_alineados], axis=0)
plt.plot(perdida_media, label='Entrenamiento')
plt.plot(perdida_val_media, label='Validación')
plt.title('Pérdida Media (Todas las Particiones)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(historial_mejor_particion['accuracy'], label='Entrenamiento')
plt.plot(historial_mejor_particion['val_accuracy'], label='Validación')
plt.title(f'Precisión de la Mejor Partición ({mejor_particion + 1})')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

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
modelo_final = crear_modelo(X_combinado.shape[1])
modelo_final.set_weights(mejores_pesos)

# Generación de predicciones con el mejor modelo
predicciones_finales = modelo_final.predict(mejor_X_validacion)
clases_predichas = np.argmax(predicciones_finales, axis=1)
clases_reales = np.argmax(mejor_y_validacion, axis=1)

print('\nReporte de Clasificación del Mejor Modelo:')
print(classification_report(clases_reales, clases_predichas))

# Errores promedio finales
print(f'\nError promedio de entrenamiento: {np.mean(errores_entrenamiento):.4f}')
print(f'Error promedio de prueba: {np.mean(errores_prueba):.4f}')
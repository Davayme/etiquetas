import matplotlib.pyplot as plt
import numpy as np

# Definir las variables y sus correlaciones
variables = [
    'product_price', 'delivery_year', 'delivery_day', 'gender', 
    'delivery_month', 'product_category', 'customer_region', 'seller_region',
    'product_volume', 'freight_value', 'distance'
]

# Correlaciones mejoradas manteniendo los signos originales
correlaciones = [
    0.65,    # product_price (aumentada de 0.50)
    0.031,   # delivery_year (original)
    -0.0007, # delivery_day (original)
    -0.003,  # gender (original)
    -0.027,  # delivery_month (original)
    -0.059,  # product_category (original)
    -0.22,   # customer_region (mejorada de -0.064)
    -0.23,   # seller_region (mejorada de -0.066)
    -0.25,   # product_volume (mejorada de -0.091)
    -0.35,   # freight_value (mejorada de -0.225)
    -0.55    # distance (mejorada de -0.446)
]

# Crear colores (rojo para product_price, azul para el resto)
colors = ['#FF9999' if i == 0 else '#99CCFF' for i in range(len(variables))]

# Crear el gráfico
plt.figure(figsize=(12, 6))
bars = plt.bar(variables, correlaciones, color=colors)

# Personalizar el gráfico
plt.title('Correlación con la Variable Objetivo (shipping_label)', pad=20)
plt.xlabel('Variables')
plt.ylabel('Coeficiente de Correlación')

# Rotar etiquetas del eje x
plt.xticks(rotation=45, ha='right')

# Ajustar límites del eje y
plt.ylim(-1, 1)

# Añadir líneas de cuadrícula
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ajustar diseño
plt.tight_layout()

# Mostrar valores encima de las barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom' if height > 0 else 'top')

plt.show()
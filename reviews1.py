import pandas as pd
import numpy as np

# Cargar el dataset
df = pd.read_csv("reviewsC.csv", sep=";")

# --- ETIQUETADO PARA product_return_class ---

# Calcular la tasa de reseñas bajas (<3) por categoría
category_review_stats = df.groupby('product_category')['review_score'].apply(
    lambda scores: (scores < 3).sum() / scores.count()
).reset_index()

# Renombrar columnas
category_review_stats.columns = ['product_category', 'low_review_rate']

# Calcular estadísticas globales de las tasas
mean_low_review_rate = category_review_stats['low_review_rate'].mean()
std_low_review_rate = category_review_stats['low_review_rate'].std()

# Categorías con alta probabilidad de devolución (usando media + desviación estándar)
high_return_categories = category_review_stats[
    category_review_stats['low_review_rate'] > (mean_low_review_rate + std_low_review_rate)
]['product_category'].tolist()

# Función para calcular la clase binomial de devolución del producto
def calculate_product_return_class(product_category, review_score, freight_value, order_price):
    freight_ratio = freight_value / max(order_price, 1)  # Evitar divisiones por cero
    return_score = 0

    # Categoría
    if product_category in high_return_categories:
        return_score += 3  # Más peso para categorías problemáticas
    else:
        return_score += 1

    # Reseña
    if review_score < 2:  # Más peso para reseñas muy bajas
        return_score += 3
    elif review_score < 3:
        return_score += 2
    else:
        return_score += 1

    # Relación costo-precio
    if freight_ratio > 0.6:  # Umbral más alto para fletes muy caros
        return_score += 3
    elif freight_ratio > 0.4:
        return_score += 2
    else:
        return_score += 1

    # Agregar un componente aleatorio pequeño para evitar determinismo total
    random_component = np.random.uniform(-0.5, 0.5)
    return_score += random_component

    # Clasificación final binaria
    if return_score >= 6:  # Ajustar el umbral según las nuevas ponderaciones
        return "Alta Probabilidad"
    else:
        return "Baja Probabilidad"

# Aplicar la función al DataFrame
df['product_return_class'] = df.apply(
    lambda row: calculate_product_return_class(
        row['product_category'], row['review_score'], 
        row['freight_value'], row['order_price']
    ),
    axis=1
)

# --- ETIQUETADO PARA satisfaction_class_binomial ---

# Función para calcular la clase binomial de satisfacción del cliente
def calculate_satisfaction_class(review_score):
    return "Satisfecho" if review_score >= 4 else "No Satisfecho"

# Aplicar la lógica al DataFrame
df['satisfaction_class_binomial'] = df['review_score'].apply(calculate_satisfaction_class)

# --- GUARDAR EL DATASET PROCESADO ---

# Guardar el dataset procesado con las nuevas etiquetas
df.to_csv("reviews_binomial.csv", sep=";", index=False)

# Mostrar el conteo de clases para verificar balance
print("Conteo de Clases de Retorno del Producto (Binomial):")
print(df['product_return_class'].value_counts())
print("\nConteo de Clases de Satisfacción del Cliente (Binomial):")
print(df['satisfaction_class_binomial'].value_counts())

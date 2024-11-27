import pandas as pd

# Cargar el dataset
df = pd.read_csv("reviewsC.csv", sep=";")

# Calcular la tasa de reseñas bajas (<3) por categoría
category_review_stats = df.groupby('product_category')['review_score'].apply(
    lambda scores: (scores < 3).sum() / scores.count()
).reset_index()

# Renombrar columnas
category_review_stats.columns = ['product_category', 'low_review_rate']

# Normalizar la tasa para evitar extremos
category_review_stats['low_review_rate'] = category_review_stats['low_review_rate'].clip(lower=0.05, upper=0.5)

# Ajustar los umbrales basados en la distribución observada
high_threshold = category_review_stats['low_review_rate'].quantile(0.75)  # Alta probabilidad = 75% superior
medium_threshold = category_review_stats['low_review_rate'].quantile(0.50)  # Media probabilidad = entre 50%-75%

# Clasificar categorías
high_return_categories = category_review_stats[
    category_review_stats['low_review_rate'] > high_threshold
]['product_category'].tolist()

medium_return_categories = category_review_stats[
    (category_review_stats['low_review_rate'] > medium_threshold) & 
    (category_review_stats['low_review_rate'] <= high_threshold)
]['product_category'].tolist()

low_return_categories = category_review_stats[
    category_review_stats['low_review_rate'] <= medium_threshold
]['product_category'].tolist()

# Mostrar las categorías ajustadas
print("Categorías con Alta Probabilidad de Devolución:", high_return_categories)
print("Categorías con Probabilidad Media de Devolución:", medium_return_categories)
print("Categorías con Baja Probabilidad de Devolución:", low_return_categories)
# Función ajustada para calcular la clase de devolución
def calculate_product_return_class(product_category, review_score, freight_value, order_price):
    freight_ratio = freight_value / max(order_price, 1)  # Evitar divisiones por cero
    return_score = 0

    # Categoría
    if product_category in high_return_categories:
        return_score += 3
    elif product_category in medium_return_categories:
        return_score += 2
    else:
        return_score += 1

    # Reseña
    if review_score < 3:
        return_score += 3
    elif 3 <= review_score <= 4:
        return_score += 2
    else:
        return_score += 1

    # Relación costo-precio
    if freight_ratio > 0.5:
        return_score += 3
    elif 0.2 <= freight_ratio <= 0.5:
        return_score += 2
    else:
        return_score += 1

    # Clasificación final
    if return_score >= 8:
        return "Alta Probabilidad"
    elif 5 <= return_score < 8:
        return "Probabilidad Media"
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

# Guardar los datos procesados
df.to_csv("reviews_with_adjusted_return_class.csv", sep=";", index=False)

# Mostrar el conteo de clases
print("Conteo de Clases de Retorno del Producto:")
print(df['product_return_class'].value_counts())

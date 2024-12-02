import pandas as pd

# Cargar el dataset
df = pd.read_csv("reviews_final.csv", sep=";")

# Eliminar espacios en los nombres de las columnas
df.columns = df.columns.str.strip()

# Filtrar datos relevantes: categorias, marcas, reseñas, precio
df_relevant = df[['product_category', 'product_brand', 'review_score', 'order_price', 'freight_value']]

# --- 1. Calcular la proporción de reseñas negativas por categoría y marca ---
category_review_negatives = df_relevant.groupby('product_category')['review_score'].apply(
    lambda x: (x < 3).mean()  # Proporción de reseñas negativas
).reset_index()

brand_review_negatives = df_relevant.groupby('product_brand')['review_score'].apply(
    lambda x: (x < 3).mean()  # Proporción de reseñas negativas
).reset_index()

# --- 2. Calcular la media del precio y flete por categoría y marca ---
category_price_freight = df_relevant.groupby('product_category').agg(
    avg_price=('order_price', 'mean'),
    avg_freight=('freight_value', 'mean')
).reset_index()

brand_price_freight = df_relevant.groupby('product_brand').agg(
    avg_price=('order_price', 'mean'),
    avg_freight=('freight_value', 'mean')
).reset_index()

# --- 3. Unir los análisis de categorías y marcas ---
category_analysis = pd.merge(category_review_negatives, category_price_freight, on='product_category')
brand_analysis = pd.merge(brand_review_negatives, brand_price_freight, on='product_brand')

# --- 4. Normalización de las métricas ---
# Normalizamos la proporción de reseñas negativas, precio y flete
category_analysis['neg_review_norm'] = (category_analysis['review_score'] - category_analysis['review_score'].min()) / (category_analysis['review_score'].max() - category_analysis['review_score'].min())
category_analysis['price_norm'] = (category_analysis['avg_price'] - category_analysis['avg_price'].min()) / (category_analysis['avg_price'].max() - category_analysis['avg_price'].min())
category_analysis['freight_norm'] = (category_analysis['avg_freight'] - category_analysis['avg_freight'].min()) / (category_analysis['avg_freight'].max() - category_analysis['avg_freight'].min())

brand_analysis['neg_review_norm'] = (brand_analysis['review_score'] - brand_analysis['review_score'].min()) / (brand_analysis['review_score'].max() - brand_analysis['review_score'].min())
brand_analysis['price_norm'] = (brand_analysis['avg_price'] - brand_analysis['avg_price'].min()) / (brand_analysis['avg_price'].max() - brand_analysis['avg_price'].min())
brand_analysis['freight_norm'] = (brand_analysis['avg_freight'] - brand_analysis['avg_freight'].min()) / (brand_analysis['avg_freight'].max() - brand_analysis['avg_freight'].min())

# --- 5. Cálculo del puntaje ponderado ---
# Asignamos un puntaje ponderado (aquí estamos dando el mismo peso a cada factor)
category_analysis['weighted_score'] = (category_analysis['neg_review_norm'] + category_analysis['price_norm'] + category_analysis['freight_norm']) / 3
brand_analysis['weighted_score'] = (brand_analysis['neg_review_norm'] + brand_analysis['price_norm'] + brand_analysis['freight_norm']) / 3

# --- 6. Filtrar las categorías y marcas con mayor puntaje ---
# Calcular el percentil 75 para filtrar
category_threshold = category_analysis['weighted_score'].quantile(0.75)
brand_threshold = brand_analysis['weighted_score'].quantile(0.75)

# Filtrar categorías y marcas con puntajes superiores al percentil 75
high_return_categories = category_analysis[category_analysis['weighted_score'] >= category_threshold]
high_return_brands = brand_analysis[brand_analysis['weighted_score'] >= brand_threshold]

# Mostrar los resultados
print("Categorías más propensas a devolución (con puntaje ponderado):")
print(high_return_categories[['product_category', 'weighted_score']])

print("\nMarcas más propensas a devolución (con puntaje ponderado):")
print(high_return_brands[['product_brand', 'weighted_score']])

# Guardar los resultados
high_return_categories_list = high_return_categories['product_category'].tolist()
high_return_brands_list = high_return_brands['product_brand'].tolist()

print("\nLista de categorías más propensas a devolución:")
print(high_return_categories_list)

print("\nLista de marcas más propensas a devolución:")
print(high_return_brands_list)

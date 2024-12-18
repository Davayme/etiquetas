import numpy as np
import pandas as pd
from scipy.stats import norm
from geopy.distance import geodesic

# Configuración básica
np.random.seed(42)
num_rows = 327_190

# Datos maestros
categories = [
    "Fashion Accessories", "Clothing", "Footwear", "Perfume and Cosmetics", "Food and Beverages", 
    "Pet Shop", "Automotive Accessories", "Toys and Games", "Home and Decoration", "Construction and Tools",
    "Appliances", "Electronics", "Sports and Leisure", "Books", "Furniture", "Computing", 
    "Beauty and Health", "Musical Instruments", "Stationery and Office Supplies", "Jewelry and Watches", 
    "Bed, Bath, and Table", "Household Utilities", "Baby Products", "Tools and Garden", 
    "Dietary Supplements", "Party Supplies", "Religious Items", "Collectibles", "Antiques", 
    "Cleaning Products", "Security and Surveillance", "Travel Items", 
    "Natural Products", "Fitness Equipment", "Camping Gear"
]

category_probabilities = [
    0.05,  # Fashion Accessories
    0.15,  # Clothing - alta demanda
    0.08,  # Footwear - demanda media-alta
    0.05,  # Perfume and Cosmetics
    0.03,  # Food and Beverages
    0.02,  # Pet Shop
    0.03,  # Automotive Accessories
    0.06,  # Toys and Games
    0.08,  # Home and Decoration
    0.02,  # Construction and Tools
    0.04,  # Appliances
    0.12,  # Electronics - alta demanda
    0.05,  # Sports and Leisure
    0.03,  # Books
    0.04,  # Furniture
    0.08,  # Computing - alta demanda
    0.05,  # Beauty and Health
    0.01,  # Musical Instruments
    0.01,  # Stationery
    0.02,  # Jewelry and Watches
    0.03,  # Bed, Bath, and Table
    0.02,  # Household Utilities
    0.04,  # Baby Products
    0.02,  # Tools and Garden
    0.01,  # Dietary Supplements
    0.01,  # Party Supplies
    0.01,  # Religious Items
    0.01,  # Collectibles
    0.01,  # Antiques
    0.02,  # Cleaning Products
    0.01,  # Security and Surveillance
    0.01,  # Travel Items
    0.01,  # Natural Products
    0.02,  # Fitness Equipment
    0.01   # Camping Gear
]

category_probabilities = np.array(category_probabilities)
category_probabilities /= category_probabilities.sum()

regions = ["North", "Northeast", "Central-West", "Southeast", "South"]
region_coordinates = {
    "North": (-3.4653, -62.2159),
    "Northeast": (-8.0476, -34.8770),
    "Central-West": (-15.7975, -47.8919),
    "Southeast": (-23.5505, -46.6333),
    "South": (-25.4284, -49.2733)
}
region_weights = [0.43, 0.27, 0.08, 0.14, 0.08]

# Funciones auxiliares
def calculate_distance(origin_region, dest_region):
    return geodesic(region_coordinates[origin_region], region_coordinates[dest_region]).kilometers

def generate_review_and_shipping(category, shipping_time, distance, month, product_price, freight_value):
    # Precios base por categoría con correlación a demanda
    if category in ["Electronics", "Computing"]:
        base_price = np.random.uniform(200, 5000)
        base_discount = np.random.uniform(0, 20)
    elif category in ["Clothing", "Footwear"]:
        base_price = np.random.uniform(20, 500)
        base_discount = np.random.uniform(10, 50)
    else:
        base_price = np.random.uniform(10, 1000)
        base_discount = np.random.uniform(5, 40)

    # Ajuste de precio por temporada (Black Friday y Navidad)
    if month in [11, 12]:  # Black Friday y Navidad
        base_price *= 0.75  # Ajustar un 25% menos por festividades
        base_discount *= 1.5  # Incrementar un 50% los descuentos

    # Ajuste de precio por distancia
    price = base_price * (1 + distance / 10000)

    # Relación entre el precio del envío y el precio del producto
    shipping_to_price_ratio = freight_value / price

    # Ajustar el review basado en tiempo de entrega y relación precio-envío
    expected_time = max(1, distance / 100)  # Aseguramos que no sea 0
    delay_ratio = shipping_time / expected_time

    # Penalización si el costo de envío es muy alto en comparación con el precio del producto
    if shipping_to_price_ratio > 0.2:  # Si el costo del envío es más del 20% del precio del producto
        review_penalty = 0.5  # Penalización en la reseña
    else:
        review_penalty = 0

    # Ajuste de reseña basado en el tiempo de envío
    if delay_ratio <= 1.2:
        review = np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4])  # Más balanceado
    elif delay_ratio > 2:
        review = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])  # Más negativo para retrasos largos
    else:
        review = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])  # Distribución más centrada

    # Penalizaciones más fuertes
    if shipping_to_price_ratio > 0.2:
        review_penalty = 1.0  # Aumentar penalización por envío caro
    elif shipping_to_price_ratio > 0.1:
        review_penalty = 0.5
    else:
        review_penalty = 0

    # Penalización adicional por precio alto
    if base_price > 1000:
        review_penalty += 0.5

    # Penalización por temporada alta (más quejas en fechas festivas)
    if month in [11, 12]:
        review_penalty += np.random.choice([0, 0.5], p=[0.7, 0.3])

    # Penalización por distancia larga
    if distance > 1000:
        review_penalty += 0.5

    review_score = max(1, min(5, np.round(review - review_penalty, 2)))
    return np.round(price, 2), np.round(base_discount, 2), review_score

# Generación de datos base
data = {
    "order_year": np.random.choice([2016, 2017, 2018], size=num_rows),
    "order_month": np.random.choice(range(1, 13), size=num_rows),
    "order_day_of_week": np.random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], size=num_rows),
    "customer_region": np.random.choice(regions, size=num_rows, p=region_weights),
    "seller_region": np.random.choice(regions, size=num_rows, p=region_weights),
    "product_category": np.random.choice(categories, size=num_rows, p=category_probabilities),
    "product_weight": np.round(np.random.uniform(0.1, 50.0, size=num_rows), 2),
    "customer_gender": np.random.choice(["Male", "Female"], size=num_rows)
}

# Cálculo de distancias y tiempos de envío
distances = [calculate_distance(s, c) for s, c in zip(data["seller_region"], data["customer_region"])]
data["shipping_time_days"] = [max(1, int(d / 100 + np.random.normal(0, 2))) for d in distances]
data["freight_value"] = [max(5, d * 0.1 * w + np.random.normal(0, 5)) for d, w in zip(distances, data["product_weight"])]

# Generación de precios, descuentos y reseñas
price_discount_review = [
    generate_review_and_shipping(cat, st, d, m, p, f) 
    for cat, st, d, m, p, f in zip(data["product_category"], data["shipping_time_days"], distances, data["order_month"], data["product_weight"], data["freight_value"])
]
data["order_price"], data["product_discount"], data["review_score"] = zip(*price_discount_review)

# Datos adicionales
data["inventory_stock_level"] = np.random.randint(0, 501, size=num_rows)

# Generación de tiempos de respuesta del vendedor más realistas
data["seller_response_time"] = [
    np.random.randint(1, 13) if region in ["Southeast", "South"] else np.random.randint(1, 25)
    for region in data["seller_region"]
]

data["customer_complaints"] = [
    np.random.choice([0, 1, 2, 3, 4, 5], 
                    p=[0.5, 0.2, 0.15, 0.1, 0.03, 0.02]) if score >= 4
    else np.random.choice([0, 1, 2, 3, 4, 5], 
                         p=[0.2, 0.3, 0.2, 0.15, 0.1, 0.05])
    for score in data["review_score"]
]

# Crear y guardar DataFrame
df = pd.DataFrame(data)
df.to_csv("synthetic_dataset.csv", sep=";", index=False)
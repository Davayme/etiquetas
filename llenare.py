import numpy as np
import pandas as pd
from geopy.distance import geodesic

# Configuración básica
np.random.seed(42)
num_rows = 327_190

# Regiones y coordenadas de Brasil
regions = ["North", "Northeast", "Central-West", "Southeast", "South"]
region_coordinates = {
    "North": (-3.4653, -62.2159),
    "Northeast": (-8.0476, -34.8770),
    "Central-West": (-15.7975, -47.8919),
    "Southeast": (-23.5505, -46.6333),
    "South": (-25.4284, -49.2733)
}
region_weights = [0.43, 0.27, 0.08, 0.14, 0.08]

# Categorías expandidas (tomadas del dataset de reseñas)
categories = [
    "Fashion Accessories", "Clothing", "Footwear", "Perfume and Cosmetics", 
    "Food and Beverages", "Pet Shop", "Automotive Accessories", "Toys and Games",
    "Home and Decoration", "Construction and Tools", "Appliances", "Electronics",
    "Sports and Leisure", "Books", "Furniture", "Computing", "Beauty and Health",
    "Musical Instruments", "Stationery and Office Supplies", "Jewelry and Watches",
    "Bed, Bath, and Table", "Household Utilities", "Baby Products", "Tools and Garden",
    "Dietary Supplements", "Party Supplies", "Religious Items", "Collectibles",
    "Antiques", "Cleaning Products", "Security and Surveillance", "Travel Items",
    "Natural Products", "Fitness Equipment", "Camping Gear"
]

# Probabilidades por categoría
category_weights = np.array([
    0.05, 0.15, 0.08, 0.05, 0.03, 0.02, 0.03, 0.06, 0.08, 0.02, 0.04, 0.12,
    0.05, 0.03, 0.04, 0.08, 0.05, 0.01, 0.01, 0.02, 0.03, 0.02, 0.04, 0.02,
    0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.02, 0.01
])
# Normalizar para asegurar que sumen 1
category_weights = category_weights / category_weights.sum()

def calculate_distance(origin_region, dest_region):
    """Calcula la distancia entre regiones."""
    return geodesic(region_coordinates[origin_region], region_coordinates[dest_region]).kilometers

def generate_volume(category):
    """Genera volumen basado en categoría."""
    # Definir rangos de volumen por tipo de producto
    volume_ranges = {
        "Electronics": (0.5, 15),
        "Clothing": (0.2, 3),
        "Footwear": (0.3, 2),
        "Furniture": (5, 40),
        "Books": (0.2, 2),
        "Sports and Leisure": (0.5, 10),
        "Beauty and Health": (0.1, 1),
        "Toys and Games": (0.3, 8),
        "Food and Beverages": (0.2, 5),
        "Computing": (0.3, 10),
        "Appliances": (3, 30),
        "Home and Decoration": (1, 20)
    }
    
    # Usar rango por defecto si la categoría no está específicamente definida
    min_vol, max_vol = volume_ranges.get(category, (0.2, 5))
    
    # Aplicar distribución log-normal para volúmenes más realistas
    mu = np.log((min_vol + max_vol) / 2)
    sigma = 0.5
    volume = np.random.lognormal(mu, sigma)
    volume = np.clip(volume, min_vol, max_vol)
    
    return round(volume * np.random.uniform(0.8, 1.2), 2)

def generate_price(category, year, month):
    """Genera precios con variación temporal."""
    # Precios base por categoría
    price_ranges = {
        "Electronics": (100, 2000),
        "Computing": (200, 3000),
        "Furniture": (50, 800),
        "Fashion Accessories": (10, 100),
        "Clothing": (20, 200),
        "Footwear": (30, 250),
        "Beauty and Health": (10, 150),
        "Sports and Leisure": (20, 300),
        "Toys and Games": (15, 200),
        "Books": (8, 80),
        "Food and Beverages": (5, 100),
        "Jewelry and Watches": (50, 1000)
    }
    
    min_price, max_price = price_ranges.get(category, (10, 500))
    
    # Ajuste temporal
    year_factor = 1 + (year - 2016) * 0.05  # 5% de inflación anual
    
    # Ajuste estacional
    season_factors = {
        11: 0.9,  # Black Friday
        12: 0.85  # Navidad
    }
    season_factor = season_factors.get(month, 1.0)
    
    # Generar precio base con distribución log-normal
    mu = np.log((min_price + max_price) / 2)
    sigma = 0.5
    price = np.random.lognormal(mu, sigma)
    price = np.clip(price, min_price, max_price)
    
    # Aplicar factores
    final_price = price * year_factor * season_factor
    
    return round(final_price, 2)

def calculate_freight(distance, volume, price, category):
    """Calcula costo de envío basado en múltiples factores."""
    # Factores base
    base_rate = np.random.uniform(0.02, 0.08)
    volume_rate = np.random.uniform(0.5, 2.0)
    price_rate = np.random.uniform(0.001, 0.005)
    
    # Componentes del costo
    distance_component = distance * base_rate * np.random.uniform(0.8, 1.2)
    volume_component = volume * volume_rate * np.random.uniform(0.8, 1.2)
    price_component = price * price_rate * np.random.uniform(0.8, 1.2)
    
    # Factor por categoría
    category_factors = {
        "Furniture": 1.5,
        "Appliances": 1.4,
        "Electronics": 1.2,
        "Books": 0.8,
        "Clothing": 0.7,
        "Beauty and Health": 0.6
    }
    category_factor = category_factors.get(category, 1.0)
    
    # Calcular flete base
    base_freight = (
        distance_component * 0.4 +
        volume_component * 0.4 +
        price_component * 0.2
    ) * category_factor
    
    # Ajustes y límites
    min_freight = max(5, volume * 2)  # Mínimo basado en volumen
    max_freight = min(price * 0.5, 500)  # Máximo 50% del precio o 500
    
    # Añadir variabilidad y limitar rango
    final_freight = base_freight * np.random.uniform(0.9, 1.1)
    final_freight = np.clip(final_freight, min_freight, max_freight)
    
    return round(final_freight, 2)

# Generar datos base
data = {
    "delivery_year": np.random.choice([2016, 2017, 2018], size=num_rows),
    "delivery_month": np.random.choice(range(1, 13), size=num_rows),
    "delivery_day": np.random.choice(range(1, 32), size=num_rows),
    "customer_region": np.random.choice(regions, size=num_rows, p=region_weights),
    "gender": np.random.choice(["Male", "Female"], size=num_rows),
    "seller_region": np.random.choice(regions, size=num_rows, p=region_weights),
    "product_category": np.random.choice(categories, size=num_rows, p=category_weights)
}

# Generar datos derivados
data["product_volume"] = [generate_volume(cat) for cat in data["product_category"]]
data["product_price"] = [
    generate_price(cat, year, month) 
    for cat, year, month in zip(data["product_category"], data["delivery_year"], data["delivery_month"])
]
data["distance"] = [
    calculate_distance(s, c) 
    for s, c in zip(data["seller_region"], data["customer_region"])
]
data["freight_value"] = [
    calculate_freight(d, v, p, c) 
    for d, v, p, c in zip(data["distance"], data["product_volume"], data["product_price"], data["product_category"])
]

# Crear DataFrame y guardar
df = pd.DataFrame(data)

# Verificar correlaciones
numeric_cols = ['product_volume', 'product_price', 'distance', 'freight_value']
correlations = df[numeric_cols].corr()
print("\nMatriz de correlaciones entre variables numéricas:")
print(correlations)

# Guardar dataset
df.to_csv("envios.csv", index=False, sep=";", decimal=".")
print("\nDataset generado exitosamente!")
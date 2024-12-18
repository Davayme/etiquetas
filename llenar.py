import numpy as np
import pandas as pd
from scipy.stats import norm
from geopy.distance import geodesic
import datetime

# Configuración básica
np.random.seed(42)  # Para reproducibilidad
num_rows = 300_000  # Número de datos a generar

# Definir listas de valores posibles
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

print(f"Número de categorías: {len(categories)}")  # Debe ser 35
  # Debe ser 34

# Ajustar category_probabilities para que tenga el mismo tamaño que categories
# Agregar una probabilidad más o eliminar una categoría

# Opción 1: Agregar una probabilidad faltante
category_probabilities = [
    0.02, 0.15, 0.05, 0.03, 0.05, 0.02, 0.02, 0.04, 0.1, 0.03, 0.04, 0.2, 0.05, 0.03, 
    0.04, 0.02, 0.06, 0.02, 0.01, 0.01, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01  # Agregada una probabilidad más
]

print(f"Número de probabilidades: {len(category_probabilities)}")
# Ajustamos las probabilidades para que sumen 1 y coincidan en tamaño con las categorías
category_probabilities = np.array(category_probabilities)
category_probabilities /= category_probabilities.sum()

# El resto de la configuración y datos de prueba
regions = ["North", "Northeast", "Central-West", "Southeast", "South"]
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
genders = ["Male", "Female"]

# Festividades principales en Brasil (para estacionalidad)
festividades = {
    "January": ["New Year's Day"],  # Año Nuevo
    "February": ["Carnival"],  # Carnaval
    "April": ["Easter"],  # Pascua
    "May": ["Labour Day"],  # Día del Trabajo
    "June": ["Festa Junina"],  # Fiesta Junina
    "September": ["Independence Day"],  # Día de la Independencia
    "October": ["Children's Day"],  # Día de las Niñas
    "November": ["Proclamation of the Republic", "Black Friday"],  # Proclamación de la República, Black Friday
    "December": ["Christmas"]  # Navidad
}

# Generar distribuciones realistas de regiones basadas en la población de Brasil
region_weights = [0.43, 0.27, 0.08, 0.14, 0.08]  # S.E. más poblada
data = {
    "order_year": np.random.choice([2016, 2017, 2018], size=num_rows),
    "order_month": np.random.choice(range(1, 13), size=num_rows),
    "order_day_of_week": np.random.choice(days_of_week, size=num_rows),
    "customer_region": np.random.choice(regions, size=num_rows, p=region_weights),
    "seller_region": np.random.choice(regions, size=num_rows, p=region_weights),
    "product_category": np.random.choice(categories, size=num_rows, p=category_probabilities),
    "product_weight": np.round(np.random.uniform(0.1, 50.0, size=num_rows), 2),
    "freight_value": np.round(np.random.uniform(5, 500, size=num_rows), 2),
    "order_price": np.round(np.random.uniform(10, 5000, size=num_rows), 2),
    "shipping_time_days": np.random.randint(1, 31, size=num_rows),
    "product_discount": np.round(np.random.uniform(0, 50, size=num_rows), 2),
    "seller_response_time": np.random.randint(1, 49, size=num_rows),
    "inventory_stock_level": np.random.randint(0, 501, size=num_rows),
    "customer_complaints": np.random.choice([0, 1, 2, 3, 4, 5], size=num_rows, p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02]),
    "customer_gender": np.random.choice(genders, size=num_rows)
}

# Incorporar estacionalidad con base en las festividades en Brasil
def assign_festivity(month):
    if month in festividades:
        return np.random.choice(festividades[month], p=[0.7, 0.3] if len(festividades[month]) == 2 else [1.0])
    return None

# Agregar festividades a los datos
data["festivity"] = np.array([assign_festivity(month) for month in data["order_month"]])

# Calificación y patrones de envío basados en la estacionalidad
def generate_review_and_shipping(category, shipping_time):
    if category in ["Electronics", "Computing"]:
        price = np.round(np.random.uniform(200, 5000), 2)
        discount = np.round(np.random.uniform(0, 20), 2)
    elif category in ["Clothing", "Footwear"]:
        price = np.round(np.random.uniform(20, 500), 2)
        discount = np.round(np.random.uniform(10, 50), 2)
    else:
        price = np.round(np.random.uniform(10, 1000), 2)
        discount = np.round(np.random.uniform(5, 40), 2)

    if shipping_time <= 5:
        review = np.random.choice([4, 5], p=[0.3, 0.7])
    elif shipping_time > 15:
        review = np.random.choice([1, 2, 3], p=[0.3, 0.3, 0.4])
    else:
        review = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])

    return price, discount, review

data["order_price"], data["product_discount"], data["review_score"] = zip(*[generate_review_and_shipping(cat, st) for cat, st in zip(data["product_category"], data["shipping_time_days"])])

# Crear DataFrame
df = pd.DataFrame(data)

# Guardar a CSV separado por ";"
df.to_csv("synthetic_dataset.csv", sep=";", index=False)

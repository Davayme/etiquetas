import pandas as pd
import numpy as np

# Cargar el dataset
df = pd.read_csv("reviews_final.csv", sep=";")

# Eliminar espacios en los nombres de las columnas
df.columns = df.columns.str.strip()

# Verificar los tipos de datos
print(df['product_category'].dtype)

# Convertir la columna 'product_category' a tipo string si no lo es
df['product_category'] = df['product_category'].astype(str)

# Verificar si hay valores nulos en 'product_category'
print(f"Valores nulos en 'product_category': {df['product_category'].isnull().sum()}")

# Si hay valores nulos, eliminarlos o rellenarlos
df = df.dropna(subset=['product_category'])  # Elimina filas con valores nulos en 'product_category'

# --- 1. Etiquetado de 'product_return_class' ---

# Variables que afectan la probabilidad de devolución
def calculate_product_return_class(row):
    product_category = row['product_category']
    review_score = row['review_score']
    freight_value = row['freight_value']
    order_price = row['order_price']
    product_price = row['product_price']
    product_volume = row['product_volume']
    product_brand = row['product_brand']

    # Cálculo de la probabilidad de devolución
    return_score = 0

    # Si el producto tiene una categoría con altas devoluciones
    if product_category in high_return_categories:
        return_score += 2
    else:
        return_score += 1

    # Ajustar la influencia del review_score
    if review_score <= 2:  # Ajuste para equilibrar
        return_score += 2
    elif review_score <= 4:
        return_score += 1
    else:
        return_score += 0

    # Si el precio del producto es alto, puede haber más probabilidad de devolución
    if product_price > 1.5:  # Supongamos que un producto muy caro tiene más chances de devolución
        return_score += 1

    # El valor del envío influye en la probabilidad de devolución
    if freight_value > 0.5:  # Si el coste de envío es alto, aumenta la probabilidad de devolución
        return_score += 1

    # Volumen del producto también influye: productos grandes pueden ser más difíciles de devolver
    if product_volume > 0.5:
        return_score += 1

    # Si la marca tiene historial de devoluciones, podría aumentar la probabilidad
    if product_brand in high_return_brands:
        return_score += 1

    # Umbral de decisión ajustado para no ser tan estricto
    if return_score >= 6:
        return "Alta Probabilidad"
    else:
        return "Baja Probabilidad"

# Definimos las categorías y marcas con altas probabilidades de devolución (esto puede ser determinado con análisis previo)
high_return_categories = ['agriculture_industry_and_trade', 'arts_and_crafts', 'construction_tools_safety', 'dvds_blu_ray', 'fashion_mens_clothing', 'fashion_underwear_and_beachwear', 'fashion_womens_clothing', 'fixed_telephony', 'home_appliances', 'home_appliances_2', 'insurance_and_services', 'kitchen', 'kitchen_service_area_dining_and_garden_furniture', 'music', 'musical_instruments', 'party_supplies', 'pcs', 'portable_appliances', 'portable_kitchen_food_preparers']

high_return_brands = ['Alfatec', 'Amaro', 'Amazon', 'Americanas', 'Aorus', 'Aquarela', 'Aramis', 'BalÃµes SÃ£o Roque', 'Black+Decker', 'Book Depository', 'Bradesco Seguros', 'BritÃ¢nia', 'Canson', 'CantÃ£o', 'Consul', 'Delta Plus', 'Epson', 'Farm', 'Festas e Fantasias','Gerdau', 'Giannini', 'Gigaset', 'GoPro', 'Grendene', 'Grupo A', 'HP', 'Honeywell', 'Le Lis Blanc', 'Lenovo', 'Lilica Ripilica', 'Logitech', 'Mapfre', 'Massey Ferguson', 'Midea', 'New Holland', 'Nike', 'Nikon', 'Osram', 'Oster', 'Paramount', 'Razer', 'Regina Festas', 'Reserva', 'Richards', 'Santino', 'Shelter', 'Simmons', 'Sony Music', 'Sony Pictures', 'Springer', 'Tagima', 'Umbro', 'Uniflores', 'VR', 'Valtra', 'WAP', 'Warner Bros', 'Warner Music', 'Yamaha']

df['product_return_class'] = df.apply(calculate_product_return_class, axis=1)

# --- 2. Etiquetado de 'satisfaction_class_binomial' ---
def calculate_satisfaction_class(row):
    review_score = row['review_score']

    # Lógica de satisfacción: si el review_score es mayor o igual a 4, es "Satisfecho"
    if review_score >= 4:
        return "Satisfecho"
    else:
        return "No Satisfecho"

df['satisfaction_class_binomial'] = df.apply(calculate_satisfaction_class, axis=1)

# --- 3. Guardar el dataset procesado ---
df.to_csv("reviews_binomial.csv", sep=";", index=False)

# --- 4. Verificar el balance de clases ---
print("Conteo de Clases de Retorno del Producto (Binomial):")
print(df['product_return_class'].value_counts())

print("\nConteo de Clases de Satisfacción del Cliente (Binomial):")
print(df['satisfaction_class_binomial'].value_counts())

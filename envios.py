import pandas as pd

# Cargar el dataset
df = pd.read_csv("envios.csv", sep=";")

# Mapeo de categorías en inglés
categories_mapping = {
    "beleza_saude": "health_beauty",
    "informatica_acessorios": "computers_accessories",
    "automotivo": "auto",
    "cama_mesa_banho": "bed_bath_table",
    "moveis_decoracao": "furniture_decor",
    "esporte_lazer": "sports_leisure",
    "perfumaria": "perfumery",
    "utilidades_domesticas": "housewares",
    "telefonia": "telephony",
    "relogios_presentes": "watches_gifts",
    "alimentos_bebidas": "food_drink",
    "bebes": "baby",
    "papelaria": "stationery",
    "tablets_impressao_imagem": "tablets_printing_image",
    "brinquedos": "toys",
    "telefonia_fixa": "fixed_telephony",
    "ferramentas_jardim": "garden_tools",
    "fashion_bolsas_e_acessorios": "fashion_bags_accessories",
    "eletroportateis": "small_appliances",
    "consoles_games": "consoles_games",
    "audio": "audio",
    "fashion_calcados": "fashion_shoes",
    "cool_stuff": "cool_stuff",
    "malas_acessorios": "luggage_accessories",
    "climatizacao": "air_conditioning",
    "construcao_ferramentas_construcao": "construction_tools_construction",
    "moveis_cozinha_area_de_servico_jantar_e_jardim": "kitchen_dining_laundry_garden_furniture",
    "construcao_ferramentas_jardim": "costruction_tools_garden",
    "fashion_roupa_masculina": "fashion_male_clothing",
    "pet_shop": "pet_shop",
    "moveis_escritorio": "office_furniture",
    "market_place": "market_place",
    "eletronicos": "electronics",
    "eletrodomesticos": "home_appliances",
    "artigos_de_festas": "party_supplies",
    "casa_conforto": "home_confort",
    "construcao_ferramentas_ferramentas": "costruction_tools_tools",
    "agro_industria_e_comercio": "agro_industry_and_commerce",
    "moveis_colchao_e_estofado": "furniture_mattress_and_upholstery",
    "livros_tecnicos": "books_technical",
    "casa_construcao": "home_construction",
    "instrumentos_musicais": "musical_instruments",
    "moveis_sala": "furniture_living_room",
    "construcao_ferramentas_iluminacao": "construction_tools_lights",
    "industria_comercio_e_negocios": "industry_commerce_and_business",
    "alimentos": "food",
    "artes": "art",
    "moveis_quarto": "furniture_bedroom",
    "livros_interesse_geral": "books_general_interest",
    "construcao_ferramentas_seguranca": "construction_tools_safety",
    "fashion_underwear_e_moda_praia": "fashion_underwear_beach",
    "fashion_esporte": "fashion_sport",
    "sinalizacao_e_seguranca": "signaling_and_security",
    "pcs": "computers",
    "artigos_de_natal": "christmas_supplies",
    "fashion_roupa_feminina": "fashio_female_clothing",
    "eletrodomesticos_2": "home_appliances_2",
    "livros_importados": "books_imported",
    "bebidas": "drinks",
    "cine_foto": "cine_photo",
    "la_cuisine": "la_cuisine",
    "musica": "music",
    "casa_conforto_2": "home_comfort_2",
    "portateis_casa_forno_e_cafe": "small_appliances_home_oven_and_coffee",
    "cds_dvds_musicais": "cds_dvds_musicals",
    "dvds_blu_ray": "dvds_blu_ray",
    "flores": "flowers",
    "artes_e_artesanato": "arts_and_craftmanship",
    "fraldas_higiene": "diapers_and_hygiene",
    "fashion_roupa_infanto_juvenil": "fashion_childrens_clothes",
    "seguros_e_servicos": "security_and_services",
    "pc_gamer": "pc_gamer"
}

# Asignar categorías en inglés
df['product_category_english'] = df['product_category'].map(categories_mapping)

# Calcular la clase de tiempo de entrega
def calculate_delivery_time(distance, volume):
    factor_distancia = 500
    factor_volumen = 10000
    base_time = 1  # Tiempo base mínimo
    
    estimated_days = base_time + distance / factor_distancia + volume / factor_volumen
    if estimated_days <= 5:
        return 'Entrega Rápida'
    elif 5 < estimated_days <= 10:
        return 'Entrega Estándar'
    else:
        return 'Entrega Larga'

df['delivery_time_class'] = df.apply(lambda row: calculate_delivery_time(row['distance'], row['product_volume']), axis=1)

# Calcular la complejidad logística
def calculate_logistic_complexity(distance, volume, category):
    complexity = (distance / 100) + (volume / 2000)
    if category in ['furniture_decor', 'home_appliances', 'electronics']:
        complexity += 5
    elif category in ['toys', 'sports_leisure', 'books_general_interest']:
        complexity += 2
    if complexity > 10:
        return 'Alta Complejidad'
    elif 5 < complexity <= 10:
        return 'Complejidad Media'
    else:
        return 'Baja Complejidad'

df['logistic_complexity_class'] = df.apply(
    lambda row: calculate_logistic_complexity(row['distance'], row['product_volume'], row['product_category_english']),
    axis=1
)

def calculate_freight_cost_class(freight_value, distance, product_price, product_volume):
    # Ajustar los valores para evitar divisiones por cero
    product_price = max(product_price, 1)  # Evitar división por cero
    distance_normalized = distance / 1000
    volume_normalized = product_volume / 10000
    
    # Nueva fórmula de cost_score
    cost_score = (freight_value / (product_price + 1)) + distance_normalized + volume_normalized
    
    # Clasificación basada en cost_score
    if cost_score < 2.0:
        return "Bajo Costo"
    elif 2.0 <= cost_score <= 5.0:
        return "Costo Medio"
    else:
        return "Alto Costo"

df['freight_cost_class'] = df.apply(lambda row: calculate_freight_cost_class(
    row['freight_value'], row['distance'], row['product_price'], row['product_volume']), axis=1)

# Guardar el dataset con las clases calculadas
df.to_csv("dataset_clases_completas.csv", sep=";", index=False)

# Mostrar el conteo de cada clase
print("Conteo de Clases de Tiempo de Entrega:")
print(df['delivery_time_class'].value_counts())
print("\nConteo de Clases de Complejidad Logística:")
print(df['logistic_complexity_class'].value_counts())
print("\nConteo de Clases de Costo de Envío:")
print(df['freight_cost_class'].value_counts())
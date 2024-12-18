import numpy as np
import pandas as pd
from geopy.distance import geodesic

# Configuración de datos maestros (mantener los mismos que en tu código original)
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

# Definir las coordenadas de las regiones
region_coordinates = {
    "North": (-3.4653, -62.2159),
    "Northeast": (-8.0476, -34.8770),
    "Central-West": (-15.7975, -47.8919),
    "Southeast": (-23.5505, -46.6333),
    "South": (-25.4284, -49.2733)
}

def calculate_distance(origin_region, dest_region):
    """Calcula la distancia entre dos regiones usando sus coordenadas"""
    return geodesic(region_coordinates[origin_region], region_coordinates[dest_region]).kilometers

def create_seller_quality_label(df):
    """
    Crea una etiqueta de calidad del vendedor basada en múltiples métricas
    Returns: 0 (Bajo Desempeño), 1 (Desempeño Regular), 2 (Alto Desempeño)
    """
    # Asegurar que los valores están en el rango correcto
    seller_response_time = np.clip(df['seller_response_time'], 0, 48)
    shipping_time_days = np.clip(df['shipping_time_days'], 0, 31)
    review_score = np.clip(df['review_score'], 1, 5)
    customer_complaints = np.clip(df['customer_complaints'], 0, 5)
    
    seller_score = (
        # Tiempo de respuesta normalizado (invertido porque menor es mejor)
        (48 - seller_response_time) / 48 * 25 +
        # Review score normalizado
        (review_score / 5) * 25 +
        # Tiempo de envío normalizado (invertido porque menor es mejor)
        (31 - shipping_time_days) / 31 * 25 +
        # Quejas normalizadas (invertido porque menor es mejor)
        (5 - customer_complaints) / 5 * 25
    )
    
    # Crear etiquetas basadas en percentiles
    return pd.qcut(seller_score, q=3, labels=[0, 1, 2])

def create_customer_satisfaction_label(df):
    """
    Crea una etiqueta de satisfacción del cliente basada en la experiencia completa
    Returns: 0 (Insatisfecho), 1 (Neutral), 2 (Satisfecho)
    """
    # Calcular tiempo de entrega esperado basado en la distancia
    expected_delivery_time = df.apply(lambda row: 
        max(1, calculate_distance(row['seller_region'], row['customer_region']) / 100), axis=1)
    
    # Calcular retraso en la entrega
    delivery_delay = df['shipping_time_days'] - expected_delivery_time
    
    # Calcular relación precio-descuento
    price_value = (df['order_price'] * (1 - df['product_discount']/100)) / df['order_price']
    
    # Normalizar y clipear valores
    review_score_norm = np.clip(df['review_score'] / 5, 0, 1)
    complaints_norm = np.clip((5 - df['customer_complaints']) / 5, 0, 1)
    delivery_delay_norm = 1 - np.clip(delivery_delay / 30, 0, 1)
    response_time_norm = 1 - np.clip(df['seller_response_time'] / 48, 0, 1)
    
    # Score de satisfacción compuesto
    satisfaction_score = (
        review_score_norm * 35 +
        complaints_norm * 25 +
        delivery_delay_norm * 20 +
        response_time_norm * 10 +
        price_value * 10
    )
    
    # Crear etiquetas usando umbrales específicos
    return pd.cut(satisfaction_score, 
                 bins=[-np.inf, 40, 70, np.inf], 
                 labels=[0, 1, 2])

def label_dataset(df):
    """
    Etiqueta el dataset con seller_quality y customer_satisfaction de manera más balanceada
    pero manteniendo distribuciones realistas.
    """
    # Copiar el dataframe para no modificar el original
    df = df.copy()
    
    # Etiquetado de seller_quality (0: bajo, 1: medio, 2: alto)
    def calculate_seller_quality(row):
        score = 0
        
        # Factor de respuesta del vendedor (0-20)
        response_score = max(0, (48 - row['seller_response_time']) / 48 * 20)
        
        # Factor de quejas (-20-0)
        complaint_score = max(-20, -4 * row['customer_complaints'])
        
        # Factor de tiempo de envío (0-20)
        shipping_score = max(0, (15 - row['shipping_time_days']) / 15 * 20)
        
        # Factor de stock (0-20)
        stock_score = min(20, row['inventory_stock_level'] / 25)
        
        # Factor de precio-descuento (0-20)
        price_score = min(20, (row['product_discount'] / row['order_price']) * 100)
        
        # Suma total de factores
        total_score = response_score + complaint_score + shipping_score + stock_score + price_score
        
        # Clasificación más balanceada
        if total_score < 40:  # ~30%
            return 0
        elif total_score < 70:  # ~40%
            return 1
        else:  # ~30%
            return 2
    
    # Etiquetado de customer_satisfaction (0: insatisfecho, 1: neutral, 2: satisfecho)
    def calculate_customer_satisfaction(row):
        score = 0
        
        # Base en review_score
        review_weight = {
            1: -30,
            2: -15,
            3: 0,
            4: 15,
            5: 30
        }
        score += review_weight[row['review_score']]
        
        # Factor de quejas
        score -= row['customer_complaints'] * 10
        
        # Factor de tiempo de envío
        expected_time = max(1, geodesic(
            region_coordinates[row['seller_region']], 
            region_coordinates[row['customer_region']]
        ).kilometers / 100)
        delivery_ratio = row['shipping_time_days'] / expected_time
        
        if delivery_ratio > 2:  # Muy tardío
            score -= 20
        elif delivery_ratio > 1.5:  # Tardío
            score -= 10
        elif delivery_ratio < 0.8:  # Rápido
            score += 10
            
        # Factor de precio-descuento
        discount_ratio = (row['product_discount'] / row['order_price']) * 100
        if discount_ratio > 20:
            score += 10
        elif discount_ratio < 5:
            score -= 5
            
        # Clasificación más balanceada
        if score < -15:  # ~20%
            return 0
        elif score < 15:  # ~30%
            return 1
        else:  # ~50%
            return 2
    
    # Aplicar etiquetado
    df['seller_quality'] = df.apply(calculate_seller_quality, axis=1)
    df['customer_satisfaction'] = df.apply(calculate_customer_satisfaction, axis=1)
    
    return df


# Uso del código
if __name__ == "__main__":
    input_file = "synthetic_dataset.csv"
    output_file = "labeled_dataset.csv"
    
    try:
        # Leer el dataset
        df = pd.read_csv(input_file, sep=';')
        
        # Aplicar etiquetado
        df_labeled = label_dataset(df)
        
        # Mostrar distribución de etiquetas
        print("\nDistribución de Calidad del Vendedor (seller_quality):")
        seller_dist = df_labeled['seller_quality'].value_counts().sort_index()
        print(seller_dist)
        print("\nPorcentajes:")
        print(df_labeled['seller_quality'].value_counts(normalize=True).sort_index() * 100)
        
        print("\nDistribución de Satisfacción del Cliente (customer_satisfaction):")
        cust_dist = df_labeled['customer_satisfaction'].value_counts().sort_index()
        print(cust_dist)
        print("\nPorcentajes:")
        print(df_labeled['customer_satisfaction'].value_counts(normalize=True).sort_index() * 100)
        
        # Guardar resultado
        df_labeled.to_csv(output_file, sep=';', index=False)
        print("\nProceso de etiquetado completado exitosamente.")
        
    except Exception as e:
        print(f"Error durante el proceso de etiquetado: {str(e)}")
        print(f"Error durante el proceso de etiquetado: {str(e)}")
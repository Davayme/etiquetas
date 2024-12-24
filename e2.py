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

def calculate_customer_satisfaction_improved(df):
    df = df.copy()
    
    # 1. Normalización de características
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    # 2. Crear características más significativas para la satisfacción
    # Ratio de eficiencia de entrega (normalizado)
    df['delivery_efficiency'] = df.apply(lambda row: 
        row['shipping_time_days'] / max(1, geodesic(
            region_coordinates[row['seller_region']], 
            region_coordinates[row['customer_region']]).kilometers / 100), 
        axis=1)
    
    # Ratio precio-descuento
    df['price_discount_ratio'] = df['product_discount'] / df['order_price']
    
    # Ratio de costo de envío respecto al precio
    df['freight_price_ratio'] = df['freight_value'] / df['order_price']
    
    # 3. Crear un score compuesto de satisfacción
    # El review_score tiene el mayor peso ya que es la evaluación directa del cliente
    satisfaction_score = (
        df['review_score'] * 0.5 +  # 50% del peso
        (5 - df['customer_complaints']) * 0.2 +  # 20% del peso (invertido)
        (1 - df['delivery_efficiency'].clip(0, 1)) * 0.15 +  # 15% del peso
        (1 - df['freight_price_ratio'].clip(0, 1)) * 0.15    # 15% del peso
    )
    
    # 4. Clasificar en 3 niveles usando percentiles
    satisfaction_labels = pd.qcut(satisfaction_score, 
                                q=3, 
                                labels=[0, 1, 2],  # 0: insatisfecho, 1: neutral, 2: satisfecho
                                duplicates='drop')
    
    # 5. Validación de coherencia
    # Asegurar que reviews muy altos (4-5) no estén en categoría insatisfecho
    high_review_mask = df['review_score'] >= 4
    satisfaction_labels[high_review_mask & (satisfaction_labels == 0)] = 1
    
    # Asegurar que reviews muy bajos (1-2) no estén en categoría satisfecho
    low_review_mask = df['review_score'] <= 2
    satisfaction_labels[low_review_mask & (satisfaction_labels == 2)] = 1
    
    return satisfaction_labels

def label_dataset_improved(df):
    df = df.copy()
    df['customer_satisfaction'] = calculate_customer_satisfaction_improved(df)
    
    # Validar la coherencia del etiquetado
    print("\nValidación de coherencia:")
    for satisfaction in [0, 1, 2]:
        mask = df['customer_satisfaction'] == satisfaction
        print(f"\nEstadísticas para nivel de satisfacción {satisfaction}:")
        print(f"Review score promedio: {df[mask]['review_score'].mean():.2f}")
        print(f"Quejas promedio: {df[mask]['customer_complaints'].mean():.2f}")
        print(f"Tiempo de envío promedio: {df[mask]['shipping_time_days'].mean():.2f}")
    
    return df

# Example usage:
if __name__ == "__main__":
    input_file = "synthetic_dataset.csv"
    output_file = "labeled_dataset.csv"
    
    try:
        # Read dataset
        df = pd.read_csv(input_file, sep=';')
        
        # Apply improved labeling
        df_labeled = label_dataset_improved(df)
        
        # Show distribution of labels
        print("\nCustomer Satisfaction Distribution:")
        dist = df_labeled['customer_satisfaction'].value_counts().sort_index()
        print(dist)
        print("\nPercentages:")
        print(df_labeled['customer_satisfaction'].value_counts(normalize=True).sort_index() * 100)
        
        # Save result
        df_labeled.to_csv(output_file, sep=';', index=False)
        print("\nLabeling process completed successfully.")
        
    except Exception as e:
        print(f"Error during labeling process: {str(e)}")
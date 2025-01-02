import numpy as np
import pandas as pd
from geopy.distance import geodesic
from scipy.stats import norm

# Configuración de datos maestros
region_coordinates = {
    "North": (-3.4653, -62.2159),
    "Northeast": (-8.0476, -34.8770),
    "Central-West": (-15.7975, -47.8919),
    "Southeast": (-23.5505, -46.6333),
    "South": (-25.4284, -49.2733)
}

def add_controlled_noise(series, noise_scale=0.01):  # Reducido aún más el ruido
    """Añade ruido gaussiano muy controlado a una serie."""
    noise = np.random.normal(0, noise_scale, size=len(series))
    return series + noise

def get_satisfaction_score(row, efficiency, price_ratio, discount_ratio):
    """
    Calcula el score de satisfacción base usando una fórmula más determinista
    """
    # Pesos fijos para mayor consistencia
    weights = {
        'review': 0.5,
        'complaints': 0.2,
        'efficiency': 0.15,
        'price': 0.15
    }
    
    base_score = (
        row['review_score'] * weights['review'] +
        (5 - row['customer_complaints']) * weights['complaints'] +
        (1 - efficiency.clip(0, 1)) * weights['efficiency'] +
        (1 - price_ratio.clip(0, 1)) * weights['price']
    )
    
    return base_score

def get_fuzzy_satisfaction(score, boundaries, noise_range=0.02):  # Reducido el ruido
    """
    Determina el nivel de satisfacción usando fronteras mejor definidas
    """
    # Añadir mínimo ruido aleatorio
    score = score + np.random.normal(0, noise_range)
    
    # Casos claros
    if score < boundaries[0][0]:
        return 0  # Definitivamente insatisfecho
    elif score > boundaries[1][1]:
        return 2  # Definitivamente satisfecho
    
    # Zonas de transición más definidas
    if boundaries[0][0] <= score <= boundaries[0][1]:
        if score <= (boundaries[0][0] + boundaries[0][1])/2:
            return 0 if np.random.random() < 0.95 else 1  # 95% seguro
        else:
            return 1 if np.random.random() < 0.95 else 0
    elif boundaries[1][0] <= score <= boundaries[1][1]:
        if score <= (boundaries[1][0] + boundaries[1][1])/2:
            return 1 if np.random.random() < 0.95 else 2
        else:
            return 2 if np.random.random() < 0.95 else 1
    
    return 1  # Zona neutral clara

def calculate_customer_satisfaction_fuzzy(df):
    df = df.copy()
    
    # 1. Calcular métricas con mínimo ruido
    df['delivery_efficiency'] = df.apply(lambda row: 
        row['shipping_time_days'] / max(1, geodesic(
            region_coordinates[row['seller_region']], 
            region_coordinates[row['customer_region']]).kilometers / 100), 
        axis=1)
    df['delivery_efficiency'] = add_controlled_noise(df['delivery_efficiency'])
    
    df['price_ratio'] = add_controlled_noise(df['freight_value'] / df['order_price'])
    df['discount_ratio'] = add_controlled_noise(df['product_discount'] / df['order_price'])
    
    # 2. Calcular scores base
    satisfaction_scores = df.apply(lambda row: get_satisfaction_score(
        row, 
        df['delivery_efficiency'].iloc[row.name],
        df['price_ratio'].iloc[row.name],
        df['discount_ratio'].iloc[row.name]
    ), axis=1)
    
    # 3. Definir fronteras más claras
    boundaries = [
        (np.quantile(satisfaction_scores, 0.30), np.quantile(satisfaction_scores, 0.35)),  # Frontera 0-1
        (np.quantile(satisfaction_scores, 0.65), np.quantile(satisfaction_scores, 0.70))   # Frontera 1-2
    ]
    
    # 4. Aplicar clasificación
    satisfaction_labels = [
        get_fuzzy_satisfaction(score, boundaries)
        for score in satisfaction_scores
    ]
    
    # 5. Reglas deterministas para casos extremos
    final_labels = []
    for i, row in df.iterrows():
        label = satisfaction_labels[i]
        
        # Casos 100% claros
        if row['review_score'] >= 4.8 and row['customer_complaints'] == 0:
            label = 2  # Definitivamente satisfecho
        elif row['review_score'] <= 1.2 and row['customer_complaints'] >= 4:
            label = 0  # Definitivamente insatisfecho
        elif row['review_score'] >= 4.0 and row['customer_complaints'] <= 1:
            label = 2 if np.random.random() < 0.9 else 1  # Muy probable satisfecho
        elif row['review_score'] <= 2.0 and row['customer_complaints'] >= 3:
            label = 0 if np.random.random() < 0.9 else 1  # Muy probable insatisfecho
            
        final_labels.append(label)
    
    return pd.Series(final_labels)

def label_dataset_fuzzy(df):
    df = df.copy()
    df['customer_satisfaction'] = calculate_customer_satisfaction_fuzzy(df)
    
    # Validar y mostrar estadísticas
    print("\nEstadísticas por nivel de satisfacción:")
    for satisfaction in [0, 1, 2]:
        mask = df['customer_satisfaction'] == satisfaction
        print(f"\nNivel {satisfaction}:")
        print(f"Cantidad de casos: {mask.sum()}")
        print(f"Review score promedio: {df[mask]['review_score'].mean():.2f}")
        print(f"Quejas promedio: {df[mask]['customer_complaints'].mean():.2f}")
        print(f"Tiempo de envío promedio: {df[mask]['shipping_time_days'].mean():.2f}")
    
    return df

if __name__ == "__main__":
    input_file = "synthetic_dataset.csv"
    output_file = "labeled_dataset_fuzzy.csv"
    
    try:
        df = pd.read_csv(input_file, sep=';')
        df_labeled = label_dataset_fuzzy(df)
        
        print("\nDistribución de Satisfacción del Cliente:")
        print(df_labeled['customer_satisfaction'].value_counts().sort_index())
        print("\nPorcentajes:")
        print(df_labeled['customer_satisfaction'].value_counts(normalize=True).sort_index() * 100)
        
        df_labeled.to_csv(output_file, sep=';', index=False)
        print("\nProceso de etiquetado completado exitosamente.")
        
    except Exception as e:
        print(f"Error durante el proceso de etiquetado: {str(e)}")
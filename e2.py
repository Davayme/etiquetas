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

def add_controlled_noise(series, noise_scale=0.05):
    """Añade ruido gaussiano controlado a una serie."""
    noise = np.random.normal(0, noise_scale, size=len(series))
    return series + noise

def get_fuzzy_satisfaction(score, boundaries, noise_range=0.1):
    """
    Determina el nivel de satisfacción usando fronteras difusas.
    """
    # Añadir pequeño ruido aleatorio al score
    score = score + np.random.normal(0, noise_range)
    
    # Inicializar array de probabilidades con tres elementos (para niveles 0, 1, 2)
    probs = np.zeros(3)
    
    # Si el score es menor que la primera frontera
    if score < boundaries[0][0]:
        probs = np.array([0.8, 0.2, 0])
    # Si el score está entre la primera y segunda frontera
    elif boundaries[0][1] <= score <= boundaries[1][0]:
        probs = np.array([0.2, 0.6, 0.2])
    # Si el score está después de la segunda frontera
    elif score > boundaries[1][1]:
        probs = np.array([0, 0.2, 0.8])
    # Si está en zonas de transición
    else:
        if score <= boundaries[0][1]:  # Primera zona de transición
            dist_ratio = (score - boundaries[0][0]) / (boundaries[0][1] - boundaries[0][0])
            probs = np.array([1 - dist_ratio, dist_ratio, 0])
        else:  # Segunda zona de transición
            dist_ratio = (score - boundaries[1][0]) / (boundaries[1][1] - boundaries[1][0])
            probs = np.array([0, 1 - dist_ratio, dist_ratio])
    
    # Normalizar probabilidades
    probs = probs / probs.sum()
    
    return np.random.choice([0, 1, 2], p=probs)

def calculate_customer_satisfaction_fuzzy(df):
    df = df.copy()
    
    # 1. Crear características con ruido controlado
    df['delivery_efficiency'] = df.apply(lambda row: 
        row['shipping_time_days'] / max(1, geodesic(
            region_coordinates[row['seller_region']], 
            region_coordinates[row['customer_region']]).kilometers / 100), 
        axis=1)
    df['delivery_efficiency'] = add_controlled_noise(df['delivery_efficiency'])
    
    # Ratio precio-descuento con ruido
    df['price_discount_ratio'] = add_controlled_noise(
        df['product_discount'] / df['order_price'])
    
    # Ratio de costo de envío respecto al precio con ruido
    df['freight_price_ratio'] = add_controlled_noise(
        df['freight_value'] / df['order_price'])
    
    # 2. Crear un score compuesto de satisfacción con pesos variables
    def get_random_weights():
        """Genera pesos aleatorios que suman 1"""
        w = np.random.dirichlet([10, 4, 3, 3])  # Alpha mayor = menos variación
        return w

    satisfaction_scores = []
    for _ in range(len(df)):
        weights = get_random_weights()
        score = (
            df['review_score'].iloc[_] * weights[0] +
            (5 - df['customer_complaints'].iloc[_]) * weights[1] +
            (1 - df['delivery_efficiency'].iloc[_].clip(0, 1)) * weights[2] +
            (1 - df['freight_price_ratio'].iloc[_].clip(0, 1)) * weights[3]
        )
        satisfaction_scores.append(score)
    
    satisfaction_scores = np.array(satisfaction_scores)
    
    # 3. Definir fronteras difusas usando np.quantile en lugar de .quantile
    boundaries = [
        (np.quantile(satisfaction_scores, 0.3), np.quantile(satisfaction_scores, 0.4)),  # Frontera 0-1
        (np.quantile(satisfaction_scores, 0.6), np.quantile(satisfaction_scores, 0.7))   # Frontera 1-2
    ]
    
    # 4. Aplicar clasificación difusa
    satisfaction_labels = [
        get_fuzzy_satisfaction(score, boundaries)
        for score in satisfaction_scores
    ]
    
    # 5. Aplicar reglas de coherencia probabilísticas
    final_labels = []
    for i, row in df.iterrows():
        label = satisfaction_labels[i]
        
        # Reglas probabilísticas para casos extremos
        if row['review_score'] >= 4 and row['customer_complaints'] <= 1:
            if label == 0:  # Si está etiquetado como insatisfecho
                # 80% de probabilidad de cambiar a neutral
                label = np.random.choice([0, 1], p=[0.2, 0.8])
        elif row['review_score'] <= 2 and row['customer_complaints'] >= 2:
            if label == 2:  # Si está etiquetado como satisfecho
                # 80% de probabilidad de cambiar a neutral
                label = np.random.choice([1, 2], p=[0.8, 0.2])
                
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

# Uso del script
if __name__ == "__main__":
    input_file = "synthetic_dataset.csv"
    output_file = "labeled_dataset_fuzzy.csv"
    
    try:
        # Leer dataset
        df = pd.read_csv(input_file, sep=';')
        
        # Aplicar etiquetado difuso
        df_labeled = label_dataset_fuzzy(df)
        
        # Mostrar distribución de etiquetas
        print("\nDistribución de Satisfacción del Cliente:")
        print(df_labeled['customer_satisfaction'].value_counts().sort_index())
        print("\nPorcentajes:")
        print(df_labeled['customer_satisfaction'].value_counts(normalize=True).sort_index() * 100)
        
        # Guardar resultado
        df_labeled.to_csv(output_file, sep=';', index=False)
        print("\nProceso de etiquetado completado exitosamente.")
        
    except Exception as e:
        print(f"Error durante el proceso de etiquetado: {str(e)}")
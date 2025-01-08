import pandas as pd
import numpy as np
from scipy.stats import norm

def add_controlled_noise(series, noise_scale=0.05):
    """Añade ruido gaussiano controlado a una serie."""
    noise = np.random.normal(0, noise_scale * np.std(series), size=len(series))
    return series + noise

def calculate_shipping_labels(df):
    """
    Calcula etiquetas de envío usando un enfoque fuzzy menos determinista
    Versión optimizada con operaciones vectorizadas
    """
    df = df.copy()
    n_samples = len(df)
    
    # 1. Calcular ratios con operaciones vectorizadas
    price_distance_ratio = add_controlled_noise(df['product_price'] / (df['distance'] + 1))
    price_volume_ratio = add_controlled_noise(df['product_price'] / (df['product_volume'] + 1))
    price_freight_ratio = add_controlled_noise(df['product_price'] / (df['freight_value'] + 1))
    
    # 2. Calcular scores base con pesos variables
    weights = np.random.normal([0.4, 0.3, 0.3], 0.05)
    weights = weights / np.sum(weights)
    
    base_scores = (
        price_distance_ratio * weights[0] +
        price_volume_ratio * weights[1] +
        price_freight_ratio * weights[2]
    )
    
    # 3. Definir fronteras
    q30, q40 = np.percentile(base_scores, [30, 40])
    q60, q70 = np.percentile(base_scores, [60, 70])
    
    # 4. Crear máscara para cada zona
    low_mask = base_scores < q30
    high_mask = base_scores > q70
    low_trans_mask = (base_scores >= q30) & (base_scores < q40)
    high_trans_mask = (base_scores >= q60) & (base_scores < q70)
    mid_mask = (base_scores >= q40) & (base_scores < q60)
    
    # 5. Inicializar array de etiquetas
    labels = np.zeros(n_samples)
    
    # Asignar etiquetas directas
    labels[high_mask] = 2
    
    # Zonas de transición
    n_low_trans = np.sum(low_trans_mask)
    if n_low_trans > 0:
        trans_scores = (base_scores[low_trans_mask] - q30) / (q40 - q30)
        probs = norm.cdf(trans_scores, loc=0.5, scale=0.2)
        labels[low_trans_mask] = np.random.binomial(1, probs)
    
    n_high_trans = np.sum(high_trans_mask)
    if n_high_trans > 0:
        trans_scores = (base_scores[high_trans_mask] - q60) / (q70 - q60)
        probs = norm.cdf(trans_scores, loc=0.5, scale=0.2)
        labels[high_trans_mask] = 1 + np.random.binomial(1, probs)
    
    # Zona media - Corregido el broadcasting
    n_mid = np.sum(mid_mask)
    if n_mid > 0:
        mid_scores = base_scores[mid_mask]
        mid_mean = np.mean([q40, q60])
        # Asignar probabilidades según la posición en la zona media
        labels[mid_mask] = np.where(
            mid_scores < mid_mean,
            np.random.choice([0, 1, 2], size=n_mid, p=[0.15, 0.75, 0.1]),
            np.random.choice([0, 1, 2], size=n_mid, p=[0.1, 0.75, 0.15])
        )
    
    # 6. Ajustar casos extremos
    extreme_ratio = price_distance_ratio * price_volume_ratio * price_freight_ratio
    extreme_high = extreme_ratio > np.percentile(extreme_ratio, 95)
    extreme_low = extreme_ratio < np.percentile(extreme_ratio, 5)
    
    # Ajustar con probabilidad
    random_mask = np.random.random(n_samples) < 0.9
    labels[extreme_high & random_mask] = 2
    labels[extreme_high & ~random_mask] = 1
    labels[extreme_low & random_mask] = 0
    labels[extreme_low & ~random_mask] = 1
    
    return pd.Series(labels)

if __name__ == "__main__":
    # Leer el dataset
    df = pd.read_csv("envios.csv", sep=';')
    
    # Calcular nuevas etiquetas
    df['shipping_label'] = calculate_shipping_labels(df)
    
    # Mostrar distribución
    print("\nDistribución de etiquetas:")
    distribution = df['shipping_label'].value_counts().sort_index()
    for label, count in distribution.items():
        percentage = (count / len(df)) * 100
        print(f"Clase {int(label)}: {count} casos ({percentage:.2f}%)")
    
    # Guardar resultados
    df.to_csv("envios_etiquetado_fuzzy1.csv", sep=';', index=False)
    print("\nProceso de etiquetado completado exitosamente.")
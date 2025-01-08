import pandas as pd
import numpy as np
from scipy.stats import norm

def add_controlled_noise(series, noise_scale=0.03):
    """Añade ruido gaussiano controlado a una serie."""
    noise = np.random.normal(0, noise_scale * np.std(series), size=len(series))
    return series + noise

def calculate_composite_score(df):
    """
    Calcula un score compuesto más robusto
    """
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min())
    
    # 1. Eficiencia precio-distancia
    price_distance = normalize(df['product_price'] / (df['distance'] + 1))
    
    # 2. Eficiencia precio-volumen
    price_volume = normalize(df['product_price'] / (df['product_volume'] + 1))
    
    # 3. Eficiencia precio-flete
    price_freight = normalize(df['product_price'] / (df['freight_value'] + 1))
    
    # 4. Ratio de beneficio por volumen
    volume_profit = normalize(
        (df['product_price'] - df['freight_value']) / (df['product_volume'] + 1)
    )
    
    weights = [0.35, 0.25, 0.25, 0.15]
    
    return (
        price_distance * weights[0] +
        price_volume * weights[1] +
        price_freight * weights[2] +
        volume_profit * weights[3]
    )

def calculate_shipping_labels(df):
    """
    Calcula etiquetas de envío usando un enfoque fuzzy mejorado y balanceado
    """
    df = df.copy()
    n_samples = len(df)
    
    # 1. Calcular score base
    base_scores = calculate_composite_score(df)
    base_scores = add_controlled_noise(base_scores)
    
    # 2. Definir fronteras para tres clases
    q33 = np.percentile(base_scores, 33.33)  # Primer tercil
    q67 = np.percentile(base_scores, 66.67)  # Segundo tercil
    
    # 3. Definir zonas de transición
    transition_width = 0.1  # 10% de zona de transición
    range_width = q67 - q33
    
    low_high = q33 + transition_width * range_width
    mid_low = q33 - transition_width * range_width
    mid_high = q67 - transition_width * range_width
    high_low = q67 + transition_width * range_width
    
    # 4. Inicializar etiquetas
    labels = np.ones(n_samples)  # Inicializar todo como clase media
    
    # 5. Asignar etiquetas con zonas de transición
    # Zonas claras
    labels[base_scores < mid_low] = 0  # Claramente baja
    labels[base_scores > high_low] = 2  # Claramente alta
    
    # Zonas de transición baja-media
    trans_low_mask = (base_scores >= mid_low) & (base_scores <= low_high)
    n_trans_low = np.sum(trans_low_mask)
    if n_trans_low > 0:
        trans_scores = (base_scores[trans_low_mask] - mid_low) / (low_high - mid_low)
        probs = norm.cdf(trans_scores, loc=0.5, scale=0.2)
        labels[trans_low_mask] = np.random.binomial(1, probs)
    
    # Zonas de transición media-alta
    trans_high_mask = (base_scores >= mid_high) & (base_scores <= high_low)
    n_trans_high = np.sum(trans_high_mask)
    if n_trans_high > 0:
        trans_scores = (base_scores[trans_high_mask] - mid_high) / (high_low - mid_high)
        probs = norm.cdf(trans_scores, loc=0.5, scale=0.2)
        labels[trans_high_mask] = 1 + np.random.binomial(1, probs)
    
    # 6. Ajustar casos extremos
    extreme_score = (
        df['product_price'] / 
        (df['freight_value'] + df['distance'] + df['product_volume'])
    )
    
    extreme_high = extreme_score > np.percentile(extreme_score, 95)
    extreme_low = extreme_score < np.percentile(extreme_score, 5)
    
    # Aplicar ajustes con probabilidad
    random_high = np.random.random(n_samples) < 0.9
    random_low = np.random.random(n_samples) < 0.9
    
    labels[extreme_high & random_high] = 2
    labels[extreme_high & ~random_high] = 1
    labels[extreme_low & random_low] = 0
    labels[extreme_low & ~random_low] = 1
    
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
    df.to_csv("envios_etiquetado_fuzzy2.csv", sep=';', index=False)
    print("\nProceso de etiquetado completado exitosamente.")
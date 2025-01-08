import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def assign_region_value(region):
    """
    Asigna valor numérico a cada región basado en su potencial económico
    """
    region_values = {
        'Southeast': 5.0,  # São Paulo, Rio (mayor PIB)
        'South': 4.0,      # Segunda región más rica
        'Central-West': 3.0, # Tercera
        'Northeast': 2.0,   # Cuarta
        'North': 1.0       # Quinta
    }
    return region_values[region]

def calculate_shipping_label(df):
    """
    Calcula las etiquetas de rentabilidad (0: baja, 1: media, 2: alta)
    """
    df = df.copy()
    
    # 1. Convertir regiones a valores numéricos para el cálculo interno
    df['customer_region_value'] = df['customer_region'].apply(assign_region_value)
    df['seller_region_value'] = df['seller_region'].apply(assign_region_value)
    
    # 2. Normalizamos variables para el cálculo
    scaler = lambda x: (x - x.min()) / (x.max() - x.min())
    
    df['price_norm'] = scaler(df['product_price'])
    df['distance_norm'] = 1 - scaler(df['distance'])
    df['freight_norm'] = 1 - scaler(df['freight_value'])
    df['volume_norm'] = 1 - scaler(df['product_volume'])
    df['customer_region_norm'] = scaler(df['customer_region_value'])
    df['seller_region_norm'] = scaler(df['seller_region_value'])
    
    # 3. Calculamos score con pesos ajustados
    df['rentability_score'] = (
        df['price_norm'] * 0.3 +             # Precio
        df['distance_norm'] * 0.25 +         # Distancia
        df['freight_norm'] * 0.25 +          # Flete
        df['volume_norm'] * 0.1 +            # Volumen
        df['customer_region_norm'] * 0.05 +  # Región cliente
        df['seller_region_norm'] * 0.05      # Región vendedor
    )
    
    # 4. Asignamos etiquetas con distribución natural
    thresholds = [0, 0.37, 0.74, 1.0]  # Aproximadamente 37-37-26
    labels = pd.qcut(df['rentability_score'], q=thresholds, labels=[0, 1, 2])
    
    return labels

def print_metrics(df):
    """
    Imprime métricas relevantes del etiquetado solo con variables originales
    """
    # 1. Distribución de clases
    print("\nDistribución de clases:")
    class_counts = df['shipping_label'].value_counts().sort_index()
    total_samples = len(df)
    
    for label, count in class_counts.items():
        percentage = (count / total_samples) * 100
        if label == 0:
            print(f"Clase {label} (Baja rentabilidad): {count} muestras ({percentage:.2f}%)")
        elif label == 1:
            print(f"Clase {label} (Media rentabilidad): {count} muestras ({percentage:.2f}%)")
        else:
            print(f"Clase {label} (Alta rentabilidad): {count} muestras ({percentage:.2f}%)")
    
    # 2. Matriz de correlaciones solo con variables originales
    print("\nMatriz de correlaciones con shipping_label:")
    
    # Solo incluimos las variables originales
    numeric_cols = ['product_volume', 'product_price', 'distance', 'freight_value', 
                   'customer_region_value', 'seller_region_value', 'shipping_label']
    correlations = df[numeric_cols].corr()
    
    label_correlations = correlations['shipping_label'].sort_values(ascending=False)
    
    print("\nCorrelaciones ordenadas con shipping_label:")
    for var, corr in label_correlations.items():
        if var != 'shipping_label':
            strength = ""
            if abs(corr) >= 0.7:
                strength = "(Correlación fuerte)"
            elif abs(corr) >= 0.5:
                strength = "(Correlación moderada)"
            else:
                strength = "(Correlación débil)"
            print(f"{var}: {corr:.3f} {strength}")

def preprocess_dataframe(df):
    """
    Preprocesa el DataFrame convirtiendo columnas numéricas
    """
    df = df.copy()
    numeric_cols = ['product_price', 'freight_value', 'distance', 'product_volume']
    
    for col in numeric_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

if __name__ == "__main__":
    try:
        # Leer y preprocesar el dataset
        df = pd.read_csv("envios.csv", sep=';')
        df = preprocess_dataframe(df)
        
        # Calcular regiones numéricas (necesario para correlaciones)
        df['customer_region_value'] = df['customer_region'].apply(assign_region_value)
        df['seller_region_value'] = df['seller_region'].apply(assign_region_value)
        
        # Calcular etiquetas
        df['shipping_label'] = calculate_shipping_label(df)
        
        # Mostrar métricas
        print_metrics(df)
        
        # Eliminar columnas auxiliares antes de guardar
        df = df.drop(['customer_region_value', 'seller_region_value'], axis=1)
        
        # Guardar dataset
        df.to_csv("etiquetado_envios.csv", sep=';', index=False)
        print("\nProceso de etiquetado completado exitosamente.")
        
    except Exception as e:
        print(f"Error durante el proceso de etiquetado: {str(e)}")
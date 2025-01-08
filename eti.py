import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_shipping_label(df):
    """
    Calcula las etiquetas de rentabilidad (0: baja, 1: media, 2: alta)
    usando una lógica que maximice las correlaciones
    """
    # 1. Calculamos el ratio precio/flete (esto tendrá correlación fuerte con rentabilidad)
    df['price_freight_ratio'] = df['product_price'] / (df['freight_value'] + 1)  # +1 para evitar división por cero
    
    # 2. Calculamos ratio margen/distancia
    df['margin_per_distance'] = (df['product_price'] - df['freight_value']) / (df['distance'] + 1)
    
    # 3. Calculamos el ratio precio/volumen para la eficiencia del envío
    df['price_volume_ratio'] = df['product_price'] / (df['product_volume'] + 1)
    
    # 4. Combinamos los ratios para crear un score
    df['rentability_score'] = (
        df['price_freight_ratio'] * 0.4 +
        df['margin_per_distance'] * 0.4 +
        df['price_volume_ratio'] * 0.2
    )
    
    # 5. Normalizamos el score usando percentiles
    df['rentability_normalized'] = df['rentability_score'].rank(pct=True)
    
    # 6. Asignamos etiquetas usando una distribución 40-35-25
    labels = pd.qcut(df['rentability_normalized'], 
                    q=[0, 0.40, 0.75, 1],
                    labels=[0, 1, 2])
    
    # 7. Limpiamos las columnas temporales
    df = df.drop(['price_freight_ratio', 'margin_per_distance', 
                  'price_volume_ratio', 'rentability_score', 
                  'rentability_normalized'], axis=1)
    
    return labels

def print_metrics(df):
    """
    Imprime métricas relevantes del etiquetado
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
    
    # 2. Matriz de correlaciones
    print("\nMatriz de correlaciones con shipping_label:")
    numeric_cols = ['product_volume', 'product_price', 'distance', 'freight_value', 'shipping_label']
    correlations = df[numeric_cols].corr()
    
    # Ordenamos las correlaciones con shipping_label de mayor a menor
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
            
    # Crear y guardar el heatmap de correlaciones
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlaciones')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

def preprocess_numeric_columns(df):
    """Convierte columnas numéricas de string a float"""
    numeric_cols = ['product_price', 'freight_value', 'distance', 'product_volume']
    df_copy = df.copy()
    
    for col in numeric_cols:
        df_copy[col] = pd.to_numeric(df_copy[col].str.replace(',', '.'), errors='coerce')
    
    return df_copy


if __name__ == "__main__":
    try:
        # Leer el dataset
        df = pd.read_csv("envios.csv", sep=';', decimal=',')
        df = preprocess_numeric_columns(df)
        # Calcular etiquetas
        df['shipping_label'] = calculate_shipping_label(df)
        
        # Mostrar métricas
        print_metrics(df)
        
        # Guardar dataset etiquetado
        df.to_csv("etiquetado_envios.csv", sep=';', decimal=',', index=False)
        print("\nProceso de etiquetado completado exitosamente.")
        
    except Exception as e:
        print(f"Error durante el proceso de etiquetado: {str(e)}")
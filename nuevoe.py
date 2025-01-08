import pandas as pd
import numpy as np

def calculate_profitability_score(row):
    """
    Calcula el score de rentabilidad con mayor énfasis en las variables clave
    """
    # 1. Ratio precio/flete (más alto = mejor rentabilidad)
    price_freight_ratio = row['product_price'] / max(1, row['freight_value'])
    price_freight_score = np.clip(price_freight_ratio / 100, 0, 1)
    
    # 2. Eficiencia volumétrica (más bajo = mejor rentabilidad)
    volume_efficiency = row['product_volume'] / (row['product_price'] + 1)
    volume_score = 1 - np.clip(volume_efficiency, 0, 1)
    
    # 3. Eficiencia de distancia (más bajo = mejor rentabilidad)
    distance_efficiency = row['distance'] / 5000  # normalizado a 5000km
    distance_score = 1 - np.clip(distance_efficiency, 0, 1)
    
    # 4. Score de categoría con mayores márgenes
    category_margins = {
        "Electronics": 0.9,
        "Beauty": 0.85,
        "Clothing": 0.8,
        "Books": 0.75,
        "Sports": 0.7,
        "Toys": 0.65,
        "Food": 0.6,
        "Furniture": 0.5
    }
    category_score = category_margins[row['product_category']]
    
    # Combinar scores con nuevos pesos para aumentar correlación
    final_score = (
        price_freight_score * 0.45 +      # Aumentamos peso del precio
        volume_score * 0.25 +             # Mantenemos importancia del volumen
        distance_score * 0.20 +           # Reducimos levemente peso de distancia
        category_score * 0.10             # Categoría como factor menor
    )
    
    # Aplicar transformación no lineal para acentuar diferencias
    return np.power(final_score, 1.5)  # Exponente > 1 acentúa las diferencias

def label_dataset(df):
    """
    Etiqueta el dataset con la rentabilidad del envío
    """
    df = df.copy()
    
    # Calcular score de rentabilidad
    df['profitability_score'] = df.apply(calculate_profitability_score, axis=1)
    
    # Usar qcut con más énfasis en la separación
    df['shipping_profitability'] = pd.qcut(
        df['profitability_score'], 
        q=3, 
        labels=['Baja', 'Media', 'Alta']
    )
    
    return df

if __name__ == "__main__":
    try:
        # Leer el dataset
        df = pd.read_csv("llenar.csv")
        
        # Aplicar etiquetado
        df_labeled = label_dataset(df)
        
        # Convertir etiquetas a numérico para análisis de correlación
        df_labeled['shipping_profitability_num'] = df_labeled['shipping_profitability'].map({
            'Baja': 0, 'Media': 1, 'Alta': 2
        })
        
        # Analizar correlaciones
        numeric_cols = [
            'product_volume', 'product_price', 'distance', 
            'freight_value', 'shipping_profitability_num'
        ]
        correlations = df_labeled[numeric_cols].corr()
        
        print("\nNuevas correlaciones con shipping_profitability_num:")
        for col in numeric_cols[:-1]:
            corr = correlations.loc[col, 'shipping_profitability_num']
            print(f"{col}: {corr:.3f}")
        
        # Guardar dataset etiquetado
        df_labeled.to_csv("etiquetado_envios.csv", index=False)
        print("\nProceso de etiquetado completado exitosamente.")
        
    except Exception as e:
        print(f"Error durante el proceso de etiquetado: {str(e)}")
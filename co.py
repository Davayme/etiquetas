import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def analyze_correlations(csv_path):
    # Leer el CSV
    df = pd.read_csv(csv_path, sep=';')
    
    # Crear copia para no modificar el original
    df_encoded = df.copy()
    
    # Convertir categóricas a numéricas usando LabelEncoder
    le = LabelEncoder()
    categorical_columns = ['order_day_of_week', 'customer_region', 'seller_region', 
                         'product_category', 'customer_gender']
    
    for col in categorical_columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    # Calcular matriz de correlación
    correlation_matrix = df_encoded.corr()
    
    # Mostrar todas las correlaciones ordenadas con customer_satisfaction
    satisfaction_corr = correlation_matrix['customer_satisfaction'].sort_values(ascending=False)
    print("\nCorrelaciones con satisfacción del cliente (ordenadas):")
    print(satisfaction_corr)
    
    # Crear el heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    
    # Añadir valores numéricos
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            text = plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black')
    
    # Configurar ejes
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    
    # Añadir barra de color y título
    plt.colorbar(im)
    plt.title('Matriz de Correlación')
    plt.tight_layout()
    
    # Obtener top 7 correlaciones absolutas
    correlations = correlation_matrix['customer_satisfaction']
    abs_correlations = correlations.abs().sort_values(ascending=False)
    top_correlations = abs_correlations[1:8]  # Excluimos la primera que es la correlación consigo misma
    
    # Añadir prints de depuración
    print("\n" + "="*60)
    print("TOP 7 VARIABLES CON MAYOR CORRELACIÓN ABSOLUTA:")
    print("="*60)
    print("\nVariable                  Correlación    Correlación Absoluta")
    print("-" * 60)
    for var in top_correlations.index:
        print(f"{var:<25} {correlations[var]:>10.3f}    {abs_correlations[var]:>10.3f}")
    
    return correlation_matrix, top_correlations

# Ejecutar el análisis
correlation_matrix, top_correlations = analyze_correlations('labeled_dataset.csv')
plt.show()
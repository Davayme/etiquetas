import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def encode_categorical_variables(df):
    """
    Convierte variables categóricas a numéricas
    """
    df = df.copy()
    
    # Codificar regiones
    region_values = {
        'Southeast': 5,
        'South': 4,
        'Central-West': 3,
        'Northeast': 2,
        'North': 1
    }
    df['customer_region'] = df['customer_region'].map(region_values)
    df['seller_region'] = df['seller_region'].map(region_values)
    
    # Codificar género
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    # Codificar categorías de productos
    product_categories = df['product_category'].unique()
    category_values = {cat: idx for idx, cat in enumerate(sorted(product_categories))}
    df['product_category'] = df['product_category'].map(category_values)
    
    return df

def plot_correlation_matrix(df):
    """
    Genera y guarda un plot de la matriz de correlación entre todas las variables
    """
    # Obtener matriz de correlación
    correlation_matrix = df.corr()
    
    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Crear el heatmap
    im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Configurar ticks
    variables = correlation_matrix.columns
    ax.set_xticks(np.arange(len(variables)))
    ax.set_yticks(np.arange(len(variables)))
    
    # Configurar labels
    ax.set_xticklabels(variables, rotation=45, ha='right')
    ax.set_yticklabels(variables)
    
    # Añadir valores numéricos en cada celda
    for i in range(len(variables)):
        for j in range(len(variables)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                         ha='center', va='center', fontsize=8,
                         color='black' if abs(correlation_matrix.iloc[i, j]) < 0.5 else 'white')
    
    # Añadir barra de color
    plt.colorbar(im)
    
    # Título
    plt.title('Matriz de Correlación entre Variables', pad=20)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig('correlation_matrix_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Imprimir correlaciones significativas
    print("\nCorrelaciones significativas (|corr| > 0.5):")
    for i in range(len(variables)):
        for j in range(i+1, len(variables)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.5:
                print(f"{variables[i]} vs {variables[j]}: {corr:.3f}")

def preprocess_dataframe(df):
    """
    Preprocesa el DataFrame convirtiendo columnas numéricas
    """
    df = df.copy()
    
    # Convertir columnas numéricas
    numeric_cols = ['delivery_year', 'delivery_month', 'delivery_day',
                   'product_volume', 'product_price', 'distance', 'freight_value']
    
    for col in numeric_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

if __name__ == "__main__":
    try:
        # Leer el dataset
        df = pd.read_csv("envios.csv", sep=';')
        
        # Preprocesar datos numéricos
        df = preprocess_dataframe(df)
        
        # Codificar variables categóricas
        df = encode_categorical_variables(df)
        
        # Generar matriz de correlación
        plot_correlation_matrix(df)
        print("\nMatriz de correlación generada exitosamente.")
        
        # Guardar el mapping de categorías para referencia
        print("\nCódigos asignados a variables categóricas:")
        print("\nRegiones:")
        print("Southeast: 5")
        print("South: 4")
        print("Central-West: 3")
        print("Northeast: 2")
        print("North: 1")
        
        print("\nGénero:")
        print("Male: 1")
        print("Female: 0")
        
        print("\nCategorías de productos:")
        product_categories = sorted(pd.read_csv("envios.csv", sep=';')['product_category'].unique())
        for idx, cat in enumerate(product_categories):
            print(f"{cat}: {idx}")
        
    except Exception as e:
        print(f"Error al generar la matriz de correlación: {str(e)}")
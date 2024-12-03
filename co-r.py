import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

def calculate_cramers_v(var1, var2):
    """Calcula V de Cramer entre dos variables categóricas"""
    contingency = pd.crosstab(var1, var2)
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

def create_correlation_matrix(df):
    """
    Crea una matriz de correlación considerando tanto variables numéricas como categóricas
    """
    # Identificar columnas numéricas y categóricas
    numeric_cols = ['review_score', 'freight_value', 'order_price', 'product_price', 
                   'product_volume', 'return_numeric', 'satisfaction_numeric']
    
    categorical_cols = ['product_category', 'customer_city_name', 'seller_city_name',
                       'order_month', 'order_trimester', 'order_semester']
    
    # Crear DataFrame para la matriz de correlación
    all_cols = numeric_cols + categorical_cols
    corr_matrix = pd.DataFrame(index=all_cols, columns=all_cols, dtype=float)
    
    # Calcular correlaciones
    for i, col1 in enumerate(all_cols):
        for j, col2 in enumerate(all_cols):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
            else:
                # Si ambas son numéricas
                if col1 in numeric_cols and col2 in numeric_cols:
                    corr_matrix.iloc[i, j] = df[col1].corr(df[col2])
                # Si al menos una es categórica
                else:
                    try:
                        corr_matrix.iloc[i, j] = calculate_cramers_v(df[col1], df[col2])
                    except:
                        corr_matrix.iloc[i, j] = np.nan
    
    return corr_matrix

def plot_correlations(df):
    """
    Crea y guarda visualizaciones de las correlaciones
    """
    # Obtener matriz de correlación
    corr_matrix = create_correlation_matrix(df)
    
    # Configurar el estilo de las gráficas
    plt.style.use('ggplot')
    
    # 1. Mapa de calor general
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f', 
                square=True, linewidths=.5)
    plt.title('Matriz de Correlaciones Completa', pad=20, size=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('correlation_matrix_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlaciones con las variables objetivo
    target_vars = ['return_numeric', 'satisfaction_numeric']
    correlations_with_target = corr_matrix[target_vars].drop(target_vars)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlations_with_target, annot=True, cmap='RdYlBu_r', center=0, 
                fmt='.2f', linewidths=.5)
    plt.title('Correlaciones con Variables Objetivo', pad=20, size=16)
    plt.tight_layout()
    plt.savefig('correlation_with_targets.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Correlaciones entre variables numéricas
    numeric_cols = ['review_score', 'freight_value', 'order_price', 'product_price', 
                   'product_volume', 'return_numeric', 'satisfaction_numeric']
    numeric_corr = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_corr, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f',
                square=True, linewidths=.5)
    plt.title('Correlaciones entre Variables Numéricas', pad=20, size=16)
    plt.tight_layout()
    plt.savefig('numeric_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Imprimir las correlaciones más importantes con las variables objetivo
    print("\nCorrelaciones más importantes con Probabilidad de Retorno:")
    return_corr = correlations_with_target['return_numeric'].sort_values(ascending=False)
    print(return_corr.head(10))
    
    print("\nCorrelaciones más importantes con Satisfacción:")
    satisfaction_corr = correlations_with_target['satisfaction_numeric'].sort_values(ascending=False)
    print(satisfaction_corr.head(10))
    
    return corr_matrix

# Cargar datos
df = pd.read_csv('reviews_binomial.csv', sep=';')

# Generar y guardar visualizaciones
correlation_matrix = plot_correlations(df)

# Sugerir variables para el perceptrón
def suggest_variables(correlation_matrix, threshold=0.1):
    """
    Sugiere variables para el perceptrón basado en correlaciones
    """
    print("\n=== SUGERENCIAS PARA VARIABLES DEL PERCEPTRÓN ===")
    
    # Variables para predicción de retorno
    return_correlations = correlation_matrix['return_numeric'].abs()
    significant_return = return_correlations[return_correlations > threshold]
    significant_return = significant_return.sort_values(ascending=False)
    
    print("\nVariables sugeridas para predicción de Retorno:")
    for var, corr in significant_return.items():
        if var != 'return_numeric':
            print(f"- {var}: correlación = {corr:.3f}")
    
    # Variables para predicción de satisfacción
    satisfaction_correlations = correlation_matrix['satisfaction_numeric'].abs()
    significant_satisfaction = satisfaction_correlations[satisfaction_correlations > threshold]
    significant_satisfaction = significant_satisfaction.sort_values(ascending=False)
    
    print("\nVariables sugeridas para predicción de Satisfacción:")
    for var, corr in significant_satisfaction.items():
        if var != 'satisfaction_numeric':
            print(f"- {var}: correlación = {corr:.3f}")

# Sugerir variables
suggest_variables(correlation_matrix)
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency

def calculate_city_risk_score(df):
    """
    Calcula un score de riesgo para cada ciudad basado en el histórico de reviews
    """
    city_scores = {}
    
    # Calcular score promedio de reviews por ciudad
    city_reviews = df.groupby('customer_city_name')['review_score'].agg(['mean', 'count']).reset_index()
    
    # Normalizar scores considerando también la cantidad de reviews
    for _, row in city_reviews.iterrows():
        weight = min(row['count'] / df['customer_city_name'].value_counts().mean(), 1)
        city_scores[row['customer_city_name']] = (5 - row['mean']) * weight
    
    return city_scores

def calculate_month_risk_score():
    """
    Asigna scores de riesgo a meses basado en patrones típicos de comercio
    """
    # Meses de alto riesgo (post-temporada alta, más devoluciones)
    high_risk_months = ['enero', 'julio']
    # Meses de riesgo medio (cambios de temporada)
    medium_risk_months = ['abril', 'octubre']
    
    month_scores = {}
    for month in ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 
                  'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']:
        if month in high_risk_months:
            month_scores[month] = 1.0
        elif month in medium_risk_months:
            month_scores[month] = 0.6
        else:
            month_scores[month] = 0.3
    return month_scores

def calculate_return_class(row, city_scores, month_scores):
    """
    Calcula la probabilidad de devolución considerando todas las variables
    """
    score = 0
    
    # Variables numéricas (50% del peso total)
    review_weight = 0.2
    score += review_weight * (6 - row['review_score']) / 5
    
    price_weight = 0.1
    price_norm = min(row['product_price'] / 150, 1)
    score += price_weight * price_norm
    
    freight_weight = 0.1
    freight_ratio = row['freight_value'] / (row['product_price'] + 1)
    score += freight_weight * min(freight_ratio * 2, 1)
    
    volume_weight = 0.1
    volume_norm = min(row['product_volume'] / 10000, 1)
    score += volume_weight * volume_norm
    
    # Variables categóricas (30% del peso total)
    city_weight = 0.15
    score += city_weight * (city_scores.get(row['customer_city_name'], 0.5) / 5)
    
    category_weight = 0.15
    high_risk_categories = ['fashion_womens_clothing', 'electronics', 'watches_gifts']
    if row['product_category'] in high_risk_categories:
        score += category_weight
    
    # Variables temporales (20% del peso total)
    month_weight = 0.1
    score += month_weight * month_scores.get(row['order_month'], 0.5)
    
    trimester_weight = 0.1
    high_risk_trimesters = ['4to Trimestre', '1er Trimestre']
    if row['order_trimester'] in high_risk_trimesters:
        score += trimester_weight
    
    return "Alta Probabilidad" if score > 0.5 else "Baja Probabilidad"

def calculate_satisfaction_class(row, city_scores, month_scores):
    """
    Calcula la satisfacción considerando todas las variables
    """
    score = 0
    
    # Variables numéricas (60% del peso total)
    score += 0.3 * (row['review_score'] / 5)
    score += 0.2 * (1 - min(row['freight_value'] / (row['product_price'] + 1), 1))
    score += 0.1 * (1 - min(row['product_price'] / 200, 1))
    
    # Variables categóricas (25% del peso total)
    city_satisfaction = 1 - (city_scores.get(row['customer_city_name'], 0.5) / 5)
    score += 0.15 * city_satisfaction
    
    same_city = row['customer_city_name'] == row['seller_city_name']
    score += 0.1 * (1 if same_city else 0)
    
    # Variables temporales (15% del peso total)
    score += 0.15 * (1 - month_scores.get(row['order_month'], 0.5))
    
    return "Satisfecho" if score > 0.5 else "No Satisfecho"

def analyze_correlations(df):
    """
    Analiza correlaciones para variables numéricas y categóricas
    """
    # Convertir clases a numéricas
    df['return_numeric'] = (df['product_return_class'] == "Alta Probabilidad").astype(int)
    df['satisfaction_numeric'] = (df['satisfaction_class_binomial'] == "Satisfecho").astype(int)
    
    # Correlaciones numéricas
    numerical_cols = ['review_score', 'freight_value', 'product_price', 'product_volume']
    
    print("\nCorrelaciones numéricas con product_return_class:")
    for col in numerical_cols:
        corr, _ = pearsonr(df[col], df['return_numeric'])
        print(f"{col}: {corr:.3f}")
    
    # Correlaciones categóricas (Chi-cuadrado)
    categorical_cols = ['product_category', 'customer_city_name', 'order_month', 'order_trimester']
    
    print("\nAsociaciones categóricas (V de Cramer):")
    for col in categorical_cols:
        contingency = pd.crosstab(df[col], df['return_numeric'])
        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim))
        print(f"{col}: {cramer_v:.3f}")
# Añadir análisis de balance de clases
def analyze_class_balance(df):
    """
    Analiza y muestra el balance de clases para ambas variables objetivo
    """
    print("\n=== ANÁLISIS DE BALANCE DE CLASES ===")
    
    # Análisis para product_return_class
    print("\nBalance de Clases - Probabilidad de Devolución:")
    return_counts = df['product_return_class'].value_counts()
    return_percentages = df['product_return_class'].value_counts(normalize=True) * 100
    
    print("\nConteo absoluto:")
    for clase, count in return_counts.items():
        print(f"{clase}: {count:,} registros")
    
    print("\nPorcentajes:")
    for clase, percentage in return_percentages.items():
        print(f"{clase}: {percentage:.2f}%")
    
    # Calcular ratio de desbalanceo para devoluciones
    return_ratio = return_counts.max() / return_counts.min()
    print(f"\nRatio de desbalanceo: {return_ratio:.2f} : 1")
    
    # Análisis para satisfaction_class_binomial
    print("\n" + "="*50)
    print("\nBalance de Clases - Satisfacción del Cliente:")
    satisfaction_counts = df['satisfaction_class_binomial'].value_counts()
    satisfaction_percentages = df['satisfaction_class_binomial'].value_counts(normalize=True) * 100
    
    print("\nConteo absoluto:")
    for clase, count in satisfaction_counts.items():
        print(f"{clase}: {count:,} registros")
    
    print("\nPorcentajes:")
    for clase, percentage in satisfaction_percentages.items():
        print(f"{clase}: {percentage:.2f}%")
    
    # Calcular ratio de desbalanceo para satisfacción
    satisfaction_ratio = satisfaction_counts.max() / satisfaction_counts.min()
    print(f"\nRatio de desbalanceo: {satisfaction_ratio:.2f} : 1")
    
    # Recomendaciones basadas en el desbalanceo
    print("\n=== RECOMENDACIONES PARA EL BALANCEO ===")
    
    for nombre, ratio in [("Devolución", return_ratio), ("Satisfacción", satisfaction_ratio)]:
        print(f"\nPara la clase de {nombre}:")
        if ratio > 4:
            print("- Altamente desbalanceado. Considerar:")
            print("  * SMOTE o ADASYN para sobremuestreo")
            print("  * Submuestreo de la clase mayoritaria")
            print("  * Combinación de sobre y submuestreo")
        elif ratio > 2:
            print("- Moderadamente desbalanceado. Considerar:")
            print("  * Random oversampling")
            print("  * Class weights en el modelo")
        else:
            print("- Relativamente balanceado. Podría:")
            print("  * Usar los datos tal como están")
            print("  * Aplicar class weights si es necesario")

# Cargar y procesar dataset
df = pd.read_csv("reviews_final.csv", sep=";")

# Limpiar datos
numeric_cols = ['review_score', 'freight_value', 'product_price', 'product_volume']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=numeric_cols)

# Calcular scores de ciudades y meses
city_scores = calculate_city_risk_score(df)
month_scores = calculate_month_risk_score()

# Aplicar etiquetado
df['product_return_class'] = df.apply(lambda row: calculate_return_class(row, city_scores, month_scores), axis=1)
df['satisfaction_class_binomial'] = df.apply(lambda row: calculate_satisfaction_class(row, city_scores, month_scores), axis=1)

# Analizar correlaciones
analyze_correlations(df)

# Analizar balance de clases
analyze_class_balance(df)

# Guardar resultados
df.to_csv("reviews_binomial.csv", sep=";", index=False)
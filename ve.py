import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Leer el dataset
df = pd.read_csv("synthetic_dataset.csv", sep=";")

# Crear grupos de calificaciones
def categorize_review(score):
    if score <= 2:
        return "1-2 (Malo)"
    elif score == 3:
        return "3 (Neutral)"
    else:
        return "4-5 (Bueno)"

# Aplicar categorización
df['review_category'] = df['review_score'].apply(categorize_review)

# Calcular distribución
review_dist = df['review_category'].value_counts()
review_percent = df['review_category'].value_counts(normalize=True) * 100

# Mostrar resultados
print("\nDistribución de Reviews:")
print(review_dist)
print("\nPorcentajes:")
print(review_percent)

# Visualizar
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='review_category', order=['1-2 (Malo)', '3 (Neutral)', '4-5 (Bueno)'])
plt.title('Distribución de Reviews')
plt.xlabel('Categoría de Review')
plt.ylabel('Cantidad')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
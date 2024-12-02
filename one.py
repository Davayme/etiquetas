import pandas as pd

# Cargar el dataset
df = pd.read_csv("reviews_binomial.csv", sep=";")

# Selecciona las columnas categóricas para One-Hot Encoding
categorical_columns = ['product_category', 'product_brand']

# Aplicar One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Convertir solo las columnas de tipo booleano (True/False) a 1/0
df_encoded = df_encoded.applymap(lambda x: 1 if x is True else (0 if x is False else x))

# Guardar el dataframe resultante en un archivo CSV parseado
df_encoded.to_csv("reviews_binomial_encoded.csv", index=False, sep=";")

# Verifica el dataframe resultante
print(df_encoded.head())

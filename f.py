import pandas as pd
import chardet

# Detectar la codificación del archivo
with open('reviews_binomial.csv', 'rb') as f:
    result = chardet.detect(f.read())

print(f"La codificación detectada es: {result['encoding']}")

# Intentar cargar el archivo con la codificación detectada
try:
    df = pd.read_csv('reviews_binomial.csv', sep=';', encoding=result['encoding'])
    print("Archivo cargado correctamente.")

    # Verificar las primeras filas para asegurarse de que se cargó correctamente
    print("\nPrimeras filas del DataFrame:")
    print(df.head())

except Exception as e:
    print(f"No se pudo cargar el archivo. Error: {e}")

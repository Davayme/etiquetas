import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Supongamos que tienes un DataFrame con una columna 'distance'
df = pd.DataFrame({'distance': [10, 20, 30, 40, 50, 100, 150, 500, 1000, 1500]})

# Graficar el histograma
plt.figure(figsize=(8, 5))
sns.histplot(df['distance'], kde=True, bins=10, color="blue")  # kde=True agrega la curva de densidad
plt.title("Distribución de la variable 'distance'")
plt.xlabel("Distancia")
plt.ylabel("Frecuencia")
plt.show()

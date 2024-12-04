import numpy as np
import matplotlib.pyplot as plt

# ---- PASO 1: Crear valores para trabajar ----
# Generamos valores de entrada 'z' (de -10 a 10) para la función sigmoide
z = np.linspace(-10, 10, 100)

# Generamos valores aleatorios entre 0 y 1 (representan probabilidades)
probabilidades = np.random.rand(100)

# ---- PASO 2: Definir las fórmulas matemáticas ----
# Función sigmoide: transforma 'z' en probabilidades
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cálculo del logaritmo natural (ln(p))
log_prob = np.log(probabilidades)

# Cálculo del logaritmo de las odds (ln(p / (1 - p)))
log_odds = np.log(probabilidades / (1 - probabilidades))

# ---- PASO 3: Calcular valores ----
# Calcular los valores de la función sigmoide
sig = sigmoid(z)

# ---- PASO 4: Crear los gráficos ----
plt.figure(figsize=(14, 7))  # Tamaño de la figura

# --- Gráfica 1: Función Sigmoide ---
plt.subplot(1, 2, 1)  # Primera gráfica en un lienzo de 1 fila, 2 columnas
plt.plot(z, sig, label="Función Sigmoide", color="blue", linewidth=2)
plt.axhline(0.5, color='red', linestyle='--', label="Umbral (p=0.5)")
plt.title("Función Sigmoide", fontsize=14)
plt.xlabel("Entrada (z)", fontsize=12)
plt.ylabel("Probabilidad (p)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# --- Gráfica 2: Funciones Logarítmicas ---
plt.subplot(1, 2, 2)  # Segunda gráfica
plt.scatter(probabilidades, log_prob, label="ln(p) (logaritmo natural)", color="orange", s=50)
plt.scatter(probabilidades, log_odds, label="ln(p / (1 - p)) (log-odds)", color="green", s=50)
plt.title("Funciones Logarítmicas", fontsize=14)
plt.xlabel("Probabilidades (p)", fontsize=12)
plt.ylabel("Valores Logarítmicos", fontsize=12)
plt.axvline(0.5, color='red', linestyle='--', label="Probabilidad p=0.5")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# Ajustar diseño
plt.tight_layout()
plt.show()

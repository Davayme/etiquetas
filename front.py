import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification

# Generar un conjunto de datos 2D ficticio para clasificación (2 características)
X, y = make_classification(n_samples=100, n_features=2, n_classes=2,
                            n_informative=2, n_redundant=0, n_repeated=0,
                            random_state=42)

# Entrenar un modelo de perceptrón
clf = Perceptron()
clf.fit(X, y)

# Crear una malla de puntos para visualizar la frontera de decisión
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predecir en cada punto de la malla
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar la frontera de decisión
plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.Paired)

# Graficar los puntos de datos reales
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100, linewidth=1, cmap=plt.cm.Paired)

# Configurar el título y etiquetas de los ejes
plt.title('Frontera de Decisión del Perceptrón')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')

# Mostrar la gráfica
plt.show()

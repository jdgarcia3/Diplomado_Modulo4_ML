import numpy as np                   # Importa la librería NumPy, usada para manejar arreglos numéricos y operaciones matemáticas eficientes.
import matplotlib.pyplot as plt      # Importa Matplotlib, una librería para crear gráficos en 2D.
from sklearn.linear_model import Perceptron  # Importa el modelo Perceptrón de scikit-learn, usado para clasificación binaria.

# ---------------------------------------------------------------
# 1. Datos del OR lógico
# ---------------------------------------------------------------
X = np.array([                      # Define las posibles combinaciones de entradas (A, B)
    [0, 0],                         # A=0, B=0
    [0, 1],                         # A=0, B=1
    [1, 0],                         # A=1, B=0
    [1, 1]                          # A=1, B=1
])

y = np.array([0, 1, 1, 1])          # Define las salidas del OR lógico:
# El OR devuelve 1 si al menos una entrada es 1.
# Tabla de verdad:
# 0 OR 0 → 0
# 0 OR 1 → 1
# 1 OR 0 → 1
# 1 OR 1 → 1

# ---------------------------------------------------------------
# 2. Crear y entrenar el perceptrón
# ---------------------------------------------------------------
modelo = Perceptron(
    max_iter=1000,     # Número máximo de iteraciones (épocas) para ajustar los pesos del modelo.
    eta0=0.1,          # Tasa de aprendizaje: controla el tamaño de los pasos en la actualización de pesos.
    random_state=42    # Semilla aleatoria para garantizar resultados reproducibles.
)

modelo.fit(X, y)        # Entrena el perceptrón con los datos (X = entradas, y = salidas esperadas).

# ---------------------------------------------------------------
# 3. Crear malla para graficar frontera de decisión
# ---------------------------------------------------------------
x_min, x_max = -0.5, 1.5     # Establece los límites del eje X del gráfico.
y_min, y_max = -0.5, 1.5     # Establece los límites del eje Y del gráfico.

# np.meshgrid crea una cuadrícula de puntos (xx, yy) en el plano.
# np.linspace genera 200 valores equidistantes entre los límites definidos.
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

# np.c_ combina las coordenadas xx y yy en pares (x, y) para predecir sobre todos los puntos.
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])  
# modelo.predict() devuelve la clase (0 o 1) que el perceptrón predice para cada punto.
Z = Z.reshape(xx.shape)  
# Se reorganiza Z para que tenga la misma forma que la cuadrícula (xx, yy),
# lo cual es necesario para graficar correctamente.

# ---------------------------------------------------------------
# 4. Dibujar la frontera de decisión
# ---------------------------------------------------------------
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
# contourf() dibuja regiones coloreadas según la clase predicha.
# alpha=0.3 → transparencia de las regiones (para ver los puntos debajo).
# cmap=plt.cm.Paired → paleta de colores predefinida de Matplotlib.

# ---------------------------------------------------------------
# 5. Dibujar los puntos de entrada
# ---------------------------------------------------------------
for i in range(len(X)):  # Recorre todos los puntos de entrada definidos en X.
    plt.scatter(
        X[i][0], X[i][1],                       # Coordenadas (x, y) del punto.
        c='red' if y[i] == 1 else 'blue',       # Color: rojo si la salida esperada es 1, azul si es 0.
        edgecolors='k',                         # Borde negro alrededor del punto (mejora la visibilidad).
        s=100                                   # Tamaño del punto.
    )
    plt.text(X[i][0] + 0.05, X[i][1], f"{y[i]}", fontsize=12)
    # Agrega el valor de salida (0 o 1) junto al punto, con una ligera separación (0.05).

# ---------------------------------------------------------------
# 6. Estilo del gráfico
# ---------------------------------------------------------------
plt.title("Perceptrón - OR lógico")  # Título del gráfico.
plt.xlabel("Entrada A")              # Etiqueta del eje X.
plt.ylabel("Entrada B")              # Etiqueta del eje Y.
plt.grid(True)                       # Activa la cuadrícula para facilitar la lectura visual.
plt.xlim(x_min, x_max)               # Define los límites del eje X.
plt.ylim(y_min, y_max)               # Define los límites del eje Y.
plt.show()                           # Muestra la ventana con el gráfico final.

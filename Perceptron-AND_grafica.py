import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# ---------------------------------------------------------------
# 1. Datos del OR lógico (aunque el título dice "AND", estos datos
#    corresponden al comportamiento de un AND: solo 1,1 -> 1)
# ---------------------------------------------------------------
X = np.array([
    [0, 0],   # Entrada A=0, B=0
    [0, 1],   # Entrada A=0, B=1
    [1, 0],   # Entrada A=1, B=0
    [1, 1]    # Entrada A=1, B=1
])
y = np.array([0, 0, 0, 1])  
# Etiquetas de salida (resultado del AND lógico):
# solo 1 cuando ambas entradas son 1

# ---------------------------------------------------------------
# 2. Crear y entrenar el perceptrón
# ---------------------------------------------------------------
modelo = Perceptron(
    max_iter=1000,   # Número máximo de iteraciones (épocas) de entrenamiento
    eta0=0.1,        # Tasa de aprendizaje (cuánto se ajustan los pesos en cada paso)
    random_state=42  # Semilla para que los resultados sean reproducibles
)
modelo.fit(X, y)     # Entrena el modelo usando las entradas X y salidas y

# ---------------------------------------------------------------
# 3. Crear una malla de puntos para graficar la frontera de decisión
# ---------------------------------------------------------------
x_min, x_max = -1, 2   # Rango de valores en el eje X (para extender el gráfico)
y_min, y_max = -1, 2   # Rango de valores en el eje Y

# np.meshgrid crea una cuadrícula de coordenadas (xx, yy)
# útil para evaluar el modelo en muchos puntos y visualizar la frontera
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),  # 200 valores equidistantes en el rango X
    np.linspace(y_min, y_max, 200)   # 200 valores equidistantes en el rango Y
)

# Se predice la clase (0 o 1) para cada punto de la cuadrícula
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
# np.c_ combina xx y yy en pares (x, y)
# .ravel() aplana el arreglo para convertirlo en una lista de puntos
# reshape vuelve a dar forma a Z para que coincida con la forma de la cuadrícula
Z = Z.reshape(xx.shape)

# ---------------------------------------------------------------
# 4. Dibujar la frontera de decisión
# ---------------------------------------------------------------
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
# contourf dibuja regiones coloreadas según la clase predicha (0 o 1)
# alpha = 0.3 → transparencia del color
# cmap=plt.cm.Paired → paleta de colores predefinida

# ---------------------------------------------------------------
# 5. Dibujar los puntos de entrada (datos originales)
# ---------------------------------------------------------------
for i in range(len(X)):
    plt.scatter(
        X[i][0], X[i][1], 
        c='red' if y[i] == 1 else 'blue',  # Rojo si salida=1, azul si salida=0
        edgecolors='k',  # Borde negro alrededor del punto
        s=100            # Tamaño del punto
    )
    # Etiqueta junto al punto indicando su salida (0 o 1)
    plt.text(X[i][0] + 0.05, X[i][1], f"{y[i]}", fontsize=12)

# ---------------------------------------------------------------
# 6. Configurar el estilo del gráfico
# ---------------------------------------------------------------
plt.title("Perceptrón - AND lógico")  # Título del gráfico
plt.xlabel("Entrada A")               # Etiqueta del eje X
plt.ylabel("Entrada B")               # Etiqueta del eje Y
plt.grid(True)                        # Activa la cuadrícula
plt.xlim(x_min, x_max)                # Límites del eje X
plt.ylim(y_min, y_max)                # Límites del eje Y

# Muestra el gráfico
plt.show()


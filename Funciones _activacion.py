# ---------------------------------------------------------------
# Librerías necesarias
# ---------------------------------------------------------------
import numpy as np                    # NumPy se usa para crear y manipular arreglos numéricos
import matplotlib.pyplot as plt       # Matplotlib se usa para generar gráficos

# ---------------------------------------------------------------
# 1. Crear los valores del eje X
# ---------------------------------------------------------------
x = np.linspace(-2.5, 2.5, 100)
# np.linspace(inicio, fin, num_puntos)
# Genera 100 valores equidistantes entre -2.5 y 2.5
# Estos serán los valores de entrada para las funciones de activación

# ---------------------------------------------------------------
# 2. Definir funciones de activación
# ---------------------------------------------------------------

# Función Sigmoid: f(x) = 1 / (1 + e^(-x))
sigmoid = 1 / (1 + np.exp(-x))
# Rango de salida: (0, 1)
# Muy usada en capas de salida para problemas de clasificación binaria.

# Función Tanh (Tangente hiperbólica): f(x) = tanh(x)
tanh = np.tanh(x)
# Rango de salida: (-1, 1)
# Similar a la Sigmoid pero centrada en 0, lo que facilita el aprendizaje.

# Función ReLU (Rectified Linear Unit): f(x) = max(0, x)
relu = np.maximum(0, x)
# Rango de salida: [0, ∞)
# Es la más usada en redes neuronales modernas porque evita el problema del "gradiente desvanecido".

# ---------------------------------------------------------------
# 3. Crear una figura con subgráficos (subplots)
# ---------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
# plt.subplots(filas, columnas, tamaño)
# Crea una figura con 1 fila y 3 columnas (3 gráficos en la misma ventana)
# figsize define el tamaño de la figura en pulgadas (ancho, alto)

# ---------------------------------------------------------------
# 4. Dibujar cada función en su respectivo gráfico
# ---------------------------------------------------------------

# --- Gráfico 1: Sigmoid ---
axs[0].plot(x, sigmoid, label='Sigmoid', color='blue')  # Dibuja la curva
axs[0].set_title("Sigmoid")                             # Título del gráfico
axs[0].set_xlabel("x")                                  # Etiqueta del eje X
axs[0].set_ylabel("f(x)")                               # Etiqueta del eje Y
axs[0].grid(True)                                       # Activa cuadrícula
axs[0].legend()                                         # Muestra la leyenda

# --- Gráfico 2: Tanh ---
axs[1].plot(x, tanh, label='Tanh', color='orange')      # Dibuja la curva
axs[1].set_title("Tanh")
axs[1].set_xlabel("x")
axs[1].set_ylabel("f(x)")
axs[1].grid(True)
axs[1].legend()

# --- Gráfico 3: ReLU ---
axs[2].plot(x, relu, label='ReLU', color='green')       # Dibuja la curva
axs[2].set_title("ReLU")
axs[2].set_xlabel("x")
axs[2].set_ylabel("f(x)")
axs[2].grid(True)
axs[2].legend()

# ---------------------------------------------------------------
# 5. Ajustar el diseño y mostrar la figura
# ---------------------------------------------------------------
plt.tight_layout()   # Ajusta automáticamente el espaciado entre los subgráficos
plt.show()           # Muestra la figura con las tres funciones

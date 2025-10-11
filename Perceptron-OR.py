# ---------------------------------------------------------------
# Librerías necesarias
# ---------------------------------------------------------------

# pip install scikit-learn   # (Ejecutar una sola vez si no tienes instalada la librería)
from sklearn.linear_model import Perceptron   # Importa el modelo Perceptrón de scikit-learn
import numpy as np                            # Importa NumPy para manejar arreglos y operaciones matemáticas

# ---------------------------------------------------------------
# Ejemplo de uso del Perceptrón para el problema OR lógico
# ---------------------------------------------------------------
# Un perceptrón simple (una neurona) puede resolver operaciones lógicas
# que sean linealmente separables, como el OR, AND, o NOT.

# ---------------------------------------------------------------
# 1. Datos de entrada (entradas A y B del OR lógico)
# ---------------------------------------------------------------
X = np.array([
    [0, 0],   # Entrada A=0, B=0
    [0, 1],   # Entrada A=0, B=1
    [1, 0],   # Entrada A=1, B=0
    [1, 1]    # Entrada A=1, B=1
])
# Cada fila representa una combinación posible de las dos entradas lógicas.

# ---------------------------------------------------------------
# 2. Salidas esperadas (valores del OR lógico)
# ---------------------------------------------------------------
y = np.array([0, 1, 1, 1])
# Según la tabla de verdad del OR:
# 0 OR 0 → 0
# 0 OR 1 → 1
# 1 OR 0 → 1
# 1 OR 1 → 1

# ---------------------------------------------------------------
# 3. Crear el modelo Perceptrón
# ---------------------------------------------------------------
modelo = Perceptron(
    max_iter=10,       # Número máximo de iteraciones (épocas) para entrenar el modelo.
                       # Cada iteración recorre todas las muestras de entrenamiento.
    eta0=0.1,          # Tasa de aprendizaje (qué tan grandes son los ajustes de los pesos).
    random_state=0     # Fija una semilla aleatoria para obtener resultados reproducibles.
)
# Este perceptrón tiene una única neurona (una sola capa de salida).

# ---------------------------------------------------------------
# 4. Entrenar el modelo con los datos
# ---------------------------------------------------------------
modelo.fit(X, y)
# fit() ajusta los pesos y el sesgo (bias) del perceptrón
# de modo que las predicciones coincidan lo mejor posible con las salidas esperadas (y).

# ---------------------------------------------------------------
# 5. Probar el modelo entrenado
# ---------------------------------------------------------------
print("Predicciones:")
for entrada in X:
    pred = modelo.predict([entrada])   # Usa el modelo para predecir la salida (0 o 1)
    print(f"Entrada: {entrada}, Predicción: {pred[0]}")

# Salidas esperadas (si el modelo aprendió correctamente):
# [0, 0] → 0
# [0, 1] → 1
# [1, 0] → 1
# [1, 1] → 1

# ---------------------------------------------------------------
# 6. Ver los pesos y el sesgo (bias) aprendidos
# ---------------------------------------------------------------
print("\nPesos:", modelo.coef_)
# modelo.coef_ muestra los pesos que el perceptrón asignó a cada entrada.
# Ejemplo: [[0.3 0.4]] → significa que las entradas A y B tienen pesos diferentes
# según su influencia en la salida.

print("Bias:", modelo.intercept_)
# modelo.intercept_ muestra el valor del sesgo (bias), que actúa como un umbral
# para decidir si la salida final será 0 o 1.

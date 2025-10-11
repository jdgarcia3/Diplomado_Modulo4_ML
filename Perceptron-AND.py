# ---------------------------------------------------------------
# Librerías necesarias
# ---------------------------------------------------------------

# pip install scikit-learn   # Comando (si fuera necesario) para instalar scikit-learn
from sklearn.linear_model import Perceptron   # Importa el modelo Perceptrón de scikit-learn
import numpy as np                            # Importa NumPy para manejar arreglos numéricos y operaciones matemáticas

# ---------------------------------------------------------------
# Ejemplo de uso del Perceptrón para el problema AND lógico
# ---------------------------------------------------------------
# Un perceptrón es un modelo de una sola capa (una "neurona artificial")
# que aprende una frontera lineal entre dos clases.

# ---------------------------------------------------------------
# 1. Datos de entrada (entradas A y B del AND)
# ---------------------------------------------------------------
X = np.array([
    [0, 0],   # Entrada A=0, B=0
    [0, 1],   # Entrada A=0, B=1
    [1, 0],   # Entrada A=1, B=0
    [1, 1]    # Entrada A=1, B=1
])
# Cada fila representa una combinación de las entradas del operador lógico AND.

# ---------------------------------------------------------------
# 2. Salidas esperadas
# ---------------------------------------------------------------
y = np.array([0, 0, 0, 1])
# Estas son las salidas correspondientes al operador lógico AND:
# 0 AND 0 → 0
# 0 AND 1 → 0
# 1 AND 0 → 0
# 1 AND 1 → 1

# ---------------------------------------------------------------
# 3. Crear el modelo Perceptrón
# ---------------------------------------------------------------
modelo = Perceptron(
    max_iter=10,       # Número máximo de iteraciones (épocas) de entrenamiento.
                       # Una época = pasar por todos los datos una vez.
    eta0=0.1,          # Tasa de aprendizaje (qué tan rápido ajusta los pesos en cada paso).
    random_state=0     # Semilla aleatoria para obtener resultados reproducibles.
)
# Este modelo tiene una sola neurona (ya que es un perceptrón simple).

# ---------------------------------------------------------------
# 4. Entrenar el modelo con los datos
# ---------------------------------------------------------------
modelo.fit(X, y)
# El método fit() ajusta los pesos internos del modelo
# buscando una combinación que separe correctamente las clases (0 y 1).

# ---------------------------------------------------------------
# 5. Probar el modelo entrenado
# ---------------------------------------------------------------
print("Predicciones:")
for entrada in X:
    pred = modelo.predict([entrada])   # predice la salida (0 o 1) para cada combinación de entrada
    print(f"Entrada: {entrada}, Predicción: {pred[0]}")

# Al recorrer todas las entradas del AND, el perceptrón debería predecir:
# [0, 0] → 0
# [0, 1] → 0
# [1, 0] → 0
# [1, 1] → 1

# ---------------------------------------------------------------
# 6. Ver los pesos y el sesgo (bias) aprendidos
# ---------------------------------------------------------------
print("\nPesos:", modelo.coef_)
# modelo.coef_ → muestra los pesos asociados a cada entrada (A y B)
# Por ejemplo, algo como [[0.2, 0.2]] indica la contribución de A y B a la salida.

print("Bias:", modelo.intercept_)
# modelo.intercept_ → es el valor del sesgo (bias), que ajusta el umbral de activación
# para decidir si la salida es 0 o 1.

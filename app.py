import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generar más puntos para el dataset
celsius = np.array([-40, -30, -20, -10, 0, 10, 20, 30, 40, 50], dtype=float)
fahrenheit = (celsius * 1.8) + 32

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
train_celsius = celsius[:8]
train_fahrenheit = fahrenheit[:8]
test_celsius = celsius[8:]
test_fahrenheit = fahrenheit[8:]

# Definir el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, input_shape=[1], activation='relu'),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

# Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_absolute_error'
)

# Entrenar el modelo
print("Comenzando entrenamiento...")
historial = modelo.fit(train_celsius, train_fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado!")

# Evaluar el modelo en el conjunto de prueba
loss = modelo.evaluate(test_celsius, test_fahrenheit)
print("Pérdida en el conjunto de prueba: ", loss)

# Hacer una predicción
resultado = modelo.predict([100.0])
print("El resultado es:", resultado[0][0], "fahrenheit")

# Graficar la evolución de la pérdida durante el entrenamiento
plt.xlabel("#Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

# Imprimir las variables internas del modelo
for capa in modelo.layers:
    pesos, sesgos = capa.get_weights()
    print("Pesos:", pesos)
    print("Sesgos:", sesgos)

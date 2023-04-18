import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generar más puntos para el dataset
celsius = np.array([-40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], dtype=float)
fahrenheit = (celsius * 1.8) + 32

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
train_celsius = celsius[:15]
train_fahrenheit = fahrenheit[:15]
test_celsius = celsius[15:]
test_fahrenheit = fahrenheit[15:]

# Definir el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=32, input_shape=[1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l2(0.01))
])

# Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_absolute_error',
    metrics=['mean_squared_error']
)

# Entrenar el modelo
print("Comenzando entrenamiento...")
historial = modelo.fit(train_celsius, train_fahrenheit, epochs=500, verbose=False, validation_data=(test_celsius, test_fahrenheit))
print("Modelo entrenado!")

# Evaluar el modelo en el conjunto de prueba
loss, mse = modelo.evaluate(test_celsius, test_fahrenheit)
print("Pérdida en el conjunto de prueba (MAE): ", loss)
print("Error cuadrático medio en el conjunto de prueba (MSE): ", mse)

# Hacer una predicción
resultado = modelo.predict([100.0])
print("El resultado es:", resultado[0][0], "fahrenheit")

# Graficar la evolución de la pérdida durante el entrenamiento
plt.figure(figsize=(8, 6))
plt.plot(historial.history['loss'])
plt.plot(historial.history['val_loss'])
plt.title('Modelo de pérdida')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
plt.show()

# Imprimir las variables internas del modelo
for capa in modelo.layers:
    pesos, sesgos = capa.get_weights()
    print("Pesos:", pesos)
    print("Sesgos:", sesgos)

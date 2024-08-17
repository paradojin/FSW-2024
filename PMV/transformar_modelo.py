import tensorflow as tf

# Cargar el modelo entrenado en formato .h5
model = tf.keras.models.load_model('modelo_bostezos_mar.h5')

# Crear el convertidor de TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Opcional: aplicar optimizaciones
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convertir el modelo
tflite_model = converter.convert()

# Guardar el modelo convertido en un archivo .tflite
with open('modelo_bostezos_mar.tflite', 'wb') as f:
    f.write(tflite_model)

print("El modelo se ha convertido y guardado como 'modelo_bostezos_mar.tflite'")

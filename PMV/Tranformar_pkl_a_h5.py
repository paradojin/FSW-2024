import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar el modelo
with open('C:/Users/anton/Desktop/Universidad Dylhan/Feria 2024/FSW-2024/PMV/modelo_bostezos_mar.pkl', 'rb') as file:
    model = pickle.load(file)

# Supongamos que el modelo es un pipeline
if isinstance(model, Pipeline):
    # Extrae el modelo entrenado
    model = model.named_steps['modelo']

# Reimplementa el modelo en TensorFlow/Keras
# Ejemplo con un modelo simple
tf_model = Sequential([
    Dense(units=64, activation='relu', input_shape=(input_dim,)),
    Dense(units=10, activation='softmax')
])

# Asume que tienes el modelo y sus pesos
tf_model.set_weights(weights)

# Guarda el modelo en formato SavedModel o HDF5
tf_model.save('modelo_tf.h5')

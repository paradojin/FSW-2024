import os
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


# Ruta al directorio descomprimido del Drowsiness Dataset
drowsiness_data_path = 'drowsiness\\train\\'  # Ajusta esta ruta según sea necesario

# Ruta al clasificador en cascada de Haar para la detección de caras
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Cargar el predictor de puntos faciales de dlib
import dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Lista para almacenar el MAR y las etiquetas
mar_values = []
labels = []

# Función para calcular el MAR (mouth aspect ratio)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    mar = (A + B + C) / (3.0 * dist.euclidean(mouth[12], mouth[16]))
    return mar

# Función para cargar las imágenes y calcular el MAR
def load_drowsiness_images(data_path):
    for label, category in enumerate(['no_yawn', 'yawn']):
        category_path = os.path.join(data_path, category)
        for file in os.listdir(category_path):
            if file.endswith('.png') or file.endswith('.jpg'):
                file_path = os.path.join(category_path, file)
                image = cv2.imread(file_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]  # Usar la primera detección
                    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    mouth = shape[face_utils.FACIAL_LANDMARKS_IDXS["mouth"][0]:face_utils.FACIAL_LANDMARKS_IDXS["mouth"][1]]
                    mar = mouth_aspect_ratio(mouth)
                    mar_values.append(mar)
                    labels.append(label)
                    # Dibujar el rectángulo alrededor de la cara detectada y los puntos de la boca
                    for (x1, y1) in mouth:
                        cv2.circle(image, (x1, y1), 1, (0, 255, 0), -1)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    #cv2.imshow("Face Detection", image)
                    #cv2.waitKey(1)
                else:
                    print(f"No face detected in file: {file_path}")

# Cargar las imágenes y calcular el MAR
load_drowsiness_images(drowsiness_data_path)

# Verificar la cantidad de datos cargados
print(f"Total samples loaded: {len(mar_values)}")

# Convertir las listas a arrays de numpy
X = np.array(mar_values).reshape(-1, 1)
y = np.array(labels)

# Verificar las primeras entradas de los datos
print(f"First 5 MAR values: {X[:5]}")
print(f"First 5 labels: {y[:5]}")





# Normaliza los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definir el modelo en TensorFlow/Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_scaled.shape[1],)),  # Capa oculta con 10 neuronas
    tf.keras.layers.Dense(2, activation='softmax')  # Capa de salida con 2 neuronas para 2 clases
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy del modelo en TensorFlow/Keras: {accuracy * 100:.2f}%')

# Guardar el modelo en formato .h5
model.save('modelo_bostezos_mar.h5')
import os
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

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
                    cv2.imshow("Face Detection", image)
                    cv2.waitKey(1)
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





# Dividir el conjunto de datos en entrenamiento y prueba
if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Crear y entrenar el modelo basado en MAR
    clf_bostezos_mar = DecisionTreeClassifier()
    clf_bostezos_mar.fit(X_train, y_train)

    # Hacer predicciones y evaluar el modelo
    y_pred = clf_bostezos_mar.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy del modelo de bostezos: {accuracy * 100:.2f}%')

    # Guardar el modelo entrenado basado en MAR
    joblib.dump(clf_bostezos_mar, 'modelo_bostezos_mar.pkl')
else:
    print("No se cargaron datos suficientes para entrenar el modelo.")
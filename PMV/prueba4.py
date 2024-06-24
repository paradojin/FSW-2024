import cv2
import dlib
import joblib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from tensorflow.keras.models import load_model
import time

# Inicializar el detector de rostros y el predictor de puntos faciales de dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Cargar los modelos entrenados
modelo_bostezos_mar = joblib.load('modelo_bostezos_mar.pkl')
model_pestanear = load_model("eye_status_model.h5")

# Indices de los puntos faciales correspondientes a los ojos y la boca
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def eye_status(eye):
    if eye.size == 0:
        return "Unknown"
    eye = cv2.resize(eye, (24, 24))
    eye = eye.astype("float") / 255.0
    eye = np.expand_dims(eye, axis=-1)
    eye = np.expand_dims(eye, axis=0)
    pred = model_pestanear.predict(eye)
    return "Closed" if pred > 0.5 else "Open"

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    mar = (A + B + C) / (3.0 * dist.euclidean(mouth[12], mouth[16]))
    return mar

def detectar_bostezo_mar(mar):
    return modelo_bostezos_mar.predict([[mar]])[0]

def detect_drowsiness_and_yawn():
    cap = cv2.VideoCapture(0)
    blink_counter = 0
    total_blinks = 0
    microsleep_start_time = None
    microsleep_threshold = 1.0  # Tiempo en segundos para considerar un microsueño
    MAR_THRESHOLD = 0.3  # Umbral para detectar bostezos

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Procesar ojos
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            leftEyeRegion = gray[leftEye[1][1]:leftEye[4][1], leftEye[0][0]:leftEye[3][0]]
            rightEyeRegion = gray[rightEye[1][1]:rightEye[4][1], rightEye[0][0]:rightEye[3][0]]

            leftEyeStatus = eye_status(leftEyeRegion)
            rightEyeStatus = eye_status(rightEyeRegion)

            if leftEyeStatus == "Closed" and rightEyeStatus == "Closed":
                if microsleep_start_time is None:
                    microsleep_start_time = time.time()
                blink_counter += 1
                if time.time() - microsleep_start_time >= microsleep_threshold:
                    cv2.putText(frame, "Microsueno detectado", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                microsleep_start_time = None
                if blink_counter > 1:  # Considera un parpadeo si ambos ojos estuvieron cerrados en al menos 2 frames consecutivos
                    total_blinks += 1
                blink_counter = 0

            cv2.putText(frame, f'Left: {leftEyeStatus}', (leftEye[0][0], leftEye[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Right: {rightEyeStatus}', (rightEye[0][0], rightEye[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Parpadeos: {total_blinks}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Procesar boca
            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)

            # Obtener coordenadas de la boca para dibujar un cuadro
            (x, y, w, h) = cv2.boundingRect(np.array([shape[mStart:mEnd]]))
            
            # Dibujar puntos faciales en la boca
            for (x1, y1) in mouth:
                cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)
            
            # Mostrar el MAR calculado en la imagen
            cv2.putText(frame, f'MAR: {mar:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if mar > MAR_THRESHOLD:
                if detectar_bostezo_mar(mar):
                    cv2.putText(frame, "Bostezo detectado", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Dibuja un cuadro alrededor de la boca

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_drowsiness_and_yawn()

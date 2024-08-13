import cv2
import dlib
import joblib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from tensorflow.keras.models import load_model
import time
import simpleaudio as sa
from collections import deque

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
    return "Closed" if pred > 0.6 else "Open"

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    mar = (A + B + C) / (3.0 * dist.euclidean(mouth[12], mouth[16]))
    return mar

def detectar_bostezo_mar(mar):
    return modelo_bostezos_mar.predict([[mar]])[0]

def calcular_puntuacion_somnolencia(microsuenos_acumulados, promedio_pestaneos, bostezos, pesos):
    puntuacion = (pesos['microsuenos'] * microsuenos_acumulados) + \
                 (pesos['promedio_pestaneos'] * promedio_pestaneos) + \
                 (pesos['bostezos'] * bostezos)
    return min(puntuacion,100)

def detectar_rostros_y_puntos(gray):
    rects = detector(gray, 0)
    shapes = [predictor(gray, rect) for rect in rects]
    return rects, [face_utils.shape_to_np(shape) for shape in shapes]

def procesar_ojos(frame, gray, shape):
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
    return leftEyeStatus, rightEyeStatus

def procesar_boca(frame, shape, current_time, yawn_start_time, MAR_THRESHOLD, yawn_duration_threshold, total_yawns, yawn_detected, yawn_count, yawn_timestamps):
    mouth = shape[mStart:mEnd]
    mar = mouth_aspect_ratio(mouth)
    (x, y, w, h) = cv2.boundingRect(np.array([shape[mStart:mEnd]]))

    for (x1, y1) in mouth:
        cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)

    cv2.putText(frame, f'MAR: {mar:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    if mar > MAR_THRESHOLD:
        if yawn_start_time is None:
            yawn_start_time = current_time
        
        if current_time - yawn_start_time >= yawn_duration_threshold:
            if detectar_bostezo_mar(mar):
                cv2.putText(frame, "Bostezo detectado", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                total_yawns += 1

        if current_time - yawn_start_time >= yawn_duration_threshold and not yawn_detected:
            if detectar_bostezo_mar(mar):
                yawn_detected = True
                yawn_count += 1
                yawn_timestamps.append(current_time)
    else:
        yawn_start_time = None
        yawn_detected = False

    return yawn_start_time, total_yawns, yawn_detected, yawn_count,yawn_timestamps

def reproducir_alarma():
    wave_obj = sa.WaveObject.from_wave_file('alarma.wav')
    play_obj = wave_obj.play()
    play_obj.wait_done()

def superponer_imagen(frame, imagen_path, x_offset, y_offset):
    imagen_critica = cv2.imread(imagen_path, cv2.IMREAD_UNCHANGED)
    y1, y2 = y_offset, y_offset + imagen_critica.shape[0]
    x1, x2 = x_offset, x_offset + imagen_critica.shape[1]

    alpha_s = imagen_critica[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (alpha_s * imagen_critica[:, :, c] +
                                  alpha_l * frame[y1:y2, x1:x2, c])
    return frame

def detect_drowsiness_and_yawn():
    cap = cv2.VideoCapture(0)
    blink_counter = 0
    total_blinks = 0
    total_yawns = 0
    total_microsueños = 0
    microsleep_start_time = None
    yawn_start_time = None
    yawn_detected = False
    yawn_count = 0
    blink_start_time = None
    microsleep_threshold = 0.5                      # Tiempo de parpadeo para considerar un microsueño
    yawn_duration_threshold = 4.0                   # Duración del bostezo para considerarlo
    MAR_THRESHOLD = 0.3                             # Umbral para detectar bostezos
    blink_rate_window = 60
    start_time = time.time()
    previous_alert_level = None

    blink_timestamps = deque()  # Cola para almacenar los tiempos de parpadeos
    yawn_timestamps = deque()

    pesos = {
        'microsuenos': 3.0,
        'promedio_pestaneos': 0.1,
        'bostezos': 0.3
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        elapsed_time = current_time - start_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects, shapes = detectar_rostros_y_puntos(gray)

        for shape in shapes:
            leftEyeStatus, rightEyeStatus = procesar_ojos(frame, gray, shape)

            if leftEyeStatus == "Closed" and rightEyeStatus == "Closed":
                if microsleep_start_time is None:
                    microsleep_start_time = current_time
                blink_counter += 1
                if current_time - microsleep_start_time >= microsleep_threshold:
                    cv2.putText(frame, "Microsueno detectado", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    total_microsueños += 1  # Incrementamos el total de microsueños
            else:
                microsleep_start_time = None
                if blink_counter > 1:
                    total_blinks += 1
                    blink_timestamps.append(current_time)
                blink_counter = 0

            cv2.putText(frame, f'Left: {leftEyeStatus}', (shape[lStart][0], shape[lStart][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Right: {rightEyeStatus}', (shape[rStart][0], shape[rStart][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Parpadeos: {total_blinks}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            yawn_start_time, total_yawns, yawn_detected, yawn_count, yawn_timestamps = procesar_boca(frame, shape, current_time, yawn_start_time, MAR_THRESHOLD, yawn_duration_threshold, total_yawns, yawn_detected, yawn_count, yawn_timestamps)

        blink_rate = (total_blinks / elapsed_time) * 60
        # Calcular la tasa de parpadeos en los últimos 60 segundos
        while blink_timestamps and current_time - blink_timestamps[0] > 60:
            blink_timestamps.popleft()

        blink_rate_60s = len(blink_timestamps)

        # Calcular la tasa de parpadeos en los últimos 60 segundos
        while yawn_timestamps and current_time - yawn_timestamps[0] > 60:
            yawn_timestamps.popleft()

        blink_rate_60s = len(blink_timestamps)
        yawn_rate_60s = len(yawn_timestamps)
        cv2.putText(frame, f'Parpadeos ultimos 60s: {blink_rate_60s}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame, f'Bostezos: {yawn_count}', (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame, f'bostezos ultimos 60s: {yawn_rate_60s}', (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        if blink_rate_60s < 15:
            somnolencia_puntuacion = calcular_puntuacion_somnolencia(total_microsueños, blink_rate, total_yawns, pesos) + 5
        else:
            somnolencia_puntuacion = calcular_puntuacion_somnolencia(total_microsueños, blink_rate, total_yawns, pesos)

        if yawn_rate_60s > 1:
            somnolencia_puntuacion += yawn_rate_60s**2
        else:
            somnolencia_puntuacion += 0
            
        somnolencia_puntuacion = min(somnolencia_puntuacion,100)
        cv2.putText(frame, f'Somnolencia: {somnolencia_puntuacion:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        alert_level = None
        if somnolencia_puntuacion < 20:
            alert_level = "ninguna"
        elif 20 <= somnolencia_puntuacion < 40:
            alert_level = "baja"
            cv2.putText(frame, "Somnolencia: Baja", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif 40 <= somnolencia_puntuacion < 60:
            alert_level = "moderada"
            cv2.putText(frame, "Somnolencia: Moderada", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif 60 <= somnolencia_puntuacion < 70:
            alert_level = "alta"
            cv2.putText(frame, "Somnolencia: Alta", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # Naranja
        else:
            alert_level = "critica"
            cv2.putText(frame, "ALERTA: Somnolencia Critica", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            frame = superponer_imagen(frame, "imagen_critica.png", x_offset=0, y_offset=150)
            # Solo reproducir la alarma y mostrar la imagen en el nivel crítico
            if alert_level != previous_alert_level:
                reproducir_alarma()
            
        previous_alert_level = alert_level

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_drowsiness_and_yawn()
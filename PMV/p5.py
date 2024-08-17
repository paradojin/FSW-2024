from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
import dlib
import joblib
from scipy.spatial import distance as dist
from imutils import face_utils
from tensorflow.lite.python.interpreter import Interpreter
from collections import deque
import time
import simpleaudio as sa
import tensorflow as tf

# Cargar los modelos y recursos
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
modelo_bostezos_mar = joblib.load('modelo_bostezos_mar.pkl')
interpreter = Interpreter(model_path="eye_status_model.tflite")
interpreter.allocate_tensors()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

interpreter = tf.lite.Interpreter(model_path='eye_status_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def eye_status(eye):
    if eye.size == 0:
        return "Unknown"
    
    # Ajustar el tamaño de la imagen y convertir a float32
    eye = cv2.resize(eye, (24, 24))
    eye = eye.astype("float32") / 255.0
    
    # Verificar dimensiones esperadas
    input_shape = input_details[0]['shape']
    
    if len(input_shape) == 4:
        eye = np.expand_dims(eye, axis=-1)  # Añadir canal
        eye = np.expand_dims(eye, axis=0)   # Añadir dimensión de batch
    elif len(input_shape) == 3:
        eye = np.expand_dims(eye, axis=0)   # Solo añadir dimensión de batch
    else:
        raise ValueError("Dimensiones inesperadas para la entrada del modelo")
    
    # Establecer el tensor
    interpreter.set_tensor(input_details[0]['index'], eye)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    
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

    return yawn_start_time, total_yawns, yawn_detected, yawn_count, yawn_timestamps

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

class DrowsinessApp(App):
    def build(self):
        self.img = Image()
        self.capture = cv2.VideoCapture(0)
        self.texture = None
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update every 1/30 seconds
        return self.img

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        current_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects, shapes = detectar_rostros_y_puntos(gray)

        blink_counter = 0
        total_blinks = 0
        total_yawns = 0
        total_microsueños = 0
        microsleep_start_time = None
        yawn_start_time = None
        yawn_detected = False
        yawn_count = 0
        blink_start_time = None
        microsleep_threshold = 0.5
        yawn_duration_threshold = 4.0
        MAR_THRESHOLD = 0.3

        blink_timestamps = deque()
        yawn_timestamps = deque()

        for shape in shapes:
            leftEyeStatus, rightEyeStatus = procesar_ojos(frame, gray, shape)

            if leftEyeStatus == "Closed" and rightEyeStatus == "Closed":
                if microsleep_start_time is None:
                    microsleep_start_time = current_time
                blink_counter += 1
                if current_time - microsleep_start_time >= microsleep_threshold:
                    cv2.putText(frame, "Microsueno detectado", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    total_microsueños += 1
            else:
                microsleep_start_time = None
                if blink_counter > 1:
                    total_blinks += 1
                    blink_timestamps.append(current_time)
                blink_counter = 0

            cv2.putText(frame, f'Left Eye: {leftEyeStatus}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Right Eye: {rightEyeStatus}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            yawn_start_time, total_yawns, yawn_detected, yawn_count, yawn_timestamps = procesar_boca(frame, shape, current_time, yawn_start_time, MAR_THRESHOLD, yawn_duration_threshold, total_yawns, yawn_detected, yawn_count, yawn_timestamps)

        cv2.putText(frame, f'Bostezos: {total_yawns}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Cálculo y visualización de puntuación de somnolencia
        puntuacion = calcular_puntuacion_somnolencia(total_microsueños, total_blinks, total_yawns, {'microsuenos': 1, 'promedio_pestaneos': 1, 'bostezos': 1})
        cv2.putText(frame, f'Somnolencia: {puntuacion}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if puntuacion > 50:  # Umbral de puntuación para activar alarma
            reproducir_alarma()

        # Mostrar el resultado en Kivy
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not self.texture:
            self.texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        self.texture.blit_buffer(frame.flatten(), colorfmt='rgb', bufferfmt='ubyte')
        self.img.texture = self.texture

    def on_stop(self):
        if self.capture:
            self.capture.release()

if __name__ == '__main__':
    DrowsinessApp().run()


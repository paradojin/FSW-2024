from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import dlib
import joblib
from scipy.spatial import distance as dist
from imutils import face_utils
from tensorflow.keras.models import load_model
from collections import deque
import time
import logging

# Configurar el registro de errores
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Inicializar el detector de rostros y el predictor de puntos faciales de dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Cargar los modelos entrenados
modelo_bostezos_mar = joblib.load('modelo_bostezos_mar.pkl')
model_pestanear = load_model("eye_status_model.h5")

# Índices de los puntos faciales correspondientes a los ojos y la boca
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Para almacenar las métricas de somnolencia
blink_counter = 0
blink_timestamps = deque()
yawn_timestamps = deque()
total_blinks = 0
total_yawns = 0
microsuenos_acumulados = 0
start_time = time.time()

# Variables para medir el tiempo de los ojos cerrados
microsleep_start_time = None
left_eye_closed_start_time = None
right_eye_closed_start_time = None
mouth_open_start_time = None

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
    return int(modelo_bostezos_mar.predict([[mar]])[0])  # Convertir a int

def calcular_puntuacion_somnolencia(microsuenos_acumulados, promedio_pestaneos, bostezos, pesos):
    puntuacion = (pesos['microsuenos'] * microsuenos_acumulados) + \
                 (pesos['promedio_pestaneos'] * promedio_pestaneos) + \
                 (pesos['bostezos'] * bostezos)
    return min(puntuacion, 100)
    

def yuv_to_rgb(y, u, v, width, height, format_type='420p'):
    try:
        yuv = np.zeros((height + height // 2, width), dtype=np.uint8)
        yuv[0:height, 0:width] = y.reshape(height, width)
        u_reshaped = u[: (width // 2) * (height // 2)].reshape((height // 2), (width // 2))
        v_reshaped = v[: (width // 2) * (height // 2)].reshape((height // 2), (width // 2))
        yuv[height:height + height // 2, 0:width:2] = u_reshaped
        yuv[height:height + height // 2, 1:width:2] = v_reshaped
        rgb_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    
    except Exception as e:
        raise ValueError(f"Error processing YUV data: {str(e)}")

    return rgb_image


@app.get("/ping/")
async def ping():
    return JSONResponse(content={"status": "ok", "message": "Server is ready to receive requests"})

@app.get("/end_trip/")
async def end_trip():
    global blink_timestamps, yawn_timestamps, total_blinks, total_yawns, microsuenos_acumulados, start_time
    global left_eye_closed_start_time, right_eye_closed_start_time

    # Resetear las variables globales
    blink_timestamps.clear()
    yawn_timestamps.clear()
    total_blinks = 0
    total_yawns = 0
    microsuenos_acumulados = 0
    start_time = time.time()  # Reiniciar el tiempo de inicio
    left_eye_closed_start_time = None
    right_eye_closed_start_time = None

    return JSONResponse(content={"status": "ok", "message": "Viaje finalizado y variables reseteadas."})

@app.post("/detect/")
async def detect(y_plane: UploadFile = File(...), u_plane: UploadFile = File(...), v_plane: UploadFile = File(...),
                 width: int = Form(...), height: int = Form(...), format_type: str = Form('420p')):
    try:
        # Leer los planos Y, U, V
        y = np.frombuffer(await y_plane.read(), dtype=np.uint8)
        u = np.frombuffer(await u_plane.read(), dtype=np.uint8)
        v = np.frombuffer(await v_plane.read(), dtype=np.uint8)

        # Convertir YUV a RGB
        frame = yuv_to_rgb(y, u, v, width, height, format_type)

        if frame is None:
            raise HTTPException(status_code=400, detail="Error decoding the image")

        # Asegurarse de que la imagen esté orientada correctamente (rotarla si es necesario)
        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)    # Para dispositivo fisico 
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)          # Para emulador
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros y puntos faciales
        rects, shapes = detectar_rostros_y_puntos(gray)

        # Verificar si se detectaron rostros
        if len(rects) == 0:
            raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")

        results = []

        global left_eye_closed_start_time, right_eye_closed_start_time, microsleep_start_time
        for shape in shapes:
            leftEyeStatus, rightEyeStatus = procesar_ojos(frame, gray, shape)

            
            # Procesar los ojos
            global total_blinks, microsuenos_acumulados, blink_counter, microsleep_start_time
            current_time = time.time()
            microsleep_threshold = 0.5

            if leftEyeStatus == "Closed" and rightEyeStatus == "Closed":
                if microsleep_start_time is None:
                    microsleep_start_time = current_time  # Inicia el conteo del tiempo con ojos cerrados
                blink_counter += 1
                # Verifica si se alcanza el umbral para un microsueño
                if current_time - microsleep_start_time >= microsleep_threshold:
                    microsuenos_acumulados += 1  # Cuenta un microsueño
                    blink_counter = 0  # Resetea el contador de parpadeos
                    microsleep_start_time = None  # Resetea el tiempo de inicio del microsueño
            else:
                if microsleep_start_time is not None:
                    # Si el tiempo acumulado es menor que el umbral, cuenta como un parpadeo
                    if blink_counter > 1 and (current_time - microsleep_start_time < microsleep_threshold):
                        total_blinks += 1
                        blink_timestamps.append(current_time)
                    # Resetea los contadores y tiempos
                    blink_counter = 0
                    microsleep_start_time = None


            # Procesar la boca
            global total_yawns, mouth_open_start_time
            mar = mouth_aspect_ratio(shape[mStart:mEnd])
            yawn_detected = detectar_bostezo_mar(mar)

            if yawn_detected:
                if mouth_open_start_time is None:
                    mouth_open_start_time = time.time()
                elif (time.time() - mouth_open_start_time) >= 4.0:
                    total_yawns += 1
                    yawn_timestamps.append(current_time)
                    mouth_open_start_time = None
            else:
                mouth_open_start_time = None

            # Calcular la puntuación de somnolencia
            blink_rate = (total_blinks / (current_time - start_time)) * 60
            somnolencia_puntuacion = calcular_puntuacion_somnolencia(microsuenos_acumulados, blink_rate, total_yawns, {'microsuenos': 3.0, 'promedio_pestaneos': 0.1, 'bostezos': 0.3})

            #calcular metricas de 60 segundos
            # Calcular la tasa de parpadeos en los últimos 60 segundos
            while blink_timestamps and current_time - blink_timestamps[0] > 60:
                blink_timestamps.popleft()

            blink_rate_60s = len(blink_timestamps)

            # Calcular la tasa de parpadeos en los últimos 60 segundos
            while yawn_timestamps and current_time - yawn_timestamps[0] > 60:
                yawn_timestamps.popleft()

            yawn_rate_60s = len(yawn_timestamps)

            results.append({
                "left_eye_status": str(leftEyeStatus),
                "right_eye_status": str(rightEyeStatus),
                "mar": float(mar),  # Asegurar que sea tipo float
                "yawn_detected": bool(yawn_detected),
                "somnolencia_puntuacion": float(somnolencia_puntuacion),  # Asegurar que sea tipo float
                "total_blinks": int(total_blinks),  # Asegurar que sea tipo int
                "total_yawns": int(total_yawns),  # Asegurar que sea tipo int
                "microsuenos_acumulados": int(microsuenos_acumulados),  # Asegurar que sea tipo int
                "blink_rate_60s": float(blink_rate_60s),  # Asegurar que sea tipo float
                "yawn_rate_60s": float(yawn_rate_60s)  # Asegurar que sea tipo float
            })


        return JSONResponse(content=results)

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


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

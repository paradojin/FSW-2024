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
microsleep_detected = False
blink_timestamps = deque()
yawn_timestamps = deque()
total_blinks = 0
total_yawns = 0
microsuenos_acumulados = 0
microsuenos_acumulados_1 = 0
start_time = time.time()
alert_level = None
color = None

# Variables para medir el tiempo de los ojos cerrados
microsleep_start_time = None
left_eye_closed_start_time = None
right_eye_closed_start_time = None
mouth_open_start_time = None

pesos = {
        'microsuenos': 7.0,
        'promedio_pestaneos': 0.3 ,
        'bostezos': 0.3
    }

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
    global blink_timestamps, yawn_timestamps, total_blinks, total_yawns, microsuenos_acumulados, start_time,blink_counter, microsleep_detected,microsuenos_acumulados_1, alert_level,color
    global left_eye_closed_start_time, right_eye_closed_start_time

    # Resetear las variables globales
    blink_counter=0
    
    blink_timestamps.clear()
    yawn_timestamps.clear()
    total_blinks = 0
    total_yawns = 0
    microsuenos_acumulados_1 = 0
    microsuenos_acumulados = 0
    start_time = time.time()  # Reiniciar el tiempo de inicio
    left_eye_closed_start_time = None
    right_eye_closed_start_time = None
    microsleep_detected = False
    alert_level = None
    color = None

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
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)    # Para dispositivo fisico 
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)          # Para emulador
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #cv2.imshow("Frame from Server", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    cv2.destroyAllWindows()

        # Detectar rostros y puntos faciales
        rects, shapes = detectar_rostros_y_puntos(gray)
    
        if len(shapes) == 0:
            raise HTTPException(status_code=400, detail="No faces detected in the image")
        results = []

        global left_eye_closed_start_time, right_eye_closed_start_time, microsleep_start_time
        for shape in shapes:
            leftEyeStatus, rightEyeStatus, ear_left, ear_rigth = procesar_ojos(frame, gray, shape)

            
            # Procesar los ojos
            global total_blinks, microsuenos_acumulados, blink_counter, microsleep_start_time, microsleep_detected,microsuenos_acumulados_1, alert_level,color

            current_time = time.time()
            microsleep_threshold = 0.5

    
            if leftEyeStatus == "Closed" and rightEyeStatus == "Closed":
                
                if microsleep_start_time is None:
                    microsleep_start_time = current_time
                blink_counter += 1
                
                # Si los ojos están cerrados por más del umbral, es un microsueño
                if current_time - microsleep_start_time >= microsleep_threshold:
                    microsuenos_acumulados_1 += 1
                    if not microsleep_detected:  # Solo contar un microsueño por periodo de ojos cerrados
                        microsuenos_acumulados += 1
                        microsleep_detected = True  # Evita que cuente más microsueños durante el mismo cierre de ojos

            else:
                # Si los ojos se abren, reiniciar el temporizador y contar parpadeo si fue corto
                if blink_counter > 1 and not microsleep_detected:
                    total_blinks += 1
                    blink_timestamps.append(current_time)
                    
                    
                    
                # Reiniciar contadores al abrir los ojos
                blink_counter = 0
                microsleep_start_time = None
                microsleep_detected = False  # Reiniciar la detección de microsueños


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

            #calcular metricas de 60 segundos
            # Calcular la tasa de parpadeos en los últimos 60 segundos
            while blink_timestamps and current_time - blink_timestamps[0] > 60:
                blink_timestamps.popleft()

            blink_rate_60s = len(blink_timestamps)

            # Calcular la tasa de parpadeos en los últimos 60 segundos
            while yawn_timestamps and current_time - yawn_timestamps[0] > 60:
                yawn_timestamps.popleft()

            yawn_rate_60s = len(yawn_timestamps)

            if blink_rate_60s < 15:
                somnolencia_puntuacion = calcular_puntuacion_somnolencia(microsuenos_acumulados_1, blink_rate, total_yawns, pesos) + 5
            else:
                somnolencia_puntuacion = calcular_puntuacion_somnolencia(microsuenos_acumulados_1, blink_rate, total_yawns, pesos)

            if yawn_rate_60s > 1:
                somnolencia_puntuacion += yawn_rate_60s**2
            else:
                somnolencia_puntuacion += 0

            somnolencia_puntuacion = min(somnolencia_puntuacion,100)

            if somnolencia_puntuacion < 20:
                alert_level = "ninguna"
                color = "negro"
            elif 20 <= somnolencia_puntuacion < 40:
                alert_level = "baja"
                color = "verde"
            elif 40 <= somnolencia_puntuacion < 60:
                alert_level = "moderada"
                color = "amarillo"
            elif 60 <= somnolencia_puntuacion < 70:
                alert_level = "alta"
                color = "naranja"
            else:
                alert_level = "critica"
                color = "rojo"


            results.append({
                "left_eye_status": str(leftEyeStatus),
                "right_eye_status": str(rightEyeStatus),
                "mar": float(mar),  # Asegurar que sea tipo float
                "yawn_detected": bool(yawn_detected),
                "somnolencia_puntuacion": round(float(somnolencia_puntuacion), 2),  # Asegurar que sea tipo float
                "total_blinks": int(total_blinks),  # Asegurar que sea tipo int
                "total_yawns": int(total_yawns),  # Asegurar que sea tipo int
                "microsuenos_acumulados": int(microsuenos_acumulados),  # Asegurar que sea tipo int
                "blink_rate_60s": float(blink_rate_60s),  # Asegurar que sea tipo float
                "yawn_rate_60s": float(yawn_rate_60s),  # Asegurar que sea tipo float,
                "ear_left": round(float(ear_left),2),
                "ear_right": round(float(ear_rigth),2),
                "alert_level": str(alert_level),
                "color": str(color)
            })


        return JSONResponse(content=results)

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


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
    ear_left = eye_aspect_ratio(leftEye)
    ear_right = eye_aspect_ratio(rightEye)
    return leftEyeStatus, rightEyeStatus, ear_left, ear_right

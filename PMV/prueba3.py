import cv2
import dlib
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model
import time

# Inicializar el detector de rostros y el predictor de puntos faciales de dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Cargar el modelo entrenado
model = load_model("eye_status_model.h5")

# Indices de los puntos faciales correspondientes a los ojos
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_status(eye):
    if eye.size == 0:
        return "Unknown"
    eye = cv2.resize(eye, (24, 24))
    eye = eye.astype("float") / 255.0
    eye = np.expand_dims(eye, axis=-1)
    eye = np.expand_dims(eye, axis=0)
    pred = model.predict(eye)
    return "Closed" if pred > 0.5 else "Open"

def detect_drowsiness():
    cap = cv2.VideoCapture(0)
    blink_counter = 0
    total_blinks = 0
    microsleep_start_time = None
    microsleep_threshold = 1.0  # Tiempo en segundos para considerar un microsueño

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Extraer las coordenadas de los ojos
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
                    cv2.putText(frame, "Microsueño detectado", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                microsleep_start_time = None
                if blink_counter > 1:  # Considera un parpadeo si ambos ojos estuvieron cerrados en al menos 2 frames consecutivos
                    total_blinks += 1
                blink_counter = 0

            cv2.putText(frame, f'Left Eye: {leftEyeStatus}', (leftEye[0][0], leftEye[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Right Eye: {rightEyeStatus}', (rightEye[0][0], rightEye[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Parpadeos: {total_blinks}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_drowsiness()

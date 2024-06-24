import cv2
import dlib
import joblib
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np

# Cargar el detector de rostros y el predictor de puntos faciales de dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Cargar el modelo entrenado basado en MAR
modelo_bostezos_mar = joblib.load('modelo_bostezos_mar.pkl')

# Indices de los puntos faciales correspondientes a la boca
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    mar = (A + B + C) / (3.0 * dist.euclidean(mouth[12], mouth[16]))
    return mar

def detectar_bostezo_mar(mar):
    return modelo_bostezos_mar.predict([[mar]])[0]

def detect_yawn():
    cap = cv2.VideoCapture(0)
    MAR_THRESHOLD = 0.3  # Ajustar el umbral si es necesario

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
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
                    cv2.putText(frame, "Bostezo detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Dibuja un cuadro alrededor de la boca

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_yawn()

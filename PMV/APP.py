from flask import Flask, request, jsonify
import cv2
import dlib
import joblib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from tensorflow.keras.models import load_model

app = Flask(__name__)

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

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Obtener la imagen del POST request
        file = request.files['image'].read()
        np_img = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        
        # Verifica si se detectaron rostros
        if len(rects) == 0:
            return jsonify({"error": "No se detectaron rostros en la imagen"}), 400
        
        shapes = [predictor(gray, rect) for rect in rects]
        results = []

        for shape in shapes:
            leftEye = []
            for i in range(lStart, lEnd):
                leftEye.append((shape.part(i).x, shape.part(i).y))

            rightEye = []
            for i in range(rStart, rEnd):
                rightEye.append((shape.part(i).x, shape.part(i).y))

            mouth = []
            for i in range(mStart, mEnd):
                mouth.append((shape.part(i).x, shape.part(i).y))

            leftEye = np.array(leftEye)
            rightEye = np.array(rightEye)
            mouth = np.array(mouth)

            leftEyeRegion = gray[leftEye[1][1]:leftEye[4][1], leftEye[0][0]:leftEye[3][0]]
            rightEyeRegion = gray[rightEye[1][1]:rightEye[4][1], rightEye[0][0]:rightEye[3][0]]

            leftEyeStatus = eye_status(leftEyeRegion)
            rightEyeStatus = eye_status(rightEyeRegion)

            mar = mouth_aspect_ratio(mouth)
            yawn_detected = detectar_bostezo_mar(mar)

            results.append({
                'left_eye_status': str(leftEyeStatus),
                'right_eye_status': str(rightEyeStatus),
                'mouth_aspect_ratio': float(mar),
                'yawn_detected': bool(yawn_detected)
            })

        return jsonify(results)

    except Exception as e:
        # Manejar cualquier otro error y devolver el mensaje de error
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

import cv2
import dlib

# Cargar el detector de rostros de dlib
detector = dlib.get_frontal_face_detector()

# Cargar el predictor de puntos faciales de dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros en la imagen
    rects = detector(gray, 0)
    
    for rect in rects:
        # Dibujar un rectángulo alrededor del rostro detectado
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Obtener los puntos faciales
        shape = predictor(gray, rect)
        for i in range(0, 68):
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    
    # Mostrar el frame con las detecciones
    cv2.imshow("Frame", frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

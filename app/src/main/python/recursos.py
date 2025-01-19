import cv2
import numpy as np
import base64

def process_image(image_bytes):
    # Convertir los bytes a un array de numpy
    np_arr = np.frombuffer(image_bytes, np.uint8)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Decodificar la imagen usando OpenCV
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Leemos la imagen como un array de OpenCV
    img = cv2.flip(img,1)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # Procesar la imagen (aquí puedes aplicar cualquier procesamiento de OpenCV)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Ejemplo: convertir a escala de grises
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Convertir la imagen procesada de vuelta a bytes
    _, processed_image_bytes = cv2.imencode('.jpeg', img)  # Puedes usar otros formatos, como JPEG

    return processed_image_bytes.tobytes()  # Regresar los bytes de la imagen procesada

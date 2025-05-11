import cv2
from google.cloud import vision
from PIL import Image
import os
import time

# Configurar las credenciales de Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\WINDOWS\Desktop\DOCS\friendly-path-459404-a8-cdbaf9e51a09.json"

# Inicializar el cliente de Vision API
client = vision.ImageAnnotatorClient()

# Inicializar la webcam
cap = cv2.VideoCapture(0)  # 0 es el índice de la cámara predeterminada

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Presiona 'c' para capturar una foto, 'q' para salir.")

while True:
    # Leer frame de la webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    # Mostrar el frame en una ventana
    cv2.imshow('Webcam', frame)

    # Capturar foto al presionar 'c'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Guardar la imagen capturada
        save_path = os.path.join(os.getcwd(), "captured.jpg")
        cv2.imwrite(save_path, frame)
        print(f"Foto capturada y guardada en: {save_path}")

        # Analizar la imagen con Google Cloud Vision
        with open(save_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

        # Detectar etiquetas (objetos)
        response = client.label_detection(image=image)
        print("Objetos detectados:")
        for label in response.label_annotations:
            print(f"- {label.description} (Confianza: {label.score:.2%})")

        # Detectar rostros
        response = client.face_detection(image=image)
        print("\nRostros detectados:")
        if response.face_annotations:
            for i, face in enumerate(response.face_annotations, 1):
                print(f"- Rostro {i}: Confianza {face.detection_confidence:.2%}")
        else:
            print("- No se detectaron rostros.")

    # Salir al presionar 'q'
    elif key == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
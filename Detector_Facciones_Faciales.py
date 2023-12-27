import cv2
import face_recognition
from deepface import DeepFace

def detectar_genero_edad_emociones(imagen_path):
    # Cargar la imagen
    imagen = face_recognition.load_image_file(imagen_path)

    # Obtener ubicaciones de los rostros en la imagen
    ubicaciones_rostros = face_recognition.face_locations(imagen)

    for ubicacion_rostro in ubicaciones_rostros:
        # Extraer las coordenadas de la ubicación del rostro
        top, right, bottom, left = ubicacion_rostro

        # Recortar la imagen para obtener solo el rostro
        rostro = imagen[top:bottom, left:right]

        # Redimensionar la imagen para DeepFace
        rostro = cv2.resize(rostro, (48, 48))

        # Realizar la predicción de género, edad y emociones con DeepFace
        resultados = DeepFace.analyze(rostro, actions=['gender', 'age', 'emotion'])

        # Mostrar los resultados
        print(f"Ubicación del rostro: {ubicacion_rostro}")
        print(f"Género: {resultados['gender']}")
        print(f"Edad: {resultados['age']}")
        print(f"Emociones: {resultados['dominant_emotion']}")
        print("-" * 50)

if __name__ == "__main__":
    # Ruta de la imagen a analizar
    imagen_path = "ruta/a/tu/imagen.jpg"

    # Realizar la detección de género, edad y emociones
    detectar_genero_edad_emociones(imagen_path)

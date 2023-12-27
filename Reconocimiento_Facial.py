import face_recognition
import cv2
import os

# Función para cargar las imágenes y nombres de personas conocidas
def load_known_faces():
    known_faces = []
    known_names = []

    for filename in os.listdir("known_faces"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join("known_faces", filename))
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(os.path.splitext(filename)[0])

    return known_faces, known_names

# Cargar caras conocidas
known_faces, known_names = load_known_faces()

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar un frame
    ret, frame = cap.read()

    # Encontrar caras en el frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comprobar si la cara es conocida
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Desconocido"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Dibujar un rectángulo alrededor de la cara y mostrar el nombre
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Mostrar la imagen
    cv2.imshow("Reconocimiento Facial", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()

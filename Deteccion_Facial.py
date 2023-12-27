import cv2

def detectar_rostros():
    # Inicializar el clasificador de rostros de OpenCV
    clasificador_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Iniciar la captura de video desde la cámara
    cap = cv2.VideoCapture(0)

    # Crear una ventana
    cv2.namedWindow('Detección Facial', cv2.WINDOW_NORMAL)

    while True:
        # Leer un frame de la cámara
        ret, frame = cap.read()

        # Convertir la imagen a escala de grises
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en la imagen
        rostros = clasificador_rostros.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=5)

        # Dibujar un rectángulo alrededor de cada rostro detectado
        for (x, y, w, h) in rostros:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Mostrar la imagen en la ventana
        cv2.imshow('Detección Facial', frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_rostros()

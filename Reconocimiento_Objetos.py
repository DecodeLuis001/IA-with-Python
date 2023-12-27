import cv2
import numpy as np

def reconocimiento_objetos(imagen):
    # Cargar el modelo preentrenado de MobileNet SSD
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

    # Obtener dimensiones de la imagen
    alto, ancho = imagen.shape[:2]

    # Preprocesar la imagen para la red neuronal
    blob = cv2.dnn.blobFromImage(imagen, 0.007843, (300, 300), 127.5)

    # Pasar la imagen a través de la red neuronal
    net.setInput(blob)
    detecciones = net.forward()

    # Procesar las detecciones
    for i in range(detecciones.shape[2]):
        confianza = detecciones[0, 0, i, 2]
        if confianza > 0.5:  # Umbral de confianza
            clase = int(detecciones[0, 0, i, 1])
            etiqueta = CLASES[clase]
            confianza = round(confianza * 100, 2)
            box = detecciones[0, 0, i, 3:7] * np.array([ancho, alto, ancho, alto])
            (inicioX, inicioY, finX, finY) = box.astype("int")

            # Dibujar el rectángulo y la etiqueta en la imagen
            cv2.rectangle(imagen, (inicioX, inicioY), (finX, finY), (0, 255, 0), 2)
            texto = f"{etiqueta}: {confianza}%"
            cv2.putText(imagen, texto, (inicioX, inicioY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return imagen

# Lista de clases disponibles en el modelo MobileNet SSD
CLASES = ["fondo", "avion", "bicicleta", "pajaro", "barco", "botella", "autobus", "auto", "gato", "silla",
          "vaca", "mesa", "perro", "caballo", "motocicleta", "persona", "planta en maceta", "oveja", "sofa",
          "tren"]

# Cargar una imagen para el reconocimiento de objetos
imagen_path = "imagen.jpg"  # Reemplaza con la ruta de tu propia imagen
imagen = cv2.imread(imagen_path)

# Realizar el reconocimiento de objetos
imagen_reconocida = reconocimiento_objetos(imagen)

# Mostrar la imagen con los objetos identificados
cv2.imshow("Reconocimiento de Objetos", imagen_reconocida)
cv2.waitKey(0)
cv2.destroyAllWindows()

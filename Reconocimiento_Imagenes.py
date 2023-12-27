import cv2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

def reconocimiento_imagen(ruta_imagen):
    # Cargar el modelo preentrenado InceptionV3
    modelo = InceptionV3(weights='imagenet')

    # Cargar la imagen y ajustar al tamaño requerido por InceptionV3 (299x299 píxeles)
    img = image.load_img(ruta_imagen, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Realizar la predicción
    predicciones = modelo.predict(img_array)

    # Decodificar y mostrar las tres principales predicciones
    decoded_predictions = decode_predictions(predicciones, top=3)[0]
    print("Predicciones:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score:.2f})")

    # Mostrar la imagen con las predicciones
    img = cv2.imread(ruta_imagen)
    cv2.imshow("Imagen", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Reemplaza con la ruta de tu propia imagen
    ruta_imagen = "imagen.jpg"
    reconocimiento_imagen(ruta_imagen)

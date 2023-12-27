import cv2
import pytesseract

# Configurar la ruta al ejecutable de Tesseract (reemplaza con tu propia ruta)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def reconocimiento_matricula(ruta_imagen):
    # Cargar la imagen
    img = cv2.imread(ruta_imagen)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral adaptativo
    _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Realizar OCR en la imagen
    texto_matricula = pytesseract.image_to_string(threshold, config='--psm 8')

    # Mostrar la imagen y el resultado del reconocimiento
    cv2.imshow("Imagen", img)
    print(f"Matrícula reconocida: {texto_matricula}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Reemplaza con la ruta de tu propia imagen de matrícula
    ruta_imagen_matricula = "matricula.jpg"
    reconocimiento_matricula(ruta_imagen_matricula)

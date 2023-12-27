from textblob import TextBlob

def analizar_sentimientos(texto):
    # Crear un objeto TextBlob
    blob = TextBlob(texto)

    # Obtener la polaridad y subjetividad del texto
    polaridad = blob.sentiment.polarity
    subjetividad = blob.sentiment.subjectivity

    # Clasificar el sentimiento
    if polaridad > 0:
        sentimiento = "Positivo"
    elif polaridad < 0:
        sentimiento = "Negativo"
    else:
        sentimiento = "Neutral"

    return sentimiento, polaridad, subjetividad

if __name__ == "__main__":
    # Solicitar al usuario que ingrese un texto
    texto_usuario = input("Ingrese un texto para analizar los sentimientos: ")

    # Analizar sentimientos
    sentimiento, polaridad, subjetividad = analizar_sentimientos(texto_usuario)

    # Mostrar los resultados
    print(f"Sentimiento: {sentimiento}")
    print(f"Polaridad: {polaridad}")
    print(f"Subjetividad: {subjetividad}")

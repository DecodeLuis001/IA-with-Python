import speech_recognition as sr

def reconocimiento_por_voz():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Di algo:")
        recognizer.adjust_for_ambient_noise(source)  # Ajustar para ruido ambiental
        audio = recognizer.listen(source)

    try:
        print("Google Speech Recognition piensa que dijiste: \n" + recognizer.recognize_google(audio, language='es'))
    except sr.UnknownValueError:
        print("Google Speech Recognition no pudo entender la entrada")
    except sr.RequestError as e:
        print("Error en la solicitud a Google Speech Recognition; {0}".format(e))

if __name__ == "__main__":
    reconocimiento_por_voz()

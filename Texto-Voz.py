from gtts import gTTS
import os
import pyttsx3

def texto_a_voz(texto, idioma='es'):
    # Crear un objeto gTTS
    tts = gTTS(text=texto, lang=idioma, slow=False)
    
    # Guardar el audio en un archivo temporal
    audio_path = "temp_audio.mp3"
    tts.save(audio_path)
    
    # Reproducir el audio
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Velocidad de reproducción
    engine.setProperty('voice', engine.getProperty('voices')[0].id)  # Seleccionar la primera voz
    engine.say("Convirtiendo texto a voz.")
    engine.runAndWait()
    
    # Reproducir el archivo de audio
    os.system("start " + audio_path)

if __name__ == "__main__":
    # Obtener el texto de entrada
    texto_a_convertir = input("Introduce el texto que deseas convertir a voz: ")
    
    # Seleccionar el idioma (por defecto, español)
    idioma_seleccionado = input("Introduce el código del idioma (por ejemplo, 'es' para español): ") or 'es'
    
    # Convertir y reproducir el texto a voz
    texto_a_voz(texto_a_convertir, idioma_seleccionado)

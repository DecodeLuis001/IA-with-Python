import spacy
from collections import Counter

def analizar_comportamiento_social(texto):
    # Cargar el modelo de spaCy en español
    nlp = spacy.load("es_core_news_sm")

    # Procesar el texto con spaCy
    doc = nlp(texto)

    # Obtener las entidades nombradas (NER)
    entidades_nombradas = [ent.text for ent in doc.ents]

    # Obtener las partes del discurso (POS)
    partes_discurso = [token.pos_ for token in doc]

    # Obtener los sustantivos y verbos
    sustantivos = [token.text for token in doc if token.pos_ == "NOUN"]
    verbos = [token.text for token in doc if token.pos_ == "VERB"]

    # Contar las entidades nombradas, partes del discurso, sustantivos y verbos
    conteo_entidades = Counter(entidades_nombradas)
    conteo_pos = Counter(partes_discurso)
    conteo_sustantivos = Counter(sustantivos)
    conteo_verbos = Counter(verbos)

    # Mostrar los resultados
    print("Entidades Nombradas:", conteo_entidades)
    print("Partes del Discurso:", conteo_pos)
    print("Sustantivos:", conteo_sustantivos)
    print("Verbos:", conteo_verbos)

if __name__ == "__main__":
    # Solicitar al usuario que ingrese el texto del posteo del foro
    texto_posteo = input("Ingrese el texto del posteo del foro: ")

    # Analizar el comportamiento social
    analizar_comportamiento_social(texto_posteo)

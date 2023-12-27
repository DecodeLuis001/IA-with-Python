from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from googletrans import Translator

def traducir_texto(texto, idioma_destino='en'):
    translator = Translator()
    texto_traducido = translator.translate(texto, dest=idioma_destino)
    return texto_traducido.text

def resumir_pagina_web(url, idioma_origen='auto', idioma_destino='en', oraciones=5):
    # Parsear el contenido HTML de la página web
    parser = HtmlParser.from_url(url, Tokenizer("spanish"))
    # Obtener el texto de la página
    texto_pagina = ' '.join(str(sent) for sent in parser.document.sentences)

    # Traducir el texto al idioma de destino
    if idioma_origen != 'auto':
        texto_pagina = traducir_texto(texto_pagina, idioma_destino)

    # Utilizar el resumidor LSA
    summarizer = LsaSummarizer()
    # Obtener las oraciones resumidas
    resumen = summarizer(parser.document, oraciones)

    # Devolver el resumen como texto
    return ' '.join(str(oracion) for oracion in resumen)

if __name__ == "__main__":
    # Reemplazar con la URL de la página web que deseas resumir
    url_pagina_web = "https://example.com"
    
    # Configurar idiomas
    idioma_origen = input("Introduce el idioma de origen (por ejemplo, 'es' para español): ")
    idioma_destino = input("Introduce el idioma de destino (por ejemplo, 'en' para inglés): ")

    # Resumir la página web
    resumen_texto = resumir_pagina_web(url_pagina_web, idioma_origen, idioma_destino)
    print("Resumen de la página web:")
    print(resumen_texto)

from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def resumir_pagina_web(url, oraciones=5):
    # Parsear el contenido HTML de la página web
    parser = HtmlParser.from_url(url, Tokenizer("spanish"))
    # Utilizar el resumidor LSA
    summarizer = LsaSummarizer()
    # Obtener las oraciones resumidas
    resumen = summarizer(parser.document, oraciones)
    # Devolver el resumen como texto
    return ' '.join(str(oracion) for oracion in resumen)

if __name__ == "__main__":
    # Reemplazar con la URL de la página web que deseas resumir
    url_pagina_web = "https://example.com"
    resumen_texto = resumir_pagina_web(url_pagina_web)
    print("Resumen de la página web:")
    print(resumen_texto)

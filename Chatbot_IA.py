from flask import Flask, request, render_template
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

app = Flask(__name__)

# Crear el chatbot
chatbot = ChatBot('AI_ML_Bot')

# Entrenar el chatbot con el corpus de ChatterBot
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train('chatterbot.corpus.spanish')

# Agregar ejemplos de diálogo específicos de AI y ML
trainer.train([
    "¿Qué es inteligencia artificial?",
    "La inteligencia artificial es la simulación de procesos de inteligencia humana mediante algoritmos y modelos matemáticos.",

    "Explícame machine learning",
    "El machine learning es una rama de la inteligencia artificial que permite a las computadoras aprender patrones y tomar decisiones sin ser programadas explícitamente.",

    "¿Cuáles son los tipos de aprendizaje en machine learning?",
    "Los tipos de aprendizaje en machine learning incluyen aprendizaje supervisado, no supervisado y por refuerzo.",

    "¿Cuál es el algoritmo más común en machine learning?",
    "El algoritmo más común en machine learning es el algoritmo de regresión lineal para problemas de regresión y el algoritmo de clasificación de vecinos más cercanos (KNN) para problemas de clasificación."
])

# Definir rutas para la página web
@app.route("/")
def home():
    return render_template("index_ai_ml.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.form["user_message"]
    response = chatbot.get_response(user_message)
    return {"response": str(response)}

if __name__ == "__main__":
    app.run(debug=True)

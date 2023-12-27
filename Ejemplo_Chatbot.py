from flask import Flask, request, render_template
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

app = Flask(__name__)

# Crear el chatbot
chatbot = ChatBot('PagosBot')

# Entrenar el chatbot con el corpus de ChatterBot
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train('chatterbot.corpus.spanish')

# Definir rutas para la página web
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.form["user_message"]
    response = chatbot.get_response(user_message)
    return {"response": str(response)}

if __name__ == "__main__":
    app.run(debug=True)

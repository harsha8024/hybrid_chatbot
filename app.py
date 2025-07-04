from flask import Flask, request, jsonify
from chatbot.router import hybrid_chatbot
from chatbot.router import generative_bot

app = Flask(__name__)
app.register_blueprint(hybrid_chatbot)

@app.route('/')
def index():
    return "Hybrid Chatbot is running!"

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = generative_bot(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)


from flask import Flask, request, jsonify
from chatbot.router import hybrid_chatbot

app = Flask(__name__)

@app.route('/')
def index():
    return "Hybrid Chatbot is running!"

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = hybrid_chatbot(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)


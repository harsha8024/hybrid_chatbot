from flask import Blueprint, request, jsonify
from chatbot.generative import generative_bot
from chatbot.retrieval_based import retrieval_bot

hybrid_chatbot = Blueprint("hybrid_chatbot", __name__)

@hybrid_chatbot.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message")

    # retrieval first
    answer = retrieval_bot(user_msg)

    if not answer:
        answer = generative_bot(user_msg)

    if not answer:
        answer = "Sorry, I couldn't understand that."

    return jsonify({"response": answer})

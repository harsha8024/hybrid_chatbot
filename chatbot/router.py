from flask import Blueprint, request, jsonify
from chatbot.generative import generative_bot
from chatbot.retrieval_based import retrieval_bot

hybrid_chatbot = Blueprint("hybrid_chatbot", __name__)

@hybrid_chatbot.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").strip()
    user_msg_lower = user_msg.lower()

    # Keywords to trigger generative bot
    generative_keywords = [
        "percent", "average", "compare", "total",
        "overspend", "budget", "most", "highest", "expenditure", "spending","trend"
    ]

    # Check if user message matches any keyword
    use_generative = any(keyword in user_msg_lower for keyword in generative_keywords)

    # Route to correct bot
    if use_generative:
        answer = generative_bot(user_msg)
    else:
        answer = retrieval_bot(user_msg)
        if not answer:
            answer = generative_bot(user_msg)

    if not answer:
        answer = "Sorry, I couldn't understand that."

    return jsonify({"response": answer})

from chatbot.rule_based import rule_based_bot
from chatbot.retrieval_based import retrieval_bot
from chatbot.generative import generative_bot

def hybrid_chatbot(user_input):
    # 1. Rule-based
    response = rule_based_bot(user_input)
    if response:
        return response

    # 2. Retrieval-based
    response = retrieval_bot(user_input)
    if response:
        return response

    # 3. Generative (fallback)
    return generative_bot(user_input)


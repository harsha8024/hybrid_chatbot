def rule_based_bot(user_input):
    user_input = user_input.lower()
    
    if "hi" in user_input or "hello" in user_input:
        return "Hello! How can I assist you with financial data today?"
    elif "help" in user_input:
        return "You can ask me things like 'What was the profit in 2022?' or 'Show me revenue for 2021.'"
    elif "exit" in user_input:
        return "Goodbye!"
    
    return None  # No rule matched


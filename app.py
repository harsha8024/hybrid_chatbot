from flask import Flask, request, jsonify
from chatbot.router import hybrid_chatbot
from chatbot.router import generative_bot
import pandas as pd
import logging
import os
# os.environ["FLASK_DEBUG"] = "development"
# os.environ["PYTHONUNBUFFERED"] = "1"


logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
# app.register_blueprint(hybrid_chatbot)

@app.route('/')
def index():
    return "Hybrid Chatbot (DEBUG MODE) is running!"

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    company_id=request.json['company_id']
    print(f"📥 Received request for company ID: {company_id}", flush=True)
    if not company_id:
        return jsonify({"response": "Missing company_id"}), 400

    try:
        path = f"data/{company_id}/samplepnl.xlsx"
        print(f"📂 Trying to load: {path}", flush=True)
        df = pd.read_excel(path, skiprows=3)
    except FileNotFoundError:
        return jsonify({"response": f"No spreadsheet found for company_id {company_id}"}), 404
    except Exception as e:
        return jsonify({"response": f"Failed to load company data: {str(e)}"}), 500


    print(f"📞 Calling generative_bot for company_id {company_id}", flush=True)
    response_text = generative_bot(user_input, df=df, use_llm=True)

    return jsonify({
        "response": response_text,
        # "debug_file_used": used_path,
        # "sample_data": df.head(2).to_dict(orient="records")  # for verification
    })

if __name__ == "__main__":
    import os
    os.environ["FLASK_DEBUG"] = "development"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Ensure reloader and debugger don’t interfere
    app.run(debug=True, use_reloader=False)



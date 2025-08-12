from flask import Flask, request, jsonify, render_template
from chatbot.generative import generative_bot,analyze_transaction_csv, extract_core_questions
import logging
import os
from flasgger import Swagger
from schema_utils import get_db_schema
from db_utils import get_all_categories, fetch_spending_for_category
from dotenv import load_dotenv
from chatbot.langchain_agent import answer_from_db_with_langchain
from flask_cors import CORS

# Load environment variables
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

# --- ADDED: Function to format the database response ---
def format_spending_response(data, category_name):
    """
    Takes the list of transaction dictionaries from the database
    and formats it into a human-readable summary.
    """
    if not data or not isinstance(data, list):
        return f"No spending data was found for {category_name}."

    month_map = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }

    total_spend = 0.0
    monthly_details = []

    for row in data:
        actual_value = row.get('actual_value', 0)
        total_spend += actual_value
        
        if actual_value > 0:
            month_name = month_map.get(row.get('mon'), 'Unknown Month')
            monthly_details.append(f"- {month_name}: ₹{actual_value:,.2f}")

    if total_spend == 0:
        return f"There was no spending recorded for {category_name} in the given period."

    summary = f"You spent a total of **₹{total_spend:,.2f}** on {category_name}."
    
    if monthly_details:
        summary += "\n\nHere is the monthly breakdown:\n" + "\n".join(monthly_details)
        
    return summary
# --- END of ADDED function ---

def extract_category_from_input(user_input, known_categories):
    """
    Attempts to find a category mentioned in user's input by matching known categories.
    Returns the matched category or None.
    """
    user_input_lower = user_input.lower()
    for category in known_categories:
        if category.lower() in user_input_lower:
            return category
    return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    
    """
      Chat endpoint for financial queries
      ---
      parameters:
        - in: body
          name: body
          required: true
          schema:
            type: object
            properties:
              user_input:
                type: string
                example: How much did I spend on Services?
              company_id:
                type: string
                example: "1"
      responses:
        200:
          description: Successful response with AI answer
          schema:
            type: object
            properties:
              response:
                type: string
                example: "Just an example output"
      """

    data = request.get_json()
    user_input = data.get("user_input")
    company_id = data.get("company_id")

    if not user_input or not company_id:
        return jsonify({"error": "Missing 'user_input' or 'company_id'"}), 400
    
    try:
        company_id = int(company_id)
    except ValueError:
        return jsonify({"error": "'company_id' must be a number"}), 400

    # --- NEW, UNIFIED LOGIC FLOW ---

    # 1. PRE-PROCESSING STEP: Clean long paragraphs first
    final_input = user_input
    if len(user_input) > 150: # Increased threshold slightly
        final_input = extract_core_questions(user_input)

    # 2. ROUTER STEP: Check for specific file analysis requests
    response_data = ""
    # 2. ROUTER STEP: Check which bot to use
    if "blueberry" in final_input.lower():
        # If the user mentions the specific file, call the CSV analyzer.
        print("INFO: 'Blueberry Data' keyword detected. Routing to CSV analyzer.")
        # Get the directory where your app.py script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Build a full, reliable path to the CSV file inside the 'data' folder
        filepath = os.path.join(base_dir, 'data', 'Blueberry Data.csv')
        response_data = analyze_transaction_csv(filepath, company_id)
    else:
        # For ALL other questions, default to the main database bot
        print("INFO: No specific file keyword. Routing to default database bot (generative_bot).")
        
        # --- CORRECTED to call generative_bot for the database ---
        schema = get_db_schema(DB_HOST, DB_NAME, DB_USER, DB_PASS)
        response_data = generative_bot(
            final_input,
            company_id=company_id,
            db_schema=schema,
            use_llm=True
        )
        # response_data = answer_from_db_with_langchain(final_input, company_id)
    
    # 3. Return the final response
    return jsonify({"response": response_data})
    

if __name__ == "__main__":
    
    os.environ["FLASK_DEBUG"] = "development"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Ensure reloader and debugger don’t interfere
    app.run(debug=True, use_reloader=False)



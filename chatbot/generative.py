import pandas as pd
import re
import os
import fuzzywuzzy
import openpyxl
from rapidfuzz import process
from fuzzywuzzy import process
import requests
import json
import psycopg2
from dotenv import load_dotenv
from schema_utils import get_db_schema
from scipy.stats import linregress
from db_utils import run_simple_query_with_llm, get_db_connection
import logging
import numpy as np
import string


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)
chat_history = []
memory = {
    "intent": None,
    "category": None
}

# load_dotenv()
# DB_HOST = os.getenv("DB_HOST")
# DB_PORT = os.getenv("DB_PORT")
# DB_NAME = os.getenv("DB_NAME")
# DB_USER = os.getenv("DB_USER")
# DB_PASS = os.getenv("DB_PASS")


# # Load Excel and setup
# excel_path = "data/samplepnl.xlsx"
# df = pd.read_excel(excel_path, skiprows=3)

# Clean up columns

import pandas as pd


def extract_core_questions(paragraph: str) -> str:
    """
    Uses the LLM to extract direct questions from a long paragraph.
    """
    print("INFO: Paragraph detected. Using LLM to extract core questions...")

    # A specific prompt that tells the LLM its task
    extraction_prompt = f"""
From the following text, please extract only the direct financial questions being asked.
- Do not answer the questions.
- Do not add any extra text or explanations.
- If there are multiple questions, list each one on a new line, joined by the word "and".

Example:
Text: "Hey, I was looking over my financials from last quarter and things seem a bit weird. My manager asked me for a report, so I need to figure out what is the trend for Services before our meeting tomorrow. Can you help?"
Extracted Questions: "what is the trend for Services"

Here is the text to analyze:
"{paragraph}"

Extracted Questions:
"""

    try:
        # Call your Ollama/LLaMA 3 API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": extraction_prompt,
                "stream": False
            }
        )
        response.raise_for_status() # Raise an exception for bad status codes
        
        # The LLM's response should be the clean, extracted question(s)
        cleaned_question = response.json().get("response", "").strip()
        print(f"INFO: Extracted question(s): '{cleaned_question}'")
        return cleaned_question
        
    except Exception as e:
        print(f"ERROR: Failed to extract questions with LLM: {e}")
        # If extraction fails, we can fall back to using the original paragraph
        return paragraph


def analyze_transaction_csv(filepath, company_id):
    """
    Loads and analyzes a transactional CSV file for a specific company ID,
    using the 'ledger_type' column to summarize income and expenses.
    """
    try:
        df = pd.read_csv(filepath)

        # --- Validation Block (no changes here) ---
        required_columns = ['tenant_company_id', 'ledger_type', 'category_name', 'actual_value']
        if not all(col in df.columns for col in required_columns):
            return f"Error: The CSV file is missing one of the required columns: {required_columns}"
        df = df[df['tenant_company_id'] == int(company_id)].copy()
        if df.empty:
            return f"Error: This file does not contain any data for company ID '{company_id}'."

        # --- Data Cleaning ---
        df['actual_value'] = pd.to_numeric(df['actual_value'], errors='coerce').fillna(0)
        df['ledger_type'] = df['ledger_type'].str.lower()

        # --- New Analysis Logic using 'ledger_type' ---
        income_df = df[df['ledger_type'] == 'income']
        expenses_df = df[df['ledger_type'] == 'expense'].copy() # Use .copy() to avoid warnings

        # --- THIS IS THE FIX ---
        # 1. Create a new column with the absolute expense values
        expenses_df['abs_expense'] = expenses_df['actual_value'].abs()

        # 2. Calculate totals using the new, clean column
        total_income = income_df['actual_value'].sum()
        total_expenses = expenses_df['abs_expense'].sum()
        net_result = total_income - total_expenses

        # 3. Group by category and sum the new, clean column
        top_5_expenses = expenses_df.groupby('category_name')['abs_expense'].sum().sort_values(ascending=False).head(5)
        # --- END OF FIX ---
        
        # --- Formatting the Output (no changes needed here) ---
        response = f"Here is the financial summary for company ID {company_id} from the file:\n\n"
        response += f"  - **Total Income:** ₹{total_income:,.2f}\n"
        response += f"  - **Total Expenses:** ₹{total_expenses:,.2f}\n"
        response += f"  - **Net Result:** ₹{net_result:,.2f}\n"

        if not top_5_expenses.empty:
            response += "\n**Top 5 Expense Categories:**\n"
            for category, total in top_5_expenses.items():
                response += f"- {category}: ₹{total:,.2f}\n"
        
        return response

    except FileNotFoundError:
        return f"Error: The file was not found at the path '{filepath}'"
    except Exception as e:
        return f"An error occurred while analyzing the file: {str(e)}"


def extract_category_for_trend(user_input):
    match = re.search(r"(?:trend\s+for|how\s+has|how\s+did|change\s+in|evolve\s+in|pattern\s+of|progress\s+of)\s+([a-zA-Z\s]+)", user_input, re.IGNORECASE)
    if match:
        return match.group(1).strip().title()
    return None



def generate_spreadsheet_summary(df, months):
    summary_lines = []
    for _, row in df.iterrows():
        category = row["Category"]
        monthly_values = row[months].tolist()
        monthly_summary = ", ".join(f"{month}: ₹{val:.2f}" for month, val in zip(months, monthly_values))
        summary_lines.append(f"{category} → {monthly_summary}")
    
    return "\n".join(summary_lines)



# def clean_llm_sql_output(raw_sql):
#     """
#     Cleans the raw output from the LLM to extract only the SQL query.
#     Handles markdown code blocks (e.g., ```sql ... ```).
#     """
#     # Pattern to find content within ```sql ... ``` or ``` ... ```
#     # re.DOTALL allows '.' to match newlines
#     match = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw_sql, re.DOTALL | re.IGNORECASE)

#     if match:
#         # If a match is found, return the captured group
#         return match.group(1).strip()

#     # If no markdown block is found, just return the raw string after stripping whitespace
#     return raw_sql.strip()


def call_llama3(user_input, df=None, months=None, history=None):

    
    # try:
    #     response = requests.post(
    #         "http://localhost:11434/api/generate",  # Ollama default
    #         json={"model": "llama3", "prompt": prompt, "stream": False}
    #     )
    #     response.raise_for_status()
    #     result = response.json()
    #     return result["response"]
    # except Exception as e:
    #     return f"LLM Call Error: {str(e)}"

    if history is None:
        history = []

    # Build a formatted history string for the prompt
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    context = ""
    if df is not None and months is not None:
        context = generate_spreadsheet_summary(df, months)
        # print(context)
    
    prompt = f"""
You are a helpful and context-aware financial assistant.
Analyze the user's new question based on the spreadsheet data and the conversation history provided below.

--- Spreadsheet Data ---
{context}
--- End of Data ---

--- Conversation History ---
{formatted_history}
--- End of History ---

Based on the data and history, answer questions about:
- Which categories they spent the most or least on
- Where they can cut costs
- Trends or changes in spending
- Budget or category comparisons

Only use the spreadsheet values for your answer. Be specific and helpful.

User: {user_input}
Assistant:
"""
    # print("=== PROMPT SENT TO LLaMA ===")
    # print(prompt)


    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    
    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        return "Error communicating with the LLaMA model."
    


def build_prompt_from_history(history, user_input):
    history.append(f"User: {user_input}")
    prompt = "\n".join(history) + "\nAssistant:"
    return prompt

def find_file_with_category(company_id, category):
    folder_path = os.path.join(str(company_id))

    if not os.path.exists(folder_path):
        return None, None

    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
                df.columns = df.columns.str.strip()
                if category in df[df.columns[0]].values:
                    return file_path, df
            except Exception:
                continue
    return None, None

def extract_monthly_values(df, category):
    try:
        months = [col for col in df.columns if re.match(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$", col)]
        category_row = df[df[df.columns[0]] == category]
        if category_row.empty:
            return None
        return category_row[months].to_dict(orient='records')[0]
    except Exception:
        return None
    
def load_spreadsheet_with_category_header(file_path):
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active
    for idx, row in enumerate(ws.iter_rows(values_only=True)):
        if row and "Category" in row:
            return pd.read_excel(file_path, skiprows=idx)
    raise ValueError(f"No row with 'Category' found in: {file_path}")


def handle_spend_on_query(df, months, category, category_map):
    """
    Handles "How much did I spend on [category]?" and provides a detailed monthly breakdown.
    """
    available_categories = df["Category_lower"].unique()
    match_result = process.extractOne(category.lower(), available_categories)

    if not match_result:
        return f"Sorry, I couldn't find a matching category for '{category.title()}'."
    
    match, score = match_result

    if score < 85:
        return f"Sorry, I couldn't find a confident match for '{category.title()}' in your data."

    best_match_lower = match_result[0]
    actual_category_casing = category_map.get(best_match_lower, best_match_lower.title())
    row = df.loc[df["Category_lower"] == best_match_lower]

    if row.empty:
        return f"Sorry, no data was found for {actual_category_casing}."

    # --- ADD THIS DEBUGGING BLOCK ---
    print("\n--- DEBUGGING: Inspecting the DataFrame row ---")
    # This will print the entire row that was found for the category
    print(f"Data for matched category '{best_match_lower}':")
    print(row.to_string())
    print("--- END DEBUGGING ---\n")
    # --- END OF DEBUGGING BLOCK ---

    total_value = row["Total"].iloc[0]

    monthly_breakdown = []
    for month in months:
        month_value = row[month].iloc[0]
        if month_value > 0:
            monthly_breakdown.append(f"* {month}: ₹{month_value:,.2f}")
    
    response = (
        f"According to the spreadsheet, for **{actual_category_casing}** you spent:\n\n"
        + "\n".join(monthly_breakdown)
        + f"\n\nIn total, you spent **₹{total_value:,.2f}**."
    )
    return response

def handle_spend_most(df, months, **_):
    df["Total"] = df[months].sum(axis=1)
    top_category = df.loc[df["Total"].idxmax(), "Category"]
    top_amount = df["Total"].max()
    return f"You spent the most on {top_category}: ₹{top_amount:,.2f}"

def handle_highest_expense_month(df, months, **_):
    monthly_totals = df[months].sum()
    top_month = monthly_totals.idxmax()
    top_amount = monthly_totals.max()
    return f"Your expenses were highest in {top_month}: ₹{top_amount:,.2f}"

def handle_steepest_increase(df, months, **_):
    month_diffs = df[months].diff(axis=1)
    if month_diffs.empty:
        return "Not enough data to calculate an increase."
    max_increases = month_diffs.max(axis=1)
    top_index = max_increases.idxmax()
    top_category = df.loc[top_index, "Category"]
    top_value = max_increases.loc[top_index]
    top_diff_row = month_diffs.loc[top_index].dropna()
    max_month = top_diff_row.idxmax()
    prev_month_idx = months.index(max_month) - 1
    prev_month = months[prev_month_idx] if prev_month_idx >= 0 else "a previous month"
    return (f"The category with the steepest single-month increase was **{top_category}**, "
            f"which rose by ₹{top_value:,.2f} from {prev_month} to {max_month}.")

def handle_trend_query(df, months, category, **_):
    print("--- CHECK 2: Entered handle_trend_query ---")
    """
    Handles trend analysis for a category, with added debugging.
    """
    # --- ADD THIS FINAL DEBUGGING BLOCK ---
    print("\n--- DEBUGGING INSIDE handle_trend_query ---")
    
    # 1. Show the exact category string we received from the user input
    print(f"Attempting to find a match for this category: {repr(category.lower())}")
    
    # 2. Show the list of all available clean categories from the DataFrame
    print("\nList of available clean categories in the DataFrame:")
    # We use repr() to reveal any hidden spaces or special characters
    available_cats = df['Category_lower'].unique().tolist()
    for cat in available_cats:
        print(f"  - {repr(cat)}")
        
    print("--- END DEBUGGING ---\n")
    # --- END OF DEBUGGING BLOCK ---

    # This is the line that is failing to find a match
    row = df[df["Category_lower"] == category.lower()]

    if not row.empty:
        y = row[months].values.flatten()
        x = np.arange(len(months))
        slope, _, _, _, _ = linregress(x, y)
        
        if slope > 50: trend = "strongly increasing"
        elif slope > 0: trend = "increasing"
        elif slope < -50: trend = "strongly decreasing"
        elif slope < 0: trend = "decreasing"
        else: trend = "stable"
        
        original_casing_category = row['Category'].iloc[0]
        return f"The trend for **{original_casing_category}** is **{trend}** over the months."
    
    return f"Sorry, I couldn't find data to analyze the trend for '{category.title()}'."

def handle_anomalies(df, months, **_):
    anomalies = []
    for _, row in df.iterrows():
        values = row[months].values
        mean = np.mean(values)
        std = np.std(values)
        if std > 0:
            z_scores = (values - mean) / std
            spike_months = [months[i] for i, z in enumerate(z_scores) if abs(z) > 2]
            if spike_months:
                anomalies.append(f"{row['Category']}: {', '.join(spike_months)}")
    if anomalies:
        return "Detected unusual spending in:\n" + "\n".join(anomalies)
    return "No significant anomalies found in spending patterns."

# Add these three functions to generative.py

def handle_percent_spend_query(df, months, category, **_):
    """
    Handles questions like "What percent of my spending was on [category]?"
    """
    available_categories = df["Category_lower"].unique()
    match_result = process.extractOne(category.lower(), available_categories)

    if not match_result:
        return f"Sorry, I couldn't find any category matching '{category.title()}'."

    # --- ADD THIS DEBUGGING BLOCK ---
    print("\n--- DEBUGGING FUZZY MATCH ---")
    print(f"User's raw category input: '{category}'")
    print(f"Best match found in data: '{match_result[0]}'")
    print(f"Calculated similarity score: {match_result[1]}")
    print("--- END DEBUGGING ---\n")
    # --- END OF DEBUGGING BLOCK ---

    match, score = match_result
    
    if score < 85:
        return f"Sorry, I couldn't find a confident match for '{category.title()}' in your data."

    row = df[df['Category_lower'] == match]
    if not row.empty:
        total_spent = df["Total"].sum()
        if total_spent == 0:
            return "Total spending is ₹0, so a percentage cannot be calculated."
            
        category_total = row["Total"].iloc[0]
        percent = (category_total / total_spent) * 100
        
        original_casing_category = row['Category'].iloc[0]
        return f"**{original_casing_category}** accounted for **{percent:.2f}%** of your total spending."
        
    return f"Sorry, I couldn't find data for '{category.title()}' to calculate a percentage."


def handle_average_spend_query(df, months, category, **_):
    """
    Handles questions like "What was the average spend on [category]?"
    """
    available_categories = df["Category"].str.lower().unique()
    match, score= process.extractOne(category.lower(), available_categories)

    if score < 60:
        return f"Sorry, I couldn't find data for '{category.title()}' to calculate an average."

    row = df[df['Category'].str.lower() == match]
    if not row.empty:
        # Calculate the average only across months with non-zero spending for a more useful metric
        non_zero_months = row[months].loc[:, (row[months] != 0).any(axis=0)]
        if non_zero_months.empty:
             avg = 0
        else:
             avg = non_zero_months.values.flatten().mean()
        return f"The average monthly expenditure on {row['Category'].iloc[0]} was **₹{avg:,.2f}**."
    return f"Sorry, I couldn't find data for '{match.title()}'."


def handle_summary_query(df, months, **_):
    """
    Handles general requests for a summary or overview.
    """
    if "Total" not in df.columns:
        df["Total"] = df[months].sum(axis=1)

    # --- ADD THIS DEBUGGING CODE ---
    print("\n--- DEBUGGING: Inspecting the 'Total' column ---")
    print(f"Data type of 'Total' column: {df['Total'].dtype}")
    print("Contents of 'Total' column:")
    print(df[['Category', 'Total']].to_string())
    print("--- END DEBUGGING ---\n")
    # --- END OF DEBUGGING CODE ---
    
    top_3 = df.sort_values("Total", ascending=False).head(3)
    monthly_totals = df[months].sum()
    highest_month = monthly_totals.idxmax()
    lowest_month = monthly_totals.idxmin()
    
    # Calculate standard deviation to find consistency
    df_std = df.set_index("Category")[months].std(axis=1).sort_values()
    consistent = df_std.head(1).index[0]
    variable = df_std.tail(1).index[0]

    return (
        f"Here's your spending summary:\n\n"
        f"🔹 **Top 3 Categories**: {', '.join(top_3['Category'].tolist())}\n"
        f"🔹 **Highest Spend Month**: {highest_month} (₹{monthly_totals.max():,.2f})\n"
        f"🔹 **Lowest Spend Month**: {lowest_month} (₹{monthly_totals.min():,.2f})\n"
        f"🔹 **Most Consistent Spending**: {consistent}\n"
        f"🔹 **Most Variable Spending**: {variable}"
        
    )

def handle_total_category_query(df, category, category_map, **_):
    """
    Handles queries for pre-calculated total categories like "total expenses".
    """
    # The `category` we get from the regex will include the word "total"
    # We use the clean 'Category_lower' column for a reliable match
    row = df[df['Category_lower'] == category.lower()]

    if not row.empty:
        total_value = row['Total'].iloc[0]
        # Get the original casing for a clean response
        original_casing = row['Category'].iloc[0]
        return f"The value for **{original_casing}** is **₹{total_value:,.2f}**."
        
    return f"Sorry, I couldn't find a result for '{category}'."

INTENT_PATTERNS = [

    {
        "intent": "get_total_category",
        # This captures phrases that start with "total", like "total expenses"
        "pattern": re.compile(r"(?:total spendings|how much did i spend on) (.+?)(?=\s+and\b|\?|,|$)", re.IGNORECASE),
        "handler": handle_spend_on_query,
        "template": "How much did I spend on [category]",
        "requires_entity": True
    },

    {
        "intent": "get_trend",
        "pattern": re.compile(r"(?:what is the trend for|trend for|trend of|how has) (.+?)(?=\s+and\b|\?|,|$)", re.IGNORECASE),
        "handler": handle_trend_query,
        "template": "what is the trend for [category]",
        "requires_entity": True
    },
    {
        "intent": "spend_most",
        "pattern": re.compile(r"spend the most", re.IGNORECASE),
        "handler": handle_spend_most,
        "template": "where did i spend the most",
        "requires_entity": False # This intent doesn't need a category
    },
    {
        "intent": "get_percent_spend",
        "pattern": re.compile(r"(?:percent of spending on|percent on|percent of) (.+?)(?=\s+and\b|\?|,|$)", re.IGNORECASE),
        "handler": handle_percent_spend_query,
        "template": "what percent of my spendings were on [category]",
        "requires_entity": True
    },
    {
        "intent": "get_average_spend",
        "pattern": re.compile(r"(?:average monthly expenditure on|average spend on|average spending on) (.+?)(?=\s+and\b|\?|,|$)", re.IGNORECASE),
        "handler": handle_average_spend_query,
        "template": "what is the average spend on [category]",
        "requires_entity": True
    },
    # --- Intents that don't require a category ---
    {
        "intent": "get_summary",
        "pattern": re.compile(r"summary|overview|insight", re.IGNORECASE),
        "handler": handle_summary_query,
        "template": "give me a summary",
        "requires_entity": False # We tell the logic no category is needed
    },
    {
        "intent": "highest_month",
        "pattern": re.compile(r"expenses the highest|most expensive month|which month did I spend the most", re.IGNORECASE),
        "handler": handle_highest_expense_month,
        "template": "which month was the most expensive",
        "requires_entity": False
    },
    {
        "intent": "steepest_increase",
        "pattern": re.compile(r"steepest increase", re.IGNORECASE),
        "handler": handle_steepest_increase,
        "template": "what was the steepest increase",
        "requires_entity": False
    },
    {
        "intent": "anomalies",
        "pattern": re.compile(r"anomalies|unusual spending", re.IGNORECASE),
        "handler": handle_anomalies,
        "template": "show me any unusual spending",
        "requires_entity": False
    }
    # ... you can continue to add more intents from your elif blocks here
]

def hybrid_chatbot_logic(user_input, df, months, category_map, history=None, use_llm=True):
    """
    Identifies potential intents and uses a strict similarity score to decide
    whether to use a fast, hardcoded function or fall back to the LLM.
    """
    input_lower = user_input.lower().strip()
    potential_tasks = []

    # 1. Find all potential intents that match keywords in the input
    for intent in INTENT_PATTERNS:
        matches = intent["pattern"].finditer(input_lower)
        for match in matches:
            entity = None
            if intent.get("requires_entity", True):
                entity = match.group(1).strip().rstrip(string.punctuation)
            
            potential_tasks.append({
                "handler": intent["handler"],
                "template": intent["template"],
                "entity": entity
            })

    if not potential_tasks:
        # If no keywords match at all, go straight to the LLM
        return call_llama3(user_input, df=df, months=months, history=history)

    # 2. Check if the question is simple enough for a hardcoded function
    # We will only proceed if there is ONE clear and simple intent.
    if len(potential_tasks) == 1:
        task = potential_tasks[0]
        template = task["template"]
        
        # Create a "perfect" version of the question using the user's category
        if task["entity"]:
            filled_template = template.replace("[category]", task["entity"])
        else:
            filled_template = template

        # Calculate the similarity between the user's question and the ideal template
        similarity_score = process.fuzz.ratio(input_lower, filled_template.lower())
        
        print(f"DEBUG: Similarity score is {similarity_score} for '{input_lower}'")

        # 3. Make the decision
        # If the score is high, the question is simple. Use the hardcoded function.
        if similarity_score > 95:
            print("INFO: High similarity. Using hardcoded function.")
            return task["handler"](df=df, months=months, category=task["entity"], category_map=category_map)

    # If there are multiple potential questions, or if the similarity score is low,
    # the question is too complex for our simple rules. Use the LLM.
    print("INFO: Low similarity or multiple intents detected. Routing to LLM.")
    return call_llama3(user_input, df=df, months=months)

def classify_intent_with_llm(user_input):
    """
    Uses the LLM to determine if a question is about the PNL or the Balance Sheet.
    """
    print("INFO: Using LLM to classify user intent...")
    
    classification_prompt = f"""
You are a financial analyst routing assistant. Your task is to determine which financial document is needed to answer the user's question.
Respond with only a single word: 'pnl', 'balance_sheet', or 'unknown'.

User question: "{user_input}"

Classification:
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": classification_prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        classification = response.json().get("response", "").strip().lower()
        print(f"INFO: LLM classified intent as: '{classification}'")
        return classification
    except Exception as e:
        print(f"ERROR: LLM classification failed: {e}")
        return 'unknown'

# ==============================================================================
# --- Main Controller Function ---
# This is the primary function your app.py will call.
# ==============================================================================

def generative_bot(user_input, company_id, history=None, db_schema=None, use_llm=True):
#     """
#     Orchestrates the two-step LLM chain:
#     1. Generate a SQL query to get data.
#     2. Summarize the resulting data into a final answer.
#     """
#     if not use_llm:
#         return "LLM is disabled."
        
#     # --- STEP 1: TEXT-TO-SQL ---
#     final_sql = generate_sql_with_llm(user_input, company_id, db_schema)
    
#     # Execute the generated SQL to get the raw data
#     raw_data_results = execute_sql_query(final_sql)

#     # --- ADD THIS DEBUGGING BLOCK ---
#     print("\n--- DEBUG: Checking data before summarization ---")
#     print(f"Data received from execute_sql_query: {raw_data_results}")
    
#     # --- Check the results from the database ---
#     # If the result is not a list, it's an error or info message (e.g., "No data found.")
#     if not isinstance(raw_data_results, list):
#         print("--> CONCLUSION: Condition is TRUE. Exiting BEFORE calling the summarizer.")
#         return raw_data_results 
    
#     else:
#         print("--> CONCLUSION: Condition is FALSE. Proceeding to call the summarizer.")
    
#     # If the list is empty, we can also stop here.
#     if not raw_data_results:
#         return "I found no data that matches your request."

#     # --- STEP 2: DATA-TO-TEXT ---
#     # Pass the original question and raw data to the summarization function
#     final_answer = summarize_data_with_llm(user_input, raw_data_results)
    
#     return final_answer

# # ==============================================================================
# # --- 1. Text-to-SQL Function ---
# # ==============================================================================

# def generate_sql_with_llm(user_input, company_id, db_schema):
#     """Builds a prompt, calls the LLM, and returns a clean SQL query string."""
#     prompt = f"""
# You are an expert PostgreSQL data analyst. Your task is to write a single, valid SQL query to answer the user's question.

# --- Instructions & Column Guide ---
# - `account_categories` table (alias `ac`):
#   - Use `ac.qbo_category` to match category names like 'Office supplies'.
# - `budget_transaction` table (alias `bt`):
#   - Use `bt.actual_value` for all spending calculations.
#   - Use `bt.tenant_company_id` when filtering by the user's company.
# - Always filter by `tenant_company_id = '{company_id}'`.

# **CRITICAL RULE for Percentages:**
# - When the user asks for the "percent of spending on [a specific category]", you MUST include a `WHERE` clause to filter for that category (e.g., `WHERE ac.qbo_category ILIKE 'Office supplies'`).
# - The calculation for the percentage MUST be: `(SUM(bt.actual_value) / (SELECT SUM(actual_value) FROM budget_transaction WHERE tenant_company_id = '{company_id}')) * 100`.

# **CRITICAL RULE for Trends:**
# - To calculate a "trend", you MUST use the function `regr_slope(Y, X)`.
# - The correct syntax is EXACTLY: `regr_slope(bt.actual_value, bt.mon) AS trend_slope`.
# - You MUST use this function with a `GROUP BY` clause. Do NOT use it with `OVER()`.

# **CRITICAL RULE for JOINs:**
# - Use this exact JOIN to connect transactions to categories:
# `... JOIN account_categories ac ON (bt.category_id = ac.id OR bt.sub_category_id = ac.id)`
# - When using a JOIN, every column must be prefixed with its table alias.

# **CRITICAL RULE for Rounding:**
# - To round a calculation to 2 decimal places, you MUST use this exact syntax:
#   `ROUND((...your calculation...)::numeric, 2)`

# --- Schema ---
# {db_schema}
# --- User Question ---
# {user_input}
# SQL Query:
# """
#     logger.info("Sending prompt to LLM for SQL generation...")
#     try:
#         llama_response = requests.post(
#             "http://localhost:11434/api/generate",
#             json={"model": "llama3", "prompt": prompt, "stream": False}
#         )
#         llama_response.raise_for_status()
#         raw_sql = llama_response.json().get("response", "")
#         logger.info(f"Raw SQL from LLM:\n{raw_sql}")
#         return clean_llm_sql_output(raw_sql)
#     except Exception as e:
#         logger.exception("LLM call error for SQL generation")
#         return None # Return None on failure

# # ==============================================================================
# # --- 2. Database Execution Function ---
# # ==============================================================================

# def execute_sql_query(sql_query):
#     """
#     Connects to the DB using the central utility, executes a query, 
#     and returns the formatted results.
#     """
#     if not sql_query or not sql_query.strip():
#         return "I'm sorry, I couldn't construct a valid database query."
        
#     conn = None # Initialize conn to None
#     try:
#         # --- MODIFIED: Use the central connection function ---
#         conn = get_db_connection()
#         cur = conn.cursor()
        
#         logger.info(f"---\nEXECUTING SQL QUERY:\n{sql_query}\n---")
#         cur.execute(sql_query)
        
#         # ... (rest of the function is the same) ...
        
#         rows = cur.fetchall()
#         colnames = [desc[0] for desc in cur.description]
        
#         logger.info(f"SQL executed successfully, returned {len(rows)} rows")
        
#         if not rows or (len(rows) == 1 and rows[0][0] is None):
#             return "I found no data that matches your request."
            
#         return [dict(zip(colnames, row)) for row in rows]

#     except Exception as e:
#         logger.exception("SQL execution error")
#         return f"Query Execution Error: {str(e)}"
#     finally:
#         if conn:
#             conn.close() # Ensure connection is closed

# # ==============================================================================
# # --- 3. Data-to-Text Summarization Function ---
# # ==============================================================================

# # In generative.py, you can add this function above summarize_data_with_llm

# def format_metric(value):
#     """Formats a value as currency if it's a number, otherwise returns it as is."""
#     if isinstance(value, (int, float)):
#         return f"₹{value:,.2f}" # Adds currency symbol, commas, and decimal places
#     return str(value) # Returns non-numeric values like "N/A" as a string

# def summarize_data_with_llm(original_question, database_results):
#     """
#     Takes raw data and asks the LLM to summarize it. It now smartly handles
#     both pre-aggregated data (like SUM) and raw monthly data.
#     """
#     # --- NEW: Smartly extract pre-calculated metrics ---
#     total_spending = 0
#     average_spending = 0
#     num_spending_months = 0

#     # Check the structure of the data to get the right numbers
#     if database_results:
#         first_row = database_results[0]
#         # Case 1: The SQL query already calculated the total (e.g., SUM)
#         if 'total_spent' in first_row:
#             total_spending = first_row.get('total_spent', 0)
#             # Average and month count aren't applicable for a simple total
#             average_spending = "N/A"
#             num_spending_months = "N/A"
#         # Case 2: The SQL query returned a list of monthly transactions
#         elif 'actual_value' in first_row:
#             total_spending = sum(row.get('actual_value', 0) for row in database_results)
#             months_with_spending = [row for row in database_results if row.get('actual_value', 0) > 0]
#             num_spending_months = len(months_with_spending)
#             average_spending = total_spending / num_spending_months if num_spending_months > 0 else 0
#     # --- END of new logic ---

#     # Build the prompt with the correctly extracted metrics
#     summary_prompt = f"""
# You are a helpful financial assistant. Your task is to answer the user's question using ONLY the pre-calculated metrics provided below.
# **CRITICAL RULE: You MUST use the exact numbers from the "Pre-Calculated Metrics" section in your answer. Do not perform your own calculations.**

# Original Question: "{original_question}"

# --- Pre-Calculated Metrics ---
# Total Spending: {format_metric(total_spending)}
# Average Monthly Spending: {format_metric(average_spending)}
# Number of Months with Spending: {num_spending_months}

# print("\n--- DEBUG: Final Prompt Sent to Summarizer LLM ---")
#     print(summary_prompt)
#     print("---------------------------------------------------\n")

# Please provide a concise, human-readable answer to the original question using these metrics.
# """
#     logger.info("Sending final metrics to LLM for summarization...")
#     try:
#         response = requests.post(
#             "http://localhost:11434/api/generate",
#             json={"model": "llama3", "prompt": summary_prompt, "stream": False}
#         )
#         response.raise_for_status()
#         return response.json().get("response", "").strip()
#     except Exception as e:
#         logger.exception("LLM call error for summarization")
#         return f"I was able to retrieve the data but failed to generate a summary. Error was: {str(e)}"

# # ==============================================================================
# # --- Helper Function ---
# # ==============================================================================

# def clean_llm_sql_output(raw_sql):
#     """ Cleans the raw output from the LLM to extract only the SQL query. """
#     match = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw_sql, re.DOTALL | re.IGNORECASE)
#     if match:
#         return match.group(1).strip()
#     return raw_sql.strip()

    """
    A more advanced bot that loads all available data sources for a company
    and intelligently selects the correct one to answer the user's question.
    """
    print(f"✅ Entered multi-file generative_bot() for company_id={company_id}", flush=True)

    base_folder = f"data/{company_id}"
    if not os.path.isdir(base_folder):
        return f"Company ID '{company_id}' has no data folder."

    # --- 1. Load ALL available data sources into a dictionary ---
    dataframes = {}
    available_files = sorted(os.listdir(base_folder)) # Sort to ensure predictable loading

    for file in available_files:
        if file.startswith("~$"): continue # Ignore temporary Excel files
        
        filepath = os.path.join(base_folder, file)
        file_lower = file.lower()
        df_temp = None
        try:
            # --- THIS IS THE CORRECTED LOGIC ---
            # It now handles different skiprows for different company IDs.
            if file_lower.endswith(".xlsx"):
                if company_id in [1, 2]:
                    df_temp = pd.read_excel(filepath, skiprows=3)
                else:
                    df_temp = pd.read_excel(filepath) # No skiprows for new companies
            elif file_lower.endswith(".csv"):
                if company_id in [1, 2]:
                     # Assuming old CSVs also needed skipping
                    df_temp = pd.read_csv(filepath, header=3)
                else:
                    # New CSVs are read from the top
                    df_temp = pd.read_csv(filepath)
            else:
                continue
            # --- END OF CORRECTION ---

            if "pnl" in file_lower:
                print(f"INFO: Loading PNL file: {file}")
                dataframes['pnl'] = df_temp
            elif "balance_sheet" in file_lower or "_bs_" in file_lower:
                print(f"INFO: Loading Balance Sheet file: {file}")
                dataframes['balance_sheet'] = df_temp
                
        except Exception as e:
            print(f"❌ Error reading {file}: {e}", flush=True)


    if not dataframes:
        return f"No valid PNL or Balance Sheet files were found for company {company_id}."

    # --- 2. Intelligently Select the Correct DataFrame using the LLM ---
    intent = classify_intent_with_llm(user_input)

    selected_df_key = None
    if intent == 'balance_sheet' and 'balance_sheet' in dataframes:
        selected_df_key = 'balance_sheet'
    elif intent == 'pnl' and 'pnl' in dataframes:
        selected_df_key = 'pnl'
    else:
        # Default behavior if classification is 'unknown' or the file doesn't exist
        # It will prioritize the PNL file if it exists.
        selected_df_key = 'pnl' if 'pnl' in dataframes else 'balance_sheet'

    df = dataframes[selected_df_key]
    print(f"INFO: Selected '{selected_df_key}' DataFrame for this query.")

    # --- This block now runs for BOTH logic paths ---
    if df is not None:
        # 1. Define base variables from the chosen DataFrame
        df.columns = df.columns.str.strip()
        df.rename(columns={df.columns[0]: "Category"}, inplace=True, errors='ignore')
        # --- THIS IS THE CRITICAL DEBUGGING BLOCK ---
        print("\n--- DEBUGGING: Final Column Headers ---")
        # This strips whitespace from headers and then prints them
        df.columns = df.columns.str.strip()
        print("Cleaned column headers being checked by the regex:")
        # We use repr() to reveal any hidden characters or extra spaces
        print([repr(col) for col in df.columns])
        print("--- END DEBUGGING ---\n")
        # --- END OF DEBUGGING BLOCK ---
        df = df[df["Category"].notna()]
         # --- ADD THIS LINE ---
    # Creates a clean column by stripping whitespace and making it lowercase
        df["Category_lower"] = df["Category"].astype(str).str.strip().str.lower()
    
        category_map = {cat_lower: cat_orig for cat_lower, cat_orig in zip(df["Category_lower"], df["Category"])}
        months = [col for col in df.columns if isinstance(col, str) and re.match(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$", col, re.IGNORECASE)]

        if not months:
            print("INFO: New month format not found, trying old format...")
            months = [col for col in df.columns if isinstance(col, str) and re.match(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$", col, re.IGNORECASE)]

        if not months:
            return "No valid month columns found in the spreadsheet."

        # 2. Perform all data cleaning
        df[months] = df[months].apply(pd.to_numeric, errors='coerce').fillna(0)
        if "Total" not in df.columns:
            df["Total"] = df[months].sum(axis=1)
        
        df["Total"] = df["Total"].astype(str).str.replace(r'[^\d.-]', '', regex=True)
        df["Total"] = pd.to_numeric(df["Total"], errors='coerce').fillna(0)
    else:
        return f"No valid spreadsheets found for company {company_id}."

    # Final call is now safe because df, months, and category_map are always defined
    return hybrid_chatbot_logic(user_input, df, months, category_map,history=history, use_llm=use_llm)


    # # 10B. Clarification handling (e.g., "yes, I meant X")
    # if input_lower.startswith("yes") or "i meant" in input_lower:
    # # Check if previous intent was a trend request and a category was stored
    #     if memory["intent"] == "trend" and memory["category"]:
    #         category = memory["category"]
    #         row = df[df["Category"].str.lower() == category.lower()]
    #         if not row.empty:
    #             import numpy as np
    #             from scipy.stats import linregress

    #             y = row[months].values.flatten()
    #             x = np.arange(len(months))
    #             slope, _, _, _, _ = linregress(x, y)

    #             if slope > 50:
    #                 trend = "strongly increasing"
    #             elif slope > 0:
    #                 trend = "increasing"
    #             elif slope < -50:
    #                 trend = "strongly decreasing"
    #             elif slope < 0:
    #                 trend = "decreasing"
    #             else:
    #                 trend = "stable"

    #             return f"Thanks for confirming. The trend for {category} is **{trend}** over the months."
    #         else:
    #             return f"Sorry, I still couldn’t find '{category}'."
    #     elif memory["intent"] == "average" and memory["category"]:
    #         row = df[df["Category"].str.lower() == memory["category"].lower()]
    #         if not row.empty:
    #             avg = row[months].values.flatten().mean()
    #             return f"Thanks for confirming. The average monthly expenditure on {memory['category']} is ₹{avg:.2f}"
    #         else:
    #             return f"Sorry, I still couldn’t find '{memory['category']}'."
    #     else:
    #         return "Can you clarify what you're referring to?"


    # # 1. Spend the most
    # if "spend the most" in input_lower:
    #     df["Total"] = df[months].sum(axis=1)
    #     top_category = df.loc[df["Total"].idxmax(), "Category"]
    #     top_amount = df["Total"].max()
    #     return f"You spent the most on {top_category}: ₹{top_amount:.2f}"

    # # 2. Highest expense month
    # elif "expenses the highest" in input_lower or "most expensive month" in input_lower:
    #     monthly_totals = df[months].sum()
    #     top_month = monthly_totals.idxmax()
    #     top_amount = monthly_totals.max()
    #     return f"Your expenses were highest in {top_month}: ₹{top_amount:.2f}"
    

    # elif any(kw in input_lower for kw in ["total spent", "how much did i spend", "total expenditure on"]):
    #     if matched_category is None:
    #         return "I couldn't identify the category you're referring to. Try asking like: 'How much did I spend on Services?'"

    #     # Map to original formatting
    #     actual_category = category_map.get(matched_category.lower().strip(), matched_category)

    #     total_column_name = "Total"
    #     try:
    #         total_value = df.loc[df["Category"] == actual_category, total_column_name].values[0]
    #     except IndexError:
    #         total_value = None

    #     if total_value is not None:
    #         print(f"✅ Using coded total response for category: {actual_category}", flush=True)
    #         return (
    #             f"According to the spreadsheet, you spent:\n\n"
    #             + "\n".join(
    #                 [f"* {month}: ₹{df.loc[df['Category'] == actual_category, month].values[0]:.2f}"
    #                 for month in months if month in df.columns]
    #             )
    #             + f"\n\nIn total, you spent ₹{total_value:.2f} on {actual_category}."
    #         )
    #     else:
    #         return f"Sorry, no total spending found for {actual_category} in the spreadsheet."

        
    # elif "spend in" in input_lower or "spent in" in input_lower:
    #     for m in month_lookup:
    #         if m in input_lower:
    #             total = df[month_lookup[m]].sum()
    #             return f"You spent ₹{total:.2f} in {month_lookup[m]}."
    #     return "I couldn't identify the month. Try asking like: 'How much did I spend in March?'"



    # # 3. Average monthly expenditure on a category
    # elif "average monthly expenditure" in input_lower:
    #     import string
    #     match = re.search(r"average monthly expenditure on (.+)", input_lower)
    #     if match:
    #         user_category = match.group(1).strip().rstrip(string.punctuation).title()
    #         row = df[df["Category"].str.lower() == user_category.lower()]
    #         if not row.empty:
    #             avg = row[months].values.flatten().mean()
    #             memory["intent"] = "average"
    #             memory["category"] = user_category
    #             return f"The average monthly expenditure on {user_category} is ₹{avg:.2f}"
    #         else:
    #             suggestion = get_closest_category(user_category)
    #             if suggestion:
    #                 return f"I couldn't find '{user_category}'. Did you mean **{suggestion}**?"
    #             else:
    #                 available = ", ".join(available_categories)
    #                 return f"I couldn't find '{user_category}'. Available categories: {available}"

    # # 4. Compare spend on X vs Y
    # elif " vs " in input_lower or " versus " in input_lower:
    #     match = re.search(r"spend on (.+?) (?:vs|versus) (.+)", input_lower)
    #     if match:
    #         cat1 = match.group(1).strip().title()
    #         cat2 = match.group(2).strip().title()
    #         row1 = df[df["Category"].str.lower() == cat1.lower()]
    #         row2 = df[df["Category"].str.lower() == cat2.lower()]

    #         if not row1.empty and not row2.empty:
    #             total1 = row1[months].values.flatten().sum()
    #             total2 = row2[months].values.flatten().sum()
    #             return f"Total spent on {cat1}: ₹{total1:.2f}, {cat2}: ₹{total2:.2f}"
    #         else:
    #             missing = []
    #             if row1.empty:
    #                 suggestion1 = get_closest_category(cat1)
    #                 missing.append(f"{cat1} (Did you mean **{suggestion1}**?)" if suggestion1 else cat1)
    #             if row2.empty:
    #                 suggestion2 = get_closest_category(cat2)
    #                 missing.append(f"{cat2} (Did you mean **{suggestion2}**?)" if suggestion2 else cat2)
    #             return f"Couldn't find data for: {', '.join(missing)}"

    # # 5. Total expenditure in a month
    # elif "total expenditure in" in input_lower:
    #     for m in month_lookup:
    #         if m in input_lower:
    #             total = df[month_lookup[m]].sum()
    #             return f"Your total expenditure in {month_lookup[m]} was ₹{total:.2f}"
    #     return "I couldn't identify the month. Try asking like: 'Total expenditure in March'."

    # # 6. Percent of spending on a category
    
    # elif "percent" in input_lower:
    # # Try multiple flexible patterns
    #     patterns = [
    #         r"percent of.*spending.*on (.+)",
    #         r"percent.*on (.+)",
    #         r"what percent.*on (.+)",
    #         r"what percent.*was spent on (.+)",
    #         r"percent.*was on (.+)",
    #         r"how much percent.*on (.+)"
    #     ]

    #     for pat in patterns:
    #         match = re.search(pat, input_lower)
    #         if match:
    #             user_input_cat = match.group(1).strip()
    #             category = user_input_cat.title()
    #             row = df[df["Category"].str.lower() == category.lower()]

    #             if not row.empty:
    #                 total_spent = df[months].sum().sum()
    #                 category_total = row[months].values.flatten().sum()
    #                 if total_spent > 0:
    #                     percent = (category_total / total_spent) * 100
    #                     return f"{category} accounted for {percent:.2f}% of your total spending."
    #                 else:
    #                     return "Total spending is ₹0, so percentage cannot be calculated."
    #             else:
    #                 suggestion = get_closest_category(category)
    #                 return f"I couldn't find '{category}'. Did you mean **{suggestion}**?"

    #     return "I couldn't identify the category. Try asking like: 'What percent of my spending was on Rent?'"




    # # 7. Budget overspend check
    # elif "overspend" in input_lower and "budget" in input_lower:
    #     match = re.search(r"overspend on (.+?) if.*budget.*?(\d+)", input_lower)
    #     if match:
    #         category = match.group(1).strip().title()
    #         budget = float(match.group(2))
    #         row = df[df["Category"].str.lower() == category.lower()]
    #         if not row.empty:
    #             category_total = row[months].values.flatten().sum()
    #             if category_total > budget:
    #                 return f"Yes, you overspent on {category}. Total: ₹{category_total:.2f}, Budget: ₹{budget:.2f}"
    #             else:
    #                 return f"No, your spending on {category} was ₹{category_total:.2f}, within your ₹{budget:.2f} budget."
    #         else:
    #             suggestion = get_closest_category(category)
    #             return f"I couldn't find '{category}'. Did you mean **{suggestion}**?"
    
    # # 8. Expense Summary Insight
    # elif "summary" in input_lower or "insight" in input_lower or "overview" in input_lower:
    #     df["Total"] = df[months].sum(axis=1)
    #     top_3 = df.sort_values("Total", ascending=False).head(3)

    #     monthly_totals = df[months].sum()
    #     highest_month = monthly_totals.idxmax()
    #     lowest_month = monthly_totals.idxmin()

    #     consistent = df.set_index("Category")[months].std(axis=1).sort_values().head(1).index[0]
    #     variable = df.set_index("Category")[months].std(axis=1).sort_values(ascending=False).head(1).index[0]

    #     return (
    #         f" Here's your spending summary:\n\n"
    #         f"🔹 Top 3 categories: {', '.join(top_3['Category'].tolist())}\n"
    #         f"🔹 Highest spend month: {highest_month}\n"
    #         f"🔹 Lowest spend month: {lowest_month}\n"
    #         f"🔹 Most consistent category: {consistent}\n"
    #         f"🔹 Most variable category: {variable}"
    # )
    # # 9. Category with steepest increase (month-over-month)
    # elif "steepest increase" in input_lower:
    #     month_diffs = df[months].diff(axis=1)
    #     max_increases = month_diffs.max(axis=1)

    #     top_index = max_increases.idxmax()
    #     top_category = df.loc[top_index, "Category"]
    #     top_value = max_increases[top_index]

    #     top_diff_row = month_diffs.loc[top_index]
    #     max_month = top_diff_row.idxmax()
    #     prev_month_idx = months.index(max_month) - 1
    #     prev_month = months[prev_month_idx] if prev_month_idx >= 0 else "previous month"

    #     return f"The category with the steepest single-month increase is **{top_category}**, which rose by ₹{top_value:.2f} from {prev_month} to {max_month}."


    
    
    # # 10. Trend over time for a category
        
       
    # elif any(keyword in input_lower for keyword in ["trend", "how has", "how did", "change", "evolve", "pattern", "progress"]):
    #     category = extract_category_for_trend(user_input)
    #     if category:
    #         row = df[df["Category"].str.lower() == category.lower()]
    #         if not row.empty:
    #             import numpy as np
    #             from scipy.stats import linregress

    #             y = row[months].values.flatten()
    #             x = np.arange(len(months))
    #             slope, _, _, _, _ = linregress(x, y)

    #             if slope > 50:
    #                 trend = "strongly increasing"
    #             elif slope > 0:
    #                 trend = "increasing"
    #             elif slope < -50:
    #                 trend = "strongly decreasing"
    #             elif slope < 0:
    #                 trend = "decreasing"
    #             else:
    #                 trend = "stable"

    #             # Save memory
    #             memory["intent"] = "trend"
    #             memory["category"] = category


    #             return f"The trend for {category} is **{trend}** over the months."
    #         else:
    #             suggestion = get_closest_category(category)
    #             if suggestion:
    #                 # Save memory for confirmation
    #                 memory["intent"] = "trend"
    #                 memory["category"] = suggestion
    #                 return f"I couldn't find '{category}'. Did you mean **{suggestion}**?"
    #             else:
    #                 return f"I couldn't find '{category}'. Available categories: {', '.join(available_categories)}"
    #     else:
    #         return "I couldn't identify the category. Try asking like: 'What is the trend for Food?'"

    





            
    # # 11. Anomalies in spending (z-score based)
    # elif "anomalies" in input_lower or "unusual spending" in input_lower:
    #     import numpy as np
    #     anomalies = []
    #     for idx, row in df.iterrows():
    #         values = row[months].values
    #         mean = np.mean(values)
    #         std = np.std(values)
    #         z_scores = (values - mean) / std if std > 0 else np.zeros_like(values)
    #         spike_months = [months[i] for i, z in enumerate(z_scores) if abs(z) > 2]
    #         if spike_months:
    #             anomalies.append(f"{row['Category']}: {', '.join(spike_months)}")
    #     if anomalies:
    #         return "Detected unusual spending in:\n" + "\n".join(anomalies)
    #     else:
    #         return "No significant anomalies found in spending patterns."





    
    # # 12. Fallback – Pass to LLaMA 3 with spreadsheet
    # # 12. Fallback – Pass to LLaMA 3 with spreadsheet
    # if use_llm:
    #     llama_response = call_llama3(user_input, df, months)
    #     chat_history.append(f"User: {user_input}")
    #     chat_history.append(f"Assistant: {llama_response}")
    #     return f"(AI response)\n{llama_response}"
    # else:
    #     return "I'm not sure how to answer that yet. Try asking about totals, comparisons, averages, or budgets."





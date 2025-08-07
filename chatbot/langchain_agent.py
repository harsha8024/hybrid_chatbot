from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import Ollama
from langchain.chains import create_sql_query_chain
from dotenv import load_dotenv
from db_utils import get_db_connection, execute_sql_query
import os

# Load your .env variables
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

def summarize_data_with_llm(original_question, database_results):
    """
    Takes raw data, calculates key metrics in Python first, then asks the LLM
    to write a summary using those pre-calculated metrics to ensure accuracy.
    """
    # --- PRE-CALCULATE METRICS IN PYTHON FOR ACCURACY ---
    total_spending = 0
    average_spending = 0
    num_spending_months = 0
    
    # Smartly handle different types of data results
    if database_results:
        first_row = database_results[0]
        if 'total_spent' in first_row:
            total_spending = first_row.get('total_spent', 0)
        elif 'actual_value' in first_row:
            total_spending = sum(row.get('actual_value', 0) for row in database_results)
            months_with_spending = [row for row in database_results if row.get('actual_value', 0) > 0]
            num_spending_months = len(months_with_spending)
            average_spending = total_spending / num_spending_months if num_spending_months > 0 else 0

    # --- BUILD THE PROMPT FOR THE SUMMARIZER LLM ---
    summary_prompt = f"""
You are a helpful financial assistant. Your task is to answer the user's question using ONLY the pre-calculated metrics provided below.
**CRITICAL RULE: You MUST use the exact numbers from the "Pre-Calculated Metrics" section in your answer. Do not perform your own calculations.**

Original Question: "{original_question}"

--- Pre-Calculated Metrics ---
Total Spending: {total_spending:.2f}
Average Monthly Spending: {average_spending:.2f}
Number of Months with Spending: {num_spending_months}

Please provide a concise, human-readable answer to the original question using these metrics.
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": summary_prompt, "stream": False}
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"LLM call error for summarization: {e}")
        return "I was able to retrieve the data but failed to generate a summary."

def answer_from_db_with_langchain(user_input, company_id):
    """
    Answers a user's question using a three-step process:
    1. Generate SQL with an LLM
    2. Execute the SQL to get raw data
    3. Summarize the raw data with another LLM call
    """
    db_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    llm = Ollama(model="codellama")

    db = SQLDatabase.from_uri(
        db_uri,
        include_tables=['account_categories', 'budget_transaction', 'cashflow_transactions'],
        sample_rows_in_table_info=3 
    )

    # 1. Generate the SQL query
    try:
        query_chain = create_sql_query_chain(llm, db)
        final_input_with_rules = f"""
Generate a SQL query to answer the following question.

--- VERY IMPORTANT SYNTAX RULES ---
1.  **NO HARDCODED NUMBERS**: You are strictly forbidden from using any hardcoded numbers in your query, especially for filtering or calculations.
2.  **AGGREGATION RULE**: When using an aggregate function (like SUM, AVG), any other columns in the SELECT clause MUST either be inside another aggregate function or be listed in the GROUP BY clause.
3.  **ROUNDING**: To round a calculation, you MUST use this exact syntax: `ROUND((...calculation...)::numeric, 2)`.
4.  **ALIASES**: When using a JOIN, every table must be given an alias (e.g., FROM budget_transaction AS bt), and every column must be prefixed with that alias.
5.  **FILTERING**: Always filter by `tenant_company_id = '{company_id}'`.
6.  **JOIN LOGIC**: If a JOIN is needed, you MUST use: `JOIN account_categories ac ON (bt.category_id = ac.id OR bt.sub_category_id = ac.id)`.
--- END OF RULES ---
        User question: {user_input}
        """
        sql_query = query_chain.invoke({"question": final_input_with_rules})
        print(f"--- Generated SQL ---\n{sql_query}\n--------------------")
    except Exception as e:
        return f"Error generating SQL query: {str(e)}"

    # 2. Execute the query to get raw data
    conn = None
    try:
        conn = get_db_connection()
        raw_data_results = execute_sql_query(sql_query, conn)
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"
    finally:
        if conn:
            conn.close()

    # --- 3. Summarize the data into a final answer ---
    # Check if the query returned data (a list) or an error message (a string)
    if isinstance(raw_data_results, list):
        if not raw_data_results:
            return "The query ran successfully, but found no data matching your request."
        # If we have data, pass it to the summarizer
        final_answer = summarize_data_with_llm(user_input, raw_data_results)
        return final_answer
    else:
        # If 'raw_data_results' is a string, it's an error message, so return it directly
        return raw_data_results

    # --- ADD THIS DEBUGGING BLOCK ---
    # print("\n--- DEBUG: Inspecting the context provided by SQLDatabase ---")
    # # This will print the exact schema and sample rows being passed to the agent
    # print(db.get_table_info())
    # print("--- END DEBUGGING ---\n")
    # # --- END OF DEBUGGING BLOCK ---

    # agent_executor = create_sql_agent(
    #     llm, 
    #     db=db, 
    #     agent_type="zero-shot-react-description", 
    #     verbose=True,
    #     handle_parsing_errors=True # Keep this as a safety net
    # )

    # final_prompt = f"""
    # You are an expert PostgreSQL agent. Your task is to answer the user's question by generating and executing SQL queries against a database.
    # You MUST filter any SQL queries by the column `tenant_company_id` with the value `{company_id}`.

    # **VERY IMPORTANT:** You MUST use the following format for your response. Do NOT add any other text.

    # Thought: I need to think about what to do to answer the user's question.
    # Action: `sql_db_query`
    # Action Input: [a valid SQL query]
    # Observation: [the result of the query]
    # ... (this can repeat)
    # Thought: I now know the final answer.
    # Final Answer: [the final human-readable answer]

    # --- Here is an example ---
    # User's question: how much was spent on rent?
    # Thought: The user is asking for the total spending on the 'rent' category. I need to find the `account_categories` for rent and then sum the `actual_value` from `budget_transaction`. I will filter by the `tenant_company_id`.
    # Action: `sql_db_query`
    # Action Input: SELECT SUM(bt.actual_value) FROM budget_transaction AS bt JOIN account_categories AS ac ON (bt.category_id = ac.id OR bt.sub_category_id = ac.id) WHERE ac.qbo_category ILIKE 'rent' AND bt.tenant_company_id = {company_id}
    # Observation: [(5000.00,)]
    # Thought: The query returned a single value, 5000.00. This is the total amount spent on rent. I can now provide the final answer.
    # Final Answer: The total amount spent on rent was ₹5,000.00.
    # --- End of Example ---

    # Now, begin!
    # """
    
    # try:
    #     result = agent_executor.invoke({"input": final_prompt})
    #     return result.get("output")
    # except Exception as e:
    #     return f"An error occurred with the LangChain agent: {str(e)}"
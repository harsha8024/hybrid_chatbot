# test_db.py
import os
import psycopg2
from dotenv import load_dotenv

# Load the same environment variables your main script uses
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# A very simple query that we know should work.
sql = "SELECT company_name FROM public.tenant_company WHERE id = 62;"

print("--- DATABASE CONNECTION TEST ---")
print(f"Connecting to DB: '{DB_NAME}' on host '{DB_HOST}'")

try:
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )
    cur = conn.cursor()
    print("Connection successful.")
    
    cur.execute(sql)
    rows = cur.fetchall()
    
    cur.close()
    conn.close()

    if not rows:
        print("\n--- RESULT: FAILURE ---")
        print("Successfully connected, but the query returned NO ROWS.")
        print("This confirms the issue is with the database connection details or permissions.")
    else:
        print("\n--- RESULT: SUCCESS ---")
        print(f"Successfully fetched data: {rows}")

except Exception as e:
    print(f"\n--- RESULT: FAILURE ---")
    print(f"An error occurred while connecting or executing: {e}")
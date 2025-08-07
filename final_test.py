# final_test.py
import psycopg2

# --- ACTION REQUIRED ---
# Please fill in your EXACT connection details from pgAdmin here.
# Double-check every single one.
DB_DETAILS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "finanfinal",
    "user": "postgres",
    "password": "pinnu3101" # <-- IMPORTANT: FILL THIS IN
}

# This is the simple query that we know should work for company 62
sanity_check_sql = "SELECT company_name FROM public.tenant_company WHERE id = 62;"

# This is the query that is failing to find data in your main app
main_query_sql = """
SELECT
    SUM(bt.actual_value) AS total_spent
FROM
    account_categories AS ac
JOIN
    budget_transaction AS bt ON (bt.category_id = ac.id OR bt.sub_category_id = ac.id)
WHERE
    ac.qbo_category ILIKE 'Vehicle insurance'
AND
    bt.tenant_company_id = 62;
"""

print("--- FINAL DIAGNOSTIC TEST ---")
conn = None
try:
    print(f"Attempting to connect to DB '{DB_DETAILS['dbname']}' on host '{DB_DETAILS['host']}'...")
    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()
    print("Connection successful.\n")

    # --- Test 1: Sanity Check ---
    print("Executing Sanity Check Query to find Company 62...")
    cur.execute(sanity_check_sql)
    sanity_rows = cur.fetchall()
    if not sanity_rows:
        print("--> RESULT 1: FAILURE. Could not find Company 62.")
    else:
        print(f"--> RESULT 1: SUCCESS. Found Company 62: {sanity_rows[0][0]}\n")

    # --- Test 2: The Main Query ---
    print("Executing the main query for 'Vehicle insurance' for Company 62...")
    cur.execute(main_query_sql)
    main_rows = cur.fetchall()
    
    # Check if the query returned a row and the value is not None
    if not main_rows or main_rows[0][0] is None:
        print("--> RESULT 2: ZERO ROWS RETURNED.")
        print("--> CONCLUSION: The data for 'Vehicle insurance' does NOT belong to company 62 in this database.")
    else:
        print(f"--> RESULT 2: SUCCESS. Found total spending: {main_rows[0][0]}")
        print("--> CONCLUSION: If you see this, the data exists and the connection is fine.")

except Exception as e:
    print(f"\nAN ERROR OCCURRED: {e}")
finally:
    if conn:
        conn.close()
        print("\nConnection closed.")
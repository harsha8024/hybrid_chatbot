from dotenv import load_dotenv
import os
import psycopg2

# Load environment variables from .env file
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")






def get_db_schema(host, dbname, user, password, port=5432, only_tables=None):
    """
    Extracts schema from PostgreSQL DB. Optionally filters to specific tables.
    """
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        cursor = conn.cursor()

        if only_tables:
            # Format tuple for IN clause safely
            placeholders = ','.join(['%s'] * len(only_tables))
            cursor.execute(f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name IN ({placeholders});
            """, tuple(only_tables))
        else:
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public';
            """)

        tables = cursor.fetchall()
        schema_str = ""
        for table in tables:
            table_name = table[0]
            schema_str += f"Table: {table_name}\n"

            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            columns = cursor.fetchall()

            for col_name, col_type in columns:
                schema_str += f"  - {col_name} ({col_type})\n"
            schema_str += "\n"

        cursor.close()
        conn.close()
        return schema_str.strip()

    except Exception as e:
        return f"Schema Extraction Error: {str(e)}"

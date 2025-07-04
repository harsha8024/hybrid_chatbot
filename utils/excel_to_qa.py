import pandas as pd

def generate_qa_from_excel(excel_path, output_csv_path):
    # Read the Excel sheet
    df = pd.read_excel(excel_path)

    # Define relevant columns
    month_names = ["Jan 2025", "Feb 2025", "Mar 2025", "Apr 2025", "May 2025"]
    total_column = "Total"

    # Rename first column to "Category"
    df.columns.values[0] = "Category"

    # Fix column headers if needed (like Unnamed: 1, Unnamed: 2, etc.)
    all_headers = df.columns.tolist()
    for i, month in enumerate(month_names, start=1):
        df.columns.values[i] = month
    df.columns.values[len(month_names) + 1] = total_column  # Rename the last column to 'Total'

    qa_pairs = []

    # Loop through rows and generate Q-A pairs
    for _, row in df.iterrows():
        category = row["Category"]

        # Monthly questions
        for month in month_names:
            value = row.get(month, None)
            if pd.notna(value):
                question = f"What was the expenditure on {category} in {month}?"
                answer = f"₹{value}"
                qa_pairs.append({"Question": question, "Answer": answer})

        # Total question
        total_value = row.get(total_column, None)
        if pd.notna(total_value):
            question = f"What was the total expenditure on {category}?"
            answer = f"₹{total_value}"
            qa_pairs.append({"Question": question, "Answer": answer})

    # Save to CSV
    qa_df = pd.DataFrame(qa_pairs)
    qa_df.to_csv(output_csv_path, index=False)
    print(f"[✅] Generated {len(qa_pairs)} QA pairs to {output_csv_path}")

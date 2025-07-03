import pandas as pd

def generate_qa_from_excel(excel_path, output_csv_path):
    df = pd.read_excel(excel_path, index_col=0)  # 👈 Tells pandas to treat first column as index

    questions = []
    answers = []

    for category in df.index:  # 👈 Now category is from index
        for month in df.columns:
            amount = df.loc[category, month]
            if pd.notna(amount):
                question = f"What was the expenditure on {category} in {month}?"
                answer = f"₹{amount}"
                questions.append(question)
                answers.append(answer)

    qa_df = pd.DataFrame({"Question": questions, "Answer": answers})
    qa_df.to_csv(output_csv_path, index=False)
    print(f"[✔] Exported {len(qa_df)} Q&A pairs to {output_csv_path}")



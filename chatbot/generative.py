import pandas as pd
import re
import os
import fuzzywuzzy
import openpyxl
from rapidfuzz import process
from fuzzywuzzy import process
import requests


chat_history = []
memory = {
    "intent": None,
    "category": None
}


# Load Excel and setup
# excel_path = "data/samplepnl.xlsx"
# df = pd.read_excel(excel_path, skiprows=3)

# Clean up columns


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


def call_llama3(user_input, df=None, months=None):
    context = ""
    if df is not None and months is not None:
        context = generate_spreadsheet_summary(df, months)
        # print(context)
    
    prompt = (
    "You are a helpful financial assistant.\n"
    "The user has provided a monthly profit and loss spreadsheet showing spending in various categories.\n"
    "Use the spreadsheet data to directly answer user questions like:\n"
    "- Which categories they spent the most or least on\n"
    "- Where they can cut costs\n"
    "- Trends or changes in spending\n"
    "- Budget or category comparisons\n\n"
    f"Spreadsheet:\n{context}\n\n"
    f"User Query:\n{user_input}\n\n"
    "Only use the spreadsheet values for your answer. Be specific, helpful, and brief.\n"
    "Assistant:"
    "Do not add summaries or totals unless the user specifically requests them."
)
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



# Generative bot logic
def generative_bot(user_input, company_id, use_llm=True):

    print(f"✅ Entered generative_bot() for company_id={company_id}", flush=True)

    # Determine base folder
    base_folder = f"data/{company_id}"
    if not os.path.isdir(base_folder):
        return f"Company ID '{company_id}' has no spreadsheet folder."

    # Search for file containing the relevant category
    best_match = None
    best_score = 0
    matched_df = None
    matched_filename = None
    matched_category = None

    category_required = any(keyword in user_input.lower() for keyword in [
    " on ", "spend on", "expenditure on", "average", "trend", "percent", "versus", "vs", "compare"
])

# Loop through all files only if category is required
    if category_required:
        best_score = 0
        best_match = None
        matched_df = None
        matched_filename = None
        matched_category = None

        for file in os.listdir(base_folder):
            if file.endswith(".xlsx"):
                filepath = os.path.join(base_folder, file)
                try:
                    df = pd.read_excel(filepath, skiprows=3)
                    df.columns = df.columns.str.strip()
                    if df.columns.values[0].lower() != "category":
                        df.columns.values[0] = "Category"
                    df = df[df["Category"].notna()]
                    df["Category"] = df["Category"].astype(str).str.strip().str.lower()
                    category_map = {cat.lower(): cat for cat in df["Category"].unique()}

                    categories = df["Category"].tolist()
                    user_input_lower = user_input.lower().strip()

                    # Try to extract a possible category from the user query
                    cat_match = re.search(r"(?:on|about|for|regarding)\s+([a-zA-Z ]+)", user_input_lower)
                    likely_cat = cat_match.group(1).strip().lower() if cat_match else user_input_lower

                    # Try direct substring match first
                    possible_categories = [cat for cat in categories if cat in likely_cat]
                    if possible_categories:
                        match_cat = possible_categories[0]
                        score = 100
                    else:
                        match = process.extractOne(likely_cat, categories)
                        if match:
                            match_cat, score = match
                        else:
                            match_cat, score = None, 0

                    print(f"🔍 Checked {file}: Matched '{likely_cat}' to '{match_cat}' with score {score}", flush=True)

                    if score > best_score and score >= 60:
                        best_score = score
                        best_match = match_cat
                        matched_df = df
                        matched_filename = file
                        matched_category = match_cat

                except Exception as e:
                    print(f"❌ Error reading {file}: {e}", flush=True)

        # Final check
        if matched_df is None:
            return f"No spreadsheet in company {company_id} contains a category matching your query."

        df = matched_df
        print(f"✅ Best match found: '{matched_category}' in file: {matched_filename}", flush=True)
    else:
        # No specific category needed — just use the first available .xlsx
        for file in os.listdir(base_folder):
            if file.endswith(".xlsx"):
                filepath = os.path.join(base_folder, file)
                try:
                    df = pd.read_excel(filepath, skiprows=3)
                    df.columns = df.columns.str.strip()
                    if df.columns.values[0].lower() != "category":
                        df.columns.values[0] = "Category"
                    df = df[df["Category"].notna()]
                    df["Category"] = df["Category"].astype(str).str.strip().str.lower()
                    category_map = {cat.lower(): cat for cat in df["Category"].unique()}
                    matched_df = df
                    matched_filename = file
                    break
                except Exception as e:
                    print(f"❌ Error reading {file}: {e}", flush=True)

        if matched_df is None:
            return f"No valid spreadsheets found for company {company_id}."

        df = matched_df
        print(f"✅ Defaulted to file: {matched_filename}", flush=True)
    

    

    # Fuzzy match helper
    def get_closest_category(user_category):
        result = process.extractOne(user_category, available_categories, score_cutoff=60)
        if result is None:
            return None
        match = result[0]
        return match

    # Detect month columns
    months = [col for col in df.columns if re.match(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$", col)]
    if not months:
        return "No valid month columns found in the spreadsheet."

    input_lower = user_input.lower()
    month_lookup = {m.lower(): m for m in months}
    available_categories = df["Category"].unique().tolist()

    global chat_history
    


    # 10B. Clarification handling (e.g., "yes, I meant X")
    if input_lower.startswith("yes") or "i meant" in input_lower:
    # Check if previous intent was a trend request and a category was stored
        if memory["intent"] == "trend" and memory["category"]:
            category = memory["category"]
            row = df[df["Category"].str.lower() == category.lower()]
            if not row.empty:
                import numpy as np
                from scipy.stats import linregress

                y = row[months].values.flatten()
                x = np.arange(len(months))
                slope, _, _, _, _ = linregress(x, y)

                if slope > 50:
                    trend = "strongly increasing"
                elif slope > 0:
                    trend = "increasing"
                elif slope < -50:
                    trend = "strongly decreasing"
                elif slope < 0:
                    trend = "decreasing"
                else:
                    trend = "stable"

                return f"Thanks for confirming. The trend for {category} is **{trend}** over the months."
            else:
                return f"Sorry, I still couldn’t find '{category}'."
        elif memory["intent"] == "average" and memory["category"]:
            row = df[df["Category"].str.lower() == memory["category"].lower()]
            if not row.empty:
                avg = row[months].values.flatten().mean()
                return f"Thanks for confirming. The average monthly expenditure on {memory['category']} is ₹{avg:.2f}"
            else:
                return f"Sorry, I still couldn’t find '{memory['category']}'."
        else:
            return "Can you clarify what you're referring to?"


    # 1. Spend the most
    if "spend the most" in input_lower:
        df["Total"] = df[months].sum(axis=1)
        top_category = df.loc[df["Total"].idxmax(), "Category"]
        top_amount = df["Total"].max()
        return f"You spent the most on {top_category}: ₹{top_amount:.2f}"

    # 2. Highest expense month
    elif "expenses the highest" in input_lower or "most expensive month" in input_lower:
        monthly_totals = df[months].sum()
        top_month = monthly_totals.idxmax()
        top_amount = monthly_totals.max()
        return f"Your expenses were highest in {top_month}: ₹{top_amount:.2f}"
    

    elif any(kw in input_lower for kw in ["total spent", "how much did i spend", "total expenditure on"]):
        if matched_category is None:
            return "I couldn't identify the category you're referring to. Try asking like: 'How much did I spend on Services?'"

        # Map to original formatting
        actual_category = category_map.get(matched_category.lower().strip(), matched_category)

        total_column_name = "Total"
        try:
            total_value = df.loc[df["Category"] == actual_category, total_column_name].values[0]
        except IndexError:
            total_value = None

        if total_value is not None:
            print(f"✅ Using coded total response for category: {actual_category}", flush=True)
            return (
                f"According to the spreadsheet, you spent:\n\n"
                + "\n".join(
                    [f"* {month}: ₹{df.loc[df['Category'] == actual_category, month].values[0]:.2f}"
                    for month in months if month in df.columns]
                )
                + f"\n\nIn total, you spent ₹{total_value:.2f} on {actual_category}."
            )
        else:
            return f"Sorry, no total spending found for {actual_category} in the spreadsheet."

        
    elif "spend in" in input_lower or "spent in" in input_lower:
        for m in month_lookup:
            if m in input_lower:
                total = df[month_lookup[m]].sum()
                return f"You spent ₹{total:.2f} in {month_lookup[m]}."
        return "I couldn't identify the month. Try asking like: 'How much did I spend in March?'"



    # 3. Average monthly expenditure on a category
    elif "average monthly expenditure" in input_lower:
        import string
        match = re.search(r"average monthly expenditure on (.+)", input_lower)
        if match:
            user_category = match.group(1).strip().rstrip(string.punctuation).title()
            row = df[df["Category"].str.lower() == user_category.lower()]
            if not row.empty:
                avg = row[months].values.flatten().mean()
                memory["intent"] = "average"
                memory["category"] = user_category
                return f"The average monthly expenditure on {user_category} is ₹{avg:.2f}"
            else:
                suggestion = get_closest_category(user_category)
                if suggestion:
                    return f"I couldn't find '{user_category}'. Did you mean **{suggestion}**?"
                else:
                    available = ", ".join(available_categories)
                    return f"I couldn't find '{user_category}'. Available categories: {available}"

    # 4. Compare spend on X vs Y
    elif " vs " in input_lower or " versus " in input_lower:
        match = re.search(r"spend on (.+?) (?:vs|versus) (.+)", input_lower)
        if match:
            cat1 = match.group(1).strip().title()
            cat2 = match.group(2).strip().title()
            row1 = df[df["Category"].str.lower() == cat1.lower()]
            row2 = df[df["Category"].str.lower() == cat2.lower()]

            if not row1.empty and not row2.empty:
                total1 = row1[months].values.flatten().sum()
                total2 = row2[months].values.flatten().sum()
                return f"Total spent on {cat1}: ₹{total1:.2f}, {cat2}: ₹{total2:.2f}"
            else:
                missing = []
                if row1.empty:
                    suggestion1 = get_closest_category(cat1)
                    missing.append(f"{cat1} (Did you mean **{suggestion1}**?)" if suggestion1 else cat1)
                if row2.empty:
                    suggestion2 = get_closest_category(cat2)
                    missing.append(f"{cat2} (Did you mean **{suggestion2}**?)" if suggestion2 else cat2)
                return f"Couldn't find data for: {', '.join(missing)}"

    # 5. Total expenditure in a month
    elif "total expenditure in" in input_lower:
        for m in month_lookup:
            if m in input_lower:
                total = df[month_lookup[m]].sum()
                return f"Your total expenditure in {month_lookup[m]} was ₹{total:.2f}"
        return "I couldn't identify the month. Try asking like: 'Total expenditure in March'."

    # 6. Percent of spending on a category
    
    elif "percent" in input_lower:
    # Try multiple flexible patterns
        patterns = [
            r"percent of.*spending.*on (.+)",
            r"percent.*on (.+)",
            r"what percent.*on (.+)",
            r"what percent.*was spent on (.+)",
            r"percent.*was on (.+)",
            r"how much percent.*on (.+)"
        ]

        for pat in patterns:
            match = re.search(pat, input_lower)
            if match:
                user_input_cat = match.group(1).strip()
                category = user_input_cat.title()
                row = df[df["Category"].str.lower() == category.lower()]

                if not row.empty:
                    total_spent = df[months].sum().sum()
                    category_total = row[months].values.flatten().sum()
                    if total_spent > 0:
                        percent = (category_total / total_spent) * 100
                        return f"{category} accounted for {percent:.2f}% of your total spending."
                    else:
                        return "Total spending is ₹0, so percentage cannot be calculated."
                else:
                    suggestion = get_closest_category(category)
                    return f"I couldn't find '{category}'. Did you mean **{suggestion}**?"

        return "I couldn't identify the category. Try asking like: 'What percent of my spending was on Rent?'"




    # 7. Budget overspend check
    elif "overspend" in input_lower and "budget" in input_lower:
        match = re.search(r"overspend on (.+?) if.*budget.*?(\d+)", input_lower)
        if match:
            category = match.group(1).strip().title()
            budget = float(match.group(2))
            row = df[df["Category"].str.lower() == category.lower()]
            if not row.empty:
                category_total = row[months].values.flatten().sum()
                if category_total > budget:
                    return f"Yes, you overspent on {category}. Total: ₹{category_total:.2f}, Budget: ₹{budget:.2f}"
                else:
                    return f"No, your spending on {category} was ₹{category_total:.2f}, within your ₹{budget:.2f} budget."
            else:
                suggestion = get_closest_category(category)
                return f"I couldn't find '{category}'. Did you mean **{suggestion}**?"
    
    # 8. Expense Summary Insight
    elif "summary" in input_lower or "insight" in input_lower or "overview" in input_lower:
        df["Total"] = df[months].sum(axis=1)
        top_3 = df.sort_values("Total", ascending=False).head(3)

        monthly_totals = df[months].sum()
        highest_month = monthly_totals.idxmax()
        lowest_month = monthly_totals.idxmin()

        consistent = df.set_index("Category")[months].std(axis=1).sort_values().head(1).index[0]
        variable = df.set_index("Category")[months].std(axis=1).sort_values(ascending=False).head(1).index[0]

        return (
            f" Here's your spending summary:\n\n"
            f"🔹 Top 3 categories: {', '.join(top_3['Category'].tolist())}\n"
            f"🔹 Highest spend month: {highest_month}\n"
            f"🔹 Lowest spend month: {lowest_month}\n"
            f"🔹 Most consistent category: {consistent}\n"
            f"🔹 Most variable category: {variable}"
    )
    # 9. Category with steepest increase (month-over-month)
    elif "steepest increase" in input_lower:
        month_diffs = df[months].diff(axis=1)
        max_increases = month_diffs.max(axis=1)

        top_index = max_increases.idxmax()
        top_category = df.loc[top_index, "Category"]
        top_value = max_increases[top_index]

        top_diff_row = month_diffs.loc[top_index]
        max_month = top_diff_row.idxmax()
        prev_month_idx = months.index(max_month) - 1
        prev_month = months[prev_month_idx] if prev_month_idx >= 0 else "previous month"

        return f"The category with the steepest single-month increase is **{top_category}**, which rose by ₹{top_value:.2f} from {prev_month} to {max_month}."


    
    
    # 10. Trend over time for a category
        
       
    elif any(keyword in input_lower for keyword in ["trend", "how has", "how did", "change", "evolve", "pattern", "progress"]):
        category = extract_category_for_trend(user_input)
        if category:
            row = df[df["Category"].str.lower() == category.lower()]
            if not row.empty:
                import numpy as np
                from scipy.stats import linregress

                y = row[months].values.flatten()
                x = np.arange(len(months))
                slope, _, _, _, _ = linregress(x, y)

                if slope > 50:
                    trend = "strongly increasing"
                elif slope > 0:
                    trend = "increasing"
                elif slope < -50:
                    trend = "strongly decreasing"
                elif slope < 0:
                    trend = "decreasing"
                else:
                    trend = "stable"

                # Save memory
                memory["intent"] = "trend"
                memory["category"] = category


                return f"The trend for {category} is **{trend}** over the months."
            else:
                suggestion = get_closest_category(category)
                if suggestion:
                    # Save memory for confirmation
                    memory["intent"] = "trend"
                    memory["category"] = suggestion
                    return f"I couldn't find '{category}'. Did you mean **{suggestion}**?"
                else:
                    return f"I couldn't find '{category}'. Available categories: {', '.join(available_categories)}"
        else:
            return "I couldn't identify the category. Try asking like: 'What is the trend for Food?'"

    





            
    # 11. Anomalies in spending (z-score based)
    elif "anomalies" in input_lower or "unusual spending" in input_lower:
        import numpy as np
        anomalies = []
        for idx, row in df.iterrows():
            values = row[months].values
            mean = np.mean(values)
            std = np.std(values)
            z_scores = (values - mean) / std if std > 0 else np.zeros_like(values)
            spike_months = [months[i] for i, z in enumerate(z_scores) if abs(z) > 2]
            if spike_months:
                anomalies.append(f"{row['Category']}: {', '.join(spike_months)}")
        if anomalies:
            return "Detected unusual spending in:\n" + "\n".join(anomalies)
        else:
            return "No significant anomalies found in spending patterns."





    
    # 12. Fallback – Pass to LLaMA 3 with spreadsheet
    # 12. Fallback – Pass to LLaMA 3 with spreadsheet
    if use_llm:
        llama_response = call_llama3(user_input, df, months)
        chat_history.append(f"User: {user_input}")
        chat_history.append(f"Assistant: {llama_response}")
        return f"(AI response)\n{llama_response}"
    else:
        return "I'm not sure how to answer that yet. Try asking about totals, comparisons, averages, or budgets."





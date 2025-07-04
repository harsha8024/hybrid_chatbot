import pandas as pd
import re
from rapidfuzz import process

# Load Excel and setup
excel_path = "data/samplepnl.xlsx"
df = pd.read_excel(excel_path, skiprows=3)

# Clean up columns
df.columns = df.columns.str.strip()
df.columns.values[0] = "Category"
df = df[df["Category"].notna()]  # remove rows with missing category

# Dynamically detect month columns (e.g., "Jan 2025", etc.)
months = [col for col in df.columns if re.match(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$", col)]

if not months:
    raise ValueError(f"❌ No month columns detected. Available columns: {list(df.columns)}")

# Month lookup and category list
month_lookup = {m.lower(): m for m in months}
available_categories = df["Category"].unique().tolist()

# Fuzzy match helper
def get_closest_category(user_category):
    match, score = process.extractOne(user_category, available_categories, score_cutoff=60)
    return match if match else None

# Generative bot logic
def generative_bot(user_input):
    input_lower = user_input.lower()

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

    # 3. Average monthly expenditure on a category
    elif "average monthly expenditure" in input_lower:
        match = re.search(r"average monthly expenditure on (.+)", input_lower)
        if match:
            user_category = match.group(1).strip().title()
            row = df[df["Category"].str.lower() == user_category.lower()]
            if not row.empty:
                avg = row[months].values.flatten().mean()
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
    # 6. Percent of spending on a category
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

    # 8. Fallback
    return "I'm not sure how to answer that yet. Try asking about totals, comparisons, averages, or budgets."

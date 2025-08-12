import requests

# messages = ["Hey, hope you're having a good day. I'm preparing for a meeting and I'm a bit swamped. I was hoping you could help me figure out two things. First, can you give me an expense summary, and also I need to know the percent of spending on Software & apps and give me reasons for the trend."]
# messages = ["Analyse my transaction month on month."]
messages=["What can be inferred from the level and consistency of payroll expenses, and what does this indicate about the company's staffing strategy and operational maturity? "]

for msg in messages:
    print("🧪 Sending request to Flask server")
    response = requests.post(
        "http://127.0.0.1:5000/chat",
        json={"user_input": msg, "company_id": 3}  # or "2" for second company
    )

    print("Status code:", response.status_code)
    print("Raw response:", repr(response.text))  # show raw text even if empty

    try:
        print("JSON response:", response.json())
    except Exception as e:
        print("Failed to parse JSON:", e)

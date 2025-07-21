import requests

messages = ["How much did I spend on Wise?"]
for msg in messages:
    print("🧪 Sending request to Flask server")
    response = requests.post(
        "http://127.0.0.1:5000/chat",
        json={"message": msg, "company_id": "1"}  # or "2" for second company
    )

    print("Status code:", response.status_code)
    print("Raw response:", repr(response.text))  # show raw text even if empty

    try:
        print("JSON response:", response.json())
    except Exception as e:
        print("Failed to parse JSON:", e)

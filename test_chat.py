import requests

msg = "Give me a summary of my expenses. "
response = requests.post("http://127.0.0.1:5000/chat", json={"message": msg})

print("Status code:", response.status_code)
print("Raw response:", repr(response.text))  # show raw text even if empty

try:
    print("JSON response:", response.json())
except Exception as e:
    print("Failed to parse JSON:", e)

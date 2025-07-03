import requests

msg = "What was the expenditure on Business Income in Unnamed: 1?"
response = requests.post("http://127.0.0.1:5000/chat", json={"message": msg})
print(response.json())


import requests

data = {"text": "Meneer Jansen was vandaag erg onrustig"}
response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())

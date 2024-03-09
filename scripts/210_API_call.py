import requests

data = {"text": "Subjectief: Ik weet het allemaal niet meer! Objectief: Mw was vandaag erg ontredderd. Evaluatie: - Plan: -"}
response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())

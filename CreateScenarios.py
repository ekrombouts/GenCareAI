import openai
from openai import OpenAI
from datetime import datetime

def lees_bestand(bestandsnaam):
    with open(bestandsnaam, 'r') as file:
        inhoud = file.read()
    return inhoud

def schrijf_bestand(bestandsnaam, tekst):
    with open(bestandsnaam, 'w') as file:
        file.write(tekst)

def genereer_scenario(model, s_role_content, u_role_content):
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": s_role_content},
            {"role": "user", "content": u_role_content}
        ]
    )
    return completion

s_role_content = lees_bestand('roles/rol_system_scenario_maker.txt')
u_role_content = lees_bestand('roles/rol_user_scenario_maker.txt')
# m='gpt-4'
m='gpt-3.5-turbo'

scenarios = []
for i in range(10):
    scenario = genereer_scenario(m, s_role_content, u_role_content)
    scenarios.append(f"Scenario {i+1}: {scenario.choices[0].message.content}")

# Scenario's opslaan
for i, scenario in enumerate(scenarios, start=1):
    bestandsnaam = f'scenarios/scenario_{m}_{datetime.now().strftime("%Y%m%d%H%M")}_{i}.txt'
    schrijf_bestand(bestandsnaam, scenario)
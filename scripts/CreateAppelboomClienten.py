'''
Auteur:     Eva Rombouts
Datum:      22-12-2023
Project:    GenCareAI
Doel:       Dit script genereert fictieve zorgdata met behulp van de OpenAI API. Allereerst worden clientprofielen gemaakt

Let op. Het instellen van een OpenAI API key is vereist (https://platform.openai.com/docs/quickstart?context=python)
Het genereren van 24 clientenprofielen met gpt-3.5-turbo kost ongeveer 3 cent. 
'''

# Setup
pass # Soms pakt ie mijn eerste statement niet...
from openai import OpenAI
from datetime import datetime
import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Voorkomt een warning

# Parameters
seed = 6
aantal_clienten = 24

#Deze functie maakt een verbinding met de OpenAI API en genereert een 'response' gebaseerd op de rollen en inhoud die worden meegegeven (s_role en u_role). De functie maakt gebruik van een specifiek model, en biedt de mogelijkheid om de seed (voor reproduceerbaarheid) en het aantal antwoorden (n) te bepalen.
def genereer_zorgdata(s_role, u_role, model='gpt-3.5-turbo', seed=None, n=1):
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": s_role},
            {"role": "user", "content": u_role}
        ],
        seed=seed,
        n=n
    )
    return completion

# Definieer inhoud rollen
s_role_content = '''
Je specialiseert je nu in het genereren van fictieve gegevens voor helpend en verzorgend personeel in verpleeghuizen, met een focus op cliënten met een psychogeriatrische aandoening. Jouw expertise omvat het creëren van realistische cliëntscenario's, cliëntendossiers en zorgplannen die specifiek zijn afgestemd op de behoeften van deze doelgroep. Deze gegevens moeten toegankelijk en relevant zijn voor personeel zonder uitgebreide medische kennis, en moeten belangrijke aspecten van de dagelijkse zorg en ondersteuning voor deze cliënten bevatten.
'''

u_role_content = '''
Schrijf een profiel van een verpleeghuiscliënt. Wees creatief met de naam.
Geef aan welk type dementie de client heeft. En welke lichamelijke klachten. 
'''

# Data generatie: Genereer zorgdata, waarbij de eerder gedefinieerde rollen worden gebruikt. 
# De naam appelboom verwijst naar een fictieve afdeling. Hier wonen 24 clienten met een vorm van dementie. 
appelboom = genereer_zorgdata(s_role=s_role_content, u_role=u_role_content, seed=seed, n=aantal_clienten)

# Wijzig formaat naar een dictionary voor opslag als json
appelboom_clienten = {
    "model": appelboom.model,
    "clienten": [{"profiel": choice.message.content} for choice in appelboom.choices],
}

# Sla op als json
with open('zorgdata/appelboom_clienten.json', 'w') as file:
    json.dump(appelboom_clienten, file)

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
import os
import json
from GenCareAIFunctions import genereer_zorgdata 
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Voorkomt een warning

# Parameters
seed = 6
aantal_clienten = 24
model = 'gpt-3.5-turbo'
filename_clienten = 'zorgdata/appelboom_profielen.json'

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
appelboom = genereer_zorgdata(s_role=s_role_content, u_role=u_role_content, model=model, seed=seed, n=aantal_clienten)

# Wijzig formaat naar een dictionary voor opslag als json
appelboom_profielen = {
    "model": appelboom.model,
    "clienten": [{"profiel": choice.message.content} for choice in appelboom.choices],
}

# Sla op als json
with open(filename_clienten, 'w') as file:
    json.dump(appelboom_profielen, file)

pass # Voorkomt foutief draaien van vorig statement
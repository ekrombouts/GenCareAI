'''
Auteur:     Eva Rombouts
Datum:      22-12-2023
Project:    GenCareAI
Doel:       Dit script genereert fictieve zorgdata met behulp van de OpenAI API.

Let op. Het instellen van een OpenAI API key is vereist (https://platform.openai.com/docs/quickstart?context=python)
Voor 10 weken rapportages van 24 clienten betaal je 40 cent met 3.5 turbo.
'''

pass # Soms pakt ie mijn eerste statement niet...
import os
import json
import time
from GenCareAIFunctions import genereer_zorgdata 
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Voorkomt een warning

# Parameters
seed = 6
num_weeks = 10
model = 'gpt-3.5-turbo'
filename_clienten = 'zorgdata/appelboom_clienten.json'
filename_rapportages = 'zorgdata/appelboom_rapportages.json'

# Lees de clientendata in
with open(filename_clienten, 'r') as file:
    appelboom_clienten = json.load(file)

# Definieer inhoud rollen
s_role_content = '''
Je specialiseert je nu in het genereren van fictieve gegevens voor helpend en verzorgend personeel in verpleeghuizen, met een focus op cliënten met een psychogeriatrische aandoening. Jouw expertise omvat het creëren van realistische cliëntscenario's, cliëntendossiers en zorgplannen die specifiek zijn afgestemd op de behoeften van deze doelgroep. Deze gegevens moeten toegankelijk en relevant zijn voor personeel zonder uitgebreide medische kennis, en moeten belangrijke aspecten van de dagelijkse zorg en ondersteuning voor deze cliënten bevatten.
'''

u_role_instruction = '''
\n\n
Genereer fictieve zorgrapportages voor deze cliënt voor een periode van één week. 
En geef voor elke rapportage een onrustscore: Een heel getal tussen 0 en 100, wat de mate weergeeft waarin de cliënt onrust vertoont in deze rapportage. 
Elke rapportage dient de volgende structuur te hebben:\n
---StartRapportage---
Dag: [Dag van de week]
Niveau: [Helpende/Verzorgende/Verpleegkundige]
Rapportage: [Inhoud van de rapportage]
Onrustscore: [Onrustscore]
--EindRapportage---\n
De rapportages moeten afwisselend geschreven worden door een helpende (20% kans), een verzorgende (60% kans), of een verpleegkundige (20% kans). Zorg voor variatie in de tijdstippen van de dag waarop de rapportages zijn geschreven, met een focus op overdag, en in mindere mate 's avonds en 's nachts. In de inhoud van elke rapportage dienen één of meer eigenschappen of beperkingen van de cliënt naar voren te komen, zowel de cognitieve als lichamelijke beperkingen.
'''

# Print een lijstje van de clienten. Daarmee kan de voortgang worden bijgehouden
for ct in appelboom_clienten['clienten']:
    print(ct['profiel'].split('\n')[0])

# Maak een lege dictionary aan voor alle cliënten
appelboom_rapportages = {}

for ct in appelboom_clienten['clienten']:
    start = time.time()
    client_id = ct['profiel'].split('\n')[0] # Op de eerste regel staat de naam
    print (client_id) # Print om voortgang bij te houden
    u_role_content = ct['profiel'] + u_role_instruction # Voeg het clientprofiel toe aan de instructie
    
    #Genereer de rapportages
    ct_rapportages = genereer_zorgdata(s_role=s_role_content, u_role=u_role_content, model=model, seed=seed, n=num_weeks)

    # Maak een lege dictionary voor de rapportages van deze cliënt
    client_rapportages = {}
    # Itereer over elke 'choice' (week) in de response
    for j in range(len(ct_rapportages.choices)):
        # Gebruik de index als weeknummer
        weeknummer = f"Week {j+1}"
        # Sla de inhoud van de response op voor deze week
        client_rapportages[weeknummer] = ct_rapportages.choices[j].message.content
    
    # Voeg het profiel van de cliënt toe aan de dictionary
    client_rapportages['Profiel'] = ct['profiel']
    # Voeg de rapportage dictionary voor deze cliënt toe aan de hoofddictionary
    appelboom_rapportages[client_id] = client_rapportages
    print (f"Verstreken tijd: {round(time.time()-start)} seconden")

# Nu bevat appelboom_rapportages de rapportages voor elke week voor elke cliënt
# Sla op als json
with open(filename_rapportages, 'w') as file:
    json.dump(appelboom_rapportages, file)

pass
'''
Auteur:     Eva Rombouts
Datum:      22-12-2023
Project:    GenCareAI
Doel:       Dit script genereert fictieve zorgdata met behulp van de OpenAI API.

Let op. Het instellen van een OpenAI API key is vereist (https://platform.openai.com/docs/quickstart?context=python)
Voor 10 weken rapportages van 24 clienten betaal je 40 cent met 3.5 turbo.
'''

pass # Soms pakt ie mijn eerste statement niet...
from openai import OpenAI
from datetime import datetime
import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Voorkomt een warning

# Lees de data in
with open('zorgdata/appelboom_clienten.json', 'r') as file:
    appelboom_clienten = json.load(file)

# Verzamel de inhoud van alle choices met een list comprehension
clienten = [client['profiel'] for client in appelboom_clienten['clienten']]

#Deze functie maakt een verbinding met de OpenAI API en genereert een 'response' gebaseerd op de rollen en inhoud die worden meegegeven (s_role en u_role). De functie maakt gebruik van het gekozen model, en biedt de mogelijkheid om de seed (voor reproduceerbaarheid) en het aantal antwoorden (n) te bepalen.
def genereer_rapportages(s_role, u_role, model='gpt-3.5-turbo', seed=1, n=1):
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

u_role_instruction = '''
\n
Schrijf fictieve zorgrapportages voor deze client voor één week. De rapportages kunnen door een helpende worden geschreven (20% kans), door een verzorgende (60% kans) of door een verpleegkundige (20% kans). Varieer met de tijd van de dag: Overdag wordt het meeste gerapporteerd, in mindere mate 's avonds en 's nachts. 
In de rapportages komen één of meer van de eigenschappen, beperkingen van de client naar voren. 
Gebruik voor de rapportages het volgende formaat:
\n
Dag:  
Niveau: 
Rapportage:
'''

u_role_instruction = '''
Genereer fictieve zorgrapportages voor deze cliënt voor een periode van één week. Elke rapportage dient de volgende structuur te hebben:
--StartRapportage
Dag: [Dag van de week]
Niveau: [Helpende/Verzorgende/Verpleegkundige]
Rapportage: [Inhoud van de rapportage]
--EindRapportage
De rapportages moeten afwisselend geschreven worden door een helpende (20% kans), een verzorgende (60% kans), of een verpleegkundige (20% kans). Zorg voor variatie in de tijdstippen van de dag waarop de rapportages zijn geschreven, met een focus op overdag, en in mindere mate 's avonds en 's nachts. In de inhoud van elke rapportage dienen één of meer eigenschappen of beperkingen van de cliënt naar voren te komen.
'''

# Voeg bovenstaande tekst toe aan de clientprofielen
u_role_content = [client + u_role_instruction for client in clienten]

# Deze clienten zijn er...
for u_role in u_role_content:
    print(u_role.split('\n')[0])

# Lege lijst om de gegenereerde rapportages op te slaan
rapportages = []

# Itereer over elk element in u_role_content en genereer 10 weken aan rapportages
for u_role in u_role_content:
    # Genereer rapportages voor elke cliënt
    weekrapportage = genereer_rapportages(s_role=s_role_content, u_role=u_role, seed=1, n=10)
    # Voeg de gegenereerde rapportage toe aan de lijst
    rapportages.append(weekrapportage)
    print(u_role.split('\n')[0]) # Om bij te houden waar we zijn

# Om op te slaan als json:
# Maak een lege dictionary aan voor alle cliënten
appelboom_rapportages = {}

# Itereer over de lijst met rapportages en de lijst met cliënten
for i in range(len(rapportages)):
    # Gebruik een unieke identificatie voor elke cliënt, bijvoorbeeld de naam
    cliënt_id = clienten[i].split('\n')[0]  # Neem aan dat de eerste regel de naam of ID bevat

    # Maak een lege dictionary voor de rapportages van deze cliënt
    cliënt_rapportages = {}

    # Itereer over elke 'choice' (week) in de response
    for j in range(len(rapportages[i].choices)):
        # Gebruik de index als weeknummer
        weeknummer = f"Week {j+1}"

        # Sla de inhoud van de response op voor deze week
        cliënt_rapportages[weeknummer] = rapportages[i].choices[j].message.content

    # Voeg de dictionary voor deze cliënt toe aan de hoofddictionary
    appelboom_rapportages[cliënt_id] = cliënt_rapportages

# Nu bevat appelboom_rapportages de rapportages voor elke week voor elke cliënt
# Sla op als json
with open('zorgdata/appelboom_rapportages.json', 'w') as file:
    json.dump(appelboom_rapportages, file)


appelboom_clienten['clienten'][0]['profiel']

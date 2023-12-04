'''
Auteur:     Eva Rombouts
Datum:      04-12-2023
Project:    GenCareAI
Doel:       Dit script genereert zorgdata met behulp van de OpenAI api.
Vereisten:  De working directory moet een subdirectory 'roles' hebben met hierin de volgende bestanden:
            - rol_system_zorgdata_maker.txt: Tekstbestand met een beschrijving van hoe het systeem moet antwoorden
            - rol_user_???.txt: De vraag of opdracht waar het antwoord op moet volgen
            De string onder ??? verwijst naar mogelijke data die we willen genereren, dus bv:
            ADL: Maak een zorgplan item over de ADL
ToDo:       Overweeg ipv de zorgdata 5x te genereren hetvolgende:
                user: 'geef een voorbeeld'
                assistent: 'voorbeeld 1'
                user: 'nog eentje'
                assistent: 'voorbeeld 2', etc...
            Hier valt eindeloos te varieren met de rollen en opdrachten.
            
'''
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

def genereer_content(model, s_role_content, u_role_content, seed):
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": s_role_content},
            {"role": "user", "content": u_role_content}
        ],
        seed=seed
    )
    return completion

def genereer_zorgdata(model, zorgdata, aantal, seed=None):
    s_role_content = lees_bestand('roles/rol_system_zorgdata_maker.txt')
    u_role_content = lees_bestand(f'roles/rol_user_{zorgdata}.txt')

    if seed is None:
        seed = random.randint(0, 10000)

    zdata = []
    for i in range(aantal):
        zd = genereer_content(model, s_role_content, u_role_content, seed + i)
        zdata.append(zd.choices[0].message.content)
    
    for i, zd in enumerate(zdata, start=1):
        bestandsnaam = f'zorgdata/{zorgdata}/{model}_{datetime.now().strftime("%Y%m%d%H%M")}_{i}.txt'
        schrijf_bestand(bestandsnaam, zd)

genereer_zorgdata(
    model='gpt-3.5-turbo', 
    zorgdata='ADL', 
    aantal=30,
    seed=6)

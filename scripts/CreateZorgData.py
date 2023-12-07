'''
Auteur:     Eva Rombouts
Datum:      04-12-2023
Project:    GenCareAI
Doel:       Dit script genereert zorgdata met behulp van de OpenAI api.
ToDo:       Overweeg ipv de zorgdata 5x te genereren hetvolgende:
                user: 'geef een voorbeeld'
                assistent: 'voorbeeld 1'
                user: 'nog eentje'
                assistent: 'voorbeeld 2', etc...            
'''

from openai import OpenAI
from datetime import datetime
from HelperFunctions import lees_bestand, schrijf_bestand
import random
import os

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

def genereer_zorgdata(model, s_role_content, u_role_content, write_to, aantal, seed=None):

    if seed is None: # Als er geen seed wordt meegegeven willen we ook een random antwoord
        seed = random.randint(0, 10000)

    pad_naam = os.path.dirname(write_to)
    if not os.path.exists(pad_naam):
        os.makedirs(pad_naam)

    zdata = []
    
    for i in range(aantal):
        zd = genereer_content(model, s_role_content, u_role_content, seed + i)
        zdata.append(zd.choices[0].message.content)
    
    for i, zd in enumerate(zdata, start=1):
        bestandsnaam = f'{write_to}{model}_{datetime.now().strftime("%Y%m%d%H%M")}_{i}.csv'
        schrijf_bestand(bestandsnaam, zd)

model = 'gpt-3.5-turbo'
genereer_zorgdata(
    model = model, 
    s_role_content = lees_bestand('roles/rol_system_zorgdata_maker.txt'),
    u_role_content = lees_bestand('roles/rol_user_ADL.txt'),
    write_to= f'zorgdata/ADL/',
    aantal=3,
    seed=1)

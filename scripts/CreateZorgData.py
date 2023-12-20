'''
Auteur:     Eva Rombouts
Datum:      04-12-2023
Project:    GenCareAI
Doel:       Dit script genereert zorgdata met behulp van de OpenAI api.
'''

pass # Soms pakt ie mijn eerste statement niet...
from openai import OpenAI
from datetime import datetime
from HelperFunctions import lees_bestand, schrijf_bestand
import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Voorkomt een warning

# Functie om chatcompletions te genereren
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
    if seed is None:
        seed = random.randint(0, 10000)

    pad_naam = os.path.dirname(write_to)
    if not os.path.exists(pad_naam):
        os.makedirs(pad_naam)

    for i in range(aantal):
        zd = genereer_content(model, s_role_content, u_role_content, seed + i).choices[0].message.content
        bestandsnaam = f'{write_to}{model}_{datetime.now().strftime("%Y%m%d%H%M")}_{i+1}.txt'
        schrijf_bestand(bestandsnaam, zd)


# Roep de functie aan om zorgdata te genereren met het gedefinieerde model
model = 'gpt-3.5-turbo'
# model = 'gpt-4'
genereer_zorgdata(
    model = model, 
    s_role_content = lees_bestand('roles/rol_system_zorgdata_maker.txt'),
    u_role_content = lees_bestand('roles/rol_user_ADL.txt'),
    write_to= f'zorgdata/ADL/',
    aantal=5)

pass # Om te voorkomen dat ik twee keer zorgdata genereer
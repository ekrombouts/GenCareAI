import os
import pandas as pd
from HelperFunctions import lees_bestand

ADL_voorbeelden = []
pad = 'zorgdata/ADL/'

for fn in os.listdir(pad):
    ffn = os.path.join(pad, fn)
    if os.path.isfile(ffn):
        ADL_voorbeelden.append(lees_bestand(ffn))

voorbeeld = ADL_voorbeelden[0]
print(voorbeeld)
delen = voorbeeld.split('\n')

# DataFrame voor Probleem en Doel
probleem_doel_data = {'ID': [], 'Probleem': [], 'Doel': [], 'Opmerking': []}
acties_data = {'ProbleemDoelID': [], 'Actie': []}

for idx, voorbeeld in enumerate(ADL_voorbeelden):
    delen = voorbeeld.split('\n')
    actie_actief = False
    opmerking = ''
    for idd, deel in enumerate(delen):
        if actie_actief:
            if deel.startswith('-'):
                acties_data['ProbleemDoelID'].append(idx)
                acties_data['Actie'].append(deel.replace('-','').strip())
            else:
                actie_actief = False
        elif deel.startswith('Probleem:'):
            probleem = deel.replace('Probleem:', '').strip()
        elif deel.startswith('Doel:'):
            doel = deel.replace('Doel:', '').strip()
        elif deel.startswith('Acties:'):
            actie_actief = True
        else:
            opmerking = opmerking + deel
    probleem_doel_data['ID'].append(idx)
    probleem_doel_data['Probleem'].append(probleem)
    probleem_doel_data['Doel'].append(doel)
    probleem_doel_data['Opmerking'].append(opmerking)

df_probleem_doel = pd.DataFrame(probleem_doel_data)
df_acties = pd.DataFrame(acties_data)

print(df_probleem_doel['Doel'])
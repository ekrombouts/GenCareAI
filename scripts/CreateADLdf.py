import os
import pandas as pd
from HelperFunctions import lees_bestand

rapportagebestanden = []
pad = 'zorgdata/ADL/'

for fn in os.listdir(pad):
    ffn = os.path.join(pad, fn) # volledige bestandsnaam
    if os.path.isfile(ffn):
      rapportagebestanden.append(lees_bestand(ffn))

rap_data = {'ID': [], 'hulp': [], 'rapportage': []}

# Er zijn meerdere regels in meerdere (10) voorbeelden in meerdere files
for idx, rapportagebestand in enumerate(rapportagebestanden):
    voorbeelden = rapportagebestand.split('\n\n')
    for idv, voorbeeld in enumerate(voorbeelden):
        regels = voorbeeld.split('\n')
        for idr, regel in enumerate(regels):
            if regel.startswith('ADL_Hulp:'):
                hulp = regel.replace('ADL_Hulp:', '').strip()
            elif regel.startswith('Rapportage:'):
                rapportage = regel.replace('Rapportage:', '').strip()
        rap_data['ID'].append(idv)
        rap_data['hulp'].append(hulp)
        rap_data['rapportage'].append(rapportage)

df = pd.DataFrame(rap_data)

df.tail(30)
df.shape

df['prompt'] = df['rapportage'] + '\n\nWelke hulp heeft deze client nodig bij de ADL?'
print(df.loc[1,'prompt'])

df.to_csv('zorgdata/gen_data.csv')
pass
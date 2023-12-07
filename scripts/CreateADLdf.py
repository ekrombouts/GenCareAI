import os
import pandas as pd
from HelperFunctions import lees_bestand

rapportagebestanden = []
pad = 'zorgdata/ADL/'

for fn in os.listdir(pad):
    ffn = os.path.join(pad, fn)
    if os.path.isfile(ffn):
      rapportagebestanden.append(lees_bestand(ffn))

# rapportage = actie_rapportages[0]
# print(rapportage)
# delen = rapportage.split('\n')

rap_data = {'ID': [], 'Hulp': [], 'Rapportage': []}

for idx, rapportagebestanden in enumerate(rapportagebestanden):
    regels = rapportagebestanden.split('\n\n')
    for idr, regel in enumerate(regels):
        delen = regel.split('\n')
        for idd, deel in enumerate(delen):
            if deel.startswith('ADL_Hulp:'):
                hulp = deel.replace('ADL_Hulp:', '').strip()
            elif deel.startswith('Rapportage:'):
                rapportage = deel.replace('Rapportage:', '').strip()
        rap_data['ID'].append(idd)
        rap_data['Hulp'].append(hulp)
        rap_data['Rapportage'].append(rapportage)

df_rapportages = pd.DataFrame(rap_data)

df_rapportages.head(20)
df_rapportages.shape

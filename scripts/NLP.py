"""
Inspiratie: @author StellaVerkijk
"""

import pandas as pd
import time
import spacy
# import numpy as np
nlp = spacy.load('nl_core_news_sm')

lens = []
note_count = 0
start_time=time.time()

doc = df.iloc[0,2]
spacy_doc = nlp(doc)

type(spacy_doc)
for token in spacy_doc:
    print(token.text, token.pos_, token.ent_type_)

sentences = []
sentence = spacy_doc.sents
for sentence in spacy_doc.sents:
    sentences.append(sentence)
    n = 40
    chunks = [sentences[i * n:(i + 1) * n] for i in range((len(sentences) + n -1) // n)]
    lens_per_doc = []
    for chunk in chunks:
        lens_per_chunk = []
        for sentence in chunk:
            lens_per_chunk.append(len(sentence))
            if str(sentence).endswith('.'):
                print(str(sentence)+(' '))
            else:
                print(str(sentence))
                print('\n')
                lens_per_doc.append(lens_per_chunk)
                lens.append(lens_per_doc)
                
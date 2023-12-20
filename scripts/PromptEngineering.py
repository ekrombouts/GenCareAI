import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
# import sentencepiece

# Stel parameters in
seed = 6

# Laad de data
df = pd.read_csv('zorgdata/gen_data.csv')

# Verdeel de data in train- en testsets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)

# Converteer naar Hugging Face 'Dataset' objecten
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

# Toon voorbeelden uit de dataset
example_indices = [3, 4]
dash_line = '-' * 100
for i, index in enumerate(example_indices):
    print(dash_line, '\nVoorbeeld', i+1, '\nADL hulp:', dataset['test'][index]['hulp'], 
          '\nRapportage:', dataset['test'][index]['rapportage'], '\n', dash_line, '\n')

# Initialiseer model en tokenizer
model_name = 't5-small'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Voorbeeld van tekstcodering en -decodering
sentence = "Hoe laat is het, Mark?"
sentence_encoded = tokenizer.encode(sentence, return_tensors='pt')
sentence_decoded = tokenizer.decode(sentence_encoded[0], skip_special_tokens=True)
print("Encoded Sentence:", sentence_encoded)
print("Decoded sentence:", sentence_decoded)

# Genereer een antwoord op de prompt
text_to_answer = dataset['test'][2]['prompt']
inputs = tokenizer(text_to_answer, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=200, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)
print(dash_line, '\n', 'prompt:', text_to_answer, '\n', dash_line, '\nADL hulp:', tokenizer.decode(outputs[0], skip_special_tokens=True), '\n', dash_line)

# Demonstratie van model output zonder en met prompt engineering
for i, index in enumerate(example_indices):
    rapportage = dataset['test'][index]['rapportage']
    hulp = dataset['test'][index]['hulp']
    prompt = dataset['test'][index]['prompt']

    print(dash_line, '\nVOORBEELD ', i, '\nRAPPORTAGE:\n', rapportage)

    # Zonder prompt engineering
    inputs = tokenizer(rapportage, return_tensors='pt')
    output = tokenizer.decode(model.generate(inputs['input_ids'], max_new_tokens=50)[0], skip_special_tokens=True)
    print('\nModel output ZONDER prompt engineering:\n', output)

    # Met prompt engineering
    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(model.generate(inputs['input_ids'], max_new_tokens=50)[0], skip_special_tokens=True)
    print('\nModel output MET prompt engineering:\n', output)

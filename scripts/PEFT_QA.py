pass
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForQuestionAnswering, pipeline, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split

# Stel parameters in
seed = 6

# Allereerst gaan we de data laden, verdelen in train/validatie en test en omzetten in 
# HuggingFace datasets. HuggingFace datasets zijn niet geschikt voor wrangling, maar zijn
# geoptimaliseerd voor grote datasets en NLP taken. Hier gebruiken we die natuurlijk omdat 
# we met HuggingFace aan de slag gaan
df = pd.read_csv("zorgdata/gen_data.csv")

# Verdeel de data in train- en testsets
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=seed)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

# Converteer naar Hugging Face 'Dataset' objecten
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

# Creëer een DatasetDict met de train-, validatie- en testsets
dataset = DatasetDict({
    'train': train_dataset, 
    'validation': valid_dataset, 
    'test': test_dataset
})

# Nu kan je dus dingen doen als
dataset['validation']['hulp'][1]
dataset['train'].column_names

# Laden van het HuggingFace BERT model voor vraag-antwoord taken
model_name = "henryk/bert-base-multilingual-cased-finetuned-dutch-squad2"

# Het eenvoudigst gebruik je de pipeline, maar hiermee verlies je flexibiliteit om met je eigen
# data aan de gang te gaan.
qa_pipeline = pipeline(
    "question-answering",
    model = model_name,
    tokenizer = model_name
)

# De pipeline is de snelste manier om antwoorden te genereren. In het geval van een QA geef je de pipeline 
# een dictionary mee met de keys context en question 
qa_pipeline({
    'context': "Dhr vergat zijn rollator. Ik heb hem eraan herinnerd",
    'question': "Welke ADL hulp heeft de cliënt nodig?"})

# De pipeline laadt het model en de tokenizer. Als we dat zelf doen...
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.to('cpu') # Comes in handy later...

# Eerst maar s kijken hoe het tokenizen er uit ziet:
tokenized_output = tokenizer('Hallo, ik heet Eva')
print(tokenized_output)
decoded_text = tokenizer.decode(tokenized_output['input_ids'])
print(decoded_text)
# Je ziet hier dat 101 [CLS] is: Begin van de text
# [SEP] staat voor seperator, wordt ook gebruikt voor einde tekst. Dit is dus, anders dan bij GPT modellen
# geen aparte einde-tekst-token.
# Verder zie je dat er token_type_ids worden gegenereerd: Die zijn belang om het type tekst aan te geven
# in ons geval moet het voor de vraag 0 worden en voor de context 1. 

# Een voorbeeld om dit te illustreren:

question = "Wat is de hoofdstad van Nederland?"
context = "De hoofdstad van Nederland is Amsterdam."

# Tokenize de vraag en context
input_ids = tokenizer.encode(question, context, add_special_tokens=True)
token_type_ids = tokenizer(question, context, add_special_tokens=True)['token_type_ids']

print("Input IDs:", input_ids)
print("Token Type IDs:", token_type_ids)

print(tokenizer.decode(input_ids))

# En dan is er nog attention_mask. Deze is 0 bij padding:
tokenizer('Hallo, ik ben Eva', padding="max_length")
tokenizer.model_max_length # De max length is 512. Dat is voor vraag en antwoord samen. 

# Verder moet je nog specificeren: truncation (om fouten bij te lange teksten te voorkomen) en 
# de gebruikte tensor bibliotheek. Wij gebruiken PyTorch
tokenizer ('Hallo', truncation=True, return_tensors="pt")

# Ok. Dan gaan we nu dus onze dataset tokenizen. De eerste regel ziet er zo uit:
tokenizer('Welke hulp heeft de cliënt nodig?', test_dataset[0]['rapportage'], add_special_tokens=True)

def tokenize_dataset(ds):
    tokenized_outputs = []

    for rapportage in ds['rapportage']:
        # Tokenizen van de vraag en de individuele rapportage
        tokenized_output = tokenizer.encode_plus(
            'Welke hulp heeft de cliënt nodig?',
            rapportage,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        tokenized_outputs.append(tokenized_output)

    return tokenized_outputs

tokenized_subset = tokenize_dataset(train_dataset.select(range(3)))
print(tokenized_subset)


# HIER BEN IK. WERKT NOG NIET
# Toepassen van de functie op de dataset
tokenized_datasets = dataset.map(tokenize_dataset, batched=True)

tokenized_datasets = dataset.map(
    lambda df_to_tokenize: tokenize_dataset(df_to_tokenize), 
    batched=True
).remove_columns(['ID', 'hulp', 'rapportage', 'prompt'])
# Nu is...
tokenized_datasets['test'][1]

def tokenize_function(df_to_tokenize):
    rapportage = [pt for pt in df_to_tokenize["rapportage"]]
    df_to_tokenize['input_ids'] = tokenizer(rapportage, padding="max_length", truncation=True, return_tensors="pt").input_ids
    df_to_tokenize['labels'] = tokenizer(df_to_tokenize["hulp"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    return df_to_tokenize

vb = dataset['test'][1]
type(vb)
print(vb)
tokenize_function(vb)

# Tokenizeer de dataset
tokenized_datasets = dataset.map(
    lambda df_to_tokenize: tokenize_function(df_to_tokenize), 
    batched=True
).remove_columns(['ID', 'hulp', 'rapportage', 'prompt'])


# PEFT (Parameter Efficient Fine-Tuning) configuratie
lora_config = LoraConfig(
    r=32,  # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM  # Specificeer taaktype
)

# Maak PEFT-model met de geconfigureerde instellingen
peft_model = get_peft_model(original_model, lora_config)

# Train PEFT adapter
output_dir = f'./peft_model'
peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,  # Hogere leerfrequentie dan volledige fine-tuning
    num_train_epochs=5,
    logging_steps=2,
    max_steps=50    
)

# Configureer de trainer voor PEFT
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)

# Train het PEFT-model
peft_trainer.train()

# Evaluatie van het PEFT-model
index = 21

rapportage = dataset['test'][index]['rapportage']
original_hulp = dataset['test'][index]['hulp']
prompt = dataset['test'][index]['prompt']

# Tokenizeer de input voor het PEFT-model
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
peft_trainer.model.to('cpu')
input_ids.to('cpu')

# Genereer samenvattingen met de verschillende modellen
original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
peft_model_outputs = peft_trainer.model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

# Print resultaten
dash_line = '-' * 100
print(dash_line, 
      f'\nORIGINELE HULP BESCHRIJVING:\n{original_hulp}\n',
      dash_line,
      f'\nRAPPORTAGE: {rapportage}\n',
      dash_line,
      f'\nORIGINAL MODEL:\n{original_model_text_output}\n',
      dash_line,
      f'\nPEFT MODEL: {peft_model_text_output}')









# -------------------------
def generate_predictions(model, tokenizer, dataset):
    model.eval()  # Zet het model in evaluatiemodus
    predictions = []
    
    for item in dataset:
        with torch.no_grad():
            input_ids = item['input_ids'].unsqueeze(0)  # Voeg batch dimensie toe
            output = model.generate(input_ids)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)
            predictions.append(prediction)
    
    return predictions

# Genereer voorspellingen
original_predictions = generate_predictions(original_model, tokenizer, test_dataset)
peft_predictions = generate_predictions(peft_model, tokenizer, test_dataset)

# Voeg voorspellingen toe aan de DataFrame
test_df['original_model_predictions'] = original_predictions
test_df['peft_model_predictions'] = peft_predictions

# Nu is `test_df` bijgewerkt met de nieuwe kolommen
# -------------------------










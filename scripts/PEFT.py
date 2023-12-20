pass
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import evaluate
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split

# Stel parameters in
seed = 6

# Laad data
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

# Laad en configureer het model
model_name = 't5-small'
# model_name = 'google/mt5-small'
# model_name = 'google/flan-t5-small'
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)

original_model.to('cpu')

def tokenize_function(to_tokenize):
    prompt = [pt for pt in to_tokenize["prompt"]]
    to_tokenize['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    to_tokenize['labels'] = tokenizer(to_tokenize["hulp"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    return to_tokenize

vb = dataset['test'][1]
tokenize_function(vb)

print(tokenize_function(df.iloc[1]))

# Tokenizeer de dataset
tokenized_datasets = dataset.map(
    lambda to_tokenize: tokenize_function(to_tokenize), 
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













# REPLACED: 
# waarderingen = rapportages
# waardering = rapportage
# human_baseline_summaries = baseline_ADL
# reactie = hulp
# original_model_summaries = original_model_ADL
# peft_model_summaries = peft_model_ADL

# Evalueer met Rouge 
# Haal een subset van dialogen en hun samenvattingen op
rapportages = dataset['test'][0:10]['rapportage']
baseline_ADL = dataset['test'][0:10]['hulp']
prompts = dataset['test'][0:10]['prompt']

# Initialiseren van lijsten voor samenvattingen
original_model_ADL = []
peft_model_ADL = []

# Genereer samenvattingen voor elke dialoog in de subset
for idx, prompt in enumerate(prompts):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    human_baseline_text_output = baseline_ADL[idx]
    
    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

    original_model_ADL.append(original_model_text_output)
    peft_model_ADL.append(peft_model_text_output)

# Maak een DataFrame met de samenvattingen
zipped_summaries = list(zip(baseline_ADL, original_model_ADL, peft_model_ADL)) 
df = pd.DataFrame(zipped_summaries, columns = ['baseline_ADL', 'original_model_ADL', 'peft_model_ADL'])

# Bereken ROUGE-score voor deze subset van de data
rouge = evaluate.load('rouge')

# Bereken ROUGE-scores voor elk model
original_model_results = rouge.compute(predictions=original_model_ADL, references=baseline_ADL, use_aggregator=True, use_stemmer=True)
peft_model_results = rouge.compute(predictions=peft_model_ADL, references=baseline_ADL, use_aggregator=True, use_stemmer=True)

# Print de ROUGE-scores
print('ORIGINAL MODEL:')
print(original_model_results)
print('PEFT MODEL:')
print(peft_model_results)

# Bereken en print de verbetering van PEFT ten opzichte van de originele en gefinetunede modellen
print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL")
improvement = (np.array(list(peft_model_results.values())) - np.array(list(original_model_results.values())))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}: {value*100:.2f}%')
# pip install flask
# pip install --upgrade charset_normalizer

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

# Load your trained BERT model
model_path = 'models/bert_classification' 

# Load the finetuned Bert model
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('wietsedv/bert-base-dutch-cased')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract text from the request
    data = request.json
    text = data['text']

    # Preprocess and tokenize text
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1).tolist()

    # Respond with the predicted probabilities
    return jsonify(probabilities)

if __name__ == '__main__':
    app.run
#    app.run(debug=True)

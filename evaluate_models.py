#File: Evaluate_Models.py
import pandas as pd
import os
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

# Constants
BASE_PATH = "C:\\Users\\Igor Carreon\\Documents\\Other\\Masters\\AI-HC\\MIMICIV\\"
DISCHARGE_FILE = 'discharge.csv'
RANDOM_FOREST_MODEL_PATH = 'random_forest_model.pkl'
BERT_MODEL_PATH = 'bert_model.pth'
BERT_PRETRAINED_MODEL = 'emilyalsentzer/Bio_ClinicalBERT'
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 2)
BATCH_SIZE = 4

# Function to load CSV files
def load_csv(filepath):
    return pd.read_csv(filepath, low_memory=False)

# Function to annotate high-risk notes
def annotate_high_risk_notes(notes_df, sa_keywords, si_keywords):
    def check_for_keywords(text, keywords):
        text = str(text).lower()
        return any(keyword in text for keyword in keywords)

    notes_df['SA_Flag'] = notes_df['text'].apply(lambda text: 1 if check_for_keywords(text, sa_keywords) else 0)
    notes_df['SI_Flag'] = notes_df['text'].apply(lambda text: 1 if check_for_keywords(text, si_keywords) else 0)
    notes_df['High_Risk_Flag'] = notes_df[['SA_Flag', 'SI_Flag']].max(axis=1)
    return notes_df

# Function to load models
def load_models(random_forest_path, bert_path, device):
    random_forest_model = joblib.load(random_forest_path)
    tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_MODEL)
    bert_model = BertForSequenceClassification.from_pretrained(BERT_PRETRAINED_MODEL, num_labels=2)
    bert_model.load_state_dict(torch.load(bert_path, map_location=device))
    bert_model.to(device)
    bert_model.eval()
    return random_forest_model, tokenizer, bert_model

# Function to evaluate the Random Forest model
def evaluate_random_forest_model(X, y, model):
    y_pred = model.predict(X)
    print("\n--- Random Forest Model Evaluation ---")
    print(classification_report(y, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

# Function to evaluate the BERT model
def evaluate_bert_model(texts, labels, tokenizer, model, device):
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    labels_tensor = torch.tensor(labels.values).to(device)

    dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids_batch, attention_mask_batch, _ = batch
            outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    print("\n--- BERT Model Evaluation ---")
    print(classification_report(labels, predictions, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, predictions))

# Main function
def main():
    # Load and prepare data
    discharge_df = load_csv(os.path.join(BASE_PATH, DISCHARGE_FILE))
    discharge_df = discharge_df.sample(n=50000, random_state=42)

    # Define SA and SI Keyword Lists
    SA_KEYWORDS = [
        'suicide attempt', 'attempted suicide', 'overdose', 'self-harm', 'self-injury',
        'self-inflicted', 'cutting', 'stabbed self', 'ingest poison', 'jumped off',
        'gunshot', 'cut wrist', 'hang', 'hanging', 'took pills', 'kill myself'
    ]

    SI_KEYWORDS = [
        'suicidal ideation', 'wish to die', 'thoughts of suicide', 'thoughts of self-harm',
        'thinking of suicide', 'want to end my life', 'hopeless', 'no reason to live',
        'ending it all', 'thinking about suicide', 'suicidal thoughts', 'feel like dying',
        'voice told to die'
    ]

    discharge_df = annotate_high_risk_notes(discharge_df, SA_KEYWORDS, SI_KEYWORDS)

    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_forest_model, tokenizer, bert_model = load_models(RANDOM_FOREST_MODEL_PATH, BERT_MODEL_PATH, device)

    # Vectorize data for Random Forest model
    tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE, stop_words='english')
    X = tfidf_vectorizer.fit_transform(discharge_df['text']).toarray()
    y = discharge_df['High_Risk_Flag']

    # Evaluate Random Forest model
    evaluate_random_forest_model(X, y, random_forest_model)

    # Evaluate BERT model
    evaluate_bert_model(discharge_df['text'], y, tokenizer, bert_model, device)

if __name__ == "__main__":
    main()    
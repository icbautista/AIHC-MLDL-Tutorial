#File: Train_Models.py
import pandas as pd
import os
import joblib  # For saving traditional ML models
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # For a progress bar

# base path for dataset 
BASE_PATH = "C:\\Users\\Igor Carreon\\Documents\\Other\\Masters\\AI-HC\\MIMICIV\\"
DISCHARGE_FILE = 'discharge.csv'

# Function to load CSV files
def load_csv(filepath):
    """Loads a CSV file into a DataFrame."""
    return pd.read_csv(filepath, low_memory=False)

# Load and sample the CSV file to reduce the number of records to 100,000
discharge_df = load_csv(os.path.join(BASE_PATH, DISCHARGE_FILE))
discharge_df = discharge_df.sample(n=100000, random_state=42)  # Sample 100,000 records

# Define SA and SI Keyword Lists. SA workds are Suicide Attempt and SI words are Suicidal Ideation
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

# Function to check for keywords in text
def check_for_keywords(text, keywords):
    text = str(text).lower()
    return any(keyword in text for keyword in keywords)

# Annotate high-risk notes
def annotate_high_risk_notes(notes_df):
    notes_df['SA_Flag'] = notes_df['text'].apply(
        lambda text: 1 if check_for_keywords(text, SA_KEYWORDS) else 0
    )
    notes_df['SI_Flag'] = notes_df['text'].apply(
        lambda text: 1 if check_for_keywords(text, SI_KEYWORDS) else 0
    )
    # Combine both flags into a High_Risk_Flag
    notes_df['High_Risk_Flag'] = notes_df[['SA_Flag', 'SI_Flag']].max(axis=1)
    return notes_df

# Annotate the discharge notes to create the 'High_Risk_Flag'
discharge_df = annotate_high_risk_notes(discharge_df)

# Model evaluation function
def evaluate_model(y_true, y_pred, model_name=""):
    print(f"\n--- {model_name} Evaluation Metrics ---")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# Function to train and save traditional ML models
def train_and_evaluate_tfidf_model(X, y, model, model_name, save_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"{model_name} model saved to {save_path}")
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, model_name)

# Function to train and save BERT/ClinicalBERT model
def train_bert_model(model, dataloader, optimizer, criterion, device, epochs=3, save_path="bert_model.pth"):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{epochs} ---")
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} Progress", leave=False)
        for step, batch in enumerate(progress_bar):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (step + 1) % 100 == 0:
                print(f"Step {step + 1}/{len(dataloader)} - Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"BERT model saved to {save_path}")
    print("\n--- Training Complete ---")

# Prepare the data for data loading
def prepare_data():
    discharge_df = load_csv(os.path.join(BASE_PATH, DISCHARGE_FILE))
    discharge_df = discharge_df.sample(n=100000, random_state=42)
    discharge_df = annotate_high_risk_notes(discharge_df)
    return discharge_df

# Vectorize the data for feature engineering
def vectorize_data(discharge_df):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words='english')
    X = tfidf_vectorizer.fit_transform(discharge_df['text']).toarray()
    y = discharge_df['High_Risk_Flag']
    return X, y

def train_dl_model(discharge_df, y):
    # BertTokenizer and Bio_ClinicalBERT model setup
    tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = BertForSequenceClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', num_labels=2)
    
    # Tokenization and DataLoader setup
    encodings = tokenizer(discharge_df['text'].tolist(), truncation=True, padding=True, max_length=512) # Truncate to 512 tokens, library tokenuzez the text and pads it to the max length
    input_ids = torch.tensor(encodings['input_ids']) # Convert the tokenized text to tensor
    attention_mask = torch.tensor(encodings['attention_mask']) # Convert the attention mask to tensor. It indicates to the model which tokens should be attended to and which should be ignored.
    labels = torch.tensor(y.values) # Converts the target labels (y, which contains the binary classification flags for high-risk notes) into a PyTorch tensor
    dataset = TensorDataset(input_ids, attention_mask, labels) # Creates a PyTorch TensorDataset containing the input IDs, attention masks, and labels. This dataset will be used to load the data into the model in batches.
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True) # Creates a PyTorch DataLoader that provides an iterable over the dataset. It allows for easy batching, shuffling, and parallel loading of the data when training the model.
    
    # Optimizer and device setup
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train the Bio_ClinicalBERT model
    train_bert_model(model, dataloader, optimizer, criterion, device, epochs=5, save_path="bert_model.pth")

def train_ml_models(X, y):
    # Define and train Logistic Regression model
    logistic_regression = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    train_and_evaluate_tfidf_model(X, y, logistic_regression, "Logistic Regression", "logistic_regression_model.pkl")
    
    # Define and train Random Forest model
    random_forest = RandomForestClassifier(class_weight='balanced', n_estimators=200, max_depth=10, random_state=42)
    train_and_evaluate_tfidf_model(X, y, random_forest, "Random Forest", "random_forest_model.pkl")

# Main function
def main():
    print("\n--- Data Preparation ---")
    discharge_df = prepare_data()
    
    print("\n--- Feature Engineering ---")
    X, y = vectorize_data(discharge_df)
    
    print("\n--- ML Model Training ---")
    train_ml_models(X, y)
    
    print("\n--- DL Model Training ---")
    train_dl_model(discharge_df, y)

if __name__ == "__main__":
    main()
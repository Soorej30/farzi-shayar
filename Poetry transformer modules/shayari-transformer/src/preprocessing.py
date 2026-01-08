import pandas as pd
from pathlib import Path
import re
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('punkt_tab')

# Removing unwanted characters
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s\-]', '', text)  # Remove punctuation except hyphens
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.strip()  # Remove leading/trailing spaces
    return text

def preprocessing():
    # resolve repository root (adjust parents[n] if your layout differs)
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / 'data' / 'raw' / 'shayari_db.csv'
    df = pd.read_csv(csv_path)[['AUTHOR', 'POEM']]

    # df = pd.read_csv('data/raw/shayari_db.csv')[['AUTHOR', 'POEM']]
    # Apply text cleaning
    df['CLEAN_POEM'] = df['POEM'].astype(str).apply(clean_text)

    # Tokenization
    df['TOKENS'] = df['CLEAN_POEM'].apply(word_tokenize)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['CLEAN_POEM'])

    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(df['CLEAN_POEM'])

    # Padding sequences to ensure uniform length
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    max_token = 0
    for i in range(1519):
        if len(df['TOKENS'][i]) > max_token:
            max_token = len(df['TOKENS'][i])

    # print(max_token)

    # df = pd.read_csv('data/shayari_db.csv')[['AUTHOR', 'POEM']]
    processed_path = repo_root / 'data' / 'processed' / 'processed_poetry.csv'
    print("Writing to - ", processed_path)
    df.to_csv(processed_path, index=False)

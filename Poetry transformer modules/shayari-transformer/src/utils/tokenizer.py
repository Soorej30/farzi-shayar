import pickle
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

def build_and_save_tokenizer(csv_path, out_dir, num_words=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    texts = df['CLEAN_POEM'].astype(str).tolist()

    # Explicit start/end tokens
    augmented_texts = [f"<start> {t} <end>" for t in texts]

    tokenizer = Tokenizer(
        num_words=num_words,
        filters='',
        lower=True,
        oov_token='<unk>'
    )
    tokenizer.fit_on_texts(augmented_texts)

    # Save tokenizer
    with open(out_dir / 'tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # Save sequences (already include start/end)
    sequences = tokenizer.texts_to_sequences(augmented_texts)
    np.save(out_dir / 'sequences.npy', np.array(sequences, dtype=object), allow_pickle=True)

    return tokenizer, sequences

def load_tokenizer(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

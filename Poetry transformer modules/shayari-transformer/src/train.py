import tensorflow as tf
from pathlib import Path
from src.dataloader import prepare_dataset
from src.transformer_model import build_model

def train():
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo/'data'/'processed'
    ds, vocab_size, start_id, end_id, tokenizer = prepare_dataset(
        tokenizer_path = data_dir/'tokenizer.pkl',
        sequences_path = data_dir/'sequences.npy',
        max_len = 64,
        batch_size = 32
    )

    seq_len = 63
    model = build_model(seq_len, vocab_size, d_model = 128, num_heads = 4, dff = 512, num_layers = 2)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    model.compile(optimizer = 'adam', loss = loss, metrics = ['sparse_categorical_accuracy'])
    model.fit(ds, epochs = 10)
    model.save(repo / 'models' / 'transformer_poetry')

if __name__ == '__main__':
    train()
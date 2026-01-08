import argparse
from pathlib import Path
import pickle

from src.preprocessing import preprocessing
from src.utils.tokenizer import build_and_save_tokenizer
from src.dataloader import prepare_dataset
from src.transformer_model import build_model
from src.generate import generate as generate_sample

import tensorflow as tf

def train_pipeline(repo_root: Path, epochs: int, batch_size: int, max_len: int, model_out: str = None):
    # 1) Preprocess and build tokenizer
    preprocessing()
    processed_csv = repo_root / 'data' / 'processed' / 'processed_poetry.csv'
    processed_dir = repo_root / 'data' / 'processed'
    build_and_save_tokenizer(processed_csv, processed_dir)

    # 2) Prepare dataset
    ds, vocab_size, start_id, end_id, tokenizer = prepare_dataset(
        tokenizer_path=processed_dir / 'tokenizer.pkl',
        sequences_path=processed_dir / 'sequences.npy',
        max_len=max_len,
        batch_size=batch_size
    )

    # determine seq_len from dataset (inputs shape)
    for batch_in, _ in ds.take(1):
        seq_len = batch_in.shape[1]
        break

    # 3) Build model
    model = build_model(seq_len, vocab_size, d_model=128, num_heads=4, dff=512, num_layers=2, rate=0.1)

    # 4) Train
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss, metrics=['sparse_categorical_accuracy'])
    model.fit(ds, epochs=epochs)

    # 5) Save model
    models_dir = repo_root / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / (model_out or 'transformer_poetry')

    # ensure a valid extension for Keras saving
    if model_path.suffix == '':
        model_path = model_path.with_suffix('.keras')  # or '.h5'

    model.save(str(model_path))

    return {
        "model_path": model_path,
        "tokenizer_path": processed_dir / 'tokenizer.pkl',
        "start_id": start_id,
        "end_id": end_id,
        "vocab_size": vocab_size,
    }

def generate_pipeline(repo_root: Path, model_path: Path, start_word: str, generate_max_len: int, tokenizer_path: Path = None):
    processed_dir = repo_root / 'data' / 'processed'
    tokenizer_path = tokenizer_path or (processed_dir / 'tokenizer.pkl')

    # recompute start/end ids from tokenizer (must match training logic)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    base_vocab = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index

    start_id = word_index['<start>']
    end_id   = word_index['<end>']

    sample = generate_sample(
        start_text=start_word,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        start_id=start_id,
        end_id=end_id,
        max_len=generate_max_len,
    )
    print("Generated:", sample)
    return sample

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "generate", "all"], default="all",
                   help="Which pipeline to run")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--model_out", type=str, default=None)
    p.add_argument("--model_path", type=str, default=None, help="Path to saved model for generation")
    p.add_argument("--start_word", type=str, default="tera")
    p.add_argument("--generate_max_len", type=int, default=40)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    trained = None
    if args.mode in ("train", "all"):
        trained = train_pipeline(
            repo_root=repo_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_len=args.max_len,
            model_out=args.model_out,
        )

    if args.mode in ("generate", "all"):
        # determine model path: prefer provided, else use path returned from training
        model_path = Path(args.model_path) if args.model_path else (trained["model_path"] if trained else None)
        if model_path is None:
            raise SystemExit("No model_path provided and nothing was trained in this run. Use --model_path or run with mode=train or mode=all.")
        generate_pipeline(
            repo_root = repo_root,
            model_path = model_path,
            start_word = args.start_word,
            generate_max_len = args.generate_max_len,
            tokenizer_path = None
        )
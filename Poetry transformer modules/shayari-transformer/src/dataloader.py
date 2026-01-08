import numpy as np
import tensorflow as tf
import pickle


def prepare_dataset(tokenizer_path, sequences_path, max_len=None, batch_size=64):
    # ----------------------------
    # Load tokenizer and sequences
    # ----------------------------
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    sequences = np.load(sequences_path, allow_pickle=True)

    word_index = tokenizer.word_index

    # ----------------------------
    # Enforce tokenizer invariants
    # ----------------------------
    if '<start>' not in word_index or '<end>' not in word_index:
        raise ValueError(
            "Tokenizer must contain <start> and <end> tokens. "
            "Do NOT inject token IDs manually."
        )

    start_id = word_index['<start>']
    end_id = word_index['<end>']

    # ----------------------------
    # Validate sequences
    # ----------------------------
    max_token_id = max(word_index.values())

    for seq in sequences:
        if seq is None or len(seq) == 0:
            continue
        if np.max(seq) > max_token_id:
            raise ValueError(
                f"Sequence contains token id {np.max(seq)} "
                f"but tokenizer max id is {max_token_id}. "
                f"Tokenizer and sequences are out of sync."
            )

    # ----------------------------
    # Add <start>/<end>
    # ----------------------------
    processed = []
    for seq in sequences:
        if seq is None or len(seq) == 0:
            continue
        processed.append([start_id] + list(seq) + [end_id])

    if not processed:
        raise ValueError("No valid sequences found")

    # ----------------------------
    # Padding
    # ----------------------------
    if max_len is None:
        max_len = max(len(s) for s in processed)

    padded = tf.keras.preprocessing.sequence.pad_sequences(
        processed,
        maxlen=max_len,
        padding='post',
        truncating='post',
        value=0
    )

    inputs = padded[:, :-1]
    targets = padded[:, 1:]

    dataset = (
        tf.data.Dataset
        .from_tensor_slices((inputs, targets))
        .shuffle(2048)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ----------------------------
    # ✅ SINGLE SOURCE OF TRUTH
    # ----------------------------
    vocab_size = max_token_id + 1

    return dataset, vocab_size, start_id, end_id, tokenizer

import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path
from src.transformer_model import CausalTransformerBlock, CausalMask

def generate(start_text, model_path, tokenizer_path, start_id, end_id, max_len=100, temperature=0.8):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "CausalTransformerBlock": CausalTransformerBlock,
            "CausalMask": CausalMask,
        }
    )

    seq_len = model.input_shape[1]

    prompt_tokens = tokenizer.texts_to_sequences([start_text])[0]
    if not prompt_tokens:
        prompt_tokens = [tokenizer.word_index.get('<unk>', 1)]

    seq = [start_id] + prompt_tokens

    for _ in range(max_len):
        pad = tf.keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=seq_len, padding='post'
        )
        logits = model(pad, training=False)[0]
        step_logits = logits[len(seq) - 1] / temperature
        probs = tf.nn.softmax(step_logits)
        next_id = int(tf.random.categorical(tf.math.log([probs]), 1)[0, 0])

        seq.append(next_id)
        if next_id == end_id:
            break

    inv = {v: k for k, v in tokenizer.word_index.items()}
    inv[0] = '<pad>'

    words = [inv.get(i, '<unk>') for i in seq if i not in (start_id, end_id)]
    return ' '.join(words)
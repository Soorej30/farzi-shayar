import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="custom")
class CausalMask(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, t):
        seq = tf.shape(t)[1]
        mask = tf.linalg.band_part(tf.ones((seq, seq), dtype=tf.float32), -1, 0)
        return tf.expand_dims(mask, axis=0)

@tf.keras.utils.register_keras_serializable(package="custom")
class CausalTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)  # ← THIS IS THE KEY FIX

        # Store config params
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(d_model),
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=None, mask=None):
        attn_output = self.mha(
            x, x,
            attention_mask=mask,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "rate": self.rate,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="custom")
class PositionalEncoding(layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_encoding = get_positional_encoding(seq_len, d_model)

    def call(self, x):
        return x + tf.cast(self.pos_encoding, x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "d_model": self.d_model,
        })
        return config

def get_positional_encoding(seq_len, d_model):
    # positions: (seq_len, 1), dims: (1, d_model)
    positions = tf.range(seq_len)[:, tf.newaxis]
    dims = tf.range(d_model)[tf.newaxis, :]

    # compute angle rates with consistent float dtype
    angle_den = tf.cast(d_model, tf.float32)
    angle_num = tf.cast(2 * (dims // 2), tf.float32)
    angle_rates = 1.0 / tf.pow(10000.0, angle_num / angle_den)

    angle_rads = tf.cast(positions, tf.float32) * angle_rates

    sines = tf.sin(angle_rads[:, 0::2])
    coses = tf.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, coses], axis=-1)
    return pos_encoding

def build_model(seq_len, vocab_size, d_model=128, num_heads=4, dff=512, num_layers=2, rate=0.1):
    inputs = layers.Input(shape=(seq_len,), dtype="int32")

    x = layers.Embedding(vocab_size, d_model, mask_zero=False)(inputs)
    x = PositionalEncoding(seq_len, d_model)(x)

    mask = CausalMask()(inputs)

    for _ in range(num_layers):
        x = CausalTransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            rate=rate
        )(x, mask=mask)

    logits = layers.Dense(vocab_size)(x)

    return tf.keras.Model(inputs, logits)

"""Text emotion recognition: Relative EA-LSTM + rule-based emoji detector.

The custom Keras layers below must stay identical to the ones used during
training so that the saved model (models/combined_final.keras) loads correctly.
"""

import re

import contractions
import nltk
import numpy as np
import tensorflow as tf
from nrclex import NRCLex
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.saving import register_keras_serializable


# ------------------------------------------------------------------
# NLTK resources
# ------------------------------------------------------------------
_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",          # used by NRCLex / TextBlob
    "punkt_tab": "tokenizers/punkt_tab",  # required by newer NLTK versions
    "wordnet": "corpora/wordnet",         # used by WordNetLemmatizer
    "omw-1.4": "corpora/omw-1.4",
}


def ensure_nltk_resources():
    """Download the NLTK data this module needs, only if it is missing."""
    for package, path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_resources()

from nltk.stem import WordNetLemmatizer  # noqa: E402  (after resources exist)

_lemmatizer = WordNetLemmatizer()


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
def preprocess(text):
    """Lower-case, expand contractions, strip noise and lemmatise."""
    text = contractions.fix(text.lower())

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"\@\w+|#", "", text)                  # mentions / hashtags
    text = re.sub(r"[^a-z\s]", "", text)                 # non-alphabetic chars

    tokens = [_lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(tokens).strip()


def get_emotion_vector(text):
    """Build the 7-dim NRC emotion-lexicon vector fed to the EA-LSTM.

    Note: the layout of this vector (including which lexicon category is
    replaced by the "neutral" flag) matches what the model saw during
    training. Do not change it without retraining the model.
    """
    scores = NRCLex(text).raw_emotion_scores

    emo_labels = ["anger", "disgust", "fear", "joy", "anticipation", "sadness", "surprise"]
    vec = np.array([scores.get(e, 0) for e in emo_labels], dtype=float)

    neutral = 1.0 if vec.sum() == 0 else 0.0
    vec = np.append(vec[:-1], neutral)

    if vec.sum() > 0:
        vec = vec / vec.sum()
    return vec


# ------------------------------------------------------------------
# Custom layers
# ------------------------------------------------------------------
@register_keras_serializable()
class RelativeMultiheadAttention(Layer):
    """Multi-head self-attention with a learnable relative positional bias."""

    def __init__(self, input_dim, d_model, num_heads, max_len, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_len = max_len

    def build(self, input_shape):
        self.qkv_layer = Dense(3 * self.d_model)
        self.linear_layer = Dense(self.d_model)

        # One learnable bias per (relative distance, head)
        self.relative_bias_table = self.add_weight(
            name="relative_bias_table",
            shape=(2 * self.max_len - 1, self.num_heads),
            initializer="zeros",
            trainable=True,
        )

        # Matrix of relative distances between every pair of positions
        coords = tf.range(self.max_len)
        relative_coords = coords[None, :] - coords[:, None]
        relative_coords += self.max_len - 1
        self.relative_index = tf.constant(relative_coords)

    def call(self, x, mask=None):
        shape = tf.shape(x)
        batch_size, seq_len = shape[0], shape[1]

        qkv = self.qkv_layer(x)
        qkv = tf.reshape(qkv, [batch_size, seq_len, self.num_heads, 3 * self.head_dim])
        qkv = tf.transpose(qkv, [0, 2, 1, 3])
        q, k, v = tf.split(qkv, 3, axis=-1)

        attn_scores = tf.matmul(q, k, transpose_b=True)
        attn_scores /= tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        # Inject relative positional bias
        rel_index = self.relative_index[:seq_len, :seq_len]
        rel_bias = tf.gather(self.relative_bias_table, rel_index)
        rel_bias = tf.transpose(rel_bias, [2, 0, 1])
        attn_scores = attn_scores + tf.expand_dims(rel_bias, 0)

        if mask is not None:
            attn_scores += mask

        attn_weights = tf.nn.softmax(attn_scores, axis=-1)

        out = tf.matmul(attn_weights, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch_size, seq_len, self.d_model])

        return self.linear_layer(out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "max_len": self.max_len,
        })
        return config


@register_keras_serializable()
class EmotionGatedLSTMCell(Layer):
    """EA-LSTM cell: a standard LSTM on text features plus an Emotion Memory
    Candidate that projects the emotion vector directly into the cell state,
    scaled by a learnable strength parameter (alpha)."""

    def __init__(self, units, embedding_dim=128, emotion_dim=7, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.embedding_dim = embedding_dim
        self.emotion_dim = emotion_dim
        self.state_size = [units, units]
        self.output_size = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.embedding_dim + self.units, 4 * self.units),
            initializer="glorot_uniform", name="lstm_kernel")

        self.bias = self.add_weight(
            shape=(4 * self.units,), initializer="zeros", name="lstm_bias")

        self.emotion_gate_kernel = self.add_weight(
            shape=(self.emotion_dim, self.units),
            initializer="glorot_uniform", name="emotion_gate_kernel")

        self.alpha = self.add_weight(shape=(1,), initializer="ones", name="alpha")

        super().build(input_shape)

    def call(self, inputs, states):
        h_prev, c_prev = states

        x_text = inputs[:, :self.embedding_dim]
        x_emotion = inputs[:, self.embedding_dim:]

        # 1. Standard LSTM gates on the text features
        concat_input = tf.concat([x_text, h_prev], axis=-1)
        gates = tf.matmul(concat_input, self.kernel) + self.bias
        i, f, c_cand, o = tf.split(gates, num_or_size_splits=4, axis=-1)

        f = tf.sigmoid(f + 1.0)
        i = tf.sigmoid(i)
        c_cand = tf.nn.tanh(c_cand)
        o = tf.sigmoid(o)

        # 2. Emotion Memory Candidate computed from the emotion vector only
        gate_activation = tf.nn.tanh(tf.matmul(x_emotion, self.emotion_gate_kernel))

        # 3. Inject emotion into long-term memory
        c_new = f * c_prev + i * c_cand + self.alpha * gate_activation
        h_new = o * tf.nn.tanh(c_new)

        return h_new, [h_new, c_new]

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


CUSTOM_OBJECTS = {
    "Attention": RelativeMultiheadAttention,
    "EmotionGatedLSTMCell": EmotionGatedLSTMCell,
}

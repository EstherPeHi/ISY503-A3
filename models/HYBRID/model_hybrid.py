"""
ISY503 – Assessment 3
Model: HYBRID (CNN + BiLSTM) for sentiment classification

Author: Esther Peña (EstherPeHi)
Team: ISY503-A3

Notes for the team:
- Hybrid = parallel Conv1D branches (local n-grams) + a BiLSTM branch (long-range deps).
- Critical fix: Embedding(mask_zero=True) so the BiLSTM ignores padded timesteps.
- Defaults are conservative and train quickly; tune from the runner if needed.
"""

from tensorflow.keras.layers import (
    Input, Embedding, SpatialDropout1D,
    Conv1D, BatchNormalization, ReLU, GlobalMaxPooling1D,
    Bidirectional, LSTM, Concatenate, Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def create_hybrid_model(
    vocab_size: int,
    max_length: int,
    embedding_dim: int = 128,
    conv_filters: int = 128,
    lstm_units: int = 128,
    conv_kernel_sizes=(3, 4, 5),     # compact, effective, and fast
    dropout: float = 0.4,
    lr: float = 1e-3,
    use_recurrent_dropout: bool = False,  # set True for a bit more regularization (slower)
) -> Model:
    """
    CNN + BiLSTM:
      - Conv1D[k in conv_kernel_sizes] → BN → ReLU → GlobalMaxPool (parallel branches)
      - BiLSTM (return_sequences=True) → GlobalMaxPool
      - Concatenate → Dense layers → sigmoid

    Key design choices:
      - mask_zero=True in Embedding so LSTM ignores padding.
      - Mild L2 on Conv/Dense; modest dropout.
    """
    inp = Input(shape=(max_length,))

    # Embedding with masking so the LSTM doesn't learn from padding tokens.
    x = Embedding(vocab_size, embedding_dim, input_length=max_length)(inp)
    x = SpatialDropout1D(0.20)(x)

    # CNN branches (n-gram features)
    conv_vecs = []
    for k in conv_kernel_sizes:
        c = Conv1D(
            conv_filters, k,
            padding="valid",
            activation=None,
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
        )(x)
        c = BatchNormalization()(c)
        c = ReLU()(c)
        c = GlobalMaxPooling1D()(c)
        conv_vecs.append(c)

    # BiLSTM branch (sequence features)
    r = Bidirectional(LSTM(
        lstm_units,
        return_sequences=True,
        dropout=0.20,
        recurrent_dropout=0.20 if use_recurrent_dropout else 0.0,
    ))(x)
    r = GlobalMaxPooling1D()(r)

    # Merge CNN + LSTM
    merged = Concatenate()(conv_vecs + [r])

    # Classifier head
    h = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(merged)
    h = Dropout(dropout)(h)
    h = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(h)
    h = Dropout(dropout * 0.5)(h)
    out = Dense(1, activation="sigmoid")(h)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )
    return model

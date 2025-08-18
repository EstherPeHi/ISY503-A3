from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, SpatialDropout1D,
    Conv1D, BatchNormalization, ReLU, GlobalMaxPooling1D,
    Bidirectional, LSTM,
    Concatenate, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam

def create_hybrid_model(vocab_size: int,
                        max_length: int,
                        embedding_dim: int = 128,
                        conv_filters: int = 128,
                        lstm_units: int = 128,
                        conv_kernel_sizes=(2, 3, 4, 5),
                        dropout: float = 0.4,
                        lr: float = 5e-4):
    """
    CNN + BiLSTM:
      - Ramas Conv1D (ngramas) + GlobalMaxPooling
      - Rama BiLSTM (dependencias largas) + GlobalMaxPooling
      - Concatenate → Densas → Sigmoid
    """
    inp = Input(shape=(max_length,))
    x = Embedding(vocab_size, embedding_dim, input_length=max_length)(inp)
    x = SpatialDropout1D(0.2)(x)

    # CNN en paralelo
    conv_vecs = []
    for k in conv_kernel_sizes:
        c = Conv1D(conv_filters, k, padding="valid", activation=None, kernel_initializer="he_normal")(x)
        c = BatchNormalization()(c)
        c = ReLU()(c)
        c = GlobalMaxPooling1D()(c)
        conv_vecs.append(c)

    # BiLSTM
    r = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    r = GlobalMaxPooling1D()(r)

    merged = Concatenate()(conv_vecs + [r])

    h = Dense(256, activation="relu")(merged)
    h = Dropout(dropout)(h)
    h = Dense(64, activation="relu")(h)
    h = Dropout(dropout * 0.5)(h)
    out = Dense(1, activation="sigmoid")(h)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", "AUC"])
    return model

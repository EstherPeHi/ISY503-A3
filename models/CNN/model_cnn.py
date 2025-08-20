from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Concatenate, Dropout, Dense, BatchNormalization
)
from tensorflow.keras.optimizers import Adam

def create_cnn_model(vocab_size: int,
                     max_length: int,
                     embedding_dim: int = 64,
                     filters: int = 128,
                     kernel_sizes=(3, 4, 5),
                     dropout: float = 0.3,
                     lr: float = 1e-3):
    """TextCNN para clasificación binaria."""
    inp = Input(shape=(max_length,))
    x = Embedding(vocab_size, embedding_dim, input_length=max_length)(inp)

    convs = []
    for k in kernel_sizes:
        c = Conv1D(filters, k, activation='relu', padding='valid')(x)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)

    x = Concatenate()(convs) if len(convs) > 1 else convs[0]
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout / 2)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])
    return model

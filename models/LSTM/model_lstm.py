#Vladimir Boiko part
from keras.src.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout, Bidirectional

def create_lstm_model(vocab_size, max_length):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, dropout= 0.3, recurrent_dropout= 0.3, return_sequences=True)),
        Bidirectional(LSTM(32, dropout= 0.3, recurrent_dropout = 0.3)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    # use optimizer for dataset
    model.compile(optimizer=Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model

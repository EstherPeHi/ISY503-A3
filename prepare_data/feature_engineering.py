#Huong Thu Le - Emma's work

import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class TextEncoder:
    def __init__(self, max_words=10000, max_length=500):
        self.max_words = max_words
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')

    def fit_tokenizer(self, texts):
        """Fit tokenizer on texts"""
        self.tokenizer.fit_on_texts(texts)
        # Save tokenizer for later use
        with open('tokenizer.pickle', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Using top {self.max_words} words")

    def encode_texts(self, texts):
        """Convert texts to sequences"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        return padded

    def prepare_data(self, reviews, labels):
        """Prepare data for training"""

        # Encode reviews
        X = self.encode_texts(reviews)
        y = np.array(labels)
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test

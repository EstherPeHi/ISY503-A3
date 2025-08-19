# inference_cnn.py
"""
Load trained CNN model and tokenizer to predict sentiment of new texts.
Usage:
    python inference_cnn.py --text "I loved this product"
Or:
    python inference_cnn.py --file sample.txt  # one review per line
"""
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils_cnn import load_tokenizer, texts_to_padded, simple_clean

DEFAULT_MODEL_PATH = os.path.join("models", "CNN", "saved_models", "cnn_model.h5")
DEFAULT_TOKENIZER_PATH = os.path.join("models", "CNN", "tokenizer.pickle")

def predict_texts(model_path, tokenizer_path, texts, max_len=200, threshold=0.5):
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    cleaned = [simple_clean(t) for t in texts]
    X = texts_to_padded(tokenizer, cleaned, max_len)
    probs = model.predict(X, verbose=0).ravel()
    labels = (probs >= threshold).astype(int)
    return probs, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--text", type=str, default=None, help="Single text to classify")
    parser.add_argument("--file", type=str, default=None, help="Path to a .txt file with one review per line")
    parser.add_argument("--max_len", type=int, default=200)
    args = parser.parse_args()

    inputs = []
    if args.text:
        inputs = [args.text]
    elif args.file and os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            inputs = [line.strip() for line in f if line.strip()]
    else:
        print("Please provide --text or --file")
        return

    probs, labels = predict_texts(args.model_path, args.tokenizer_path, inputs, args.max_len)
    for t, p, l in zip(inputs, probs, labels):
        sentiment = "Positive" if l == 1 else "Negative"
        print(f"[{sentiment}] {p:.3f} :: {t[:80]}{'...' if len(t) > 80 else ''}")

if __name__ == "__main__":
    main()

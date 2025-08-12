import argparse, json, pickle
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

def load_assets(model_path="models/CNN/last_cnn.keras",
                tok_path="models/CNN/tokenizer.pickle",
                thr_path="models/CNN/threshold.json"):
    model = load_model(model_path)
    with open(tok_path, "rb") as f:
        tok = pickle.load(f)
    thr = 0.5
    if Path(thr_path).exists():
        with open(thr_path, "r") as f:
            thr = float(json.load(f)["threshold"])
    maxlen = int(model.inputs[0].shape[1])
    return model, tok, maxlen, thr

def predict_texts(texts, model, tok, maxlen, thr):
    seq = tok.texts_to_sequences(texts)
    pad = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    prob = model.predict(pad, verbose=0).ravel()
    labels = ["Negative" if p < thr else "Positive" for p in prob]
    return prob, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, help='Texto a clasificar, ej. --text "I loved it"')
    ap.add_argument("--file", type=str, help="Ruta a .txt con 1 texto por línea")
    ap.add_argument("--model", type=str, default="models/CNN/last_cnn.keras")
    ap.add_argument("--tokenizer", type=str, default="models/CNN/tokenizer.pickle")
    ap.add_argument("--threshold", type=str, default="models/CNN/threshold.json")
    args = ap.parse_args()

    model, tok, maxlen, thr = load_assets(args.model, args.tokenizer, args.threshold)

    texts = []
    if args.text: texts.append(args.text)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts += [line.strip() for line in f if line.strip()]
    if not texts:
        print('Usa --text "..." o --file ruta.txt'); return

    prob, labels = predict_texts(texts, model, tok, maxlen, thr)
    for t, p, lab in zip(texts, prob, labels):
        print(f"[{lab}] prob={p:.3f} thr={thr:.3f} :: {t}")

if __name__ == "__main__":
    main()

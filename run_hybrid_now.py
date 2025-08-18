# run_hybrid_now.py
import argparse, re, json, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             precision_recall_fscore_support, accuracy_score,
                             average_precision_score, f1_score)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from models.HYBRID.model_hybrid import create_hybrid_model

def read_dir_reviews(data_dir, domains=None):
    data_dir = Path(data_dir)
    if domains is None:
        domains = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']
    texts, labels = [], []
    for d in domains:
        for fname, y in [('positive.review', 1), ('negative.review', 0)]:
            p = data_dir / d / fname
            if not p.exists():
                continue
            raw = p.read_text(encoding='latin-1', errors='ignore')
            chunks = re.findall(r'<review_text>\s*(.*?)\s*</review_text>', raw, flags=re.S|re.I)
            for t in chunks:
                t = re.sub(r'<br\s*/?>', ' ', t)
                t = re.sub(r'\s+', ' ', t).strip()
                if len(t) >= 10:
                    texts.append(t); labels.append(y)
    return texts, np.array(labels, dtype=int)

def plot_training(history, out_prefix):
    h = history.history
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(); plt.plot(h.get("loss",[]),label="loss"); plt.plot(h.get("val_loss",[]),label="val_loss")
    plt.legend(); plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.tight_layout(); plt.savefig(out_prefix+"_loss.png"); plt.close()
    if "accuracy" in h:
        plt.figure(); plt.plot(h["accuracy"],label="acc")
        if "val_accuracy" in h: plt.plot(h["val_accuracy"],label="val_acc")
        plt.legend(); plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.tight_layout(); plt.savefig(out_prefix+"_acc.png"); plt.close()
    if "auc" in h:
        plt.figure(); plt.plot(h["auc"],label="auc")
        if "val_auc" in h: plt.plot(h["val_auc"],label="val_auc")
        plt.legend(); plt.title("AUC"); plt.xlabel("Epoch"); plt.ylabel("AUC")
        plt.tight_layout(); plt.savefig(out_prefix+"_auc.png"); plt.close()

def plot_confusion(cm, out_png, names=("Neg","Pos")):
    plt.figure(); plt.imshow(cm, interpolation='nearest'); plt.title("Confusion Matrix"); plt.colorbar()
    t = np.arange(len(names)); plt.xticks(t, names); plt.yticks(t, names)
    thr = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i,j]), ha="center", color="white" if cm[i,j]>thr else "black")
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_roc_pr(y_true, y_prob, out_prefix):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob); auc = roc_auc_score(y_true, y_prob)
        plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}"); plt.plot([0,1],[0,1],"--")
        plt.legend(); plt.title("ROC"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.tight_layout(); plt.savefig(out_prefix+"_roc.png"); plt.close()
    except Exception:
        pass
    ap = average_precision_score(y_true, y_prob)
    ths = np.linspace(0,1,101); prec, rec = [], []
    for th in ths:
        y_pred = (y_prob >= th).astype(int)
        p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        prec.append(p); rec.append(r)
    plt.figure(); plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.legend(); plt.title("Precision-Recall"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.tight_layout(); plt.savefig(out_prefix+"_pr.png"); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="sorted_data_acl")
    ap.add_argument("--max-words", type=int, default=80000)
    ap.add_argument("--max-length", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    print("Cargando dataset desde", args.data_dir)
    texts, y = read_dir_reviews(args.data_dir)
    print(f"Total={len(texts)}  Pos={int(y.sum())}  Neg={int((1-y).sum())}")

    X_train, X_tmp, y_train, y_tmp = train_test_split(texts, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    tok = Tokenizer(num_words=args.max_words, oov_token="<OOV>")
    tok.fit_on_texts(X_train)
    def to_pad(xs):
        return pad_sequences(tok.texts_to_sequences(xs), maxlen=args.max_length, padding='post', truncating='post')
    X_train, X_val, X_test = map(to_pad, [X_train, X_val, X_test])
    vocab_size = min(args.max_words, len(tok.word_index)+1)
    print(f"vocab_size={vocab_size}  max_length={args.max_length}")

    model = create_hybrid_model(vocab_size=vocab_size, max_length=args.max_length)

    Path("models/HYBRID").mkdir(parents=True, exist_ok=True)
    cbs = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=2, factor=0.5, monitor='val_loss'),
        ModelCheckpoint("models/HYBRID/best_hybrid.keras", monitor='val_auc', mode='max', save_best_only=True)
    ]
    hist = model.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     epochs=args.epochs, batch_size=args.batch_size,
                     callbacks=cbs, verbose=1)

    # Threshold óptimo en VALID
    y_val_prob = model.predict(X_val, verbose=0).ravel()
    ths = np.linspace(0, 1, 501)
    f1s = [f1_score(y_val, (y_val_prob >= t).astype(int), zero_division=0) for t in ths]
    best_th = float(ths[int(np.argmax(f1s))])
    print(f"\nBest threshold (VAL F1): {best_th:.3f}")

    # Evaluación en TEST con ese threshold
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= best_th).astype(int)

    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    try: auc = roc_auc_score(y_test, y_prob)
    except: auc = float("nan")
    print(f"\n=== TEST (threshold={best_th:.3f}) ===")
    print(f"Accuracy: {acc:.4f}  Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

    # Guardar artefactos + plots
    rec_dir = Path("record_results/HYBRID"); rec_dir.mkdir(parents=True, exist_ok=True)
    plot_training(hist, str(rec_dir / "history_hybrid"))
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, str(rec_dir / "confusion_matrix.png"))
    plot_roc_pr(y_test, y_prob, str(rec_dir / "curves"))

    with open("models/HYBRID/tokenizer.pickle", "wb") as f: pickle.dump(tok, f)
    model.save("models/HYBRID/last_hybrid.keras")
    with open("models/HYBRID/threshold.json", "w") as f:
        json.dump({"threshold": best_th}, f)

    # Demo rápida
    demo = ["I absolutely loved this! would buy again.",
            "This is the worst experience I've had. Terrible."]
    demo_pad = to_pad(demo)
    demo_prob = model.predict(demo_pad, verbose=0).ravel()
    for t, p_ in zip(demo, demo_prob):
        lab = "Positive" if p_ >= best_th else "Negative"
        print(f"[DEMO] {lab} (prob={p_:.3f}, thr={best_th:.3f}) :: {t}")

    print("\nListo. Revisa: record_results/HYBRID/ y models/HYBRID/")

if __name__ == "__main__":
    main()

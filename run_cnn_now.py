"""
ISY503 – Assessment 3
Runner: CNN (TextCNN) for sentiment classification

Author: Esther Peña (EstherPeHi)
Team: ISY503-A3

Description:
    - Load → split (70/15/15) → train (EarlyStopping/Checkpoint) → evaluate (TEST).
    - TextCNN model defined below.
    - Single summary figure (6 subplots) rather than many pop-up windows.
    - Optional interactive demo after training.

CLI examples (still work):
    python run_cnn_now.py --data-dir sorted_data_acl --monitor val_accuracy --show-plots
    python run_cnn_now.py --data-dir sorted_data_acl --threshold-mode auto --show-plots
"""

from __future__ import annotations
import argparse
import re
import sys as _sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_fscore_support, accuracy_score, average_precision_score, f1_score
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate,
    Dropout, Dense, BatchNormalization, ReLU, SpatialDropout1D
)
from tensorflow.keras.optimizers import Adam

# ---------- RUN PRESET (for the green "Run" button) ----------
# If no CLI args are provided, these values will be used so the script runs automatically.
USE_RUN_PRESET = True
RUN_PRESET = dict(
    data_dir="sorted_data_acl",
    monitor="val_accuracy",   # or "val_loss"
    epochs=10,
    early_stop_patience=4,
    batch_size=64,
    max_length=300,
    max_words=50000,
    threshold_mode="fixed",
    threshold=0.5,
    show_plots=False,         # set True to display the summary figure
    save_plots=True,          # set True to save the summary figure
    interactive=False
)
# -------------------------------------------------------------


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def read_dir_reviews(data_dir: str | Path, domains=None):
    data_dir = Path(data_dir)
    if domains is None:
        domains = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']

    texts, labels = [], []
    for d in domains:
        for fname, y in [('positive.review', 1), ('negative.review', 0)]:
            fpath = data_dir / d / fname
            if not fpath.exists():
                continue
            raw = fpath.read_text(encoding='latin-1', errors='ignore')
            chunks = re.findall(r'<review_text>\s*(.*?)\s*</review_text>', raw, flags=re.S | re.I)
            for t in chunks:
                t = re.sub(r'<br\s*/?>', ' ', t)
                t = re.sub(r'\s+', ' ', t).strip()
                if len(t) >= 10:
                    texts.append(t)
                    labels.append(y)
    return texts, np.array(labels, dtype=int)


# ---------------------------------------------------------------------
# Model definition (TextCNN)
# ---------------------------------------------------------------------
def create_cnn_model(
    vocab_size: int,
    max_length: int,
    embedding_dim: int = 128,
    filters: int = 128,
    kernel_sizes: tuple[int, ...] = (3, 4, 5),
    dropout: float = 0.5,
    lr: float = 1e-3,
) -> Model:
    """
    TextCNN: convolutional blocks with different kernel sizes + max pooling.
    """
    inp = Input(shape=(max_length,))
    x = Embedding(vocab_size, embedding_dim, input_length=max_length)(inp)
    x = SpatialDropout1D(0.2)(x)

    convs = []
    for k in kernel_sizes:
        c = Conv1D(filters, k, padding='valid', activation=None, kernel_initializer='he_normal')(x)
        c = BatchNormalization()(c)
        c = ReLU()(c)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)

    x = Concatenate()(convs) if len(convs) > 1 else convs[0]
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout * 0.5)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])
    return model


# ---------------------------------------------------------------------
# Plot: single summary figure with 6 subplots
# ---------------------------------------------------------------------
def plot_summary_panel(history, y_true, y_prob, y_pred, cm,
                       show=True, save_dir: str | None = None, prefix="cnn"):
    """
    One figure with:
      (1) Loss, (2) Accuracy, (3) AUC, (4) ROC, (5) Precision-Recall, (6) Confusion Matrix
    """
    h = history.history
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    # (1) Loss
    ax = axes[0]
    ax.plot(h.get("loss", []), label="loss")
    if "val_loss" in h: ax.plot(h["val_loss"], label="val_loss")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()

    # (2) Accuracy
    ax = axes[1]
    if "accuracy" in h: ax.plot(h["accuracy"], label="acc")
    if "val_accuracy" in h: ax.plot(h["val_accuracy"], label="val_acc")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend()

    # (3) AUC (robust to metric key names)
    ax = axes[2]
    has = False
    for k in ("auc", "AUC", "roc_auc"):
        if k in h:
            ax.plot(h[k], label=k); has = True
    for k in ("val_auc", "val_AUC", "val_roc_auc"):
        if k in h:
            ax.plot(h[k], label=k); has = True
    ax.set_title("AUC"); ax.set_xlabel("Epoch"); ax.set_ylabel("AUC")
    if has:
        ax.legend()

    # (4) ROC
    ax = axes[3]
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "--")
        ax.legend(loc="lower right")
    except Exception:
        ax.text(0.5, 0.5, "ROC not available", ha="center")
    ax.set_title("ROC"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")

    # (5) Precision–Recall
    ax = axes[4]
    ap = average_precision_score(y_true, y_prob)
    ths = np.linspace(0, 1, 101)
    prec, rec = [], []
    for th in ths:
        yp = (y_prob >= th).astype(int)
        p, r, _, _ = precision_recall_fscore_support(y_true, yp, average='binary', zero_division=0)
        prec.append(p); rec.append(r)
    ax.plot(rec, prec, label=f"AP={ap:.3f}")
    ax.set_title("Precision-Recall"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()

    # (6) Confusion Matrix
    ax = axes[5]
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(2)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(["Neg", "Pos"]); ax.set_yticklabels(["Neg", "Pos"])
    thr = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center",
                    color="white" if cm[i, j] > thr else "black")
    ax.set_ylabel('True'); ax.set_xlabel('Predicted')

    fig.tight_layout()

    if save_dir:
        out = Path(save_dir); out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{prefix}_summary_panel.png", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def interactive_loop(pad_fn, model, thr: float):
    print("\n[Interactive] Type a sentence to classify. Blank line or 'q' to quit.")
    while True:
        s = input("> ").strip()
        if not s or s.lower() in {"q", "quit", "exit"}:
            print("Bye!")
            break
        x = pad_fn([s])
        prob = float(model.predict(x, verbose=0).ravel()[0])
        label = "Positive" if prob >= thr else "Negative"
        print(f"  [{label}] prob={prob:.3f} (thr={thr:.3f})")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="sorted_data_acl")
    ap.add_argument("--max-words", type=int, default=50_000)
    ap.add_argument("--max-length", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)

    ap.add_argument("--monitor", choices=["val_accuracy", "val_loss"], default="val_accuracy")
    ap.add_argument("--early-stop-patience", type=int, default=4)
    ap.add_argument("--threshold-mode", choices=["fixed", "auto"], default="fixed")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--show-plots", action="store_true")
    ap.add_argument("--save-plots", action="store_true")
    ap.add_argument("--interactive", action="store_true")

    # Use PRESET if no CLI args and enabled
    if USE_RUN_PRESET and len(_sys.argv) == 1:
        args = ap.parse_args([])  # parse defaults
        for k, v in RUN_PRESET.items():
            setattr(args, k, v)
        print("[RUN_PRESET] Using in-code parameters:", RUN_PRESET)
    else:
        args = ap.parse_args()

    # 1) Load
    texts, y = read_dir_reviews(args.data_dir)
    print(f"Loaded samples: {len(texts)} | Pos={int(y.sum())} Neg={len(y)-int(y.sum())}")

    # 2) Split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    # 3) Tokenize/pad
    tok = Tokenizer(num_words=args.max_words, oov_token="<OOV>")
    tok.fit_on_texts(X_train)

    def to_pad(xs):
        return pad_sequences(
            tok.texts_to_sequences(xs),
            maxlen=args.max_length, padding='post', truncating='post'
        )

    X_train, X_val, X_test = map(to_pad, [X_train, X_val, X_test])
    vocab_size = min(args.max_words, len(tok.word_index) + 1)
    print(f"vocab_size={vocab_size}  max_length={args.max_length}")

    # 4) Build & train
    model = create_cnn_model(vocab_size=vocab_size, max_length=args.max_length)
    Path("models/CNN").mkdir(parents=True, exist_ok=True)
    mode = "max" if args.monitor == "val_accuracy" else "min"
    cbs = [
        EarlyStopping(patience=args.early_stop_patience, restore_best_weights=True, monitor=args.monitor),
        ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_loss"),
        ModelCheckpoint("models/CNN/best_cnn.keras", monitor=args.monitor, mode=mode, save_best_only=True),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=cbs, verbose=1
    )

    # Best epoch (for the monitored metric)
    best_key = args.monitor
    if best_key not in history.history:
        best_key = "val_accuracy" if "val_accuracy" in history.history else "val_loss"
    best_epoch = int(np.argmax(history.history[best_key])) + 1
    best_val = float(np.max(history.history[best_key]))
    print(f"[Model selection] best epoch = {best_epoch} ({best_key}={best_val:.4f})")

    # 5) Threshold
    if args.threshold_mode == "auto":
        y_val_prob = model.predict(X_val, verbose=0).ravel()
        ths = np.linspace(0, 1, 501)
        f1s = [f1_score(y_val, (y_val_prob >= t).astype(int), zero_division=0) for t in ths]
        best_th = float(ths[int(np.argmax(f1s))])
        print(f"\nBest threshold (VALID F1 sweep): {best_th:.3f}")
    else:
        best_th = float(args.threshold)
        print(f"\nUsing fixed threshold = {best_th:.3f}")

    # 6) Evaluate on TEST
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= best_th).astype(int)

    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float("nan")

    print(f"\n=== TEST (threshold={best_th:.3f}) ===")
    print(f"Accuracy: {acc:.4f}  Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

    # 7) Single-window plots (optionally save)
    save_dir = "record_results/CNN" if args.save_plots else None
    cm = confusion_matrix(y_test, y_pred)
    plot_summary_panel(history, y_test, y_prob, y_pred, cm,
                       show=args.show_plots, save_dir=save_dir, prefix="cnn")

    # 8) Persist artifacts
    import pickle, json
    with open("models/CNN/tokenizer.pickle", "wb") as f:
        pickle.dump(tok, f)
    model.save("models/CNN/last_cnn.keras")
    with open("models/CNN/threshold.json", "w") as f:
        json.dump({"threshold": float(best_th)}, f)

    # 9) Quick demo + optional interactive
    demo = [
        "I absolutely loved this! would buy again.",
        "This is the worst experience I've had. Terrible."
    ]
    demo_pad = to_pad(demo)
    demo_prob = model.predict(demo_pad, verbose=0).ravel()
    for text, pr in zip(demo, demo_prob):
        lab = "Positive" if pr >= best_th else "Negative"
        print(f"[DEMO] {lab} (prob={pr:.3f}, thr={best_th:.3f}) :: {text}")

    if args.interactive:
        interactive_loop(to_pad, model, best_th)


if __name__ == "__main__":
    main()

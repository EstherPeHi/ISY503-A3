#Vladimir Boiko part
import pandas as pd
from models.LSTM.model_lstm import create_lstm_model
from train import train_model, plot_training_history
from evaluation import evaluate_model
from prepare_data.preprocessing import Preprocessor
from prepare_data.feature_engineering import TextEncoder

def main():

    # Step 1: Load labeled data
    print("\n1. Loading labeled data from raw files...")
    preprocessor = Preprocessor()
    reviews, labels = preprocessor.load_reviews()

    # Count the number of positive and negative reviews
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = sum(1 for y in labels if y == 0)
    print(f"Loaded {len(reviews)} labeled reviews | Positive: {n_pos} | Negative: {n_neg}")

    # Display the first 200 characters of a positive and a negative review (raw)
    for label_value, label_name in [(1, "Positive"), (0, "Negative")]:
        try:
            idx = labels.index(label_value)
            print(f"\n--- {label_name} review sample (raw, first 200 chars) ---")
            print(reviews[idx][:200])
        except ValueError:
            print(f"No {label_name} review found.")

    # Step 2: Preprocess
    print("\n2. Preprocessing...")
    processed_reviews = [preprocessor.clean_text(r) for r in reviews]

    # Display the first 5 processed reviews
    print("\n--- 5 processed reviews & labels ---")
    for i in range(min(5, len(processed_reviews))):
        print(f"Label: {labels[i]} | Text: {processed_reviews[i][:100]}...")

    # Step 3: Remove duplicates
    print("\n3. Removing duplicates...")    
    df = pd.DataFrame({'text': processed_reviews, 'label': labels})
    before = len(df)
    df = preprocessor.remove_duplicates(df)
    after = len(df)
    print(f"Removed {before - after} duplicate reviews")
    processed_reviews = df['text'].tolist()
    labels = df['label'].tolist()

    # Step 4: Remove outliers
    print("\n4. Removing outliers...")
    processed_reviews, labels, n_outliers = preprocessor.remove_outliers(processed_reviews, labels)
    print(f"Removed {n_outliers} outlier reviews")

    # Step 5: Encode text
    print("\n5. Encoding text...")
    encoder = TextEncoder(max_words=10000, max_length=500)
    encoder.fit_tokenizer(processed_reviews)

    # Step 6: Prepare data
    print("\n6. Preparing datasets...")
    X_train, X_val, X_test, y_train, y_val, y_test = encoder.prepare_data(processed_reviews, labels)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

    #setting up model
    MAX_WORDS = encoder.max_words
    print(f"Max words: {MAX_WORDS}")
    MAX_LENGHT = encoder.max_length
    print(f"Max length: {MAX_LENGHT}")

    model = create_lstm_model(MAX_WORDS, MAX_LENGHT)
    print("\n7. Training the model...")
    history = train_model(model, X_train, y_train, X_val, y_val, epochs = 2)
    plot_training_history(history)

    #evaluating the model
    evaluate_model(model, X_test, y_test, encoder)

if __name__ == "__main__":
    main()

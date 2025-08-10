# main.py
import pandas as pd
from prepare_data.preprocessing import Preprocessor
from prepare_data.feature_engineering import TextEncoder

def main():
    print("=== Sentiment Analysis Project ===")
    # Step 1: Load labeled data
    print("\n1. Loading labeled data from raw files...")
    preprocessor = Preprocessor()
    reviews, labels = preprocessor.load_reviews()

    # Đếm số lượng positive và negative
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = sum(1 for y in labels if y == 0)
    print(f"Loaded {len(reviews)} labeled reviews | Positive: {n_pos} | Negative: {n_neg}")

    # Step 2: Preprocess
    print("\n2. Preprocessing...")
    processed_reviews = [preprocessor.clean_text(r) for r in reviews]

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
    processed_reviews, labels = preprocessor.remove_outliers(processed_reviews, labels)

    # Step 5: Encode text
    print("\n5. Encoding text...")
    encoder = TextEncoder(max_words=10000, max_length=500)
    encoder.fit_tokenizer(processed_reviews)

    # Step 6: Prepare data
    print("\n6. Preparing datasets...")
    X_train, X_val, X_test, y_train, y_val, y_test = encoder.prepare_data(processed_reviews, labels)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

if __name__ == "__main__":
    main()

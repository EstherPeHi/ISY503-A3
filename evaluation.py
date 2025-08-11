# evaluation.py
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from prepare_data.preprocessing import Preprocessor


def evaluate_model(model, X_test, y_test, encoder):
    """Evaluate model performance"""
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Negative', 'Positive']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Test on custom examples
    test_custom_reviews(model, encoder)


def test_custom_reviews(model, encoder):
    """Test model on custom review examples"""
    custom_reviews = [
        "This product is absolutely fantastic! Best purchase ever!",
        "Terrible quality. Complete waste of money. Very disappointed.",
        "It's okay, nothing special but does the job.",
        "Amazing service and great product quality. Highly recommend!",
        "Broke after one day. Do not buy this garbage.",
        "Good value for money. Satisfied with my purchase."
    ]

    # Preprocess custom reviews
    preprocessor = Preprocessor()
    processed = [preprocessor.clean_text(review) for review in custom_reviews]

    # Encode and predict
    encoded = encoder.encode_texts(processed)
    predictions = model.predict(encoded)

    print("\nCustom Review Predictions:")
    for review, pred in zip(custom_reviews, predictions):
        sentiment = "Positive" if pred > 0.5 else "Negative"
        confidence = pred[0] if pred > 0.5 else 1 - pred[0]
        print(f"Review: {review}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2%})\n")

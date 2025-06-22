import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))

from data_loader import IMDBDataLoader
from preprocessing import TextPreprocessor
from Neural import *
from vectorizer_spacy import vectorize_reviews


def train_sentiment_model():
    """Complete training pipeline with caching and CORRECTED spaCy usage"""
    np.random.seed(42)
    print("🚀 STARTING SENTIMENT ANALYSIS TRAINING (Corrected Pipeline)")
    print("=" * 50)

    # Step 1: Load data
    print("\n📁 STEP 1: Loading data...")
    loader = IMDBDataLoader()
    reviews, labels = loader.load_training_data()

    # Step 2: Create train/validation split (from the ORIGINAL reviews)
    print("\n✂️  STEP 2: Creating train/validation split...")
    train_reviews, val_reviews, train_labels, val_labels = train_test_split(
        reviews,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,  # Use original reviews
    )
    print(f"Training samples: {len(train_reviews)}")
    print(f"Validation samples: {len(val_reviews)}")

    # We will still create a preprocessor object for use later, but we DON'T
    # use it on the main dataset before vectorization.
    preprocessor = TextPreprocessor()

    # Step 3: Load or Generate Vectors
    print("\n🔢 STEP 3: Converting text to spaCy vectors (or loading from cache)...")
    os.makedirs("models", exist_ok=True)
    # Give them new names to avoid using the old, bad vectors
    train_vectors_path = "models/X_train_spacy_raw.npy"
    val_vectors_path = "models/X_val_spacy_raw.npy"

    if os.path.exists(train_vectors_path) and os.path.exists(val_vectors_path):
        print("Pre-computed RAW vectors found! Loading from disk...")
        X_train = np.load(train_vectors_path)
        X_val = np.load(val_vectors_path)
    else:
        print("Pre-computed RAW vectors not found. Generating from original reviews...")
        # ### CRITICAL: Call vectorize_reviews on the original text ###
        X_train = vectorize_reviews(train_reviews)
        X_val = vectorize_reviews(val_reviews)
        np.save(train_vectors_path, X_train)
        np.save(val_vectors_path, X_val)
        print("RAW vectors saved to disk for future runs.")

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    print("\n🧠 STEP 5: Building and training a regularized neural network...")

    model = NeuralNetwork()

    # Add the first dense layer
    model.add_layer(
        Layer(input_size=X_train.shape[1], output_size=128, activation_name="relu")
    )
    # Add a Dropout layer after it
    model.add_layer(Dropout(rate=0.5))

    # Add the second dense layer
    model.add_layer(Layer(input_size=128, output_size=64, activation_name="relu"))
    # Add another Dropout layer
    model.add_layer(Dropout(rate=0.5))

    # Add the final output layer
    model.add_layer(Layer(input_size=64, output_size=1, activation_name="sigmoid"))

    # Compile with the powerful Adam optimizer
    model.compile(optimizer=Adam(learning_rate=0.0004), loss=BinaryCrossEntropy())

    # Train the model
    model.train(X_train, train_labels, X_val, val_labels, epochs=300, batch_size=128)

    # Step 6: Evaluate model
    print("\n📊 STEP 6: Evaluating model...")
    train_predictions = model.predict_classes(X_train)
    train_accuracy = np.mean(train_predictions == train_labels)
    val_predictions = model.predict_classes(X_val)
    val_accuracy = np.mean(val_predictions == val_labels)
    print(f"Final Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")

    # Step 7: Save model
    print("\n💾 STEP 7: Saving model...")
    model.save_model("models/sentiment_model_spacy.npy")
    print("✅ Model saved successfully!")

    # Step 8: Test with sample predictions
    print("\n🧪 STEP 8: Testing with sample reviews...")
    # For single predictions, it's still okay to clean the text first
    test_reviews = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film. Waste of time. The acting was not good at all.",
        "It was okay, nothing special but not bad either.",
    ]
    for review in test_reviews:
        # Here we pass the raw text directly to the vectorizer, matching our training process
        vector = vectorize_reviews([review])
        probability = model.predict(vector)[0][0]
        sentiment = "Positive" if probability > 0.5 else "Negative"
        print(f"Review: '{review[:50]}...'")
        print(f"Prediction: {sentiment} (confidence: {probability:.3f})")
        print()

    # Plot training history
    try:
        model.plot_training_history()
    except:
        print("Could not display training plots (matplotlib might not be available)")

    return model, preprocessor


if __name__ == "__main__":
    train_sentiment_model()

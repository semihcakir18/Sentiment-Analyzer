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
    """Complete training pipeline with spaCy embeddings"""
    np.random.seed(42)  # For reproducible results
    print("🚀 STARTING SENTIMENT ANALYSIS TRAINING (with spaCy Embeddings)")
    print("=" * 50)

    # Step 1: Load data 
    print("\n📁 STEP 1: Loading data...")
    loader = IMDBDataLoader()
    reviews, labels = loader.load_training_data()

    # --- Data quality analysis  ---
    def analyze_data_quality(reviews, labels):
        print("=== DATA QUALITY ANALYSIS ===")
        print(f"Total samples: {len(reviews)}")
        print(f"Positive samples: {np.sum(labels == 1)}")
        print(f"Negative samples: {np.sum(labels == 0)}")
        print(f"Class balance: {np.sum(labels == 1) / len(labels):.3f}")
        lengths = [len(review.split()) for review in reviews]
        print(f"Average review length: {np.mean(lengths):.1f} words")
        unique_reviews = len(set(reviews))
        print(f"Unique reviews: {unique_reviews}/{len(reviews)} ({unique_reviews/len(reviews):.3f})")
        return True
    analyze_data_quality(reviews, labels)
    # -----------------------------------------------

    # Step 2: Preprocess text 
    print("\n🧹 STEP 2: Preprocessing text...")
    preprocessor = TextPreprocessor()
    cleaned_reviews = preprocessor.clean_reviews(reviews)

    # Step 3: Create train/validation split 
    print("\n✂️  STEP 3: Creating train/validation split...")
    train_reviews, val_reviews, train_labels, val_labels = train_test_split(
        cleaned_reviews, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Training samples: {len(train_reviews)}")
    print(f"Validation samples: {len(val_reviews)}")

    # Step 4: Vectorize text using spaCy
    print("\n🔢 STEP 4: Converting text to spaCy vectors...")
    
    # We no longer fit a vectorizer. We directly convert our text sets to vectors.
    X_train = vectorize_reviews(train_reviews)
    X_val = vectorize_reviews(val_reviews)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # Step 5: Create and train neural network
    print("\n🧠 STEP 5: Training neural network...")
    
    model = NeuralNetwork()
    # Wide input processing
    model.add_layer(input_size=X_train.shape[1], output_size=512, activation_name="relu")
    model.add_layer(input_size=512, output_size=256, activation_name="relu")
    model.add_layer(input_size=256, output_size=128, activation_name="relu")
    # Bottleneck for feature compression
    model.add_layer(input_size=128, output_size=32, activation_name="relu")
    # Expansion for classification
    model.add_layer(input_size=32, output_size=64, activation_name="relu")
    model.add_layer(input_size=64, output_size=32, activation_name="relu")
    model.add_layer(input_size=32, output_size=1, activation_name="sigmoid")

    model.compile(optimizer=SGD(learning_rate=0.005), loss=BinaryCrossEntropy())

    # Train the model
    model.train(X_train, train_labels, X_val, val_labels, epochs=100, batch_size=64)

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
    os.makedirs("models", exist_ok=True)
    model.save_model("models/sentiment_model_spacy.npy")
    print("✅ Model saved successfully!")

    # Step 8: Test with sample predictions
    print("\n🧪 STEP 8: Testing with sample reviews...")
    test_reviews = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film. Waste of time. Very disappointed.",
        "It was okay, nothing special but not bad either.",
        "The plot was confusing and the acting was wooden. I would not recommend this."
    ]

    for review in test_reviews:
        cleaned = preprocessor.clean_text(review)
        # Vectorize the single cleaned review using our function
        vector = vectorize_reviews([cleaned])
        # Predict
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
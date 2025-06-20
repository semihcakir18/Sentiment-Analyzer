import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))

from data_loader import IMDBDataLoader
from preprocessing import TextPreprocessor
from vectorizer import BagOfWordsVectorizer
from Neural import *


def train_sentiment_model():
    """Complete training pipeline"""
    np.random.seed(42)  # For reproducible results
    print("🚀 STARTING SENTIMENT ANALYSIS TRAINING")
    print("=" * 50)

    # Step 1: Load data
    print("\n📁 STEP 1: Loading data...")
    loader = IMDBDataLoader()
    reviews, labels = loader.load_training_data()

    # Validation of loaded data
    
    print(f"Total samples loaded: {len(reviews)}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")

    # Check for data consistency
    if len(reviews) != len(labels):
        print("❌ Mismatch between reviews and labels!")
        return

    if len(reviews) == 0:
        print("❌ No data loaded! Make sure dataset is in data/aclImdb/")
        return

    # Step 2: Preprocess text
    print("\n🧹 STEP 2: Preprocessing text...")
    preprocessor = TextPreprocessor()
    cleaned_reviews = preprocessor.clean_reviews(reviews)

    # Step 3: Create train/validation split
    print("\n✂️  STEP 3: Creating train/validation split...")

    # Better stratified split to ensure balanced classes
    train_reviews, val_reviews, train_labels, val_labels = train_test_split(
        cleaned_reviews, labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels 
    )

    print(f"Training samples: {len(train_reviews)}")
    print(f"Validation samples: {len(val_reviews)}")

    # Step 4: Vectorize text
    print("\n🔢 STEP 4: Converting text to numbers...")
    vectorizer = BagOfWordsVectorizer(max_features=5000, min_word_freq=5)

    # Fit vectorizer on training data only
    X_train = vectorizer.fit_transform(train_reviews)
    X_val = vectorizer.transform(val_reviews)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Vocabulary size: {vectorizer.vocabulary_size}")

    # Step 5: Create and train neural network
    print("\n🧠 STEP 5: Training neural network...")

    model = NeuralNetwork() 
    model.add_layer(input_size=vectorizer.vocabulary_size, output_size=256, activation_name="relu")
    model.add_layer(input_size=256, output_size=128, activation_name="relu")  
    model.add_layer(input_size=128, output_size=64, activation_name="leaky_relu")
    model.add_layer(input_size=64, output_size=1, activation_name="sigmoid")

    # Better optimizer settings
    model.compile(optimizer=SGD(learning_rate=0.001), loss=BinaryCrossEntropy())  

    # Train the model
    model.train(X_train, train_labels, X_val, val_labels, epochs=50, batch_size=64)

    # Step 6: Evaluate model
    print("\n📊 STEP 6: Evaluating model...")

    # Training accuracy
    train_predictions = model.predict_classes(X_train)
    train_accuracy = np.mean(train_predictions == train_labels)

    # Validation accuracy
    val_predictions = model.predict_classes(X_val)
    val_accuracy = np.mean(val_predictions == val_labels)

    print(f"Final Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")

    # Step 7: Save model and vectorizer
    print("\n💾 STEP 7: Saving model...")
    os.makedirs("models", exist_ok=True)

    model.save_model("models/sentiment_model.npy")
    vectorizer.save_vocabulary("models/vocabulary.pkl")

    print("✅ Training completed successfully!")

    # Step 8: Test with sample predictions
    print("\n🧪 STEP 8: Testing with sample reviews...")
    test_reviews = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film. Waste of time. Very disappointed.",
        "It was okay, nothing special but not bad either.",
    ]

    for review in test_reviews:
        # Preprocess
        cleaned = preprocessor.clean_text(review)
        # Vectorize
        vector = vectorizer.transform([cleaned])
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

    return model, vectorizer, preprocessor


if __name__ == "__main__":
    train_sentiment_model()

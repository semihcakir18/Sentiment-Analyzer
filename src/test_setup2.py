# This test were checking if the BagOfWordsVectorizer working correctly. Just like the vectorizer , this is also is not being used anymore due to change of vectorizer.
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(__file__))

from vectorizer import BagOfWordsVectorizer
from Neural import *


def test_vectorizer():
    """Test the bag of words vectorizer"""
    print("🧪 Testing Bag of Words Vectorizer...")

    try:
        # Simple test data
        texts = [
            "good movie great acting",
            "bad film terrible plot",
            "amazing movie good acting great story",
        ]

        vectorizer = BagOfWordsVectorizer(max_features=10, min_word_freq=1)
        vectors = vectorizer.fit_transform(texts)

        print(f"✅ Vectorizer works! Shape: {vectors.shape}")
        print(f"   Vocabulary: {vectorizer.get_feature_names()}")
        return True

    except Exception as e:
        print(f"❌ Vectorizer error: {e}")
        return False


def test_neural_network():
    """Test the neural network"""
    print("🧪 Testing Neural Network...")

    try:
        # Create dummy data
        X = np.random.randn(100, 50)  # 100 samples, 50 features
        y = np.random.randint(0, 2, 100)  # Binary labels

        # Create network
        input_size = 50
        hidden_size = 32
        network = NeuralNetwork()
        network.add_layer(
            input_size=input_size, output_size=hidden_size, activation_name="relu"
        )
        network.add_layer(
            input_size=hidden_size, output_size=1, activation_name="sigmoid"
        )
        network.compile(optimizer=SGD(learning_rate=0.01), loss=BinaryCrossEntropy())
        # Train for 3 epochs
        network.train(X, y, epochs=3)

        # Make predictions
        predictions = network.predict_classes(X[:5])

        print(f"✅ Neural network works!")
        print(f"   Sample predictions: {predictions}")
        return True

    except Exception as e:
        print(f"❌ Neural network error: {e}")
        return False


def test_integration():
    """Test vectorizer + neural network together"""
    print("🧪 Testing Integration...")

    try:
        # Sample data
        texts = [
            "great movie loved it amazing",
            "terrible film really bad",
            "good acting great story",
            "awful movie waste time",
            "fantastic film highly recommend",
        ]
        labels = np.array([1, 0, 1, 0, 1])  # 1=positive, 0=negative

        # Vectorize
        vectorizer = BagOfWordsVectorizer(max_features=20, min_word_freq=1)
        X = vectorizer.fit_transform(texts)

        # Train network
        network = NeuralNetwork()
        network.add_layer(
            input_size=X.shape[1], output_size=16, activation_name="leaky_relu"
        )
        network.add_layer(input_size=16, output_size=1, activation_name="sigmoid")
        network.compile(optimizer=SGD(learning_rate=0.01), loss=BinaryCrossEntropy())
        network.train(X, labels, epochs=200)

        # Test prediction
        test_text = "amazing movie great acting"
        test_vector = vectorizer.transform([test_text])
        prediction = network.predict(test_vector)[0][0]

        print(f"✅ Integration works!")
        print(f"   Test text: '{test_text}'")
        print(
            f"   Prediction: {prediction:.3f} ({'Positive' if prediction > 0.5 else 'Negative'})"
        )
        return True

    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False


def main():
    """Run all Week 2 tests"""
    print("🚀 WEEK 2 TESTING")
    print("=" * 40)

    vectorizer_test = test_vectorizer()
    network_test = test_neural_network()
    integration_test = test_integration()

    print("\n" + "=" * 40)
    print("📊 WEEK 2 TEST RESULTS")
    print("=" * 40)
    print(f"Vectorizer:    {'✅ PASS' if vectorizer_test else '❌ FAIL'}")
    print(f"Neural Network: {'✅ PASS' if network_test else '❌ FAIL'}")
    print(f"Integration:   {'✅ PASS' if integration_test else '❌ FAIL'}")

    if all([vectorizer_test, network_test, integration_test]):
        print("\n🎉 Week 2 setup complete! Ready to train on real data!")
        print("   Run: python src/train_model.py")
    else:
        print("\n🔧 Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()

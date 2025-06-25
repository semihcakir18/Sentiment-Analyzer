# train_lstm.py

import numpy as np
import os
import time
from sklearn.model_selection import train_test_split

# --- Step 1: Import All Our Components ---
print("➡️ STEP 0: Importing all our custom modules...")
from data_loader import IMDBDataLoader
from preprocessing import TextPreprocessor
from tokenizer import SimpleTokenizer, pad_sequences
from Neural import Layer, LSTMLayer, EmbeddingLayer, Adam, BinaryCrossEntropy

# --- Step 2: Set Up Configuration ---
print("➡️ STEP 1: Setting up configuration...")
# Data parameters
max_words = 5000  # The size of our vocabulary
max_len = 100  # The length of our sequences (shorter is faster for from-scratch)

# Training parameters
learning_rate = 0.005  # A good starting point for Adam
epochs = 10  # Start with a few epochs, as this will be slow
batch_size = 64

# --- Step 3: Load and Preprocess Raw Text ---
print("\n➡️ STEP 2: Loading and cleaning raw review data...")
loader = IMDBDataLoader()
raw_reviews, labels = loader.load_training_data()

preprocessor = TextPreprocessor()
cleaned_reviews = preprocessor.clean_reviews(raw_reviews)

# --- Step 4: Tokenize and Pad the Sequences ---
print("\n➡️ STEP 3: Tokenizing text into padded integer sequences...")
tokenizer = SimpleTokenizer(num_words=max_words)
tokenizer.fit_on_texts(cleaned_reviews)

sequences = tokenizer.texts_to_sequences(cleaned_reviews)
X = pad_sequences(sequences, maxlen=max_len)
y = labels  # Keep the original labels

print(f"Data shape: X={X.shape}, y={y.shape}")

# --- Step 5: Create Train/Validation Split ---
print("\n➡️ STEP 4: Creating train/validation split...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

# --- Step 6: Build the From-Scratch Sequential Model ---
print("\n➡️ STEP 5: Building the from-scratch model architecture...")
# We are not using the NeuralNetwork class here, as it's not built for this.
# Instead, we instantiate our layers directly.

embedding_layer = EmbeddingLayer(vocab_size=max_words, output_dim=64)
lstm_layer = LSTMLayer(input_size=64, hidden_size=128)
output_layer = Layer(input_size=128, output_size=1, activation_name="sigmoid")

# Instantiate our loss function and optimizer
loss_func = BinaryCrossEntropy()
# Note: We can't use our Adam class directly as it was tied to the old structure.
# We will perform the Adam update logic manually inside the training loop.
# For simplicity, let's start with a manual SGD update.
print("Model built successfully.")

# --- Step 7: The Custom Training Loop ---
print("\n➡️ STEP 6: Starting custom training loop...")
print(
    "⚠️ WARNING: This will be VERY slow. This is expected for from-scratch LSTMs in Python."
)

for epoch in range(epochs):
    epoch_start_time = time.time()
    print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    X_shuffled, y_shuffled = X_train[indices], y_train[indices]

    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[i : i + batch_size]
        y_batch = y_shuffled[i : i + batch_size].reshape(-1, 1)
        m_batch = len(X_batch)

        # --- FORWARD PASS ---
        embedded_out = embedding_layer.forward(X_batch)
        lstm_out = lstm_layer.forward(embedded_out)
        predictions = output_layer.forward(lstm_out)

        # --- BACKWARD PASS ---
        # Calculate loss gradient
        d_loss = loss_func.derivative(y_batch, predictions)

        # Propagate gradients backwards through layers
        d_lstm_out = output_layer.backward(d_loss)
        d_embedded = lstm_layer.backward(d_lstm_out)
        embedding_layer.backward(d_embedded)  # This calculates delta_embeddings

        # --- PARAMETER UPDATE (Manual SGD) ---
        # This is the part that frameworks like TensorFlow automate for us.

        # Update Output Layer
        output_layer.weights -= learning_rate * (output_layer.delta_weights / m_batch)
        output_layer.biases -= learning_rate * (output_layer.delta_bias / m_batch)

        # Update LSTM Layer (all 8 weights and 4 biases)
        lstm_layer.Wf -= learning_rate * (lstm_layer.dWf / m_batch)
        lstm_layer.Wi -= learning_rate * (lstm_layer.dWi / m_batch)
        lstm_layer.Wg -= learning_rate * (lstm_layer.dWg / m_batch)
        lstm_layer.Wo -= learning_rate * (lstm_layer.dWo / m_batch)
        lstm_layer.Uf -= learning_rate * (lstm_layer.dUf / m_batch)
        lstm_layer.Ui -= learning_rate * (lstm_layer.dUi / m_batch)
        lstm_layer.Ug -= learning_rate * (lstm_layer.dUg / m_batch)
        lstm_layer.Uo -= learning_rate * (lstm_layer.dUo / m_batch)
        lstm_layer.bf -= learning_rate * (lstm_layer.dbf / m_batch)
        lstm_layer.bi -= learning_rate * (lstm_layer.dbi / m_batch)
        lstm_layer.bg -= learning_rate * (lstm_layer.dbg / m_batch)
        lstm_layer.bo -= learning_rate * (lstm_layer.dbo / m_batch)

        # Update Embedding Layer
        embedding_layer.embeddings -= learning_rate * (
            embedding_layer.delta_embeddings / m_batch
        )

        if (i // batch_size) % 10 == 0:
            print(
                f"  Batch {i // batch_size}/{len(X_train) // batch_size} processed..."
            )

    # --- End of Epoch Evaluation ---
    print("Evaluating model performance for the epoch...")

    # Get training accuracy
    train_embedded = embedding_layer.forward(X_train)
    train_lstm_out = lstm_layer.forward(train_embedded)
    train_preds = output_layer.forward(train_lstm_out)
    train_acc = np.mean((train_preds > 0.5).astype(int) == y_train.reshape(-1, 1))

    # Get validation accuracy
    val_embedded = embedding_layer.forward(X_val)
    val_lstm_out = lstm_layer.forward(val_embedded)
    val_preds = output_layer.forward(val_lstm_out)
    val_acc = np.mean((val_preds > 0.5).astype(int) == y_val.reshape(-1, 1))

    epoch_time = time.time() - epoch_start_time
    print(
        f"Epoch Summary: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s"
    )


print("\n✅ LSTM training complete.")

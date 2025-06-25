import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Step 1: Import All Components ---
print("➡️ STEP 0: Importing all our custom modules...")
from data_loader import IMDBDataLoader
from preprocessing import TextPreprocessor
from tokenizer import SimpleTokenizer, pad_sequences
from Neural import Layer, LSTMLayer, EmbeddingLayer, BinaryCrossEntropy

# --- Step 2: Configuration ---
print("➡️ STEP 1: Setting up configuration...")
max_words = 5000
max_len = 100
learning_rate = 0.005 # A good rate for Adam
epochs = 10
batch_size = 64
grad_clip_threshold = 5.0 # For gradient clipping

# Adam optimizer parameters
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# --- Step 3 & 4: Load, Clean, Tokenize, Pad ---
print("\n➡️ STEP 2 & 3: Loading, cleaning, and tokenizing...")
loader = IMDBDataLoader()
raw_reviews, labels = loader.load_training_data()
preprocessor = TextPreprocessor()
cleaned_reviews = preprocessor.clean_reviews(raw_reviews) 
tokenizer = SimpleTokenizer(num_words=max_words)
tokenizer.fit_on_texts(cleaned_reviews)
sequences = tokenizer.texts_to_sequences(cleaned_reviews)
X = pad_sequences(sequences, maxlen=max_len)
y = labels

# --- Step 5: Train/Val Split ---
print("\n➡️ STEP 4: Creating train/validation split...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Step 6: Build the Model ---
print("\n➡️ STEP 5: Building the from-scratch model architecture...")
embedding_layer = EmbeddingLayer(vocab_size=max_words, output_dim=64)
lstm_layer = LSTMLayer(input_size=64, hidden_size=128)
output_layer = Layer(input_size=128, output_size=1, activation_name="sigmoid")
loss_func = BinaryCrossEntropy()

# ### NEW & FIXED ### Initialize Adam's moment vectors for all parameters explicitly
print("Initializing Adam optimizer state...")
layers_with_params = [embedding_layer, lstm_layer, output_layer]
m, v = {}, {}

# Define all possible parameter names for our layers
known_param_names = ['embeddings', 'Wf', 'Wi', 'Wg', 'Wo', 'Uf', 'Ui', 'Ug', 'Uo', 
                       'bf', 'bi', 'bg', 'bo', 'weights', 'biases']

for i, layer in enumerate(layers_with_params):
    for name in known_param_names:
        if hasattr(layer, name):
            param = getattr(layer, name)
            key = f'{i}_{name}'
            m[key] = np.zeros_like(param)
            v[key] = np.zeros_like(param)
            print(f"  Initialized Adam state for: {key}")

t = 0 # Adam timestep

# --- Step 7: The Final Training Loop ---
print("\n➡️ STEP 6: Starting final training loop with Adam and Gradient Clipping...")
train_acc_history, val_acc_history = [], []

for epoch in range(epochs):
    epoch_start_time = time.time()
    print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
    
    indices = np.random.permutation(len(X_train))
    X_shuffled, y_shuffled = X_train[indices], y_train[indices]

    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[i : i + batch_size]
        y_batch = y_shuffled[i : i + batch_size].reshape(-1, 1)
        m_batch = len(X_batch)

        # FORWARD PASS
        embedded_out = embedding_layer.forward(X_batch)
        lstm_out = lstm_layer.forward(embedded_out)
        predictions = output_layer.forward(lstm_out)

        # BACKWARD PASS
        d_loss = loss_func.derivative(y_batch, predictions)
        d_lstm_out = output_layer.backward(d_loss)
        d_embedded = lstm_layer.backward(d_lstm_out)
        embedding_layer.backward(d_embedded)

        # --- PARAMETER UPDATE (Manual Adam with Gradient Clipping) ---
        t += 1
        
        # Define a mapping from parameter names to their corresponding gradient names
        param_to_grad_map = {
            'embeddings': 'delta_embeddings', 'weights': 'delta_weights', 'biases': 'delta_bias',
            'Wf': 'dWf', 'Wi': 'dWi', 'Wg': 'dWg', 'Wo': 'dWo',
            'Uf': 'dUf', 'Ui': 'dUi', 'Ug': 'dUg', 'Uo': 'dUo',
            'bf': 'dbf', 'bi': 'dbi', 'bg': 'dbg', 'bo': 'dbo'
        }
        
        for layer_idx, layer in enumerate(layers_with_params):
            for param_name, grad_name in param_to_grad_map.items():
                if hasattr(layer, param_name) and hasattr(layer, grad_name):
                    param = getattr(layer, param_name)
                    grad = getattr(layer, grad_name) / m_batch
                    
                    # Gradient Clipping
                    norm = np.linalg.norm(grad)
                    if norm > grad_clip_threshold:
                        grad = grad * grad_clip_threshold / norm
                    
                    # Adam Update
                    key = f'{layer_idx}_{param_name}'
                    m[key] = beta1 * m[key] + (1 - beta1) * grad
                    v[key] = beta2 * v[key] + (1 - beta2) * (grad**2)
                    
                    m_hat = m[key] / (1 - beta1**t)
                    v_hat = v[key] / (1 - beta2**t)
                    
                    # Update the parameter on the original layer object
                    updated_param = param - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                    setattr(layer, param_name, updated_param)

    # --- End of Epoch Evaluation ---
    val_embedded = embedding_layer.forward(X_val)
    val_lstm_out = lstm_layer.forward(val_embedded)
    val_preds = output_layer.forward(val_lstm_out)
    val_acc = np.mean((val_preds > 0.5).astype(int) == y_val.reshape(-1, 1))
    val_acc_history.append(val_acc)
    
    print(f"Epoch Summary: Val Acc: {val_acc:.4f} | Time: {time.time() - epoch_start_time:.2f}s")

print("\n✅ LSTM training complete.")

# Plot the results
plt.plot(val_acc_history, label='Validation Accuracy')
plt.title('From-Scratch LSTM Performance')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
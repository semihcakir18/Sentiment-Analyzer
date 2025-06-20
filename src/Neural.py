# ---------------------------------------------------------------------------
# This code is modified version of my own code from the Neural-Network-From-Scratch repo.
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import time 

# ---------------------------------------------------------------------------
# Core Components 
# ---------------------------------------------------------------------------


# --- Activation Functions ---
class ActivationFunctions:
    """A collection of static methods for activation functions and their derivatives."""

    @staticmethod
    def sigmoid(x):
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, x * alpha)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1.0, alpha)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)


class Loss:
    """Base class for all loss functions."""

    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class MSE(Loss):
    """Mean Squared Error loss function."""

    def loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy loss function, suitable for binary classification."""

    def loss(self, y_true, y_pred):
        # Clip predictions to prevent log(0) errors.
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true, y_pred):
        # The derivative of BCE combined with sigmoid's derivative simplifies nicely.
        # This assumes the last layer uses a sigmoid activation.
        # (y_pred - y_true) is the derivative of BCE w.r.t. the pre-activation output (z).
        # Your original backward pass handles this correctly by multiplying by the activation derivative,
        # but for compatibility with Code 1's simplified backprop, we just need the error signal.
        return y_pred - y_true


class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        """Updates the layer's weights and biases using the calculated gradients."""
        # --- MODIFIED --- Averaging gradients over the batch size for stability
        m = layer.input.shape[0]
        layer.weights -= self.learning_rate * (layer.delta_weights / m)
        layer.bias -= self.learning_rate * (layer.delta_bias / m)


# ---------------------------------------------------------------------------
# Network Architecture 
# ---------------------------------------------------------------------------


class Layer:
    """Represents a single dense layer in the neural network."""

    def __init__(self, input_size, output_size, activation_name="sigmoid"):
        self.input_size = input_size  # --- NEW --- Store for saving model
        self.output_size = output_size  # --- NEW --- Store for saving model
        self.activation_name = activation_name

        if self.activation_name in ["relu", "leaky_relu"]:
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(
                2.0 / input_size
            )
        elif self.activation_name == "sigmoid":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(
                1.0 / input_size
            )
        else:
            self.weights = np.random.randn(input_size, output_size) * 0.01

        self.bias = np.zeros((1, output_size))

        activations = {
            "sigmoid": (
                ActivationFunctions.sigmoid,
                ActivationFunctions.sigmoid_derivative,
            ),
            "relu": (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            "leaky_relu": (
                lambda x: ActivationFunctions.leaky_relu(x, 0.01),
                lambda x: ActivationFunctions.leaky_relu_derivative(x, 0.01),
            ),
            "linear": (
                ActivationFunctions.linear,
                ActivationFunctions.linear_derivative,
            ),
        }

        if self.activation_name not in activations:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")
        self.activation_func, self.activation_derivative_func = activations[
            self.activation_name
        ]

        self.input = None
        self.output = None
        self.delta_weights = None
        self.delta_bias = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.activation_func(self.output)

    def backward(self, output_error):
        # --- MODIFIED --- The derivative logic is slightly adjusted to work with the simplified BCE derivative
        # The output_error is now (a - y) if coming from the loss function, or the propagated error from the next layer.
        # For the last layer, the error passed in is dz, so we don't need to multiply by the derivative.
        # For hidden layers, the error is da, so we do.
        # Your original code was general. This makes it slightly more specific to the BCE+Sigmoid case.
        # We can simplify this by checking if it's the last layer. A simpler approach is to adjust the BCE derivative.
        # Let's use the simplified BCE derivative: (y_pred - y_true) which is dL/dz.
        # This makes the backward pass consistent.
        activated_error = output_error * self.activation_derivative_func(self.output)
        self.delta_weights = np.dot(self.input.T, activated_error)
        self.delta_bias = np.sum(activated_error, axis=0, keepdims=True)
        input_error = np.dot(activated_error, self.weights.T)
        return input_error


# ---------------------------------------------------------------------------
# Main Neural Network Class (Your Code with Compatibility Additions)
# ---------------------------------------------------------------------------


class NeuralNetwork:
    """A fully-connected feedforward neural network."""

    def __init__(self):
        """Initializes the network components."""
        self.layers = []
        self.loss_func = None
        self.optimizer = None
        # --- NEW --- 
        self.loss_history = []
        self.accuracy_history = []

    def add_layer(self, input_size, output_size, activation_name="sigmoid"):
        """Adds a new layer to the network."""
        self.layers.append(Layer(input_size, output_size, activation_name))

    def compile(self, optimizer, loss):
        """Configures the model for training with a specified optimizer and loss function."""
        self.optimizer = optimizer
        self.loss_func = loss

    def forward(self, input_data):
        """Propagates input data through all layers."""
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_true, y_pred):
        """Initiates the backpropagation process starting from the loss derivative."""
        # --- MODIFIED --- Now using the simplified derivative (y_pred - y_true), which is dL/dz for the final layer
        error = self.loss_func.derivative(y_true, y_pred)

        # --- MODIFIED --- The backward pass for the last layer is now simpler
        # because the error is already dL/dz. For other layers, it remains dL/da.
        last_layer = self.layers[-1]
        # Calculate gradients for the last layer directly
        last_layer.delta_weights = np.dot(last_layer.input.T, error)
        last_layer.delta_bias = np.sum(error, axis=0, keepdims=True)
        # Propagate error to the previous layer
        error = np.dot(error, last_layer.weights.T)

        # For all other layers in reverse
        for layer in reversed(self.layers[:-1]):
            error = layer.backward(
                error
            )  # Your original backward logic is perfect for hidden layers

    # --- MODIFIED --- Changed from static to instance method 
    def calculate_accuracy(self, y_true, y_pred):
        """Calculates classification accuracy for binary problems."""
        predicted_classes = (y_pred > 0.5).astype(int).flatten()
        y_true_flat = y_true.flatten()
        return np.mean(predicted_classes == y_true_flat)

    # --- MODIFIED --- Updated train method 
    def train(
        self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=64
    ):
        """
        Trains the neural network using mini-batch gradient descent.
        """
        best_val_accuracy = 0
        patience = 5
        patience_counter = 0
        best_epoch_number = 0
        if self.optimizer is None or self.loss_func is None:
            raise ValueError("Network must be compiled before training.")

        print(f"Training neural network for {epochs} epochs...")
        n_samples = X_train.shape[0]
        
        # Track total training time
        total_start_time = time.time()

        for epoch in range(epochs):
            # Start timing this epoch
            epoch_start_time = time.time()
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size].reshape(
                    -1, 1
                )  # Ensure y is a column vector

                # 1. Forward pass
                predictions = self.forward(X_batch)

                # 2. Backward pass
                self.backward(y_batch, predictions)

                # 3. Update weights
                for layer in self.layers:
                    self.optimizer.update(layer)

            # Calculate and store loss and accuracy for this epoch
            train_predictions = self.forward(X_train)
            train_loss = self.loss_func.loss(y_train.reshape(-1, 1), train_predictions)
            train_accuracy = self.calculate_accuracy(y_train, train_predictions)

            self.loss_history.append(train_loss)
            self.accuracy_history.append(train_accuracy)

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print progress with timing
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f} - Time: {epoch_time:.2f}s")
                    
            # Validation metrics if provided
            if X_val is not None and y_val is not None:
                val_predictions = self.forward(X_val)
                val_loss = self.loss_func.loss(
                    y_val.reshape(-1, 1), val_predictions
                )
                val_accuracy = self.calculate_accuracy(y_val, val_predictions)
                print(
                    f"                    Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f} \n"
                )
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_epoch_number = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

        # Calculate total training time
        total_time = time.time() - total_start_time
        avg_epoch_time = total_time / epochs
        
        print("Training completed!")
        print(f"Best epoch number: {best_epoch_number}")
        print(f"Val_acc of the best epoch: {best_val_accuracy}")
        print(f"Total training time: {total_time:.2f}s")
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")

    # --- NEW --- Added predict method for compatibility
    def predict(self, X):
        """Make predictions on new data (alias for forward pass)."""
        return self.forward(X)

    # --- NEW --- Added predict_classes method for compatibility
    def predict_classes(self, X):
        """Make class predictions (0 or 1)."""
        probabilities = self.predict(X)
        return (probabilities > 0.5).astype(int).flatten()

    # --- NEW --- Added plot_training_history method for compatibility
    def plot_training_history(self):
        """Plot training loss and accuracy."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.loss_history)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Binary Cross Entropy Loss")
        ax1.grid(True)

        ax2.plot(self.accuracy_history)
        ax2.set_title("Training Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    # --- NEW --- Added save_model method for compatibility
    def save_model(self, filepath):
        """Saves the model's architecture, weights, and optimizer state."""
        architecture = []
        for layer in self.layers:
            architecture.append(
                {
                    "input_size": layer.input_size,
                    "output_size": layer.output_size,
                    "activation_name": layer.activation_name,
                }
            )

        model_data = {
            "architecture": architecture,
            "weights": [layer.weights for layer in self.layers],
            "biases": [layer.bias for layer in self.layers],
            "optimizer_config": {"learning_rate": self.optimizer.learning_rate},
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history,
        }
        np.save(filepath, model_data)
        print(f"Model saved to {filepath}")

    # --- NEW --- Added load_model method for compatibility
    def load_model(self, filepath):
        """Loads a model from a file."""
        model_data = np.load(filepath, allow_pickle=True).item()

        # Rebuild architecture
        self.layers = []
        for layer_config in model_data["architecture"]:
            self.add_layer(
                input_size=layer_config["input_size"],
                output_size=layer_config["output_size"],
                activation_name=layer_config["activation_name"],
            )

        # Load weights and biases
        for i, layer in enumerate(self.layers):
            layer.weights = model_data["weights"][i]
            layer.bias = model_data["biases"][i]

        # Re-compile model
        optimizer = SGD(learning_rate=model_data["optimizer_config"]["learning_rate"])
        loss = BinaryCrossEntropy()  # Assuming BCE for sentiment analysis
        self.compile(optimizer, loss)

        # Load history
        self.loss_history = model_data.get("loss_history", [])
        self.accuracy_history = model_data.get("accuracy_history", [])

        print(f"Model loaded from {filepath}")


# ---------------------------------------------------------------------------
# Test Execution 
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🧪 TESTING MODIFIED NEURAL NETWORK")
    print("=" * 40)

    # 1. Prepare dummy data
    n_samples = 1000
    input_size = 100
    hidden_size = 32

    X = np.random.randn(n_samples, input_size)
    y = np.random.randint(0, 2, n_samples)

    # 2. Build the Neural Network using your modular design
    network = NeuralNetwork()
    network.add_layer(
        input_size=input_size, output_size=hidden_size, activation_name="relu"
    )
    network.add_layer(input_size=hidden_size, output_size=1, activation_name="sigmoid")

    # 3. Compile the model
    network.compile(optimizer=SGD(learning_rate=0.01), loss=BinaryCrossEntropy())

    # 4. Train the model using the new compatible `train` method
    network.train(X, y, epochs=500, batch_size=64)

    # 5. Make predictions using the new compatible methods
    predictions = network.predict(X[:10])
    class_predictions = network.predict_classes(X[:10])

    print(f"\nSample predictions (probabilities): {predictions.flatten()[:5]}")
    print(f"Sample class predictions: {class_predictions[:5]}")
    print(f"Actual labels: {y[:5]}")

    # 6. Test saving and loading
    model_path = "./models/my_custom_model.npy"
    network.save_model(model_path)

    # Create a new, empty network and load the saved state
    new_network = NeuralNetwork()
    new_network.load_model(model_path)

    print("\nTesting loaded model...")
    loaded_predictions = new_network.predict_classes(X[:10])
    print(f"Loaded model predictions: {loaded_predictions[:5]}")
    assert np.array_equal(
        class_predictions, loaded_predictions
    ), "Loaded model predictions do not match!"
    print("✅ Model save/load test passed!")

    # 7. Plot history
    network.plot_training_history()

    print("✅ Neural network test completed!")
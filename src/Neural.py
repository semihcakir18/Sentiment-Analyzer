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
    @staticmethod
    def sigmoid(x):
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


# --- Loss Functions ---
class Loss:
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class MSE(Loss):
    def loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(Loss):
    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true, y_pred):
        return y_pred - y_true


# --- Optimizers ---
class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        """Updates the layer's weights and biases using the calculated gradients."""
        m = layer.input.shape[0]  # Get batch size
        layer.weights -= self.learning_rate * (layer.delta_weights / m)
        # ### FIXED ### - Changed layer.bias to layer.biases
        layer.biases -= self.learning_rate * (layer.delta_bias / m)


class Adam:
    """Adam optimizer."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment vector
        self.v = {}  # Second moment vector
        self.t = 0  # Timestep

    def update(self, layer):
        """Updates the layer's weights and biases using the Adam algorithm."""
        self.t += 1
        layer_id = id(layer)  # Use object id as a unique key for the layer

        # Initialize moment vectors
        if layer_id not in self.m:
            # ### FIXED ### - Changed layer.bias to layer.biases
            self.m[layer_id] = {
                "dW": np.zeros_like(layer.weights),
                "db": np.zeros_like(layer.biases),
            }
            self.v[layer_id] = {
                "dW": np.zeros_like(layer.weights),
                "db": np.zeros_like(layer.biases),
            }

        m_batch = layer.input.shape[0]
        grad_w = layer.delta_weights / m_batch
        grad_b = layer.delta_bias / m_batch

        # Update biased first moment estimate
        self.m[layer_id]["dW"] = (
            self.beta1 * self.m[layer_id]["dW"] + (1 - self.beta1) * grad_w
        )
        self.m[layer_id]["db"] = (
            self.beta1 * self.m[layer_id]["db"] + (1 - self.beta1) * grad_b
        )

        # Update biased second raw moment estimate
        self.v[layer_id]["dW"] = self.beta2 * self.v[layer_id]["dW"] + (
            1 - self.beta2
        ) * (grad_w**2)
        self.v[layer_id]["db"] = self.beta2 * self.v[layer_id]["db"] + (
            1 - self.beta2
        ) * (grad_b**2)

        # Compute bias-corrected first moment estimate
        m_hat_dW = self.m[layer_id]["dW"] / (1 - self.beta1**self.t)
        m_hat_db = self.m[layer_id]["db"] / (1 - self.beta1**self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat_dW = self.v[layer_id]["dW"] / (1 - self.beta2**self.t)
        v_hat_db = self.v[layer_id]["db"] / (1 - self.beta2**self.t)

        # Update parameters
        layer.weights -= (
            self.learning_rate * m_hat_dW / (np.sqrt(v_hat_dW) + self.epsilon)
        )
        # ### FIXED ### - Changed layer.bias to layer.biases
        layer.biases -= (
            self.learning_rate * m_hat_db / (np.sqrt(v_hat_db) + self.epsilon)
        )


# ---------------------------------------------------------------------------
# Network Architecture
# ---------------------------------------------------------------------------


class Layer:
    """Represents a single dense layer in the neural network."""

    def __init__(self, input_size, output_size, activation_name="sigmoid"):
        self.type = "dense"  # ### NEW ### For saving/loading
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation_name

        # Weight Initialization
        if self.activation_name in ["relu", "leaky_relu"]:
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(
                2.0 / input_size
            )
        else:
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(
                1.0 / input_size
            )
        self.biases = np.zeros((1, output_size))

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

        self.activation_func, self.activation_derivative_func = activations[
            self.activation_name
        ]
        self.input, self.output, self.delta_weights, self.delta_bias = (
            None,
            None,
            None,
            None,
        )

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.activation_func(self.output)

    def backward(self, output_error):
        activated_error = output_error * self.activation_derivative_func(self.output)
        self.delta_weights = np.dot(self.input.T, activated_error)
        self.delta_bias = np.sum(activated_error, axis=0, keepdims=True)
        return np.dot(activated_error, self.weights.T)


# ### NEW ### - Dropout Layer
class Dropout:
    """Dropout layer for regularization."""

    def __init__(self, rate):
        self.type = "dropout"  # ### NEW ### For saving/loading
        self.rate = rate
        self.mask = None
        self.is_training = True  # Controlled by the NeuralNetwork class

    def forward(self, inputs):
        if self.is_training:
            # We use inverted dropout
            self.mask = (np.random.rand(*inputs.shape) > self.rate) / (1.0 - self.rate)
            return inputs * self.mask
        else:
            # During evaluation/testing, this layer does nothing
            return inputs

    def backward(self, dvalues):
        # Apply the same mask to the gradients
        return dvalues * self.mask


# ---------------------------------------------------------------------------
# Main Neural Network Class
# ---------------------------------------------------------------------------


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_func = None
        self.optimizer = None
        self.train_loss_history = []
        self.train_accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []

    # ### MODIFIED ### - Now accepts a layer object directly
    def add_layer(self, layer):
        """Adds a layer object (e.g., Layer, Dropout) to the network."""
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_func = loss

    # ### NEW ### - Methods to control training/evaluation mode for dropout
    def train_mode(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.is_training = True

    def eval_mode(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.is_training = False

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_true, y_pred):
        error = self.loss_func.derivative(y_true, y_pred)
        # Propagate error backwards through all layers
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def calculate_accuracy(self, y_true, y_pred):
        predicted_classes = (y_pred > 0.5).astype(int).flatten()
        y_true_flat = y_true.flatten()
        return np.mean(predicted_classes == y_true_flat)

    def train(
        self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=64
    ):
        # Set network to training mode
        self.train_mode()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            indices = np.random.permutation(X_train.shape[0])
            X_shuffled, y_shuffled = X_train[indices], y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size].reshape(-1, 1)

                predictions = self.forward(X_batch)
                self.backward(y_batch, predictions)

                # ### MODIFIED ### - Only update layers with weights (skips Dropout)
                for layer in self.layers:
                    if hasattr(layer, "weights"):
                        self.optimizer.update(layer)

            # --- Logging and Validation ---
            # Set to evaluation mode for accurate metrics
            self.eval_mode()

            train_predictions = self.forward(X_train)
            train_loss = self.loss_func.loss(y_train.reshape(-1, 1), train_predictions)
            train_accuracy = self.calculate_accuracy(y_train, train_predictions)
            self.train_loss_history.append(train_loss)
            self.train_accuracy_history.append(train_accuracy)

            log_message = f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}"

            if X_val is not None:
                val_predictions = self.forward(X_val)
                val_loss = self.loss_func.loss(y_val.reshape(-1, 1), val_predictions)
                val_accuracy = self.calculate_accuracy(y_val, val_predictions)
                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(val_accuracy)
                log_message += (
                    f" - Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                )

            print(log_message + f" - Time: {time.time() - epoch_start_time:.2f}s")

            # Switch back to training mode for the next epoch
            self.train_mode()

    def predict(self, X):
        self.eval_mode()  # Ensure dropout is off for prediction
        return self.forward(X)

    def predict_classes(self, X):
        self.eval_mode()  # Ensure dropout is off for prediction
        probabilities = self.predict(X)
        return (probabilities > 0.5).astype(int).flatten()

    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.train_accuracy_history, label="Train Accuracy")
        ax1.plot(self.val_accuracy_history, label="Validation Accuracy")
        ax1.set_title("Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.grid(True)
        ax1.margins(x=0)
        ax1.legend()
        ax2.plot(self.train_loss_history, label="Train Loss")
        ax2.plot(self.val_loss_history, label="Validation Loss")
        ax2.set_title("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Binary Cross Entropy Loss")
        ax2.grid(True)
        ax2.margins(x=0)
        ax2.legend()
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        architecture = []
        weights = []
        biases = []
        for layer in self.layers:
            if layer.type == "dense":
                architecture.append(
                    {
                        "type": "dense",
                        "input_size": layer.input_size,
                        "output_size": layer.output_size,
                        "activation_name": layer.activation_name,
                    }
                )
                weights.append(layer.weights)
                biases.append(layer.biases)
            elif layer.type == "dropout":
                architecture.append({"type": "dropout", "rate": layer.rate})

        model_data = {
            "architecture": architecture,
            "weights": weights,
            "biases": biases,
        }
        np.save(filepath, model_data)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        model_data = np.load(filepath, allow_pickle=True).item()
        self.layers = []
        weight_idx, bias_idx = 0, 0
        for layer_config in model_data["architecture"]:
            if layer_config["type"] == "dense":
                new_layer = Layer(
                    input_size=layer_config["input_size"],
                    output_size=layer_config["output_size"],
                    activation_name=layer_config["activation_name"],
                )
                new_layer.weights = model_data["weights"][weight_idx]
                new_layer.biases = model_data["biases"][bias_idx]
                self.add_layer(new_layer)
                weight_idx += 1
                bias_idx += 1
            elif layer_config["type"] == "dropout":
                self.add_layer(Dropout(rate=layer_config["rate"]))
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("🧪 TESTING NEWLY IMPLEMENTED NEURAL NETWORK")
    print("=" * 40)
    # Testing Adam and Dropout
    network = NeuralNetwork()
    network.add_layer(Layer(input_size=100, output_size=64, activation_name="relu"))
    network.add_layer(Dropout(rate=0.5))  # Add dropout
    network.add_layer(Layer(input_size=64, output_size=1, activation_name="sigmoid"))

    network.compile(
        optimizer=Adam(learning_rate=0.001), loss=BinaryCrossEntropy()
    )  # Use Adam

    X = np.random.randn(1000, 100)
    y = np.random.randint(0, 2, 1000)
    network.train(X, y, epochs=10, batch_size=32)

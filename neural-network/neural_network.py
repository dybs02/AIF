import numpy as np


class NeuralNetwork:
    def __init__(self, layers: list['Layer']):
        self.layers = layers


    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.predict(inputs)
        return inputs


    def set_learing_rate(self, step, restart_period, min_lr=0.0001, max_lr=0.1):
        current_cycle = step % restart_period
        lr = min_lr + (max_lr - min_lr) * (1 + np.cos(current_cycle / restart_period * np.pi))
        for layer in self.layers:
            layer.learning_rate = lr


    def train(self, X: np.ndarray, Y: np.ndarray, epochs=1, batch_size=32, lr_restarts=5):
        # shuffle data
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]

        # variable learning rate with warm restarts
        max_steps = (X.shape[0] // batch_size) * epochs
        restart_period = max(1, max_steps // lr_restarts)
        step = 0

        # train in batches
        for start_idx in range(0, X.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, X.shape[0])
            X_batch = X_shuffled[start_idx:end_idx]
            Y_batch = Y_shuffled[start_idx:end_idx]

            for _ in range(epochs):
                self.predict(X_batch)

                output_error = None
                next_layer = None
                step += 1
                self.set_learing_rate(step, restart_period)

                # iterate in reverse order
                for i in range(len(self.layers)-1, -1, -1):
                    if i < len(self.layers) - 1:
                        next_layer = self.layers[i + 1]
                    output_error = self.layers[i].backpropagate(
                        X_batch, Y_batch, output_error, next_layer
                    )


class Layer:
    def __init__(self, inputs: int, neurons: int):
        self.weights = np.random.random((inputs, neurons)) * 0.01
        self.biases = np.zeros((1, neurons))
        self.activation_function = self.sigmoid
        self.activation_function_d = self.sigmoid_derivative
        self.learning_rate = 0.1
        self._last_x = None
        self._last_sigmoid = None


    def sigmoid(self, x: np.ndarray):
        # uses cached value if possible, ~8% faster
        if self._last_x is not None and np.array_equal(x, self._last_x):
            return self._last_sigmoid
        
        self._last_x = x
        self._last_sigmoid = 1 / (1 + np.exp(-x))
        return self._last_sigmoid


    def sigmoid_derivative(self, x: np.ndarray):
        val = self.sigmoid(x)
        return val * (1 - val)


    def predict(self, X: np.ndarray):        
        self.inputs = X
        self.z = np.dot(X, self.weights) + self.biases
        self.prediction = self.activation_function(self.z)
        return self.prediction


    def backpropagate(self, X: np.ndarray, Y: np.ndarray, output_error=None, next_layer: 'Layer'=None):
        if output_error is None:
            error = self.prediction - Y  # output layer - loss function
        else:
            error = np.dot(output_error, next_layer.weights.T)  # hidden layer

        delta = error * self.activation_function_d(self.z)
        batch_size = X.shape[0]

        weights_gradient = np.dot(self.inputs.T, delta) / batch_size
        biases_gradient = np.sum(delta, axis=0, keepdims=True) / batch_size  # ???
        self.weights -= self.learning_rate * weights_gradient
        self.biases -= self.learning_rate * biases_gradient

        return delta
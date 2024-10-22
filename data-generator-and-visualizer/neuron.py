import numpy as np

ACTIVATION_FUNCTIONS = ["Heaviside step", "sigmoid", "sin", "tanh", "sign", "ReLu", "Leaky ReLU"]

class Neuron():

    def __init__(self):
        # np.random.seed(1)
        self.bias = 1
        self.weights = np.random.random((2,))
        self.learning_rate = 0.1
        self.activation_function = self.tanh
        self.activation_function_d = self.tanh_derivative


    def heaviside_step(self, x):
        return 1 if x >= 0 else 0


    def heaviside_step_derivative(self, x):
        return 1


    def sigmoid(self, x):
        # should it also have Beta variable
        return 1 / (1 + np.exp(-x))


    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


    def sin(self, x):
        return np.sin(x)


    def sin_derivative(self, x):
        return np.cos(x)


    def tanh(self, x):
        return np.tanh(x)


    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x)**2


    def sign(self, x):
        return np.sign(x)


    def sign_derivative(self, x):
        return 1


    def relu(self, x):
        return max(0, x)


    def relu_derivative(self, x):
        return 1 if x > 0 else 0


    def leaky_relu(self, x):
        return max(0.01*x, x)


    def leaky_relu_derivative(self, x):
        return 1 if x > 0 else 0.01


    def set_activation_function(self, choice):
        match choice:
            case "Heaviside step":
                self.activation_function = self.heaviside_step
                self.activation_function_d = self.heaviside_step_derivative
            case "sigmoid":
                self.activation_function = self.sigmoid
                self.activation_function_d = self.sigmoid_derivative
            case "sin":
                self.activation_function = self.sin
                self.activation_function_d = self.sin_derivative
            case "tanh":
                self.activation_function = self.tanh
                self.activation_function_d = self.tanh_derivative
            case "sign":
                self.activation_function = self.sign
                self.activation_function_d = self.sign_derivative
            case "ReLu":
                self.activation_function = self.relu
                self.activation_function_d = self.relu_derivative
            case "Leaky ReLU":
                self.activation_function = self.leaky_relu
                self.activation_function_d = self.leaky_relu_derivative


    def train(self, data, epochs):
        self.weights = np.random.random((2,))

        for _ in range(epochs):
            for label, arrays in data.items():
                for array in arrays:
                    for point in array:
                        prediction = self.predict(point)
                        error = int(label) - prediction
                        
                        for i, w in enumerate(self.weights):
                            self.weights[i] += self.learning_rate * error * self.activation_function_d(w*point[i]) * point[i]


    def predict(self, x):
        z = np.dot(x, self.weights.T) + self.bias
        y = self.activation_function(z)
        return y
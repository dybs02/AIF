import numpy as np


class Neuron():

    def __init__(self):
        np.random.seed(1)

        self.bias = 1
        self.weights = np.random.random((2,))
        self.learning_rate = 0.1


    def tanh(self, x):
        return np.tanh(x)


    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x)**2


    def train(self, data, epochs):
        self.weights = np.random.random((2,))

        for _ in range(epochs):
            for label, arrays in data.items():
                for array in arrays:
                    for point in array:
                        prediction = self.predict(point)
                        error = int(label) - prediction
                        
                        for i, w in enumerate(self.weights):
                            self.weights[i] += self.learning_rate * error * self.tanh_derivative(w*point[i]) * point[i]


    def predict(self, x):
        z = np.dot(x, self.weights.T) + self.bias
        y = self.tanh(z)
        return y
import numpy as np


class Net():
    def __init__(self, weights_path):
        self.weights = np.load(weights_path, allow_pickle=True).item()

    def forward(self, x: np.ndarray):
        x = x.flatten()
        x = self.weights['fc1.weight'] @ x + self.weights['fc1.bias']
        x[x < 0] = 0  # ReLU
        x = self.weights['fc2.weight'] @ x + self.weights['fc2.bias']
        return np.exp(x) / np.sum(np.exp(x))

    def predict(self, x: np.ndarray):
        return np.argmax(self.forward(x))

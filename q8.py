import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


if __name__ == "__main__":
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    y = y.flatten()
    model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)

    predictions = model.predict(X)
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")
    from sklearn.metrics import mean_squared_error
    print(f"Mean Squared Error: {mean_squared_error(y, predictions)}")



# OUTPUT
# Weights: [3.12593652]
# Bias: 4.0446871544206235
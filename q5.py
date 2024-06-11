import numpy as np

class LinearRegression:
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
    y = y.reshape(-1)  
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")
    print(predictions)
    print(predictions.shape)



# OUTPUT
# Weights: [3.12593652]
# Bias: 4.0446871544206235
# [ 7.4757995   8.51596027  7.81308725  7.45122763  6.69332317  8.08273514
#   6.78042684  9.61993873 10.06938438  6.44191485  8.99445157  7.35127104
#   7.59602963  9.83139982  4.48879557  4.58940847  4.17109001  9.25012072
#   8.90962436  9.48389264 10.16288478  9.04092503  6.92979754  8.92445647
#   4.78412385  8.04539213  4.94091371  9.95063728  7.30721661  6.63710096
#   5.69865525  8.88509788  6.89648112  7.59846403  4.1621586   7.90606586
#   7.8714319   7.90168017  9.94488032  8.3073411   6.2922849   6.77695544
#   8.40618882  4.42120916  8.213228    8.23742997  5.35997222  4.850718
#   6.01670516  6.31856072  7.60948497  6.78676813 10.2238749   4.68265836
#   5.35055811  5.05317378  8.12783748  5.62823409  6.9600029   5.57280492
#   5.03854481  4.73473852  8.14797642  4.90858942  5.27369512  6.34991011
#   9.17743259  4.651752    9.28341233  4.6454822  10.14938775  6.97463497
#  10.15127346  7.82610455  8.66646919  4.28968426  5.81276038  4.79614079
#   5.89611807  4.78695778  6.03267762  6.6346068   4.44572916  8.37393493
#   7.58700751  5.70386856  7.31596755  4.6319913   7.64543152  9.85452899
#   6.0363398   8.21725211  4.86867066  8.52307389  5.8540173   5.18997629
#   7.71149156  4.17039698  9.22711497  4.07404268]
# (100,)
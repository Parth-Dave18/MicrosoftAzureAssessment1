import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5
    
    cost = (-1 / m) * (y.dot(np.log(h + epsilon)) + (1 - y).dot(np.log(1 - h + epsilon)))
    return cost

if __name__ == "__main__":
    X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
    y = np.array([0, 0, 1, 1])
    theta = np.zeros(X.shape[1])
    
    print("Initial Cost:", cost_function(theta, X, y))



# OUTPUT
# Initial Cost: 0.6931271807599427
import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_iters):
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return labels, centroids


np.random.seed(0)
X = np.random.rand(100, 2)
k = 3
labels, centroids = kmeans(X, k)
print(labels,centroids)


# OUTPUT
# [2 0 2 2 0 0 2 1 2 2 0 2 2 2 0 2 2 2 0 0 1 0 0 1 1 0 0 1 0 1 1 0 1 0 0 0 0
#  0 1 1 1 2 0 1 2 2 2 1 1 0 0 2 2 0 2 2 0 0 0 0 0 0 2 1 0 0 1 0 2 0 0 2 0 2
#  2 2 2 2 0 1 0 2 0 1 0 1 1 0 2 0 2 0 0 2 0 1 2 0 1 1] [[0.76167338 0.40765364]
#  [0.27223715 0.21097997]
#  [0.35217863 0.78488734]]

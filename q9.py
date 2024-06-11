import numpy as np

def pca(X, k):
    X_meaned = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_meaned, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigen_values)[::-1]
    topk_indices = sorted_indices[:k]
    topk_eigen_vectors = eigen_vectors[:, topk_indices]
    X_pca = np.dot(X_meaned, topk_eigen_vectors)
    return X_pca


np.random.seed(0)
X = np.random.rand(10, 5)
k = 2  
X_pca = pca(X, k)

print(X.shape)
print(X_pca.shape)

print(X_pca)


# OUTPUT
# (10, 5)
# (10, 2)
# [[-0.09150808 -0.10412935]
#  [-0.03471536  0.41757135]
#  [-0.42427325  0.19299645]
#  [ 0.76549898  0.26305905]
#  [-0.62059276  0.02702175]
#  [ 0.26488503  0.09145729]
#  [-0.27959735 -0.24832636]
#  [-0.01023595  0.29202606]
#  [ 0.4358998  -0.43832102]
#  [-0.00536107 -0.49335522]]
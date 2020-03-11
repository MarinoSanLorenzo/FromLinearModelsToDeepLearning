import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import polynomial_kernel
### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    return polynomial_kernel(X, Y, p, gamma=None, coef0=c)

ex_name = "Polynomial kernel"
n, m, d = 3, 5, 7
c = 1
p = 2
X = np.random.random((n, d))
Y = np.random.random((m, d))
for i in range(n):
    for j in range(m):
        exp = (X[i] @ Y[j] + c) ** d
print(exp)

from sklearn.metrics.pairwise import rbf_kernel
def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    return rbf_kernel(X, Y, gamma)

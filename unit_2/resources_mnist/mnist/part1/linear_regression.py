import numpy as np

### Functions for you to fill in ###

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    # YOUR CODE HERE
    result_to_invert = np.dot(X.transpose(), X) + lambda_factor
    # result_inverted = np.linalg.pinv(result_to_invert)
    result_inverted = np.linalg.pinv(result_to_invert)
    right_term = (np.dot(X.transpose(), Y))
    return np.dot(result_inverted, right_term)
    # raise NotImplementedError

# inv(X.transpose().dot(X) + lambda_factor).dot(X.transpose()).dot(y)

X = np.arange(1, 16).reshape(3, 5)
Y = np.arange(1, 4)
lambda_factor = 0.5
exp_res = np.array([-0.03411225,  0.00320187,  0.04051599,  0.07783012,  0.11514424])
result_to_invert = np.dot(X.transpose(), X) + lambda_factor
result_inverted = np.linalg.pinv(result_to_invert, hermitian = False)
right_term = (np.dot(X.transpose(), Y))
np.dot(result_inverted, right_term)
#θ=(XTX+λI)−1XTY
### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)

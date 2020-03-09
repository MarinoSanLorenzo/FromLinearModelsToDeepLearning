import sys
sys.path.append("..")
from tqdm import tqdm
try:
    import FromLinearModelsToDeepLearning.unit_2.resources_mnist.mnist.utils
    from FromLinearModelsToDeepLearning.unit_2.resources_mnist.mnist.utils import *
except ModuleNotFoundError:
    import unit_2.resources_mnist.mnist.utils
    from unit_2.resources_mnist.mnist.utils import *



import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def get_max_per_label(inner):
    return np.array([np.max(inner[::,j]) for j in range(inner.shape[1])])

def avoid_over_flow(inner, c_j):
    for j in range(inner.shape[1]):
        inner[::,j] = inner[::,j] -c_j[j]
    return inner

def get_total_proba(inner_exp):
    return np.array([np.sum(inner_exp[::,i]) for i in range(inner_exp.shape[1])])

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    #YOUR CODE HERE

    inner = np.dot(theta, X.transpose())/temp_parameter
    c_j = get_max_per_label(inner)
    inner = avoid_over_flow(inner, c_j)
    inner_exp = np.exp(inner)
    total_probas = get_total_proba(inner_exp)
    return inner_exp/total_probas

# #test1
# ex_name = "Compute probabilities"
# n, d, k = 3, 5, 7
# X = np.arange(0, n * d).reshape(n, d)
# zeros = np.zeros((k, d))
# temp = 0.2
# exp_res = np.ones((k, n)) / k
# output = compute_probabilities(X, zeros, temp)
#
# #test2
# theta = np.arange(0, k * d).reshape(k, d)
# compute_probabilities(X, theta, temp)
# exp_res = np.zeros((k, n))
# exp_res[-1] = 1

def one_if_true(Y,i,j):
    expr = Y[i-1]==j
    if expr:return 1
    elif not expr:return 0
    else: raise NotImplementedError

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    probas = compute_probabilities(X, theta, temp_parameter).transpose()
    N, K = probas.shape
    deviance_term = 0
    for i in range(1, N + 1):
        for j in range(K):
            deviance_term += one_if_true(Y, i, j) * np.log(probas[i - 1, j])

    regularization_error = 0
    n_row, n_col = theta.shape
    for j in range(n_row):
        for i in range(n_col):
            regularization_error += theta[j, i] ** 2
    regularization_error = (lambda_factor / 2) * regularization_error

    return ((-deviance_term) / N + regularization_error)

#
# regularization_error = 0
# n_row, n_col = theta.shape
# for j in range(n_row):
#     for i in range(n_col):
#         regularization_error+= theta[j,i]**2
# regularization_error = (lambda_factor/2)*regularization_error
#
#
# probas = compute_probabilities(X,theta,temp).transpose()
# N, K = probas.shape
# deviance_term = 0
# for i in range(1,N+1):
#     for j in range(K):
#         deviance_term += one_if_true(Y,i,j)*np.log(probas[i-1,j])
# cost = ((-deviance_term)/N +regularization_error)
#
# ex_name = "Compute cost function"
# n, d, k = 3, 5, 7
# X = np.arange(0, n * d).reshape(n, d)
# Y = np.arange(0, n)
# zeros = np.zeros((k, d))
# temp = 0.2
# lambda_factor = 0.5
# compute_cost_function(X,Y,zeros,lambda_factor, temp)
# exp_res = 1.9459101490553135
#
# # TEST 2
# X = np.array([[ 1, 44, 80, 85, 57,  3, 73, 80, 18, 32, 94],
#  [ 1, 81, 93, 76, 57, 97, 93, 54, 83, 32, 63],
#  [ 1, 11, 84, 77, 23, 11, 31, 16, 11, 55, 32],
#  [ 1,  5, 76, 53, 88, 45, 52, 25, 25, 92, 46],
#  [ 1, 21, 74, 97, 96, 83, 30, 15, 95, 13, 72],
#  [ 1, 71, 47, 64, 52, 14, 48, 41, 31,  5, 31],
#  [ 1,  8, 14, 11, 74, 87, 96, 15, 89, 74, 49],
#  [ 1, 87,  3,  9, 57, 12, 98,  4, 70, 59, 69],
#  [ 1, 20, 88, 55, 37, 77,  7,  8, 49, 55, 77],
#  [ 1, 92, 37, 16, 67, 36, 21, 83, 83, 49, 91]] )
#
# theta = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#
# Y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# temp_parameter = 1.0
# lambda_factor = 0.0001
# compute_cost_function(X,Y,theta, lambda_factor,temp_parameter)

# =============================================================================
# Run Gradient Descent Iteration
# =============================================================================
def get_max_per_label(inner):
    return np.array([np.max(inner[::,j]) for j in range(inner.shape[1])])

def avoid_over_flow(inner, c_j):
    for j in range(inner.shape[1]):
        inner[::,j] = inner[::,j] -c_j[j]
    return inner

def get_total_proba(inner_exp):
    return np.array([np.sum(inner_exp[::,i]) for i in range(inner_exp.shape[1])])

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    #YOUR CODE HERE

    inner = np.dot(theta, X.transpose())/temp_parameter
    c_j = get_max_per_label(inner)
    inner = avoid_over_flow(inner, c_j)
    inner_exp = np.exp(inner)
    total_probas = get_total_proba(inner_exp)
    return inner_exp/total_probas

def get_proba_yi_equal_j(i,j, probas):
    return probas[j,i]

def one_if_true(Y,i,j):
    expr = Y[i-1]==j
    if expr:return 1
    elif not expr:return 0
    else: raise NotImplementedError

def get_unregularized_deviance(X,Y,i,m, probas):
    yi_eq_m = one_if_true(Y,i,m )
    xi = X[(i-1), ::]
    p_yi_eq_m = get_proba_yi_equal_j((i-1),m, probas)
    return xi*(yi_eq_m-p_yi_eq_m)

def update_theta_m_sparse(X, Y, m, probas, alpha, theta, lambda_factor, temp_parameter):
    from scipy.sparse import coo_matrix

    D = []
    for i in range(1, X.shape[0] + 1):
        D.append(get_unregularized_deviance(X, Y, i, m, probas))
    sum_ = np.sum(np.array(D), axis = 0)
    gradient_m = (-1 / (temp_parameter * X.shape[0])) * (sum_) + lambda_factor * theta[m, ::]
    k,d = theta.shape
    row = np.array([[i]*d for i in range(0,m+1)]).flatten()
    col = np.array([np.arange(0,d) for i in range(0,m+1)]).flatten()
    updated_theta = theta[m, ::] - alpha * gradient_m
    data = np.concatenate((theta[:m,::], updated_theta), axis = None)
    return coo_matrix((data, (row, col)), shape=(k,d)).toarray()





def update_theta_m(X, Y, m, probas, alpha, theta, lambda_factor, temp_parameter):
    D = []
    for i in range(1, X.shape[0] + 1):
        D.append(get_unregularized_deviance(X, Y, i, m, probas))
    sum_ = np.sum(np.array(D), axis = 0)
    gradient_m = (-1 / (temp_parameter * X.shape[0])) * (sum_) + lambda_factor * theta[m, ::]

    theta[m, ::] = theta[m, ::] - alpha * gradient_m
    return theta

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    from tqdm import tqdm
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    probas = compute_probabilities(X, theta, temp_parameter)
    for m in tqdm( range( theta.shape[0] ) ):
        theta = update_theta_m_sparse(X, Y,m, probas, alpha, theta, lambda_factor, temp_parameter)
        # update_theta_m(X, Y,m, probas, alpha, theta, lambda_factor, temp_parameter)z
    return theta


n, d, k = 3, 5, 7
X = np.arange(0, n * d).reshape(n, d)
Y = np.arange(0, n)
theta = np.zeros((k, d))
alpha = 2
temp_parameter = 0.2
lambda_factor = 0.5
exp_res = np.zeros((k, d))
exp_res = np.array([
   [ -7.14285714,  -5.23809524,  -3.33333333,  -1.42857143, 0.47619048],
   [  9.52380952,  11.42857143,  13.33333333,  15.23809524, 17.14285714],
   [ 26.19047619,  28.0952381 ,  30.        ,  31.9047619 , 33.80952381],
   [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
   [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
   [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
   [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286]
])

output = run_gradient_descent_iteration(X,Y,theta, alpha, lambda_factor,  temp_parameter)




# A[i[k], j[k]] = data[K]


def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE
    raise NotImplementedError

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE
    raise NotImplementedError

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in tqdm(range(num_iterations)):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)

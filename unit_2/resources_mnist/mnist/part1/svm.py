import numpy as np
from sklearn.svm import LinearSVC


### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    random_state = 0
    C = 0.1
    clf = LinearSVC(random_state = random_state, C =C)
    clf.fit(train_x,train_y)
    return clf.predict(test_x)
#     raise NotImplementedError
#
# ex_name = "One vs rest SVM"
# n, m, d = 5, 3, 7
# train_x = np.random.random((n, d))
# test_x = train_x[:m]
# train_y = np.zeros(n)
# train_y[-1] = 1
# exp_res = np.zeros(m)

# train_y = np.ones(n)
# train_y[-1] = 0
# exp_res = np.ones(m)

def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    raise NotImplementedError


def compute_test_error_svm(test_y, pred_test_y):
    count = 0
    for y, pred in zip(test_y, pred_test_y):
        if y!=pred:
            count+=1
    return count/len(test_y)
    # raise NotImplementedError

# test_y = np.array([1,1,1,1,0])
# pred_test_y = np.array([1,1,1,1,0])
# assert 0 == compute_test_error_svm(test_y, pred_test_y)
#
# test_y = np.array([1,1,1,0,0])
# pred_test_y = np.array([1,1,1,1,0])
# assert 0.2 == compute_test_error_svm(test_y, pred_test_y)

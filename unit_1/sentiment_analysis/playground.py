import numpy as np


ex_name = "Classifier accuracy"

train_feature_matrix = np.array([[1, 0], [1, -1], [2, 3]])
val_feature_matrix = np.array([[1, 1], [2, -1]])
train_labels = np.array([1, -1, 1])
val_labels = np.array([-1, 1])
exp_res = 1, 0
T = 1
if check_tuple(
        ex_name, classifier_accuracy,
        exp_res,
        perceptron,
        train_feature_matrix, val_feature_matrix,
        train_labels, val_labels,
        T = T):
    return

train_feature_matrix = np.array([[1, 0], [1, -1], [2, 3]])
val_feature_matrix = np.array([[1, 1], [2, -1]])
train_labels = np.array([1, -1, 1])
val_labels = np.array([-1, 1])
exp_res = 1, 0
T = 1
L = 0.2
if check_tuple(
        ex_name, p1.classifier_accuracy,
        exp_res,
        p1.pegasos,
        train_feature_matrix, val_feature_matrix,
        train_labels, val_labels,
        T = T, L = L):

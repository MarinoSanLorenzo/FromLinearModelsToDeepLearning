import numpy as np

feature_matrix = np.array([[1, 2]])
labels = np.array([1])
T = 1
exp_res = (np.array([1, 2]), 1)

feature_matrix = np.array([[1, 2], [-1, 0]])
labels = np.array([1, 1])
T = 1
exp_res = (np.array([0, 2]), 2)

feature_matrix = np.array([[1, 2]])
labels = np.array([1])
T = 2
exp_res = (np.array([1, 2]), 1)

feature_matrix = np.array([[1, 2], [-1, 0]])
labels = np.array([1, 1])
T = 2
exp_res = (np.array([0, 2]), 2)

feature_matrix = np.array()
theta, theta_0 = np.array([0,0]), 0
for t in range(T):

	for i in get_order(feature_matrix.shape[0]):
		theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
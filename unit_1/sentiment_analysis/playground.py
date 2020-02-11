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

feature_matrix = np.array([[ 0.23123037,  0.18175917, -0.36295135, -0.15556077,  0.04119443, -0.1136364,
   0.16438287, -0.34733527,  0.28005155, -0.40200779],
 [-0.18511076,  0.49424554,  0.43193831, -0.17146269,  0.30561265,  0.28989715,
  -0.42961405,  0.49500207, -0.01990647,  0.29533806],
 [-0.31090418, -0.07503057,  0.24826015,  0.34725439, -0.33253195, -0.35562365,
   0.10648727, -0.04711672, -0.36646585, -0.09705627],
 [ 0.03231649,  0.40061052,  0.41139918, -0.25099698,  0.2879098,   0.06260034,
   0.2253733,  -0.10981416,  0.38739225, -0.01229647],
 [-0.22925739,  0.40412187,  0.1379125,   0.44622053, -0.04774341,  0.07147578,
  -0.47994534,  0.29922071, -0.28418616, -0.11664387]])
labels = np.array( [-1,  1,  1,  1,  1])
T = 5

dim_feature_space = get_dim_feature_space(feature_matrix)

theta, theta_0 = np.zeros(dim_feature_space), 0
for t in range(T):
	for i in get_order(feature_matrix.shape[0]):
		print(f'BEFORE: {theta, theta_0}')
		theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
		print(f'AFTER: {theta, theta_0}')

def get_dim_feature_space(feature_matrix):
	return feature_matrix[0].transpose().shape[0]

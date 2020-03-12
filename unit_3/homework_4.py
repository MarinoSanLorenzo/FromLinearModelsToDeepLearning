# =============================================================================
# 1. Neural Networks
# =============================================================================

import numpy as np

def run_test():
	relu_test()
	get_output_layer_test()

def relu(z):
	def relu_not_vectorized(z):
		return max(0,z)
	return np.vectorize(relu_not_vectorized)(z)

def relu_test():
	v = np.array([-1,0,1])
	exp = np.array([0,0,1])
	output = relu(v)
	assert all(exp == output), f'exp:\t{exp}\noutput:\t{output}'

def get_output_layer(W,X, has_bias= True,activation_function=relu):
	if has_bias:
		bias_col_idx = W.shape[1] - 1
		B = W[::, bias_col_idx]
		W = W[::, :bias_col_idx]
	elif not has_bias:
		B = np.zeros((W.shape[0]))
	z = np.dot(W, X) + B
	f_z = activation_function(z)
	return dict(z=z,f_z=f_z)


def get_output_layer_test():
	W = np.array([1, 0, -1, 0, 1, -1, -1, 0, -1, 0, -1, -1]).reshape(4, 3)
	V = np.array([1, 1, 1, 1, 0, -1, -1, -1, -1, 2]).reshape(2, 5)
	x = np.array([3, 14])
	has_bias = True
	if has_bias:
		bias_col_idx = W.shape[1] - 1
		B = W[::, bias_col_idx]
		W_no_bias = W[::, :bias_col_idx]
	z = np.dot(W_no_bias, x) + B
	activation_function = relu
	exp = activation_function(z)
	output = get_output_layer(W,x)
	assert all(output['f_z'] == exp), f'exp:\t{exp}\noutput:\t{output}'

def softmax(u):
	def exp_not_vectorized(u):
		import math
		return math.exp(u)
	exp_vec = np.vectorize(exp_not_vectorized)(u)
	sum_ = np.sum(exp_vec)
	try:
		return exp_vec/sum_
	except ZeroDivisionError:
		print('warning: zero division error')
		return exp_vec / (sum_ +0.001)

def softmax_numpy(u):
	exp_ = np.exp(u)
	sum_ = np.sum(exp_)
	return exp_/sum_



z=get_output_layer(W, x)
print(f'fz:\n{fz}')
o = get_output_layer(V,z['f_z'],activation_function = softmax)
o = get_output_layer(V,fz,activation_function = softmax_numpy)
print(f'o:\n{o}')

# =============================================================================
# decision boundaries
# =============================================================================
W = np.array([1, 0, -1, 0, 1, -1, -1, 0, -1, 0, -1, -1]).reshape(4, 3)
V = np.array([1, 1, 1, 1, 0, -1, -1, -1, -1, 2]).reshape(2, 5)
x = np.array([3, 14])
bias_col_idx = W.shape[1] - 1
B = W[::, bias_col_idx]
W_no_bias = W[::, :bias_col_idx]
z = np.dot(W_no_bias,x) + B

inv_W_no_bias = np.linalg.pinv(W_no_bias)
min_b = -B
boundary = np.dot(inv_W_no_bias,min_b)

# =============================================================================
# Output of Neural Network
# =============================================================================

V = np.array([1, 1, 1, 1, 0, -1, -1, -1, -1, 2]).reshape(2, 5)
sum_f = 1
np.dot(V,sum_f)
v1 = V[0,::]

u1 = 1 + 0
u2 = -1 + 2= 1
o1 = 'e^(1)/e^(1)+e^(1)'

from math import exp as exp

sum_f = 0
u1 = 0 + 0 = 0
u2 = 0 + 2= 2
o1 = 'e^(0)/e^(0)+e^(2)'

#

sum_f = 3
u1 = 3 + 0 = 3
u2 = -3 + 2= -1
o1 = 'e^(3)/(e^(3)+e^(-1))'

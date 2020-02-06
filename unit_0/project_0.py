import numpy as np
import math

# np.transpose
# np.matmul
# np.shape
# np.exp np.cos, np.sin, np.tanh
def operations(h, w):
	"""
	Takes two inputs, h and w, and makes two Numpy arrays A and B of size
	h x w, and returns A, B, and s, the sum of A and B.

	Arg:
	  h - an integer describing the height of A and B
	  w - an integer describing the width of A and B
	Returns (in this order):
	  A - a randomly-generated h x w Numpy array.
	  B - a randomly-generated h x w Numpy array.
	  s - the sum of A and B.	"""
	# Your code her
	A = np.random.random([h, w])
	B = np.random.random([h, w])
	s = A + B
	return A, B, s


# np.max, min arr.max()
# np.linalg.norm
A, B, s = operations(2, 1)


def norm(A, B):
	"""
	Takes two Numpy column arrays, A and B, and returns the L2 norm of their
	sum.

	Arg:
	  A - a Numpy array
	  B - a Numpy array
	Returns:
	  s - the L2 norm of A+B.
	"""
	# Your code here
	s = A + B
	return np.linalg.norm(s)


# =============================================================================
# Neural Network
# =============================================================================

import numpy as np
def neural_network(inputs, weights):
	"""
	 Takes an input vector and runs it through a 1-layer neural network
	 with a given weight matrix and returns the output.

	 Arg:
	   inputs - 2 x 1 NumPy array
	   weights - 2 x 1 NumPy array
	 Returns (in this order):
	   out - a 1 x 1 NumPy array, representing the output of the neural network
	"""
	return np.array([[np.tanh(np.sum(inputs*weights))]])


def neural_network_test():
	inputs = np.array([1, 2])
	weights = np.array([2, 4])
	output = neural_network(inputs, weights)
	expected = np.array([[np.tanh(10)]])
	assert output == expected
	assert 	output.shape == expected.shape


# =============================================================================
# Vectorize function
# =============================================================================

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code here
    if x<=y:return x*y
	else:return x/y

def scalar_function_test():
	x=4
	y=2
	output = scalar_function(x,y)
	expected = 2
	assert math.isclose(output, expected)
	output2= scalar_function(y,x)
	expected2 = 8
	assert math.isclose(output2, expected2)


def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y
    """
    #Your code here
	func = np.vectorize(scalar_function)
	return func(x,y)


def vector_function_test():
	x=np.array([0,1,2,3])
	y=2
	assert vector_function(x,y) == np.array([0,2,4,1.5])


# =============================================================================
# important ML package
# =============================================================================
# see ==> https://nbviewer.jupyter.org/github/Varal7/ml-tutorial/blob/master/Part2.ipynb

# =============================================================================
# Debugger
# =============================================================================

import pdb; pdb.set_trace()
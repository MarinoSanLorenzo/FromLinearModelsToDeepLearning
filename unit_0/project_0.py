import numpy as np


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

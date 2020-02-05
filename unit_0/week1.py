import math
import matplotlib.pyplot as plt
from itertools import islice


f =   lambda k: k/(k+1)


K = 10000

def gen(K):
	for k in range(1,K+1):
		yield f(k)

import numpy as np


gen_ = gen(K)


def multi(gen_):
	array_ = np.array ( [] )
	while True:
		try:
			value = next(gen_)
			array_ =  np.append(array_,value)
			print(array_[len(array_)-1])
		except StopIteration:
			break
	return array_

product_ = np.prod(multi(gen_))

f = lambda x : (1/(1+math.exp(-x)))

x_array = np.linspace(-5,5,1000)
y_array = np.asarray([f(i) for i in x_array])

plt.plot(x_array, y_array)
plt.show()

# =============================================================================
# Points and vectors
# =============================================================================

from fractions import Fraction

class Point:

	def __init__(self,name, coord):
		self.name = name
		self.coord = coord
	def __repr__(self):
		return f'Point({self.name}={self.coord})'

a = Point('a', Fraction(4,10))
b = Point('b', Fraction(3,10))


class Vector:

	def __init__(self, name, *points):
		self.name=name
		self.points=points

	def __repr__(self):
		return f'Vector({self.name}={self.points})'

	def __len__(self):
		return len(self.points)

	def __getitem__(self, item):
		if item > len(self.points):
			raise IndexError
		else:
			return self.points[item].coord


	def norm(self):
		it = iter(self)
		return Fraction(math.sqrt(sum([point for point in it])))

vector = Vector('vector', a,b)
vector.norm()

norm_a = math.sqrt(Fraction(25,100))
norm_b = math.sqrt(Fraction(625,10000))
dot_product = 0.4*(-0.15) + 0.3*0.2
cos_theta = dot_product/(norm_a*norm_b) # pi/2


(a_1 +a_2+a_3)*cos_theta

x1*x2 = ||x1||*|x2|¨cos_theta # pi/2

p1 = ||x1||*cos_theta = ||x1||*u

# =============================================================================
# Get the distance between a point P and the plane
# =============================================================================

n_hat  = (theta_1 + theta_0)/sqrt((theta_1^2 + theta_0^2 ) ) # unit vectpr
# ((x*theta)+theta_0)/norm(theta)

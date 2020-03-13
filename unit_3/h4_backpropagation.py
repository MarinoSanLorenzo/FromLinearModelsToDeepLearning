import numpy as np
from matplotlib import pyplot as plt
import math
from sympy import*

z = symbols('z')
diff(1/(1+exp(-z)), z)


d = lambda z : math.exp(-z)/(1 + math.exp(-z))**2


z = np.arange(0,10)
d_z = np.vectorize(d)(z)

plt.plot(z, d_z)

# =============================================================================
# Simple Network
# =============================================================================

class SimpleNetwork:

	def __init__(self, x,t, w1,w2, b):
		import math
		self.x = x
		self.t = t
		self.w1 = w1
		self.w2 = w2
		self.b = b
		self._z1 = None
		self._a1 = None
		self._z2 = None
		self._y = None
		self._cost = None

	def __repr__(self):
		from pprint import pprint
		pprint(self.__dict__)
		return ''

	def reinit(self):
		self._z1 = None
		self._a1 = None
		self._z2 = None
		self._y = None
		self._cost = None
		
	@property
	def x(self):
		return self._x
	
	@x.setter
	def x(self, value):
		self.reinit()	
		self._x = value

	@property
	def t(self):
		return self._t

	@t.setter
	def t(self, value):
		self.reinit()
		self._t = value

	@property
	def w1(self):
		return self._w1

	@w1.setter
	def w1(self, value):
		self.reinit()
		self._w1 = value

	@property
	def w2(self):
		return self._w2

	@w2.setter
	def w2(self, value):
		self.reinit()
		self._w2 = value

	@property
	def b(self):
		return self._b

	@b.setter
	def b(self, value):
		self.reinit()
		self._b = value

	@property
	def z1(self):
		if self._z1 is None:
			print('calculating z1...')
			self._z1 = self.w1*self.x
		return self._z1

	@property
	def a1(self):
		if self._a1 is None:
			print('calculating a1...')
			self._a1 =  max(0,self.z1)
		return self._a1

	@property
	def z2(self):
		if self._z2 is None:
			print('calculating a2...')
			self._z2 =  self.w2*self.a1 + self.b
		return self._z2

	@property
	def y(self):
		if self._y is None:
			print('calculating y...')
			self._y =  1/(1+math.exp(-self.z2))
		return self._y

	@property
	def cost(self):
		if self._cost is None:
			print('calculating cost...')
			C = lambda y, t: 0.5 * (y - t) ** 2
			self._cost =  C(self.y, self.t)
		return self._cost




x = 3
t = 1
w1 =0.01
w2 = -5
b = -1

n = SimpleNetwork(x,t,w1,w2, b)
n.cost

diff_c_y = (n.y-n.t)
print(diff_c_y)
diff_y_z2 = d(n.z2)
print(diff_y_z2)
diff_z2_a1 = n.w2
print(diff_z2_a1)
diff_a1_z1 = [0 if n.z1 < 0 else 1][0]
diff_z1_w1 = n.x
diff_c_w1 =diff_c_y*diff_y_z2*diff_z2_a1*diff_a1_z1*diff_z1_w1

#
diff_z2_w2 = n.a1
diff_c_w2 = diff_c_y*diff_y_z2*diff_z2_w2
#
diff_z2_b = 1
diff_c_b = diff_c_y*diff_y_z2*diff_z2_b

# SGD
# new_w1 = n.w1 - 'eta'*diff_c_w1
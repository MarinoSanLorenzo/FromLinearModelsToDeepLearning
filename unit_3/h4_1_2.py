# =============================================================================
#
# 1. Neural Networks
# =============================================================================

import math
from math import exp as exp
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


W = np.array([1, 0, -1, 0, 1, -1, -1, 0, -1, 0, -1, -1]).reshape(4, 3)
V = np.array([1, 1, 1, 1, 0, -1, -1, -1, -1, 2]).reshape(2, 5)
x = np.array([3, 14])
z=get_output_layer(W, x)

o = get_output_layer(V,z['f_z'],activation_function = softmax)
o = get_output_layer(V,z['f_z'],activation_function = softmax_numpy)
print(f'o:\n{o}')

# =============================================================================
# decision boundaries
# =============================================================================
W = np.array([1, 0, -1, 0, 1, -1, -1, 0, -1, 0, -1, -1]).reshape(4, 3)
V = np.array([1, 1, 1, 1, 0, -1, -1, -1, -1, 2]).reshape(2, 5)
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
u2 = -1 + 2#= 1
o1 = 'e^(1)/e^(1)+e^(1)'

from math import exp as exp

sum_f = 0
u1 = 0 + 0# = 0
u2 = 0 + 2#= 2
o1 = 'e^(0)/e^(0)+e^(2)'

#

sum_f = 3
u1 = 3 + 0 #= 3
u2 = -3 + 2#= -1
o1 = 'e^(3)/(e^(3)+e^(-1))'

# =============================================================================
# Inverse temperature
# =============================================================================
from fractions import Fraction
from sympy.solvers import solve
from sympy import Symbol
from sympy.functions.elementary.exponential import exp as exp_sympy
a = Symbol('a')
b = Symbol('b')
solve(exp_sympy(a)/(exp_sympy(a) + exp_sympy(b)) - 1/1000)
#[{a: log(0.001001001001001*exp(b))}]
math.log(0.001001001001001) # -6.906754778648555
softmax_temp = lambda fu1, fu2, beta: Fraction(exp(beta*fu1), exp(beta*fu1)+exp(beta*fu2))

solve(exp_sympy(3*a)/(exp_sympy(3*a) + exp_sympy(3*b)) - 1/1000)
#[{a: log((-0.050016677786427 - 0.0866314271518931*I)*exp(b))}, {a: log((-0.050016677786427 + 0.0866314271518931*I)*exp(b))}, {a: log(0.100033355572854*exp(b))}]
#{a: log(0.100033355572854*exp(b))}
math.log(0.100033355572854) # -2.3022515928828504

# =============================================================================
# 2. LSTM
# =============================================================================

from collections import namedtuple

class W:
	def __init__(self,gate_name,h,x,b):
		self.n = gate_name
		self.h = h
		self.x = x
		self.b = b
	def __repr__(self):
		return f'W_{self.n}(h={self.h}, x={self.x}, b={self.b})'

WF = W('f', h=0,x=0,b=-100)
WI = W('i', h=0,x=100,b=100)
WO = W('o', h=0,x=100,b=0)
WC = W('c', h=-100,x=50,b=0)
h_m1, c_m1 = 0,0
x = np.array([0,0,1,1,1,0])
sigmoid = lambda x : 1/(1+exp(-x))
tanh = lambda x : (exp(x)-exp(-x))/(exp(x)+exp(-x))

def get_output_test():
	w = W('f', h = 0, x = 0, b = -100)
	h =0
	x = 0
	z = w.h * h + w.x * x + w.b
	expected = 0
	output = get_output(w,h,x)
	assert expected ==output, f'expected:\t{expected}\noutput:\t{output}'

	w = W('f', h = 0, x = 0, b = 100)
	h = 0
	x = 0
	z = w.h * h + w.x * x + w.b
	expected = 1
	output = get_output(w, h, x)
	assert expected == output, f'expected:\t{expected}\noutput:\t{output}'

	w = W('f', h = 0, x = 0, b = 100)
	h = 0
	x = 0
	z = w.h * h + w.x * x + w.b
	expected = 1
	output = get_output(w, h, x, act_func = tanh)
	assert expected == output, f'expected:\t{expected}\noutput:\t{output}'

	w = W('f', h = 0, x = 0, b = -100)
	h = 0
	x = 0
	z = w.h * h + w.x * x + w.b
	expected = -1
	output = get_output(w, h, x, act_func = tanh)
	assert expected == output, f'expected:\t{expected}\noutput:\t{output}'

	w = W('f', h = 0, x = 0, b = 0)
	h = 0
	x = 0
	z = w.h * h + w.x * x + w.b
	expected = 0
	from math import exp as exp
	output = get_output(w, h, x, act_func = tanh)
	assert expected == output, f'exp:\t{expected}\noutput:\t{output}'


def get_output(w, h, x, act_func =sigmoid, approximation=True):
	from math import exp as exp
	z = w.h*h+w.x*x+w.b
	if approximation:
		if z >=1:
			return 1
		elif z <=-1:
			if act_func==sigmoid:
				return 0
			elif act_func==tanh:
				return -1
	return act_func(float(z))

class RNN:
	def __init__(self, x, hm1=0, cm1=0,**w):
		self.x = x
		self.H = []
		self.C = []
		self.update_output_h(hm1)
		self.update_memory_c(cm1)
		self.init_weights(**w)

	def __repr__(self):
		from pprint import pprint
		pprint(self.__dict__)
		return str('')

	def init_weights(self,**w):
		try:
			self.wf=w['f']
			self.wi=w['i']
			self.wo=w['o']
			self.wc=w['c']
		except KeyError:
			raise KeyError('Wrong Keyword arguments for weights!')

	def update_output_h(self, h):
		self.H.append(h)

	def update_memory_c(self,c):
		self.C.append(c)

	def get_forget_state_at_time(self,t):
		return get_output(self.wf, self.get_h_at_time(t-1), self.x[t])

	def get_input_state_at_time(self,t):
		return get_output(self.wi, self.get_h_at_time(t-1), self.x[t])

	def get_output_state_at_time(self,t):
		return get_output(self.wo, self.get_h_at_time(t-1), self.x[t])

	def get_memory_at_time(self,t):
		memory_input = get_output(self.wc, self.get_h_at_time(t-1), self.x[t], act_func = tanh)
		input = self.get_input_state_at_time(t)
		power_new_info = input*memory_input
		power_past_info = self.get_forget_state_at_time(t)*self.get_c_at_time(t-1)
		new_memory = power_past_info + power_new_info
		self.update_memory_c(new_memory)
		return new_memory

	def get_new_state_and_update_at_time(self, t, is_debug= True):
		new_state = self.get_output_state_at_time(t)*tanh(self.get_memory_at_time(t))
		self.update_output_h(new_state)
		if is_debug:
			print(f"new_state:\t{new_state}")
			print(f"new_states (h):\t{self.H}")
		return new_state


	def get_h_at_time(self,t):
		try:
			return self.H[t+1]
		except IndexError:
			raise IndexError('H list not yet populated!')

	def get_c_at_time(self,t):
		try:
			return self.C[t+1]
		except IndexError:
			raise IndexError('C list not yet populated!')

	def run_lstm_states(self):
		for t in range(len(self.x)):
			self.get_new_state_and_update_at_time(t)

# =============================================================================
# LSTM states 1
# =============================================================================

WF = W('f', h=0,x=0,b=-100)
WI = W('i', h=0,x=100,b=100)
WO = W('o', h=0,x=100,b=0)
WC = W('c', h=-100,x=50,b=0)
h_m1, c_m1 = 0,0
x = np.array([0,0,1,1,1,0])

rnn = RNN(x, f= WF, i =WI, o = WO, c = WC)

rnn.run_lstm_states()
h = [round(i) for i in rnn.H[1:]]

print(f'x:\t{x}')
print(f'h:\t{h}')
# =============================================================================
# LSTM states 2
# =============================================================================
x = np.array([1,1,0,1,1])

rnn = RNN(x, f= WF, i =WI, o = WO, c = WC)
rnn.run_lstm_states()
h = [round(i) for i in rnn.H[1:]]
print(f'x:\t{x}')
print(f'h:\t{h}')
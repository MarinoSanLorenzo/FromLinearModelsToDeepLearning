import math

math.log((1_000*1_000)**2, 10)#12

# =============================================================================
# CNN 1D discrete case
# =============================================================================
def mul_(f,g):
	return f*g

def pool(f,g):
	return sum(list(map(mul_, f[:3], g)))

def kernel(f, g):
	if len(f) >= len(g):
		ini = f
		to_be_moved = g
	else:
		ini = g
		to_be_moved = f
	sum_ = 0
	l = []
	for i in range(len(to_be_moved)):
		window = ini[i:i+len(to_be_moved)]
		r  = pool(window, to_be_moved)
		l.append(r)
		sum_+=r
	return sum_, l


f = [1,3,-1,1,-3]
g = [1,0,-1]

r,l = kernel(f,g)

#

f = [0,1,3,-1,1,-3,0]
g = [1,0,-1]

r,l = kernel(f,g)


def convolutional_product(fs, gs, n):
	l = []
	for m, f in enumerate(fs):
		if (n - m) >= 0 and (n - m) < len(gs):
			r = f * gs[n - m]
			print(f'f_{m}*g_{n - m}:\nr:\t{r}')
			l.append(r)
	sum_ = sum(l)
	print(f'sum:\t{sum_}')
	return l, sum_



# other test
f = [1,2,3]
g = [2,1]
_,h0 = convolutional_product(f,g,0)
assert h0 == 2, f'h0:\t{h0}\nexp:2'
_,h1 = convolutional_product(f,g,1)
assert h1 == 5, f'h1:\t{h1}\nexp:5'
_,h2 = convolutional_product(f,g,2)
assert h2 == 8, f'h2:\t{h2}\nexp:8'
_,h3 = convolutional_product(f,g,3)
assert h3 == 3, f'h3:\t{h0}\nexp:3'

# discrete case
f = [1,3,-1,1,-3]
f = [0,1,3,-1,1,-3,0]
g = [1,0,-1]

h0 = convolutional_product(f,g,0)
h1 = convolutional_product(f,g,1)
h2 = convolutional_product(f,g,2)
h3 = convolutional_product(f,g,3)
h5 = convolutional_product(f,g,5)


from collections import deque
F = [1,3,-1,1,-3]
G = [1,0,-1]
# step 0:
#f[0]*g[2]
# f[0*g[1]+ f[1]*g[2]
# f0*g0 + f1*g1 + f2*g2
#
F = [0,1,3,-1,1,-3,0]
G = [1,0,-1]

def pool2(f,g ):
	return sum(list(map(mul_,g, f[:len(g)])))
len_iteration = len(F)-len(G)+1
conv_prod = []
for i in range(len_iteration):
	conv = pool2(F[i:], G)
	conv_prod.append(conv)

# conv prod : [-3, 2, 2, 2, 1]

# =============================================================================
# Convolution: 2D Discrete Case
# =============================================================================
import numpy as np
f = np.array([1,2,1,2,1,1,1,1,1]).reshape(3,3)
print(f)
g = np.array([1,0.5,0.5,1]).reshape(2,2)
print(g)


nb_iterations =(f.shape[1] -g.shape[1]+1)*(f.shape[0] -g.shape[0]+1)
nb_col_screens = (f.shape[1] -g.shape[1]+1)
nb_row_screens = (f.shape[0] -g.shape[0]+1)



def convolution(f,g):
	conv_prod = []
	iteration = 0
	for i in range(nb_row_screens):
		for j in range(nb_col_screens):
			print(f'----- iteration:{iteration} --------------')
			f_subset = f[i:i+g.shape[0],j:j+g.shape[1]].reshape(g.shape[0], g.shape[1])
			print(f'f_subset:\t{f_subset}')
			conv = f_subset*g
			print(f'conv:\t{conv}')
			s = np.sum(conv)
			conv_prod.append(s)
		iteration += 1
	return conv_prod
f[0:2,0:2], f[0:2,1:3],f[1:3,0:2],f[1:3,1:3]
(0,0) # (0,+g.shaoe[0], 0+g.shape[1])
(0,1)
(1,0)
(1,1)

# conv prod: [4.0, 4.0, 4.0, 3.0] sum == > 15

# =============================================================================
# Numerical example
# =============================================================================

f = np.array([1,0,2,3,1,0,0,0,4]).reshape(3,3)
print(f)
g = np.array([1,0,0,1]).reshape(2,2)
print(g)

convolution(f,g)
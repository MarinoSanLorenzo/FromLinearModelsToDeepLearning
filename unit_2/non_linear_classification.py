
# =============================================================================
# Introduction to Non-linear Classification
# =============================================================================
import math
import numpy as np
dimension = lambda d, deg : (math.factorial(d+ deg)/(math.factorial(deg) * math.factorial(d)) ) -1

assert dimension(2,3) == 9
dimension(150,3)

# =============================================================================
# Kernel as Dot Products 2
# =============================================================================

# [x1, x2, x3]
# [x3,x2, x1]
# x1*x3 x2*x2 + x3*x1
# [x1, x2 + x3 ]
# [x2'+x3', x1']
# [x1*x2' + x1*x3', x2*x1'+ x3*x1']

a = np.array([1,0,0])
b = np.array([0,1,0])
dist = np.linalg.norm(a-b)

# =============================================================================
# Collaborative filtering with matrix multiplication
# =============================================================================

# Fixing V and finding U

Y = np.array([
			 [1,8, np.nan],
			 [2, np.nan, 5]
             ])
nb_user, nb_movie = Y.shape[0], Y.shape[1]
print(f'nb of users is {nb_user}, \nnb of movies is {nb_movie}')

# inital_v = np.array([[4,2,1]]).transpose()

# np.array([
#       [ 4*u1 ; 2*u1 ; 1*u1],
#       [ 4*u2 ; 2*u2 ; 1*u2],
# ])

# f(u1, lambda) = ((1-4u1)**2)/2 + ((8-2*u1)**2)/2 + (lambda/2)*u1**2
# derivative(f)/u1 = 0
# -4*(1-4u1) + -2*(8-2u1) + lambda*u1 = 0
# -4 +16u1 -16 + 4u1 + lambda*u1 = 0
# -20 + (20+lambda)*u1 = 0
# u1 = 20/(20+lambda)

# f(u2, lambda) = ((2-4u2)^2)/2) + ((5-1u2)^2)/2 + (lambda/2)*u2**2
# derivate(f)/du2 = 0
# -4*(2-4u2) + -(5-u2) + lambda*u2 = 0
# -8 + 16u2 -5 +u2 + lambda*u2 = 0
# -13 + (17+lambda)*u2 = 0
#u2 = 13/(17+lambda)

# =============================================================================
# 1. Collaborative Filtering, Kernels, Linear Regression
# =============================================================================


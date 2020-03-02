import numpy as np
# =============================================================================
#  Collaborative Filtering, Kernels, Linear Regression
# =============================================================================
Y = np.array([
	[5,None, None],
	[None, 2, None],
	[4, None, None],
	[None, 3, 6]
])
nb_user, nb_movie = Y.shape[0], Y.shape[1]
print(f'nb of users is {nb_user}, \nnb of movies is {nb_movie}')
k = l = 1

U_0 =np.array([[6,0,3,6]]).transpose()
V_0 = np.array([[4,2,1]]).transpose()

# 1a
X = U_0*V_0.transpose()

# [[24, 12,  6],
# [ 0,  0,  0],
# [12,  6,  3],
# [24, 12,  6]]

#1b Collaborative Filtering, Kernels, Linear Regression

def get_err_term(Y,X):
	A, I = Y.shape
	E = []
	for a in range(A):
		for i in range(I):
			if Y[a,i]:
				err = (Y[a,i] - X[a,i])**2
				print(f'y:\t{Y[a,i]}\nx:\t{X[a,i]}\nerr:\t{(Y[a,i] - X[a,i])**2}')
				E.append(err)
				print(f'sum:\t{sum(E)}')
	return sum(E)/2

squared_error_term = get_err_term(Y,X)

def get_squared_elt_of_matrix(U):
	N,K = U.shape
	U_SQUARRED = []
	for a in range(N):
		for j in range(K):
			U_SQUARRED.append(U[a,j]**2)
	return sum(U_SQUARRED)

def get_regularization_err_term(U,V, lambda_):
	u_err = get_squared_elt_of_matrix(U)
	v_err = get_squared_elt_of_matrix(V)
	return (lambda_*(u_err +v_err))/2

regularisation_error_term = get_regularization_err_term(U_0,V_0,l)

def loss_function(Y,U, V, l):
	X = U*V.transpose()
	return get_err_term(Y,X) + get_regularization_err_term(U,V,l)


# 3

# V = V_0


Y = np.array([
	[5,None, None],
	[None, 2, None],
	[4, None, None],
	[None, 3, 6]
])

U = np.array([[6,0,3,6]]).transpose()
V = np.array([[4,2,1]])

# [ [6u
# UVT =
# [ [4u1, 2u1, u1   ],
#   [4u2, 2u2, u2  ],
#   [4u3, 2u3,  u3 ],
##   [4u4, 2u4,  u4 ],

# (5-4u1)^2/2 +(2-2u2)^2/2 + (4-4u3)^2/2 + (3-2u4)^2/2 +(6-u4)^2/2 + lambda/2*(u1^2 + u2^2 + u3^2 +u4^2)

# derivate w.r.t U1
# -4(5-4u1) + lambdau1=0
# -20+16u1+lambdau1=0
# u1(16+lambda)=20
#u1 = 20/(16+lambda) ==> 20/17

# (2-2u2)^2/2
# -2(2-2u2) + lambda*u2 = 0
# -4 + 4 u2 + lambda*u2 = 0
# u2 = 4/(4+lambda) ==> u2 = 4/5

#(4-4u3)^2/2
# -4(4-4u3) + lambda*u3 = 0
# -16+16u3 + lambda*u3 = 0
# u3 = 16/(16+lambda) == > u3 = 16/17

#(3-2u4)^2/2 +(6-u4)^2/2

# -2(3-2u4) + -(6-u4) + lambda*u4 = 0
# -6+4u4 -6+u4 + lambda*u4 = 0
#-12+5u4 + 2lambda*u4 = 0
# u4 = 12/(5+2lambda) == > u4 = 12/6

# =============================================================================
# 2. Feature Vectors Transformation
# =============================================================================

# z = A x    (m*1) = (m*n)*(n*1) == > (2,1) = (2,6)*(6 1)
# zi = Axi
# z1 = average(x)
# z2 = average(x1, x2,x3) - average(x4,x5,x6)

#(z1    (a11 a21 .... a16       ( x1
# z2 == a21 a22 .... a26    *      x2
# )              )                   x3
#                                        x4
#                                         x5
#                                          x6)
# [ [1/6,1/6,1/6,1/6,1/6,1/6],[1/3,1/3,1/3,-1/3,-1/3,-1/3]]

# Note that  θz⋅z=θz⋅Ax=θ⊤zAx , and compare to the  θx
#trans(A)*theta_z

# =============================================================================
# 3.Kernels
# =============================================================================

K = lambda x,q :  (np.dot(x.transpose(),q) + 1)**2
# S = lambda x : np.array([x[0]+x[1], x[0]**2, x[1]**2, (x[0]+x[1])**2])
S = lambda x : np.array([ x[0]**2, x[1]**2, x[0]+x[1], x[0]*x[1], x[0], x[1]])
x, q = np.array([1,2]), np.array([2,5])
K_r, S_r = K(x,q),np.inner(S(x), S(q))
print(f'K={K_r}\nS={S_r}')

#q2


# (x1*q1*q2 + x2*q1*q2+1)^2 =
# (q1*q2(x1+x2)+1)^2
# ((q1q2^2*(x1^2+x2^2+2x1x2)+1+2*q1*q2(x1+x2))
# q1^2q2^2*x1^2 + q1^2q2^2*x2^2 + +  q1^2q2^2*2x1x2 + 1+ 2*q1*q2*x1 +2*q1*q2*x2

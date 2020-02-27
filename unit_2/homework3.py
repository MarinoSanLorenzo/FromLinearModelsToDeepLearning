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

V = V_0



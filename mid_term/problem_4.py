import numpy as np
Y = np.array([
			[np.nan, 1],
			[np.nan,np.nan]
			  ]
			 )


import scipy.linalg as l

A= np.array([1,-1,-1,1]).reshape(2,2)
P, L, U = la.lu(A)
P, L, U = la.lu(A)
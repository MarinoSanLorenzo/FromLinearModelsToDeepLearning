import numpy as np

def hinge_loss(y, x, theta):
	z= y -np.matmul(theta,x.transpose())
	if z >=1:
		return 0
	else:
		return 1-z

def squared_loss(y, x, theta):
	z= y -np.matmul(theta,x.transpose())
	return z**2/2

x1, y1 = np.array([1,0,1]), 2
x2, y2 = np.array([1,1,1]), 2.7
x3, y3 = np.array([1,1,-1]), -0.7
x4, y4 = np.array([-1,1,1]), 2

theta = np.array([0,1,2])

def empirical_risk(ys, xs, theta, loss_func):
	return sum([loss_func(y,x, theta ) for y,x in zip(ys, xs)])/len(xs)
xs, ys = [x1, x2, x3, x4], [y1, y2, y3, y4]
e_hinge = empirical_risk(ys, xs, theta, hinge_loss)
e_squared = empirical_risk(ys, xs, theta, squared_loss)


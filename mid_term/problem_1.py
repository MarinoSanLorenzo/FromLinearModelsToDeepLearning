# =============================================================================
# Problem 1.1
# =============================================================================
import numpy as np
from collections import defaultdict
y = np.array([-1,-1,-1,-1,-1,1,1,1,1,1])
c= [(0,0), (2,0), (3,0), (0,2), (2,2), (5,1), (5,2), (2,4), (4,4), (5,5)]


theta = np.array([0,0])
theta_0 = 0
mistakes_dic = defaultdict(int)

def run_perceptron_iteration(y,c, theta,theta_0, mistakes_dic):
	has_made_mistakes = False
	for i in range(len(c)):
		if y[i]*(np.matmul(theta,np.array([*c[i]])) + theta_0) <=0:
			has_made_mistakes = True
			theta = theta + y[i]*np.array([*c[i]])
			theta_0 = theta_0 + y[i]
			mistakes_dic[c[i]] += 1
	return theta, theta_0, has_made_mistakes, mistakes_dic


def run_perceptron_algorithm():
	y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
	c = [(0, 0), (2, 0), (3, 0), (0, 2), (2, 2), (5, 1), (5, 2), (2, 4), (4, 4), (5, 5)]
	theta = np.array([0, 0])
	theta_0 = 0
	mistakes_dic = defaultdict(int)
	nb_iter = 0
	has_made_mistakes = True
	while has_made_mistakes:
		theta, theta_0, has_made_mistakes, mistakes_dic = run_perceptron_iteration(y,c,theta,theta_0, mistakes_dic)
		nb_iter +=1
		print(f'Iteration:\t{nb_iter}', end='\n----------------------------\n')
		print(f'theta is:\t{theta}', end='\n----------------------------\n')
		print(f'theta0 is:\t{theta_0}', end='\n----------------------------\n')
		print(f'mistakes dic is:\t{mistakes_dic}', end='\n----------------------------\n')
	else:
		print(f'algorithm converged after {nb_iter} iterations.'.upper())
		print(f'theta is:\t{theta}')
		print(f'theta0 is:\t{theta_0}')
		print(f'mistakes dic is:\t{mistakes_dic}')
	return theta,theta_0, mistakes_dic

run_perceptron_algorithm()


theta = np.array([0,0])
theta_0 = 0

def update_theta(theta, theta_0, y,c,i,nb_mistakes):
	for _ in range(nb_mistakes):
		theta = theta + y[i] * np.array([*c[i]])
		theta_0 = theta_0 + y[i]
	return theta, theta_0

class Perceptron:

	def __init__(self,y,c):
		self.y = y
		self.c = c
		self.theta = np.array([0,0])
		self.theta_0 = 0
		self.mistakes_dic = {}

	def __repr__(self):
		return f'Perceptron(y,c)\ny={self.y}\nc={self.c}'

	def is_mistake(self,i):
		return self.y[i]*(np.matmul(self.theta,np.array([*self.c[i]])) + self.theta_0) <=0

	def update_theta(self, i, nb_mistakes, is_debug = True	):
		self.mistakes_dic[self.c[i]] = nb_mistakes
		if nb_mistakes==0:
			return None
		for _ in range(nb_mistakes):
			self.theta = self.theta + self.y[i] * np.array([*self.c[i]])
			self.theta_0 = self.theta_0 + self.y[i]

		if is_debug:
			print(f'{self}\n made {nb_mistakes} to classify coordinate: {self.c[i]}')


p = Perceptron(y,c)
mistakes = [1,9,10,5,9,11,0,3,1,1]
for i, mistake in enumerate(mistakes):
	p.update_theta(i, mistake)

p.mistakes_dic
p.theta
p.theta_0


# =============================================================================
# Problem 1.2
# =============================================================================
p = Perceptron(y,c)
p.is_mistake(6)

# =============================================================================
# Problem 1.3
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

X, y = np.array([[i[0], i[1]] for i in c]), y

clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 10)
yy = np.linspace(ylim[0], ylim[1], 10)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()

yt_func = lambda x : -x + 5
yt = yt_func(xx)
ax.scatter(xx, yt)

lf = lambda x: -x +3
uf = lambda x: -x +7

ax.scatter(xx, lf(xx))
ax.scatter(xx, uf(xx))

# =============================================================================
# Problem 2
# =============================================================================

def run_perceptron_algorithm(c,y,T=1_00_000):
	theta = np.array([0, 0])
	theta_0 = 0
	mistakes_dic = defaultdict(int)
	nb_iter = 0
	has_made_mistakes = True
	for _ in range(T):
		theta, theta_0, has_made_mistakes, mistakes_dic = run_perceptron_iteration(y,c,theta,theta_0, mistakes_dic)
		nb_iter +=1
		print(f'Iteration:\t{nb_iter}', end='\n----------------------------\n')
		print(f'theta is:\t{theta}', end='\n----------------------------\n')
		print(f'theta0 is:\t{theta_0}', end='\n----------------------------\n')
		print(f'mistakes dic is:\t{mistakes_dic}', end='\n----------------------------\n')
	else:
		print(f'algorithm converged after {nb_iter} iterations.'.upper())
		print(f'theta is:\t{theta}')
		print(f'theta0 is:\t{theta_0}')
		print(f'mistakes dic is:\t{mistakes_dic}')
	return theta,theta_0, mistakes_dic
x = [(0,0),(0,2),(1,1),(1,4),(2,0),(3,3),(4,1),(4,4),(5,2),(5,5)]
y = np.array([-1,-1,-1,1,-1,-1,1,1,1,1])

run_perceptron_algorithm(x,y)

# =============================================================================
# Problem 2.2
# =============================================================================
import math

feature_map = lambda x1,x2 : np.array([x1**2, math.sqrt(2)*x1*x2, x2**2 ])

def fmap(x1,x2):
	return np.array([x1**2, math.sqrt(2)*x1*x2, x2**2 ])

x = [(0,0),(0,2),(1,1),(1,4),(2,0),(3,3),(4,1),(4,4),(5,2),(5,5)]
y = np.array([-1,-1,-1,1,-1,-1,1,1,1,1])
mistakes = [1,65,11,31,72,30,0,21,4,15]

mistakes = np.zeros(10)
def theta_feature_map(mistakes, y, fmap, x, i):
	return np.sum([mistakes[j]*y[j]*np.matmul(fmap(*x[j]), fmap(*x[i]).transpose()) for j in range(len(x))])

mistakes = [1,65,11,31,72,30,0,21,4,15]
theta = np.array([0,0])
for i in range(len(x)):
	if y[i]*theta_feature_map(mistakes, y,fmap, x,i) <=0:
		print(i)



def run_kernel_perceptron_algorithm(x,y, t = 10):
	pass

mistakes = [1,65,11,31,72,30,0,21,4,15]
s = []
for j in range(len(x)):
	s.append(mistakes[j]*y[j]*fmap(*x[j]))

theta = np.sum(s, axis = 0)

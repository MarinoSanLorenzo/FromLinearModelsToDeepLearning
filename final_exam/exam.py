import numpy as np
import matplotlib.pyplot as plt
import math
from collections import namedtuple, defaultdict
from tqdm import tqdm
points = []
Point = namedtuple('Point', 'x y')
for i in range(1,10+1):
	points.append(Point(math.pi*i, (-1)**i))

x = np.array([point.x for point in points])

k_quadratic = lambda x1,x2 : (x1*x2)**2
k_pol20 = lambda x1,x2 : (x1*x2)**20
k_exp = lambda x1,x2 : np.exp(-10*(x1-x2)**2)
k_cos = lambda x1,x2 : np.cos(x1)*np.cos(x2)
k_sin = lambda x1,x2 : np.sin(x1)*np.sin(x2)

kernel_funcs = [k_quadratic, k_pol20, k_exp, k_cos, k_sin]


def bad_classification(points, i, theta, kernel_func):
	s = []
	for j in range(len(points)):
		s.append(theta[j]*points[j].y*kernel_func(points[i].x, points[j].x))
	classification_result = points[i].y*sum(s)
	return True if classification_result <=0 else False


def run_kernel_perceptron(points, kernel_func, T = 100):
	wrong, good = defaultdict(int), defaultdict(int)
	theta = np.zeros(len(points))
	for t in tqdm(range(T)):
		for i in tqdm(range(len(points))):
			for j in range(len(points)):
				pred = theta[j] * points[j].y * kernel_func(points[i].x, points[j].x)
				classification = points[i].y*pred
				if classification <= 0:
					theta[j]+=points[i].y*points[i].x
					wrong[t]+=1
				elif classification >0:
					good[t]+=1
	return theta, wrong, good

theta, wrong, good = run_kernel_perceptron(points, kernel_func = k_sin)
plt.plot(list(wrong.values()))
plt.plot(list(good.values()))

# =============================================================================
# Simple NN with 2 outputs
# =============================================================================

x = -2
theta = np.array([0.1,-1,1])
t= np.array([1,-1])
z1 = x*theta[0]
a1 = max(z1,0)
sigmoid = lambda x : 1/(1+np.exp(-x))
y1,y2 = sigmoid(theta[1]*a1), sigmoid(theta[2]*a1)
L = 0.5*(y1-t[0])**2 + 0.5*(y1-t[1])**2

d_l1_y1 = y1-t[0]
d_l2_y2 = y2-t[1]

d_y1_a1 = np.exp(-theta[1]*a1)/(1+ np.exp(-theta[1]*a1))**2
d_y2_a1 = np.exp(-theta[2]*a1)/(1+ np.exp(-theta[2]*a1))**2

d_a1_w1 = theta[0] if theta[0]*x > 0 else 0
d_l_w1 = d_l1_y1*d_y1_a1*d_a1_w1 + d_l2_y2*d_y2_a1*d_a1_w1

w1 = w1 - d_l_w1

x = np.linspace(0,1)
z1 = x*theta[0]
a1 = np.vectorize(lambda x : max(x,0))(z1)
y1,y2 = sigmoid(theta[1]*a1), sigmoid(theta[2]*a1)

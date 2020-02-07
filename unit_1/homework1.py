import numpy as np
from collections import namedtuple
from recordtype import recordtype
Point = namedtuple('Point', 'x y')
Weight = recordtype('Weight', 'theta theta_0')
point_1 = Point(np.array([-4,2]), 1)
point_2 = Point(np.array([-2, 1]), 1)
point_3 = Point(np.array([-1,-1]), -1)
point_4 = Point(np.array([2,2]), -1)
point_5 = Point(np.array([1,-2]), -1)

points = [point_1, point_2, point_3, point_4, point_5]

def is_in_agreement(point, weight, is_debug =True):
	if point.y*(np.matmul(weight.theta,point.x) +weight.theta_0)<=0:
		if is_debug:
			print(f'point:{point} has been misclassified')
		return False
	else:
		if is_debug:
			print(f'point:{point} has been rightly classified!')
		return True

def update_weight(point, weights):
	weights.theta += point.y*point.x
	weights.theta_0 += point.y
	return weights

def perceptron_algorithm(points, init_w=None):
	if not init_w:
		w = Weight(np.array([0,0]), 0)
	else:
		w =init_w
	for point in points:
		if not is_in_agreement(point,w):
			w = update_weight(point, w)
	return w

def perceptron_algorithm_iter(points, nb_iter = 15, init_w=None):
	if not init_w:
		w = Weight(np.array([0,0]), 0)
	else:
		w =init_w
	for i in range(nb_iter):
		print(f'------ITERATION: {i}')
		nb_misclassification=0
		for point in points:
			if not is_in_agreement(point,w):
				w = update_weight(point, w)
				nb_misclassification +=1
			print(w)
		if nb_misclassification ==0:
			break
	return w

perceptron_algorithm(points,Weight(np.array([-3,3]), -3))
new_points= [Point(np.array([-1,1]),1), Point(np.array([1,-1]),1), Point(np.array([1,1]),-1), Point(np.array([2,2]),-1)]
perceptron_algorithm(new_points, Weight(theta=np.array([-1, -1]), theta_0=1))
new_points2= [Point(np.array([-1,1]),1), Point(np.array([1,0]),-1), Point(np.array([-1,1.5]),1)]
new_points3= [ Point(np.array([1,0]),-1), Point(np.array([-1,1.5]),1), Point(np.array([-1,1]),1)]
new_points4= [Point(np.array([-1,1]),1), Point(np.array([1,0]),-1), Point(np.array([-1,10]),1)]
new_points3= [ Point(np.array([1,0]),-1),Point(np.array([-1,10]),1), Point(np.array([-1,1]),1)]

perceptron_algorithm_iter(new_points2)
perceptron_algorithm_iter(new_points3)
perceptron_algorithm_iter(new_points4)


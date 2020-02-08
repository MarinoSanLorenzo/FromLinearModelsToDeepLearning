import numpy as np
import copy
import math
from collections import Counter
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


# =============================================================================
#  Homework 2: perceptron update
# =============================================================================

def is_not_right(point, weight, is_offset = False):
	if not is_offset:
		return point.y*(np.matmul(weight.theta,get_coord_as_vector(point.x))) <=0
	elif is_offset:
		return point.y * (np.matmul(weight.theta, point.x) + weight.theta_0) <= 0
	else:
		raise NotImplementedError

def is_right(point, weight, is_offset = False):
	if not is_offset:
		return point.y*(np.matmul(weight.theta,get_coord_as_vector(point.x))) >0
	elif is_offset:
		return point.y * (np.matmul(weight.theta, point.x) + weight.theta_0) > 0
	else:
		raise NotImplementedError

def update_point_status(point):
	try:
		point.is_right = True
		return point
	except AttributeError:
		raise 'Unknown attribute of point called!'

def is_all_right(vector):
	try:
		return all( [point.is_right for point in vector.points])
	except AttributeError:
		raise 'Unknown attribute of vector called!'


def update_weight(point, weights, is_offset= False):
	weights.theta = np.add(weights.theta, point.y * get_coord_as_vector(point.x))
	if not is_offset:
		return weights
	elif is_offset:
		weights.theta_0 += point.y
		return weights
	else:
		raise NotImplementedError

def perceptron_algorithm_iter(vector, init_w=None, is_offset = False,
							  max_iter = 20):
	Run = recordtype('Run','init_w final_w nb_iter')
	Weight = recordtype('Weight', 'theta theta_0')
	if not init_w:
		w = Weight(np.array([0,0]), 0)
	else:
		init_w_copy = init_w.theta.copy()
		w =init_w
	iter = 0
	while not is_all_right(vector) and iter < max_iter:
		for point in vector.points:
			if is_not_right(point,w):
				update_weight(point, w)
			elif is_right(point,w ):
				update_point_status(point)
		else:
			iter +=1

	return Run(Weight(init_w_copy,0), w, iter)


def create_coord(d):
	x = " ".join([f'x{i}' for i in range(1,d+1)])
	return recordtype('Coord',x)

def value_coord(i, t):
	if i == t:return np.cos(math.pi*i)
	else:return 0
def get_coord_as_vector(coord):
	return np.array( [x for x in coord])

d = 2

Coord2 = create_coord(d)
Point = recordtype('Point', 'name x y is_right')
Vector = recordtype('Vector','points')
Weight = recordtype('Weight', 'theta theta_0')
w_init = Weight(np.zeros(d), 0)
vector = Vector([])

n=100
for i in range(1, n+1):
	coord = Coord2(*[value_coord(i,d) for d in range(1,d+1)])
	point = Point(f'x_{i}', coord, 1,False )
	vector.points.append(point)

perceptron_algorithm_iter(vector, init_w = w_init, max_iter = 10000)
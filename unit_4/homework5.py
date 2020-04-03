import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from recordtype import recordtype
d = {'x1': [0,4,0,-5], 'x2':[-6,4,0,2]}
df = pd.DataFrame.from_dict(d)

K = 2
z1 = (-5,2)
z2 = (0,-6)

def gen_plot(df, z1_idx, z2_idx):
	plt.scatter(df['x1'], df['x2'])
	plt.scatter(df['x1'][z1_idx], df['x2'][z1_idx], color='red', label='z1')
	plt.scatter(df['x1'][z2_idx], df['x2'][z2_idx], color='green', label='z2')
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.legend()
	plt.show()

def initalize_representatives(df, z1_idx, z2_idx):
	return (df['x1'][z1_idx], df['x2'][z1_idx]), (df['x1'][z2_idx], df['x2'][z2_idx])

def get_dist_from_pi_to_z(i,z,df):
	return np.linalg.norm(np.array(z) - np.array([df['x1'][i], df['x2'][i]]))

def clusterize_vec(dist_z1, dist_z2):
	def clusterize(dist_z1,dist_z2):
		if dist_z1 <= dist_z2:
			return 'c1'
		else:
			return 'c2'
	return np.vectorize(clusterize)(dist_z1, dist_z2)
z1, z2 = initalize_representatives(df, 0,3)
gen_plot(df, 0,3)
s = 0
dist = {}
dist['dist_from_z1'],dist['dist_from_z2'] = [], []
for i in range(df.shape[0]):
	dist_z1 = get_dist_from_pi_to_z(i,z1, df)
	dist_z2 = get_dist_from_pi_to_z(i,z2, df)
	dist['dist_from_z1'].append(dist_z1), dist['dist_from_z2'].append(dist_z2)

dist_df = pd.DataFrame.from_dict(dist)
df = pd.concat([df,dist_df], axis = 1)

clusterize_vec(df['dist_from_z1'], df['dist_from_z2'])

df = pd.concat([df, pd.DataFrame.from_dict({'cluster':clusterize_vec(df['dist_from_z1'], df['dist_from_z2'])})], axis = 1)

from itertools import permutations
from collections import namedtuple
def get_all_coordinates(df):
	return [(x1, x2) for x1, x2 in zip(df['x1'], df['x2'])]
def get_combination_of_coord_clusters(df):
	all_coordinates = get_all_coordinates(df)
	cluster_comb = list(permutations(all_coordinates,r=2))
	first_cluster = cluster_comb[2]
	cluster_to_try = [first_cluster] + cluster_comb
	Cluster = namedtuple('Cluster', 'z1 z2')
	return [Cluster(cluster[0], cluster[1]) for cluster in cluster_to_try]


def get_coord_array(df, idx):
	return np.array([df['x1'][idx],df['x2'][idx]])

def get_distance(xi,zj, order = 1 ):
	return np.linalg.norm(xi-zj, ord = order)

def get_points_from_df(df):
	Point = namedtuple('Point', 'x1 x2 array')
	return[Point(*get_coord_array(df, i), get_coord_array(df, i)) for i in df.index]


class Cost:

	def __init__(self, cluster, point,order):
		self.cluster = cluster
		self.point = point
		self.costs = []
		self.order = order
		self.CostInfo = namedtuple('CostInfo', 'rep rep_coord cost')
	def __repr__(self):
		return f'Cost({self.cluster},\n{self.point})'

	def get_cost_xi_clusters(self):

		for  value, repr_ in zip(self.cluster,self.cluster._fields):
			cost = get_distance(self.point.array, value, self.order)
			self.cost_info = self.CostInfo(repr_, value, cost)
			self.costs.append(self.cost_info)
		return sorted(self.costs, key=lambda x : x.cost)[0]



points = get_points_from_df(df)
clusters = get_combination_of_coord_clusters(df)

total_cost_cluster_dic = {}
for cluster in clusters:
	total_cost = 0
	members_list = []
	for point in points:
		c = Cost(cluster, point, 1)
		total_cost += c.get_cost_xi_clusters()

	total_cost_cluster_dic[cluster] = total_cost

def get_cluster_idx(df, cluster):
	d_idx = {}
	for i in df.index:
		if (df['x1'][i], df['x2'][i]) == cluster.z1:
			d_idx['z1']= i
		elif (df['x1'][i], df['x2'][i]) == cluster.z2:
			d_idx['z2'] = i
	return d_idx

def k_medoids_naive(df, order = 1):
	from collections import defaultdict
	points = get_points_from_df(df)
	clusters = get_combination_of_coord_clusters(df)
	InfoCluster = namedtuple('InfoCluster', 'total_cost members')
	total_cost_cluster_dic = {}
	for cluster in clusters:
		total_cost = 0
		# members_list = {'z1':[], 'z2':[]}
		members_list = defaultdict(lambda: [])
		for point in points:
			c = Cost(cluster, point, order)
			cost_info = c.get_cost_xi_clusters()
			total_cost += cost_info.cost
			members_list[cost_info.rep].append((point.x1,point.x2))

		info_cluster = InfoCluster(total_cost, members_list)
		total_cost_cluster_dic[cluster] = info_cluster

	sorted_costs = sorted(total_cost_cluster_dic.items(), key=lambda d : d[1].total_cost)
	minimal_cost_cluster, min_info_cost = sorted_costs[0]
	z_idx = get_cluster_idx(df, minimal_cost_cluster)
	z1_idx, z2_idx = z_idx['z1'], z_idx['z2']
	gen_plot(df, z1_idx, z2_idx)
	from pprint import pprint
	pprint(sorted_costs)
	print('--------------------------')
	print(f'The best cluster is:{minimal_cost_cluster}\nwith a cost of :\t{min_info_cost.total_cost}')
	print('And the following members composition:')
	print(list(min_info_cost.members.items()))

# =============================================================================
# Question1
# =============================================================================
k_medoids_naive(df)

# =============================================================================
# Question2
# =============================================================================
k_medoids_naive(df, order = 2)


# =============================================================================
# Question 3
# =============================================================================

def k_means_naive(df, order = 1):
	from collections import defaultdict
	points = get_points_from_df(df)
	clusters = get_combination_of_coord_clusters(df)
	InfoCluster = namedtuple('InfoCluster', 'total_cost members')
	total_cost_cluster_dic = {}
	for cluster in clusters:
		total_cost = 0
		# members_list = {'z1':[], 'z2':[]}
		members_list = defaultdict(lambda: [])
		for point in points:
			c = Cost(cluster, point, order)
			cost_info = c.get_cost_xi_clusters()
			total_cost += cost_info.cost
			members_list[cost_info.rep].append((point.x1,point.x2))

		info_cluster = InfoCluster(total_cost, members_list)
		total_cost_cluster_dic[cluster] = info_cluster

	sorted_costs = sorted(total_cost_cluster_dic.items(), key=lambda d : d[1].total_cost)
	minimal_cost_cluster, min_info_cost = sorted_costs[0]
	z_idx = get_cluster_idx(df, minimal_cost_cluster)
	z1_idx, z2_idx = z_idx['z1'], z_idx['z2']
	gen_plot(df, z1_idx, z2_idx)
	from pprint import pprint
	pprint(sorted_costs)
	print('--------------------------')
	print(f'The best cluster is:{minimal_cost_cluster}\nwith a cost of :\t{min_info_cost.total_cost}')
	print('And the following members composition:')
	print(list(min_info_cost.members.items()))





# =============================================================================
# 2. Maximum Likelihood Estimation
# =============================================================================
from collections import Counter
sequence = 'A B A B B C A B A A B C A C'.split()
c = Counter(sequence)
Theta = namedtuple('Theta', 'w count pw')

def get_proba(c, w):
	total = sum(list(c.values()))
	return c[w]/total, f'{c[w]}/{total}'

class MLE:

	def __init__(self, s):
		self.s = s
		self.c = Counter(s)
		self.mle = {}
		self.mle_estimate()
		self.likelihoods = {}

	def __repr__(self):
		return f'MLE(\ns={self.s},\nestimator={[ {k:v.pws}  for k,v in self.mle.items()]}\n)'
	def mle_estimate(self):
		Theta = namedtuple('Theta', 'w count pw pws')
		for k,v in self.c.items():
			proba, proba_str =get_proba(c, k)
			self.mle[k] = Theta(k,v,proba, proba_str)

	def likelihood(self, new_s):
		c = Counter(new_s)
		l = 1
		for w,count_w in c.items():
			l*=self.mle[w].pw**count_w
		print(f'Sequence:\t{new_s}\nhas likelihood:{l}')
		self.likelihoods["".join(new_s)]=l

	def get_element_most_likely(self):
		return sorted(self.likelihoods.items(), key= lambda d : d[1], reverse = True)



m = MLE(sequence)

s1 = ['A', 'B', 'C']
s2 = ['B', 'B', 'B']
s3 = ['A', 'B', 'B']
s4 = ['A', 'A', 'C']
s_lst = [s1,s2,s3,s4]

for ss in s_lst:
	m.likelihood(ss)
m.get_element_most_likely()

s = 'A B A B B C A B A A B C A C'.split()
m= MLE(s)


# Bigram Model p(w2|w1) = count w2 after w1 / count w1


# =============================================================================
# EM algorithm
# =============================================================================

from recordtype import recordtype

class Vector:

	def __init__(self, value_dic,  vector_name = 'Vector' ):
		value_tuple = tuple(value_dic.values())
		self.len_tuple = len(value_tuple)
		self.validate(value_tuple)
		self.vector_nt = recordtype(vector_name,' '.join([k for k in value_dic.keys()]))
		self._vector = value_tuple

	def __repr__(self):
		return self.vector.__repr__()

	def validate(self, value):
		if not isinstance(value, tuple):
			raise TypeError(f'{value} should be a tuple!')
		if not len(value) == self.len_tuple:
			raise ValueError(f'{value} should be of len {self.len_tuple} and not {len(value)}')

	@property
	def vector(self):
		return self.vector_nt(*self._vector)

	@vector.setter
	def vector(self, value):
		self.validate(value)
		self._vector = value

class Theta(Vector):
	def __init__(self, value_dic,  vector_name = 'Theta'):
		super().__init__(value_dic, vector_name)


def norm_pdf(x, mu, sigma):
	import math
	part1= 1/(math.sqrt(2*math.pi*sigma))
	inside_exp = (-1/(2*sigma))*(x-mu)**2
	part2 = math.exp(inside_exp)
	return part1*part2

from FromLinearModelsToDeepLearning.unit_4.mixture_gaussian import pji

def pji(j, xi, p, mu, sigma, is_debug = True):
	pj = p.__getattribute__(f'p{j}')
	pdfij = norm_pdf(xi, mu.__getattribute__(f'mu{j}'), sigma.__getattribute__(f'sigma{j}'))
	proba_total = total_proba(xi,p, mu, sigma)
	if is_debug:
		print(f'pj:\t:{pj}')
		print(f'pdfij:\t:{pdfij}')
		print(f'proba_total:\t:{proba_total}')
	return (pj*pdfij)/proba_total

def total_proba(x, p, mu, sigma):
	K = len(p)
	proba_total = 0
	for j in range(1,K+1):
		pj = p.__getattribute__(f'p{j}')
		pdfij = norm_pdf(x, mu.__getattribute__(f'mu{j}'), sigma.__getattribute__(f'sigma{j}'))
		pij = pj*pdfij
		proba_total += pij
	return proba_total

class EM:

	def __init__(self, x, t):
		self.x = x
		self.t = t

	def get_pj(self, j):
		return self.t.vector.__getattribute__(f'pj{j}')

	def get_muj(self, j):
		return self.t.vector.__getattribute__(f'mu{j}')

	def get_sigmaj(self,j):
		return self.t.vector.__getattribute__(f's{j}')

	def get_param_j(self, j):
		return self.get_pj(j), self.get_muj(j), self.get_sigmaj(j)

	def get_k(self):
		return sum([1 for k,v in self.t.vector._asdict().items() if 'p' in k])

	def total_proba(self, xi):
		K = self.get_k()
		proba_total = 0
		for j in range(1,K+1):
			pj, muj, sigmaj = self.get_param_j(j)
			pdfij = norm_pdf(xi, muj, sigmaj)
			pji = pj*pdfij
			proba_total += pji
		return proba_total

	def get_pji(self, j, i):
		pj,muj,sigmaj = self.get_param_j(j)
		xi = self.x[i]
		pdfij = norm_pdf(xi, muj, sigmaj)
		proba_total = self.total_proba(xi)
		return (pj * pdfij) / proba_total

	def get_log_likelihood(self):
		pji_dic = {}
		for j in range(1, self.get_k() + 1):
			for i in range(len(self.x)):
				pji_dic[f'p{j}|{i}'] = self.get_pji(j, i)
		self.likelihoods = np.array(list(pji_dic.values()), np.float64)
		self.log_likelihoods = np.log(self.likelihoods)
		return np.sum(self.log_likelihoods)

x = np.array([-1,0,4,5,6])
theta_dic = {'pj1': 0.5, 'pj2':0.5, 'mu1':6, 'mu2':7, 's1':1, 's2':4}
t = Theta(theta_dic)
em = EM(x,t)
em.get_log_likelihood()


import numpy as np
try:
	import kmeans
	import common
	import naive_em
	import em
except ModuleNotFoundError:
	import FromLinearModelsToDeepLearning.unit_4.netflix.kmeans as kmeans
	import FromLinearModelsToDeepLearning.unit_4.netflix.common as common
	import FromLinearModelsToDeepLearning.unit_4.netflix.naive_em as naive_em
	import FromLinearModelsToDeepLearning.unit_4.netflix.em as em

X = np.loadtxt(r'C:\Users\sam\Documents\Trainings\FromLinearModelsToDeepLearning\FromLinearModelsToDeepLearning\unit_4\netflix\toy_data.txt')

seeds = [0,1,2,3,4]
mixture, post = common.init(X, 4, 0)
mixture, post, cost = kmeans.run(X,mixture, post )
ks = [1,2,3,4]

from collections import namedtuple
results = namedtuple('results', 'k seed cost')
costs =[]
for k in ks:
	for seed in seeds:
		mixture, post = common.init(X, k, seed)
		mixture, post, cost = kmeans.run(X, mixture, post)
		r = results(k,seed,cost)
		costs.append(r)
		print(r)

def get_best_cost_for_k(costs,k):
	best_cost = np.float('inf')
	d = {}
	for res in costs:
		if (res.k == k)  and (res.cost <= best_cost):
			best_cost = res.cost
			d['best_cost'] = best_cost
	return d['best_cost']

for k in ks:
	best_cost_k = get_best_cost_for_k(costs,k)
	print(f'The best cost for k={k}:\n{best_cost_k}\n-------------------------')

# =============================================================================
# NAIVE EM
# =============================================================================
K=3
d = X.shape[1]

# mu = np.random.uniform(-10,10,K*d).reshape(K,d)
# var = np.random.uniform(0,5, K)
# p = np.random.randint(0,5, K)

# mixture = common.GaussianMixture(mu,var,p)

init_mixture, post = common.init(X,K)


def multivariate_gaussian_pdf(xi, muj, varj):
	import math
	if xi.shape[0]!=muj.shape[0]:
		raise ValueError(f'xi and muj should have the same number of dimensions\n\
				d of xi:\t{xi.shape[0]}\n'
						 f'd of muj:\t{muj.shape[0]}')

	d = xi.shape[0]
	scaling_factor = 1/((2*math.pi*varj)**(d/2))
	center_factor = np.sum((xi-muj)**2)
	position_factor = math.exp(-(center_factor/(2*varj)))
	return scaling_factor*position_factor

def get_param_j(mixture,j):
	pj = mixture.p[j]
	muj = mixture.mu[j]
	varj = mixture.var[j]
	return pj,muj,varj

def get_total_proba(xi, mixture):
	K = len(mixture.p)
	s =0
	for j in range(K):
		pj, muj, varj = get_param_j(mixture, j)
		s+= pj*multivariate_gaussian_pdf(xi, muj, varj)
	return s

def pji(xi, mixture,j):
	pj, muj, varj = get_param_j(mixture, j)
	return (pj*multivariate_gaussian_pdf(xi, muj,varj))/get_total_proba(xi, mixture)

def get_pi(xi, mixture):
	K = len(mixture.p)
	return np.array([pji(xi, mixture,j) for j in range(K) ])

def get_p(X, mixture):
	return np.apply_along_axis(lambda x,m:get_pi(x,m),1, X, mixture)

def lij(xi,mixture, pji, j):
	pj, muj, varj = get_param_j(mixture, j)
	return pji*np.log((pj*multivariate_gaussian_pdf(xi,muj, varj))/pji)

def li(xi, mixture, post, i):
	K = len(mixture.p)
	return np.sum([lij(xi, mixture, post[i,j], j) for j in range(K) ])

def likelihood(X, mixture, post):
	return np.sum([li(X[i], mixture,post, i) for i in range(X.shape[0])])


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
	post = get_p(X, mixture)
	l = likelihood(X, mixture, post)
	return post, l

# def multivariate_gaussian_pdf(xi, muj, varj):
# 	import math
# 	if xi.shape[0]!=muj.shape[0]:
# 		raise ValueError(f'xi and muj should have the same number of dimensions\n\
# 				d of xi:\t{xi.shape[0]}\n'
# 						 f'd of muj:\t{muj.shape[0]}')
#
# 	d = xi.shape[0]
# 	scaling_factor = 1/((2*math.pi*varj)**(d/1))
# 	center_factor = np.sum((xi-muj)**2)
# 	position_factor = math.exp(-(center_factor/(2*varj)))
# 	return scaling_factor*position_factor
#
# def get_param_j(mixture,j):
# 	pj = mixture.p[j]
# 	muj = mixture.mu[j]
# 	varj = mixture.var[j]
# 	return pj,muj,varj
# def get_total_proba(xi, mixture):
# 	K = len(mixture.p)
# 	s =0
# 	for j in range(K):
# 		pj, muj, varj = get_param_j(mixture, j)
# 		s+= pj*multivariate_gaussian_pdf(xi, muj, varj)
# 	return s
#
# def pji(xi, mixture,j):
# 	pj, muj, varj = get_param_j(mixture, j)
# 	return (pj*multivariate_gaussian_pdf(xi, muj,varj))/get_total_proba(xi, mixture)
#
# def get_pi(xi, mixture):
# 	K = len(mixture.p)
# 	return np.array([pji(xi, mixture,j) for j in range(K) ])
#
# def get_p(X, mixture):
# 	return np.apply_along_axis(lambda x,m:get_pi(x,m),1, X, mixture)
#
# def lij(xi,mixture, pji, j):
# 	pj, muj, varj = get_param_j(mixture, j)
# 	return pji*np.log((pj*multivariate_gaussian_pdf(xi,muj, varj))/pji)
#
# def li(xi, mixture, post, i):
# 	K = len(mixture.p)
# 	return np.sum([lij(xi, mixture, post[i,j], j) for j in range(K) ])
#
# def likelihood(X, mixture, post):
# 	return np.sum([li(X[i], mixture,post, i) for i in range(X.shape[0])])


np.random.seed(0)
K = 3
mixture, post = common.init(X,K)

n = X.shape[0]
K= len(mixture.p)

post = get_p(X, mixture)
likelihood(X, mixture, post)

# test case 2

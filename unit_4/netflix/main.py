
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
	position_factor = np.exp(-(center_factor/(2*varj)))
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


def test_muj_hat():
	X = np.arange(10).reshape(5,2)
	post = np.arange(15).reshape(5,3)
	j = 0
	postj = post[:,j]
	sum_pji = np.sum(postj)
	sum_ = np.array([180,210])
	expected = sum_/sum_pji
	output = muj_hat(X, postj)
	assert output == expected

def muj_hat(X, postj):
	sum_pji= np.sum(postj)
	weighted_sum = np.matmul(postj, X)
	return weighted_sum/sum_pji

def get_mu_hat(X, post):
	return np.apply_along_axis(muj_hat,0, X, post)

def get_mu_hat_non_vec(X, post):
	K, d = post.shape[1], X.shape[1]
	mu_hat_array = np.empty((K,d))
	for j in range(K):
		mu_hat_array[j] = muj_hat(X, post[:,j])
	return mu_hat_array



def sigmaj_hat(X, postj, mu_hatj):
	d = X.shape[1]
	sum_pji = d*np.sum(postj)
	centered_data = np.subtract(X, mu_hatj)**2
	weighted_sum = np.matmul(postj.transpose(), centered_data)
	return np.sum(weighted_sum)/sum_pji

def get_sigma_hat(X, post, mu_hat):
	K = post.shape[1]
	var_array = np.array([])
	for j in range(K):
		sigmaj = sigmaj_hat(X, post[:, j], mu_hat[j])
		var_array = np.append(var_array, sigmaj)
	return var_array

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
	p_hat = np.apply_along_axis(np.mean,0,post)
	mu_hat = get_mu_hat_non_vec(X, post)
	sigma_hat = get_sigma_hat(X,post, mu_hat)
	return GaussianMixture(mu_hat, sigma_hat, p_hat)


# =============================================================================
# Run
# =============================================================================
class GaussianMixture(NamedTuple):
    mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component

np.random.seed(0)
K = 3
mixture, post = common.init(X,K)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    new_l = -np.float('inf')
    eval_ = lambda n,o : (n - o) > (10 ** (-6)) * np.abs(n)
    while True:
        old_l = new_l
        post, new_l = estep(X, mixture)
        mixture = mstep(X, post)
        if not eval_(new_l, old_l):
            break
    return mixture, new_l



# =============================================================================
# 4. Comparing K means and EM
# =============================================================================

seeds = [0,1,2,3,4]
# mixture, post = common.init(X, 4, 0)
# mixture, post, cost = kmeans.run(X,mixture, post )
ks = [1,2,3,4]

from collections import namedtuple
results = namedtuple('results', 'k seed cost')
costs =[]
for k in ks:
	for seed in seeds:
		mixture, post = common.init(X, k, seed)
		mixture, cost = run(X, mixture, post)
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


def run_with_post(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    new_l = -np.float('inf')
    eval_ = lambda n,o : (n - o) > (10 ** (-6)) * np.abs(n)
    while True:
        old_l = new_l
        post, new_l = estep(X, mixture)
        mixture = mstep(X, post)
        if not eval_(new_l, old_l):
            break
    return mixture, new_l, post

em_mixtures, kmeans_mixtures = [],[]
for k in ks:
	init_mixture, init_post = common.init(X, k)
	em_mixture, em_l, em_post = run_with_post(X, init_mixture, init_post)
	em_mixtures.append([em_mixture, em_l, em_post])
	kmeans_mixture, kmeans_post, kmeans_cost = kmeans.run(X, init_mixture, init_post)
	kmeans_mixtures.append([kmeans_mixture, kmeans_post, kmeans_cost])

k=4
i=k-1
plot(X, em_mixtures[i][0], em_mixtures[i][2], f'EM for {k}')
plot(X, kmeans_mixtures[i][0], kmeans_mixtures[i][1], f'Kmeans for {k}')


# =============================================================================
# BEST BIC
# =============================================================================
def bic_func(X: np.ndarray, mixture: GaussianMixture,
        log_likelihood: float) -> float:
    """Computes the Bayesian Information Criterion for a
    mixture of gaussians

    Args:
        X: (n, d) array holding the data
        mixture: a mixture of spherical gaussian
        log_likelihood: the log-likelihood of the data

    Returns:
        float: the BIC for this mixture
    """
    n = X.shape[0]
    var_p = mixture.var.shape[0]
    mean_p = mixture.mu.shape[0]*mixture.mu.shape[1]
    p = mixture.p.shape[0]-1
    free_params = var_p+mean_p+p
    return log_likelihood -((1/2)*free_params*np.log(n))


em_mixtures= []
bic_dic = {}
for k in ks:
	init_mixture, init_post = common.init(X, k)
	em_mixture, em_l, em_post = run_with_post(X, init_mixture, init_post)
	em_mixtures.append([em_mixture, em_l, em_post])
	bic = bic_func(X, em_mixture, em_l)
	bic_dic[f'bic_{k}']=bic
	print(f'for K={k}\nBIC:={bic}\n----------------------')


# =============================================================================
# Implementing EM for matrix completion
# =============================================================================

def make_it_cu(xi, muj, varj):
	mask = xi!=0
	xi_cu = xi[mask]
	muj_cu = muj[mask]
	varj_cu = varj
	return xi_cu, muj_cu, varj_cu


def multivariate_gaussian_pdf(xi, muj, varj):
	import math
	xi, muj, varj = make_it_cu(xi, muj, varj)

	if xi.shape[0]!=muj.shape[0]:
		raise ValueError(f'xi and muj should have the same number of dimensions\n\
				d of xi:\t{xi.shape[0]}\n'
						 f'd of muj:\t{muj.shape[0]}')

	d = xi.shape[0]
	scaling_factor = 1/((2*math.pi*varj)**(d/2))
	center_factor = np.sum((xi-muj)**2)
	position_factor = np.exp(-(center_factor/(2*varj)))
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
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
	post = get_p(X, mixture)
	l = likelihood(X, mixture, post)
	return post, l

path_to_test = r'C:\Users\sam\Documents\Trainings\FromLinearModelsToDeepLearning\FromLinearModelsToDeepLearning\unit_4\netflix\test_incomplete.txt'

X = np.loadtxt(path_to_test)
K=4
mu = np.array([[2, 4, 5, 5, 0],
 [3, 5, 0, 4, 3],
 [2, 5, 4, 4, 2],
 [0, 5, 3, 3, 3]])
var = np.array([5.93, 4.87, 3.99, 4.51])
p = np.array([0.25, 0.25, 0.25, 0.25])
mixture = GaussianMixture(mu, var, p)

post,l = estep(X, mixture)

def test_mu_l_k_hat():
	l ,k = 0, 0
	x_l = X[:,l]
	post_k = post[:, k]
	eta_l  = eta[:,l]

def eta_vec(X):
	def eta(xij):
		if xij:
			return 1
		return 0
	return np.vectorize(eta)(X)


def mu_l_k_hat(x_l, post_k, eta_l):
	sum_pku = np.sum(post_k*eta_l)
	weighted_sum = np.sum(post_k*eta_l*x_l)
	return weighted_sum/sum_pku

def mu_k_hat(X, post_k, eta):
	mu_k = []
	for l in range(X.shape[1]):
		x_l = X[:, l]
		eta_l = eta[:, l]
		mu_k.append(mu_l_k_hat(x_l, post_k, eta_l))
	return np.array(mu_k)

def get_mu_hat(X, post, eta):
	mu_hat= []
	for k in range(post.shape[1]):
		mu_hat.append(mu_k_hat(X, post[:,k], eta))
	return np.array(mu_hat)

def sigma_k_hat(X, mu_hat, post, eta, k):
	sigma_k = 0
	for u in range(X.shape[0]):
		den_k_u = np.sum(eta[u])*post[u,k]
		s = 0
		for l in range(X.shape[1]):
			if eta[u,l]:
				diff = (X[u, l]-mu_hat[k,l])**2
				num_k_u = post[u,k]*diff
				s+=num_k_u/diff
		sigma_k+=s/den_k_u
	return sigma_k

def get_sigma_hat(X, mu_hat, post, eta, min_variance):
	sigma = []
	for k in range(post.shape[1]):
		sigma_k =sigma_k_hat(X, mu_hat, post, eta, k)
		sigma.append(max(sigma_k, min_variance))
	return np.array(sigma)

def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
	eta = eta_vec(X)
	p_hat = np.apply_along_axis(np.mean, 0, post)
	mu_hat = get_mu_hat(X, post, eta)
	sigma_hat = get_sigma_hat(X, mu_hat, post, eta, min_variance)
	return GaussianMixture(mu_hat, sigma_hat, p_hat)
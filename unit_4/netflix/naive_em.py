"""Mixture model using EM"""
from typing import Tuple
import numpy as np
import math

try:
    from common import GaussianMixture
except ModuleNotFoundError:
    from FromLinearModelsToDeepLearning.unit_4.netflix.common import GaussianMixture





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
    raise NotImplementedError


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
    raise NotImplementedError

import numpy as np

d = {}
for i in range(1,4):
	d[f'x{i}'] = None

d['x1'] = np.array([-1.2,-0.8])
d['x2'] = np.array([-1,-1.2])
d['x3'] = np.array([-0.8,-1])

for i in range(1,4):
	d[f'x{i+3}'] = -d[f'x{i}']

c1 = np.empty((3,2))
for i in range(1,4):
	c1[i-1] = d[f'x{i}']


c2 = np.empty((3,2))
for i in range(4,7):
	c2[i-4] = d[f'x{i}']


# =============================================================================
# EM
# =============================================================================
from scipy.stats import norm
from recordtype import recordtype

mu1, mu2 = -3,2
Mu = recordtype('Mu', 'mu1 mu2')
mu = Mu(mu1,mu2)
sigma1 = sigma2 = 4
Sigma = recordtype('Sigma', 'sigma1 sigma2')
sigma = Sigma(sigma1,sigma2)
p1 = p2 = 1/2
P = recordtype('P', 'p1 p2')
p = P(p1,p1)
x1,x2,x3,x4,x5 = 0.2,-0.9,-1,1.2,1.8
x_lst = [x1,x2,x3,x4,x5]

def norm_pdf(x, mu, sigma):
	import math
	part1= 1/(math.sqrt(2*math.pi*sigma))
	inside_exp = (-1/(2*sigma))*(x-mu)**2
	part2 = math.exp(inside_exp)
	return part1*part2
def total_proba(x, p, mu, sigma):
	K = len(p)
	proba_total = 0
	for j in range(1,K+1):
		pj = p.__getattribute__(f'p{j}')
		pdfij = norm_pdf(x, mu.__getattribute__(f'mu{j}'), sigma.__getattribute__(f'sigma{j}'))
		pij = pj*pdfij
		proba_total += pij
	return proba_total

def pji(j, xi, p, mu, sigma, is_debug = True):
	pj = p.__getattribute__(f'p{j}')
	pdfij = norm_pdf(xi, mu.__getattribute__(f'mu{j}'), sigma.__getattribute__(f'sigma{j}'))
	proba_total = total_proba(xi,p, mu, sigma)
	if is_debug:
		print(f'pj:\t:{pj}')
		print(f'pdfij:\t:{pdfij}')
		print(f'proba_total:\t:{proba_total}')
	return (pj*pdfij)/proba_total


class PJI:
	def __init__(self,j,i,xi,p,mu,sigma):
		self.xi = xi
		self.p = p
		self.mu = mu
		self.sigma = sigma
		self.i = i
		self.j = j
		self.pji = pji(j,xi,p,mu,sigma, is_debug = False)
	def __repr__(self):
		return f'P({self.j}|{self.i})={round(self.pji,5)}'

pji_list = []
for i,x in enumerate(x_lst):
	pji_elt = PJI(1,i+1,x,p,mu, sigma)
	print(pji_elt)
	pji_list.append(pji_elt.pji)

import matplotlib.pyplot as plt

pji_values_list = [(x,pji_value) for x, pji_value in zip(x_lst, pji_list)]
sorted_pji = sorted(pji_values_list, key=lambda x : x[0])
plt.plot([x[0] for x in sorted_pji], [x[1] for x in sorted_pji])

pj = np.sum(pji_list)/len(pji_list)
uj = np.sum(list(map(lambda a,b : a*b, pji_list, x_lst)))/np.sum(pji_list)

x_array = np.array(x_lst)
dist = x_array - uj
dist = dist **2
pji_array = np.array(pji_list)

sigmaj = np.dot(pji_array, dist)/(np.sum(pji_list))
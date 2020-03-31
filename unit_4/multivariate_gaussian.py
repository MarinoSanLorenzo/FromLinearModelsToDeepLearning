import numpy as np

import math
x= np.array([1/(math.sqrt(math.pi)), 2])
mu = np.array([0,2])
sigma = math.sqrt(1/(2*math.pi))

cov = np.cov(x)


def pdf_gaussian(x, mu, sigma):
	d = len(x)
	part1 = (1/(2*math.pi*sigma**2)**(d/2))
	part2 = math.exp(-((1/(2*sigma**2))*(np.linalg.norm(x-mu))**2))
	print(part1)
	print(part2)
	return part1*part2

r = pdf_gaussian(x, mu, sigma)

math.log(r)



import numpy as np
from itertools import product
from pprint import pprint
l = [(0,0),(0,1), (1,0), (1,1)]
comb = list(product([*l, *l],repeat = 2))
pprint(comb)

def calc_nb_steps(tuple_):
	a, b = tuple_
	diff = np.array(a)- np.array(b)
	return np.sum(np.abs(diff))

set_comb = set(comb)
nb_steps_list = []
for c in set_comb:
	nb_steps = calc_nb_steps(c)
	print(f'c:\t{c}')
	print(f'nb_steps:{nb_steps}')
	nb_steps_list.append(nb_steps)


def gen_discounted_value(nb_steps, gamma = 0.5):
	rewards = []
	for t in range(nb_steps):
		 rewards.append((gamma**t)*-0.01)
	else:
		rewards.append((gamma**nb_steps)*1)
	print(rewards)
	return sum(rewards)

rewards = []
for nb_step in nb_steps_list:
	rewards.append(gen_discounted_value(nb_step))

print(sum(rewards)/len(rewards))
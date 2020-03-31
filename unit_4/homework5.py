import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
	return [first_cluster] + cluster_comb

cluster_to_try = get_combination_of_coord_clusters(df)
Cluster = namedtuple('Cluster', 'z1 z2')
clusters = [Cluster(cluster[0], cluster[1]) for cluster in cluster_to_try]
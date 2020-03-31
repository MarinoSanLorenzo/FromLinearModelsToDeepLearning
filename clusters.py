# =============================================================================
# Calculating Costs
# =============================================================================

from scipy.spatial import distance
import numpy as  np
x1 = np.array([-1,2])
x2 = np.array([-2,1])
x3 = np.array([-1,0])
x4 = np.array([2,1])
x5 = np.array([3,2])

z1 = np.array([-1,1])
z2 = np.array([2,2])

class Cluster:
	def __init__(self, *coord, representative):
		self.coord = [c for c in coord]
		self.z = representative
	def get_euclidean_distance(self):
		s = 0
		for p in self.coord:
			s+= np.linalg.norm(p-self.z)
		return s



c1 = Cluster(x1,x2,x3,representative = z1)

c1.get_euclidean_distance()
c2 = Cluster(x4,x5, representative = z2)
c2.get_euclidean_distance()

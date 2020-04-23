from pprint import pprint
import numpy as np
from numbers import Real
from recordtype import recordtype
# =============================================================================
# Another Example of Value Iteration (Software Implementation)
# =============================================================================

def validate_type(value, type_, msg = None):
	if not isinstance(value,type_):
		msg = f'value:{value} should be of type {type_.__name__}' if not msg else msg
		raise TypeError(msg)


class State1D:

	def __init__(self,coord):
		validate_type(coord, int)
		self.coord = coord

	def __repr__(self):
		return f'State(s={self.coord})'

	@property
	def state(self):
		return self.coord

class Reward:

	def __init__(self, coord, reward):
		validate_type(coord,int)
		validate_type(reward, Real)
		self.coord = coord
		self.reward = reward




nb_states = 5
states = [State1D(state) for state in range(nb_states)]
from pprint import pprint
import numpy as np
from numbers import Real
from recordtype import recordtype
from collections import namedtuple
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

	def __call__(self):
		return self.coord

class Reward:

	def __init__(self, coord_ini, action, coord_final, reward):
		validate_type(coord_ini,int)
		validate_type(action, str)
		validate_type(coord_final, int)
		validate_type(reward, Real)
		self.coord_ini = coord_ini
		self.action = action
		self.coord_final = coord_final
		self._reward = reward

	def __repr__(self):
		return f'R(s={self.coord_ini}, a={self.action}, sf={self.coord_final})={self.reward}'

	@property
	def reward(self):
		return self._reward

	def __call__(self):
		return self._reward

class Action:

	def __init__(self):
		self.actions = ['LEFT', 'STAY', 'RIGHT']
		Actions = namedtuple('Actions', ' '.join(self.actions))
		self.action_nt = Actions(*self.actions)

	def move_left(self):
		return self.action_nt.LEFT

	def stay(self):
		return self.action_nt.STAY

	def move_right(self):
		return self.action_nt.RIGHT

	def get_list_actions(self):
		return self.actions

class TransitionProb:

	def __init__(self):
		transitions_probas = {"STAY": np.array([
			[1 / 2, 1 / 2, 0, 0, 0],
			[1 / 4, 1 / 2, 1 / 4, 0, 0],
			[0, 1 / 4, 1 / 2, 1 / 4, 0],
			[0, 0, 1 / 4, 1 / 2, 1 / 4],
			[0, 0, 0, 1 / 2, 1 / 2]

		]), 'LEFT': np.array([
			[1, 0, 0, 0, 0],
			[1 / 3, 2 / 3, 0, 0, 0],
			[0, 1 / 3, 2 / 3, 0, 0],
			[0, 0, 1 / 3, 2 / 3, 0],
			[0, 0, 0, 1 / 3, 2 / 3]

		]), 'RIGHT': np.array([
			[2 / 3, 1 / 3, 0, 0, 0],
			[0, 2 / 3, 1 / 3, 0, 0],
			[0, 0, 2 / 3, 1 / 3, 0],
			[0, 0, 0, 2 / 3, 1 / 3],
			[0, 0, 0, 0, 1]

		])

		}
		self.transitions_probas = transitions_probas

	def proba_move_from_to_after_action(self, pos_ini, pos_final, action):
		return self.transitions_probas[action][pos_ini, pos_final]

class ValueNotInCache(KeyError):
	pass

class Value:

	def __init__(self, states, rewards, actions, trans_proba, discount = 0.5):
		self.discount = discount
		self.states = states
		self.rewards = rewards
		self.actions = actions
		self.trans_proba = trans_proba.transitions_probas
		self.value_dic = {}
		self.value = np.array([0 for _ in range(len(self.rewards))])
		self.value_dic[0] = self.value

	def __repr__(self):
		return f'{self.value_dic}'

	def check_cache(self,k):
		try:
			a = self.value_dic[k]
		except KeyError:
			msg = f'Key:{k} is not in {self.value_dic}'
			raise ValueNotInCache(msg)


	def get_trans_proba(self, action, state):
		return self.trans_proba[action][state]

	def get_value_k_for_state_and_action(self, state, action , k):
		self.check_cache(k-1)
		self.value_k_1 = self.value_dic[k-1]
		return np.sum(self.get_trans_proba(action, state)*(self.rewards[state]() + self.discount*self.value_k_1))

	def get_value_k_for_all_states_for_action(self, action, k):
		return [ self.get_value_k_for_state_and_action(state=state(),
																action = action,
																k=k) for state in self.states]


# =============================================================================
# VARIABLES
# =============================================================================

def test():
	nb_states = 5
	states = [State1D(state) for state in range(nb_states)]
	rewards_array = np.array([0,0,0,0,1])
	rewards = [Reward(i,'R', i+1, reward) for i,reward in zip(range(len(states)), rewards_array)  ]
	actions = Action()
	trans_proba = TransitionProb()
	discount = 1/2
	v = Value(states, rewards, actions, trans_proba)
	v.get_value_k_for_state_and_action(state=0,action='STAY',k=1)

	assert np.all(np.array([0,0,0,0,1]) ==
				  v.get_value_k_for_all_states_for_action(action = 'STAY', k=1))

if __name__ == '__main__':
    test()
import numpy as np
from collections import namedtuple
from FromLinearModelsToDeepLearning.unit_5.value_iteration import Value, State1D

class States:

	def __init__(self):
		self._A = ('A', 0)
		self._B = ('B', 1)
		self._C = ('C', 2)
		self._D = ('D', 3)
		self.states = [getattr(self, state_str) for state_str in ['A', 'B', 'C', 'D'] ]

	@property
	def A(self):
		return self._A[1]

	@property
	def B(self):
		return self._B[1]


	@property
	def C(self):
		return self._C[1]


	@property
	def D(self):
		return self._D[1]

	def __len__(self):
		return len(self.states)

	def __getitem__(self, item):
		return self.states[item]



class Action:

	def __init__(self):
		self.actions = ['UP', 'DOWN']
		Actions = namedtuple('Actions', ' '.join(self.actions))
		self.action_nt = Actions(*self.actions)

	def get_list_actions(self):
		return self.actions

class TransitionProb:

	def __init__(self):
		self.transitions_probas = {"UP": np.array([
			[0, 1, 0, 0],
			[0,0, 1,0],
			[0,0,0,1],
			[0,0,0,0]

		]), 'DOWN': np.array([
			[0,0,0,0],
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1, 0]

		])

		}

		self.reward_proba ={'UP': np.array([
			[0, 1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 10],
			[0, 0, 0, 0]
		]),
			'DOWN': np.array([
				[0, 0, 0, 0],
				[1, 0,0 , 0],
				[0, 1, 0, 0],
				[0, 0, 10, 0]
			])
		}

	def proba_move_from_to_after_action(self, pos_ini, pos_final, action):
		return self.transitions_probas[action][pos_ini, pos_final]


def run():
	import sys
	print(sys.argv)
	RANGE = int(sys.argv[1])
	nb_states = 4
	states = [State1D(state) for state in range(nb_states)]
	actions = Action()
	trans_proba = TransitionProb()
	v = Value(states, actions, trans_proba, discount = 0.75)

	for k in range(1,RANGE):
		v.get_value_star_k(k)
		print(f'----------------- iteration:\t{k} -------------------')
		v.print_info()
	else:
		print('You reached convergence and finished the exploration mode!')
		print(v.get_final_output())
		# [24.25, 31.0, 40.0, 40.0]
		# 		# [24.25, 31.0, 40.0, 40.0]

if __name__ == '__main__':
    run()
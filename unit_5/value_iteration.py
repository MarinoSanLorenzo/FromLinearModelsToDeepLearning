from pprint import pprint
import pandas as pd
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

	# transitions_probas = {"STAY": np.array([
	#
	# 	[1 / 2, 1 / 2, 0, 0, 0],
	# 	[1 / 4, 1 / 2, 1 / 4, 0, 0],
	# 	[0, 1 / 4, 1 / 2, 1 / 4, 0],
	# 	[0, 0, 1 / 4, 1 / 2, 1 / 4],
	# 	[0, 0, 0, 0 , 1]
	#
	# ]), 'LEFT': np.array([
	# 	[1/2, 1/2, 0, 0, 0],
	# 	[1 / 3, 2 / 3, 0, 0, 0],
	# 	[0, 1 / 3, 2 / 3, 0, 0],
	# 	[0, 0, 1 / 3, 2 / 3, 0],
	# 	[0, 0, 0, 0, 1]
	#
	# ]), 'RIGHT': np.array([
	# 	[2 / 3, 1 / 3, 0, 0, 0],
	# 	[0, 2 / 3, 1 / 3, 0, 0],
	# 	[0, 0, 2 / 3, 1 / 3, 0],
	# 	[0, 0, 0, 2 / 3, 1 / 3],
	# 	[0, 0, 0, 0, 1]
	#
	# ])
	#
	# }
	def __call__(self):
		pass

class TransitionProb:

	def __init__(self):
		self.transitions_probas = {"STAY": np.array([
			[1 / 2, 1 / 2, 0, 0, 0],
			[1 / 4, 1 / 2, 1 / 4, 0, 0],
			[0, 1 / 4, 1 / 2, 1 / 4, 0],
			[0, 0, 1 / 4, 1 / 2, 1 / 4],
			[0, 0, 0, 0, 1]

		]), 'LEFT': np.array([
			[1 / 2, 1 / 2, 0, 0, 0],
			[1 / 3, 2 / 3, 0, 0, 0],
			[0, 1 / 3, 2 / 3, 0, 0],
			[0, 0, 1 / 3, 2 / 3, 0],
			[0, 0, 0, 0, 1]

		]), 'RIGHT': np.array([

			[2 / 3, 1 / 3, 0, 0, 0],
			[0, 2 / 3, 1 / 3, 0, 0],
			[0, 0, 2 / 3, 1 / 3, 0],
			[0, 0, 0, 2 / 3, 1 / 3],
			[0, 0, 0, 0, 1]

		])

		}

		self.reward_proba ={'STAY': np.array([
			[0, 0, 0, 0, 1],
			[0, 0, 0, 0, 1],
			[0, 0, 0, 0, 1],
			[0, 0, 0, 0, 1],
			[0, 0, 0, 0, 1]
		]),
			'RIGHT': np.array([
				[0, 0, 0, 0, 1],
				[0, 0, 0, 0, 1],
				[0, 0, 0, 0, 1],
				[0, 0, 0, 0, 1],
				[0, 0, 0, 0, 1]
			]),
			'LEFT': np.array([
				[0, 0, 0, 0, 1],
				[0, 0, 0, 0, 1],
				[0, 0, 0, 0, 1],
				[0, 0, 0, 0, 1],
				[0, 0, 0, 0, 1]
			])
		}

	def proba_move_from_to_after_action(self, pos_ini, pos_final, action):
		return self.transitions_probas[action][pos_ini, pos_final]

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

class ValueNotInCache(KeyError):
	pass

class Value:

	def __init__(self, states, actions, trans_proba, discount = 0.5):
		self.discount = discount
		self.states = states
		self.rewards = trans_proba.reward_proba
		self.actions_list = actions.get_list_actions()
		self.trans_proba = trans_proba.transitions_probas
		self.value_star_dic = {}
		self.action_star_dic = {}
		self.value = np.array([0 for _ in range(len(self.states))])
		self.value_star_dic[0] = self.value

	def __repr__(self):
		return f'{self.value_star_dic}'

	def check_cache(self,k):
		try:
			a = self.value_star_dic[k]
		except KeyError:
			msg = f'Key:{k} is not in {self.value_star_dic}'
			raise ValueNotInCache(msg)


	def getT_s_a_s_prime(self, state, action, state_prime):
		return self.trans_proba[action][state, state_prime]

	def get_v_k_s_a_sp(self, k, state, action, state_prime):
		self.check_cache(k-1)
		value_k_1 = self.value_star_dic[k-1][state_prime]
		reward = self.rewards[action][state, state_prime]
		proba = self.getT_s_a_s_prime(state, action, state_prime)
		return proba*(reward + self.discount*value_k_1)

	def get_v_k_s_a(self,k, state,action):
		v_k_s_a = []
		for state_prime in self.states:
			v_k_s_a.append(self.get_v_k_s_a_sp(k = k,
										  state = state,
										  action = action,
										  state_prime = state_prime()))
		return np.sum(np.array(v_k_s_a))

	def get_v_k_a(self, k, action):
		v_k_a = []
		for state in self.states:
			v_k_a.append(self.get_v_k_s_a(k=k,
										  state= state(),
										  action = action))
		return np.array(v_k_a)

	def get_vk_for_all_a(self, k):
		value_k = {}
		for action in self.actions_list:
			value_k['k']= k
			value_k[action] = self.get_v_k_a( k = k, action = action)

		return value_k
	# def get_value_k_for_state_and_action(self, state, action , k):
	# 	self.check_cache(k-1)
	# 	self.value_k_1 = self.value_star_dic[k-1]
	# 	rewards = self.rewards[action][state]
	# 	# if state == 4 and (k-1) >= 1:
	# 	# 	return self.value_k_1[state]
	# 	return np.sum(self.get_trans_proba(action, state)*(rewards + self.discount*self.value_k_1))
	#
	# def get_value_k_for_all_states_for_action(self, action, k):
	# 	return [ self.get_value_k_for_state_and_action(state=state(),
	# 															action = action,
	# 															k=k) for state in self.states]

	# def get_value_k_for_all_actions(self,k):
	# 	value_k = {}
	# 	for action in self.actions_list:
	# 		value_k['k']= k
	# 		value_k[action] = self.get_value_k_for_all_states_for_action( action = action, k = k)
	# 	return value_k

	def get_value_star_k(self, k):
		df = pd.DataFrame(self.get_vk_for_all_a(k))
		df_actions = df[self.actions_list]
		value_star_k = df_actions.values.max(axis=1)
		self.value_star_dic[k] = value_star_k
		self.action_star_dic[k] = df_actions.idxmax(axis =1)

	def print_info(self, last_row = True):
		df_action_star = pd.DataFrame.from_dict(self.action_star_dic, orient = 'index')
		df_value_star = pd.DataFrame.from_dict(self.value_star_dic, orient = 'index')
		if last_row:
			last_idx = df_value_star.last_valid_index()
			last_value = list(df_value_star.iloc[last_idx, ::])
			last_act_idx = df_action_star.last_valid_index()
			last_action = list(df_action_star.iloc[last_act_idx-1, ::])
			print(f'last value stars:\n{last_value}')
			print('-----------------------------------------------------')
			print(f'last action stars:\n{last_action}')

	def get_actions_df(self):
		return pd.DataFrame.from_dict(self.action_star_dic, orient = "index")

	def get_values_df(self):
		return pd.DataFrame.from_dict(self.value_star_dic, orient = "index")

	def get_final_output(self):
		df = self.get_values_df()
		last_output = list(df.iloc[df.last_valid_index(),::])
		return [round(e,4) for e in last_output]
# =============================================================================
# VARIABLES
# =============================================================================

def run():
	import sys
	print(sys.argv)
	RANGE = int(sys.argv[1])
	nb_states = 5
	states = [State1D(state) for state in range(nb_states)]
	actions = Action()
	trans_proba = TransitionProb()
	v = Value(states, actions, trans_proba)

	for k in range(1,RANGE):
		v.get_value_star_k(k)
		v.print_info()
	else:
		print('You reached convergence and finished the exploration mode!')
		print(v.get_final_output())

		#[0.016667,0.05,0.2,0.8,1.2]
if __name__ == '__main__':
    run()
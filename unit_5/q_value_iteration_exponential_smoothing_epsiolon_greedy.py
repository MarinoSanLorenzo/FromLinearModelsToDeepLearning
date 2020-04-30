import numpy as np
from collections import defaultdict
def validate_type(value, type_):
	if not isinstance(value, type_):
		raise TypeError(f'value:{value} should be of type:{type_.__name__} instead of type:{type(value).__name__}')
class QValue:

	def __init__(self, states, trans_proba, rewards, action_list, alpha=1, gamma=1):
		self.states = states
		self.trans_proba = trans_proba
		self.rewards = rewards
		self.action_list = action_list
		self.alpha = alpha
		self.gamma = gamma
		self.q_value_0 = np.zeros(self.states.shape)
		self.q_value_dic = defaultdict(dict)
		for action in self.action_list:
			q_value_action = {}
			q_value_action[action] = self.q_value_0
			self.q_value_dic[0].update(q_value_action)

		def check_cache(self, i):
			class NonExistingQValueIterationError(KeyError):
				pass
			try:
				_ = self.q_value_dic[i]
			except KeyError:
				msg = f'key:{i} was not found in {self}.q_value_dic'
				raise NonExistingQValueIterationError(msg)

		def get_q_i_s_a(self, i,s,a):
			validate_type(i, int)
			validate_type(s, tuple)
			validate_type(a, str)
			check_cache(i)
			return self.q_value_dic[i][a][s]

		def get_r_s_a_sp(self, s,a,sp):
			validate_type(s, tuple)
			validate_type(sp, tuple)
			validate_type(a, str)
			return self.rewards[a][s][sp]




if __name__ == '__main__':

	alpha = 1  # smoothing parameter
	gamma = 1  # discount
	actions_list = ['UP', 'DOWN', 'LEFT', 'RIGHT']
	dim = 12  # number of states
	shape = (3, 4)  # 2d representation
	states = np.arange(dim).reshape(shape)

	up_trans_proba = np.zeros((*shape, *shape))  # tensor T((s1,s2), 'UP', (s1', s2'))
	up_trans_proba[1, 0][0, 0] = 1
	up_trans_proba[1, 2][0, 2] = 1
	up_trans_proba[1, 3][0, 3] = 1
	up_trans_proba[2, 0][1, 0] = 1
	up_trans_proba[2, 2][1, 2] = 1
	up_trans_proba[2, 3][1, 3] = 1

	down_trans_proba = np.zeros((*shape, *shape))
	down_trans_proba[0, 0][1, 0] = 1
	down_trans_proba[0, 2][1, 2] = 1
	down_trans_proba[0, 3][1, 3] = 1
	down_trans_proba[1, 0][2, 0] = 1
	down_trans_proba[1, 2][2, 2] = 1
	down_trans_proba[1, 3][2, 3] = 1

	left_trans_proba = np.zeros((*shape, *shape))
	left_trans_proba[0, 1][0, 0] = 1
	left_trans_proba[0, 2][0, 1] = 1
	left_trans_proba[0, 3][0, 2] = 1
	left_trans_proba[1, 3][1, 2] = 1
	left_trans_proba[2, 1][2, 0] = 1
	left_trans_proba[2, 2][2, 1] = 1
	left_trans_proba[2, 3][2, 2] = 1

	right_trans_proba = np.zeros((*shape, *shape))
	right_trans_proba[0, 0][0, 1] = 1
	right_trans_proba[0, 1][0, 2] = 1
	right_trans_proba[0, 2][0, 3] = 1
	right_trans_proba[1, 2][1, 3] = 1
	right_trans_proba[2, 0][2, 1] = 1
	right_trans_proba[2, 1][2, 2] = 1
	right_trans_proba[2, 2][2, 3] = 1

	trans_proba = {'UP': up_trans_proba,
				   'DOWN': down_trans_proba,
				   'LEFT': left_trans_proba,
				   'RIGHT': right_trans_proba
				   }

	up_rewards = np.zeros((*shape, *shape))
	up_rewards[1, 0][0, 0] = -0.04
	up_rewards[1, 2][0, 2] = -0.04
	up_rewards[1, 3][0, 3] = -0.04
	up_rewards[2, 0][1, 0] = -0.04
	up_rewards[2, 2][1, 2] = -0.04
	up_rewards[2, 3][1, 3] = -1

	down_rewards = np.zeros((*shape, *shape))
	down_rewards[0, 2][1, 2] = -0.04
	down_rewards[0, 3][1, 3] = -0.04
	down_rewards[0, 0][1, 0] = -0.04
	down_rewards[1, 0][2, 0] = -0.04
	down_rewards[1, 2][2, 2] = -0.04
	down_rewards[1, 3][2, 3] = -0.04

	left_rewards = np.zeros((*shape, *shape))
	left_rewards[0, 1][0, 0] = -0.04
	left_rewards[0, 2][0, 1] = -0.04
	left_rewards[0, 3][0, 2] = -0.04
	left_rewards[1, 3][1, 2] = -0.04
	left_rewards[2, 1][2, 0] = -0.04
	left_rewards[2, 2][2, 1] = -0.04
	left_rewards[2, 3][2, 2] = -0.04

	right_rewards = np.zeros((*shape, *shape))
	right_rewards[0, 0][0, 1] = -0.04
	right_rewards[0, 1][0, 2] = -0.04
	right_rewards[0, 2][0, 3] = 1
	right_rewards[1, 2][1, 3] = -1
	right_rewards[2, 0][2, 1] = -0.04
	right_rewards[2, 1][2, 2] = -0.04
	right_rewards[2, 2][2, 3] = -0.04

	rewards = {'UP': up_rewards,
			   'DOWN': down_rewards,
			   'LEFT': left_rewards,
			   'RIGHT': right_rewards
			   }


	q = QValue(states, trans_proba, rewards,actions_list)
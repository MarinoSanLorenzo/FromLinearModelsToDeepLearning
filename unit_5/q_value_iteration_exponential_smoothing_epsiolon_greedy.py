	import numpy as np
	from collections import defaultdict, namedtuple


	class ImpossibleToMoveFromThisCell(IndexError):
		pass


	def validate_type(value, type_):
		if not isinstance(value, type_):
			raise TypeError(f'value:{value} should be of type:{type_.__name__} instead of type:{type(value).__name__}')


	class QValue:

		def __init__(self, states, trans_proba, rewards, action_list, alpha = 1, gamma = 1):
			self.states = states
			self.trans_proba = trans_proba
			self.rewards = rewards
			self.action_list = action_list
			self.alpha = alpha
			self.gamma = gamma
			# defaultdict(lambda:np.zeros(self.states.shape))
			self.q_value_dic = defaultdict(lambda: defaultdict(lambda: np.zeros(self.states.shape)))
			for action in self.action_list:
				# self.q_value_action[action] = np.zeros(self.states.shape)
				self.q_value_dic[0][action] = np.zeros(self.states.shape)

			self.q_max_nt = namedtuple('Qmax', 'q_max policy')

			self.get_coordinates()

		def get_coordinate(self, idx_state):
			np_coord = np.where(self.states == idx_state)
			return np_coord[0][0], np_coord[1][0]

		def get_coordinates(self):
			coord_list = []
			N, K = self.states.shape
			for i in range(N):
				for k in range(K):
					coord_list.append(self.get_coordinate(self.states[i, k]))
			self.coord_list = coord_list

		def check_cache(self, i):
			class NonExistingQValueIterationError(KeyError):
				pass

			try:
				_ = self.q_value_dic[i]
			except KeyError:
				msg = f'key:{i} was not found in {self}.q_value_dic'
				raise NonExistingQValueIterationError(msg)

		def get_q_i_s_a(self, i, s, a):
			validate_type(i, int)
			validate_type(s, tuple)
			validate_type(a, str)
			self.check_cache(i)
			return self.q_value_dic[i][a][s]

		def get_r_s_a_sp(self, s, a, sp):
			validate_type(s, tuple)
			validate_type(sp, tuple)
			validate_type(a, str)
			return self.rewards[a][s][sp]

		def get_max_q_i_s(self, i, s):

			qs = {}
			for a in self.action_list:
				qs[a] = self.get_q_i_s_a(i, s, a)

			max_policy = sorted(qs, key = lambda k :qs[k], reverse = True)[0]
			max_q = qs[max_policy]

			return self.q_max_nt(max_q, max_policy)

		def get_q_i_plus1_s_a(self, i, s, a):
			try:
				sp = self.get_sp_from_s_a(s, a)
			except ImpossibleToMoveFromThisCell:
				sp = s
			qi_s_a = self.get_q_i_s_a(i - 1, s, a)
			r_s_a_sp = self.get_r_s_a_sp(s, a, sp)
			max_q_i = self.get_max_q_i_s(i - 1, sp).q_max
			return qi_s_a + self.alpha * (r_s_a_sp + self.gamma * max_q_i - qi_s_a)

		def update_q_i_plus_s_a(self, i, s, a):
			qi_plus_1_s_a = self.get_q_i_plus1_s_a(i, s, a)
			self.q_value_dic[i][a][s] = qi_plus_1_s_a

		def get_sp_from_s_a(self, s, a):
			p = self.trans_proba[a][s]
			# N,K = p.shape
			# for i in range(N):
			# 	for k in range(K):
			# 		print(i,k)
			# 		print(p[i,j])
			# 		if p[i,j] == 1:
			# 			sp = (i,j)
			# 			break
			# return sp

			sp = np.where(p == 1)
			try:
				return (sp[0][0], sp[1][0])
			except IndexError:
				msg = f'The action:{a} is impossible to perform from cell:{s}'
				raise ImpossibleToMoveFromThisCell(msg)

		def update_qi_plus1_s(self, i, s):
			for a in self.action_list:
				self.update_q_i_plus_s_a(i, s, a)

		def update_qi_plus1(self, i):
			for s in self.coord_list:
				self.update_qi_plus1_s(i, s)


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

		up_rewards = np.repeat(-0.04, dim**2).reshape(*shape, *shape)
		up_rewards[2, 3][1, 3] = -1.04

		down_rewards = np.repeat(-0.04, dim**2).reshape(*shape, *shape)

		left_rewards = np.repeat(-0.04, dim**2).reshape(*shape, *shape)

		right_rewards = np.repeat(-0.04, dim**2).reshape(*shape, *shape)
		right_rewards[0, 2][0, 3] = 0.96
		right_rewards[1, 2][1, 3] = -1.04


		rewards = {'UP': up_rewards,
				   'DOWN': down_rewards,
				   'LEFT': left_rewards,
				   'RIGHT': right_rewards
				   }

		q = QValue(states, trans_proba, rewards, actions_list)
		# x = q.get_q_i_plus1_s_a(1,(0,2), 'RIGHT')
		# print(f'x:\t{x}')
		# y = q.get_q_i_plus1_s_a(1, (1, 2), 'RIGHT')
		# print(f'y:\t{y}')
		# z = q.get_q_i_plus1_s_a(1, (2, 3), 'UP')
		# print(f'z:\t{z}')
		# assert x == 0.96
		# assert y == -1.04
		# assert z == -1.04
		q.update_qi_plus1(1)
		# q.q_value_dic[1]['RIGHT'][0,2]
		# q.q_value_dic[1]['RIGHT'][1, 2]
		# q.q_value_dic[1]['UP'][2, 3]
		q.q_value_dic[1]['RIGHT'][0, 1]
		q.get_max_q_i_s(1,(0,2))
		q.q_value_dic[1]['UP'][1, 2]
		q.rewards['UP'][(1,2)][0,2]
		q.get_max_q_i_s(1, (0, 2))
	# q.get_q_i_plus1_s_a(2, (0, 1), 'RIGHT')
	# q.get_q_i_plus1_s_a(2, (1,2), 'UP')
	# q.get_q_i_plus1_s_a(2, (1,2), 'DOWN')
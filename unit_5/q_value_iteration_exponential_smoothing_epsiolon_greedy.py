import numpy as np
actions_list = ['UP', 'DOWN', 'LEFT', 'RIGHT']
dim = 12
states = np.arange(dim).reshape(3,4)

up_trans_proba = np.zeros((dim, 3, 4))
up_trans_proba[1,1,0] = 1
transitions_proba = { 'UP': np.array([

									]),
						'DOWN': None,
						'LEFT': None,
						'RIGHT': None




}

import math
import numpy as np
reward_move = lambda s, sp : np.abs((sp-s)**(1/3))
reward_stay = lambda s : (s+4)**(-1/2)

def reward(s, sp):
	if s !=sp:
		return reward_move(s, sp)
	elif s == sp:
		return reward_stay(s)
	else:
		raise NotImplemented

reward(1,1)
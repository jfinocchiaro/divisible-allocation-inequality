import numpy as np
import mechs
import players
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import inequality
import utils


def main():
	#holder vars
	player_dict = {}

	#global vars
	n = 5 #number of agents
	g = 2 #number of goods
	c = 0.15 #value for inequality tolerance.  curent same for everyone

	#initialize players
	for player in range(n):
		player_dict[player] = players.Player(list(np.random.rand(g)), c, 'log', g) #random alpha_i in [0,1] parameterizing the utility and c_i constant

	

	alloc = mechs.nsw_ceei(player_dict, g)
	np.set_printoptions(suppress=True)
	print(alloc)
	print(np.sum(alloc, axis = 0))
	b = {key : 1. for key in player_dict.keys()}
	burned_alloc = mechs.partial_allocation(player_dict, g, b)
	print(list(burned_alloc))
	print(np.sum(burned_alloc, axis = 0))

	


if __name__ == "__main__":
	main()
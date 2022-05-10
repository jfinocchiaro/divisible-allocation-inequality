import numpy as np
import mechs
import players
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import inequality
import utils
from sklearn.preprocessing import normalize
from tabulate import tabulate
from random import uniform


def main():
	#holder vars
	player_dict = {}
	data = {}
	data['inequality tolerance'] = []
	data['utilities'] = []
	data['usw alloc u'] = []
	data['usw alloc v'] = []
	data['usw u'] = []
	data['usw v'] = []
	data['usw PoI'] = []


	num_reps = 10

	#global vars
	n = 2 #number of agents
	g = 3 #number of goods
	alpha = 0.3 #value for inequality tolerance.  currently same for everyone
	util_type = 'linear'

	for _ in range(num_reps):

		data['inequality tolerance'].append(alpha)

		#eps = 2. - np.sqrt(2)
		#init_u = [[0.5, 0.5, 0.], [(1. -eps) / 2, (1. -eps) / 2, eps]]
		#eps = 0.1
		#init_u = list(normalize([[1., 1., uniform(0, eps)], [1 - eps + uniform(-eps, eps), 1. - eps + uniform(-eps, eps), 2. * eps + uniform(-eps, eps)]], axis=1, norm='l1'))
		init_u = list(normalize(np.random.rand(n,g), axis=1, norm='l1'))
		#print(init_u)
		data['utilities'].append([init_u])

		#initialize players
		for player in range(n):
			player_dict[player] = players.Player(player, init_u[player], alpha, util_type, g, n, init_u) #random u_ij in [0,1] parameterizing the utility and c_i constant

		# standard allocation setting
		alloc = mechs.usw(player_dict, g)
		data['usw alloc u'].append([alloc])
		np.set_printoptions(suppress=True)
		#print(alloc)
		##print('FS inequality')
		##print(utils.FS_inequality(player_dict, alloc))

		# allocations with social preferences
		v_alloc = mechs.usw_v(player_dict, g)
		#print(v_alloc)
		data['usw alloc v'].append([v_alloc])
		##print(np.sum(v_alloc, axis = 0)) #should be a ones vector if it allocates everything
		##print('FS inequality')
		##print(utils.FS_inequality(player_dict, v_alloc))

	    # print social welfare and price of inequality
		#print('\n')
		#print('USW')
		#print('Selfish preferences')
		usw_u = utils.utilitariansocialwelfare(player_dict, alloc)
		data['usw u'].append(usw_u)
		#print(usw_u)
		#print('Social preferences')
		usw_v = utils.utilitariansocialwelfare(player_dict, v_alloc)
		data['usw v'].append(usw_v)
		#print(usw_v)
		#print('PoI: ' + str(float(usw_v / usw_u)))
		data['usw PoI'].append(float(usw_v / usw_u))


	df = pd.DataFrame(data=data)

	print(tabulate(df, headers = 'keys', tablefmt = 'psql'))

	from os.path import exists
	filename = 'simulation_results.csv'
	if exists(filename) == False:
		df.to_csv(filename,index=False)
	else:
		df.to_csv(filename, mode='a', index=False, header=False)


if __name__ == "__main__":
	main()
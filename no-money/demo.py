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
from scipy import stats


def main():
	np.set_printoptions(suppress=True)
	#holder vars
	player_dict = {}
	data = {'inequality tolerance': [], 'utilities': [],\
	'usw alloc u': [], 'usw alloc v': [], \
	'nsw alloc u': [], 'nsw alloc v': [], \
	'maxmin alloc u': [], 'maxmin alloc v': [], \
	'usw u': [], 'usw v': [], \
	'nsw u': [], 'nsw v': [], \
	'maxmin u': [], 'maxmin v': [], \
	'usw PoI' : [], 'nsw PoI' : [], 'maxmin PoI' : [],
	'usw xstar FS inequality' : [], 'nsw xstar FS inequality' : [], 'maxmin xstar FS inequality' : [],
	'usw xalpha FS inequality' : [], 'nsw xalpha FS inequality' : [], 'maxmin xalpha FS inequality' : []
	}


	num_reps = 100

	#global vars
	n = 3 #number of agents
	g = 3 #number of goods
	alpha = [0.2, 0.2, 0.2] #value for inequality tolerance.
	util_type = 'linear'

	for _ in range(num_reps):

		data['inequality tolerance'].append(alpha)

		# different variations of correlating preferences
		#eps = 2. - np.sqrt(2)
		#init_u = [[0.5, 0.5, 0.], [(1. -eps) / 2, (1. -eps) / 2, eps]]
		#eps = 0.05
		#init_u = list(normalize([[1., 1., uniform(0, eps)], [1 - eps + uniform(-eps, eps), 1. - eps + uniform(-eps, eps), 2. * eps + uniform(-eps, eps)]], axis=1, norm='l1'))
		
		# example studied in meeting
		eps = 0.05
		init_u = [[1. - eps, eps, 0.], [0.5 + eps, 0, 0.5 - eps], [0., 0.5, 0.5]]
		

		# uniformly random utilities
		#init_u = list(normalize(np.random.rand(n,g), axis=1, norm='l1'))
		data['utilities'].append([init_u])

		#initialize players
		for player in range(n):
			player_dict[player] = players.Player(player, init_u[player], alpha[player], util_type, g, n, init_u) #random u_ij in [0,1] parameterizing the utility and c_i constant

		# standard allocation setting
		alloc = mechs.usw(player_dict, g)
		data['usw alloc u'].append([alloc])
		data['usw xstar FS inequality'].append(inequality.FehrSchmidtIneq(player_dict, alloc))
		alloc_nsw = mechs.nsw_ceei(player_dict, g)
		data['nsw alloc u'].append([alloc_nsw])
		data['nsw xstar FS inequality'].append(inequality.FehrSchmidtIneq(player_dict, alloc_nsw))
		alloc_maxmin = mechs.maxmin_u(player_dict, g)
		data['maxmin alloc u'].append([alloc_maxmin])
		data['maxmin xstar FS inequality'].append(inequality.FehrSchmidtIneq(player_dict, alloc_maxmin))

		
		# allocations with social preferences
		v_alloc = mechs.usw_v(player_dict, g)
		v_alloc_nsw = mechs.nsw_ceei_v(player_dict, g)
		v_alloc_maxmin = mechs.maxmin_v(player_dict, g)
		data['usw alloc v'].append([v_alloc])
		data['usw xalpha FS inequality'].append(inequality.FehrSchmidtIneq(player_dict, v_alloc))
		data['nsw alloc v'].append([v_alloc_nsw])
		data['nsw xalpha FS inequality'].append(inequality.FehrSchmidtIneq(player_dict, v_alloc_nsw))
		data['maxmin alloc v'].append([v_alloc_maxmin])
		data['maxmin xalpha FS inequality'].append(inequality.FehrSchmidtIneq(player_dict, v_alloc_maxmin))
		

	    #compute social welfare and price of inequality
		usw_u = utils.utilitariansocialwelfare(player_dict, alloc)
		nsw_u = utils.nashsocialwelfare(player_dict, alloc_nsw)
		maxminsw_u = utils.rawlsiansocialwelfare(player_dict, alloc_maxmin)
		data['usw u'].append(usw_u)
		data['nsw u'].append(nsw_u)
		data['maxmin u'].append(maxminsw_u)

		usw_v = utils.utilitariansocialwelfare(player_dict, v_alloc)
		nsw_v = utils.nashsocialwelfare(player_dict, v_alloc_nsw)
		maxminsw_v = utils.rawlsiansocialwelfare(player_dict, v_alloc_maxmin)
		data['usw v'].append(usw_v)
		data['nsw v'].append(nsw_v)
		data['maxmin v'].append(maxminsw_v)

		data['usw PoI'].append(float(usw_v / usw_u))
		data['nsw PoI'].append(float(nsw_v / nsw_u))
		data['maxmin PoI'].append(float(maxminsw_v / maxminsw_u))


	df = pd.DataFrame(data=data)


	print(tabulate(df, headers = 'keys', tablefmt = 'psql'))

	from os.path import exists
	filename = 'simulation_results_demo.csv'
	if exists(filename) == False:
		df.to_csv(filename,index=False)
	else:
		df.to_csv(filename, mode='a', index=False, header=False)




	slope, intercept, r_value, p_value, std_err = stats.linregress(df['usw PoI'], df['nsw PoI'])
	utils.spPoI(df, 'usw PoI', 'nsw PoI', 'Price of Inequality: NSW vs USW optimization', 'figs/PoI_NSW_USW_corrprefs_3players_wr2.png', rsq=r_value, slope=slope, yint=intercept)
	#utils.spPoI(df, 'usw PoI', 'maxmin PoI', 'Price of Inequality: Max-Min vs USW optimization', 'figs/PoI_maxmin_USW_uniformprefs_4players.png')
	#utils.spPoI(df, 'nsw PoI', 'maxmin PoI', 'Price of Inequality: Max-Min vs NSW optimization', 'figs/PoI_maxmin_NSW_uniformprefs.png')

	#utils.boxplot(df, ['usw PoI', 'nsw PoI', 'maxmin PoI'],plottitle='Price of Inequalities: Uniformly random Preferences', filename='figs/boxplot_PoIs_uniformprefs.png')




if __name__ == "__main__":
	main()
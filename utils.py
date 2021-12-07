import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import inequality
import players

def socialwelfare(player_dict, bids, price, good_alloc):
	utilities = [player.u(bids[key]) for key, player in player_dict.items()]
	return np.sum(utilities), np.sum(utilities) - price * good_alloc

def plotutilities(player_dict, price_u, price_v):

	df_u = pd.DataFrame({key: {i: player.u(i) for i in np.arange(0.01, 1, 0.05)} for key, player in player_dict.items()})
	plt.figure()
	df_u.plot()
	plt.xlabel('$b_i$')
	plt.ylabel('$u_i(b_i)$')
	plt.title('Utility to be maxed as function of bid')
	plt.savefig('figs/util_u_gini.png');
	
	df_up = pd.DataFrame({key: {i: player.u(i) - price_u * i for i in np.arange(0, 1, 0.05)} for key, player in player_dict.items()})
	plt.figure()
	df_up.plot()
	plt.xlabel('$b_i$')
	plt.ylabel('$u_i(b_i) - p \\cdot b_i$')
	plt.title('Utility - price to be maxed as function of bid')
	plt.savefig('figs/util_up_gini.png');
	
	df_v = pd.DataFrame({key: {i: player.u(i) - player.c * (inequality.gen_ent_bc(len(list(player_dict.keys()))))(i) for i in np.arange(0.01, 1, 0.05)} for key, player in player_dict.items()})
	plt.figure()
	df_v.plot()
	plt.xlabel('$b_i$')
	plt.ylabel('$v_i(b_i)$')
	plt.title('Utility - ineq to be maxed as function of bid')
	plt.savefig('figs/util_v_gini.png');

	df_vp = pd.DataFrame({key: {i: player.u(i) - player.c * (inequality.gen_ent_bc(len(list(player_dict.keys()))))(i) - price_v * i for i in np.arange(0.01, 1, 0.05)} for key, player in player_dict.items()})
	plt.figure()
	df_vp.plot()
	plt.xlabel('$b_i$')
	plt.ylabel('$v_i(b_i) - p \\cdot b_i$')
	plt.title('Utility - ineq - price to be maxed as function of bid')
	plt.savefig('figs/util_vp_gini.png');
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import inequality
import auction
import players

def socialwelfare(player_dict, bids, price, good_alloc=1):
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


def ploteffectc(player_dict,ineq='gini'):
	final_prices_c = {}
	sw = {}
	iq = {}

	cs = np.arange(0., 1.5, 0.075)
	for c in cs:
		for player in player_dict.values():
			player.setc(c)
		bv, fp_v = auction.simultaneous_clock_auction_asc(player_dict, uorv='v', ineq=ineq)
		final_prices_c[c] = fp_v
		sw[c] = socialwelfare(player_dict, bv, fp_v, good_alloc=np.sum(list(bv.values())))[1]
		iq[c] = inequality.total_ineq(player_dict, bv, good_or_util='u', iq_metric='gini')

	print(final_prices_c)

	df_c_gini = pd.DataFrame({c: [final_prices_c[c], sw[c], iq[c]] for c in cs} , index=['Final prices', 'Social Welfare', 'Gini coefficient']).T
	print(df_c_gini)
	plt.figure()
	df_c_gini.plot()
	plt.xlabel('$c$')
	#plt.ylabel('Final price $p$ / Social Welfare')
	plt.title('Effect of inequality aversion $c$ on final price, social welfare, and inequality')
	plt.savefig('figs/c_impact_price_gini.png');
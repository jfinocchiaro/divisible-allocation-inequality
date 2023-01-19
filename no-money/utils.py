import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import inequality
import mechs
import players

def utilitariansocialwelfare(player_dict, alloc):
	utilities = [player.u(alloc[key]) for key, player in player_dict.items()]
	return np.sum(utilities)

def nashsocialwelfare(player_dict, alloc):
	utilities = [player.u(alloc[key]) for key, player in player_dict.items()]
	return np.prod(utilities)

def rawlsiansocialwelfare(player_dict, alloc):
	utilities = [player.u(alloc[key]) for key, player in player_dict.items()]
	return np.min(utilities)

def FS_inequality(player_dict, alloc):
	n = len(list(player_dict.values()))
	FS = lambda x : (1. / n) * ((1 / (n - 1.)) * np.sum([np.sum([np.abs(np.dot(primary.u_i, x[identifier]) - np.dot(player.u_i, x[k])) for (k, player) in player_dict.items()]) for (identifier, primary) in player_dict.items()]))
	return FS(alloc)
	
	
def max_util_loss(player_dict, alloc_u, alloc_v):
	return max([float(player.u(alloc_u[key])) / player.u(alloc_v[key]) for (key, player) in player_dict.items()])
	
	
	
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


def ploteffectc(player_dict, price_include_sw = True, ineq='gini', auct='asc'):
	final_prices_c = {}
	sw = {}
	iq = {}

	cs = np.arange(0., 1.2, 0.05)
	for c in cs:
		for player in player_dict.values():
			player.setc(c)
		if 	auct == 'asc':
			bv, fp_v = auction.simultaneous_clock_auction_asc(player_dict, uorv='v', ineq=ineq)
		else:
			bv, fp_v = auction.simultaneous_clock_auction_desc(player_dict, uorv='v', ineq=ineq)
		final_prices_c[c] = fp_v
		if price_include_sw:
			pi_sw='price_in_sw_'
			sw[c] = socialwelfare(player_dict, bv, fp_v, good_alloc=np.sum(list(bv.values())))[1]
		else:
			pi_sw='no_price_in_sw_'
			sw[c] = socialwelfare(player_dict, bv, fp_v, good_alloc=np.sum(list(bv.values())))[0]
		iq[c] = inequality.total_ineq(player_dict, bv, good_or_util='u', iq_metric='gini')

	
	df_c_gini = pd.DataFrame({c: [final_prices_c[c], sw[c], iq[c]] for c in cs} , index=['Final prices', 'Social Welfare', 'Gini coefficient']).T
	print(df_c_gini)
	plt.figure()
	df_c_gini.plot()
	plt.xlabel('$c$')
	#plt.ylabel('Final price $p$ / Social Welfare')

	if price_include_sw:
		plt.title('Effect of inequality aversion $c$ on final price, \n social welfare (incl price), and inequality')
		plt.savefig('figs/c_impact_price_gini_' + str(pi_sw) + str(auct) + '.png');

	else:
		plt.title('Effect of inequality aversion $c$ on final price, \n social welfare (excl price), and inequality')
		plt.savefig('figs/c_impact_price_gini_' + str(pi_sw) + str(auct) + '.png');		


def spPoI(df, xname, yname, plottitle, filename, rsq=None, slope=None, yint=None):
	plt.figure()
	df.plot.scatter(x=xname, y=yname)
	if slope is not None:
		mn=np.min(df[xname])
		mx=np.max(df[xname])
		x1=np.linspace(mn,mx,500)
		y1=slope*x1+yint
		plt.plot(x1,y1,'-r')
	if rsq is not None:
		plt.title(plottitle + ' $r^2$ = {rsq:.2f}'.format(rsq=rsq))
	else:
		plt.title(plottitle)

	
	
	plt.savefig(filename);

def boxplot(df, cols, plottitle='Price of Inequality by varying SW metrics', filename='figs/boxplot_PoIs.png'):
	plt.figure()
	df.boxplot(column=cols, grid=False)
	plt.title(plottitle)
	plt.savefig(filename);

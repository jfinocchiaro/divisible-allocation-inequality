import numpy as np
import auction
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
	n = 10

	#initialize players
	for player in range(n):
		player_dict[player] = players.Player(np.random.rand(), 0.15) #alpha_i parameterizing the utility and c_i

	print([player.u_i for player in player_dict.values()])

	#run auction
	bids_u, final_price_u = auction.simultaneous_clock_auction_desc(player_dict, uorv='u')
	bids_v, final_price_v = auction.simultaneous_clock_auction_desc(player_dict, uorv='v', ineq='gini')
	#print(bids_u)
	#print(bids_v)
	#print(final_price_u)
	#print(final_price_v)

	
	#calculate inequality at the end
	#ginicoeff_u = inequality.total_ineq(player_dict, bids_u)
	#ginicoeff_v = inequality.total_ineq(player_dict, bids_v)
	#print(ginicoeff_u)
	#print(ginicoeff_v)

	#plot utility of all the players in the auction with and without inequality aversion
	utils.plotutilities(player_dict, final_price_u, final_price_v)
	utils.ploteffectc(player_dict, price_include_sw=True, auct='asc')


if __name__ == "__main__":
	main()
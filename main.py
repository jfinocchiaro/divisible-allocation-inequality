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
	n = 5

	#initialize players
	for player in range(n):
		player_dict[player] = players.Player(np.random.rand(), 0.2)


	#run auction
	bids_u, final_price_u = auction.simultaneous_clock_auction_asc(player_dict, uorv='u')
	bids_v, final_price_v = auction.simultaneous_clock_auction_asc(player_dict, uorv='v', ineq='gini')
	print(bids_u)
	print(bids_v)
	print(final_price_u)
	print(final_price_v)

	
	#calculate inequality at the end
	#TODO

	#plot utility of all the players in the auction with and without inequality aversion
	utils.plotutilities(player_dict, final_price_u, final_price_v)



if __name__ == "__main__":
	main()
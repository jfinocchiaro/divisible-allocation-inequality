import numpy as np
import players

def simultaneous_clock_auction_asc(player_dict,uorv = 'u',ineq='genent',reserve_price = 0.1,step_size = 0.05,max_price=10.):
	price = reserve_price
	overdemand = True
	while overdemand == True and price <= max_price:
		bids = {}
		for key, player in player_dict.items():
			if uorv == 'u':
				bids[key] = player.bid_u(price)
			else:
				bids[key] = player.bid_v(price,len(list(player_dict.values())),ineq)
		if np.sum(list(bids.values())) > 1:
			price = price + step_size
		else:
			overdemand = False
			

	return bids, price


def simultaneous_clock_auction_desc(player_dict,uorv = 'u',ineq='genent',reserve_price = 0.1,step_size = 0.05,max_price=1.2):
	price = max_price
	overdemand = False
	while overdemand == False and price >= reserve_price:
		bids = {}
		for key, player in player_dict.items():
			if uorv == 'u':
				bids[key] = player.bid_u(price)
			else:
				bids[key] = player.bid_v(price,len(list(player_dict.values())),ineq)
		if np.sum(list(bids.values())) < 1:
			price = price - step_size
		else:
			overdemand = True
			return bids, price
	print("not enough bidding at reserve price")
	return bids, reserve_price			

	

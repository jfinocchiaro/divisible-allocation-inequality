import numpy as np
import pandas as pd
import matplotlib

import auction
import players
import inequality
import utils
from utils import bids2str

# matplotlib.use('Agg')

rng = np.random.default_rng(1)


def main():
    # Holder vars
    player_dict = {}

    # Global vars
    n = 10

    # Initialize players
    for player in range(n):
        player_dict[player] = players.Player(4*rng.random(), 0.5)  # alpha_i parameterizing the utility and c_i

    # print([player.u_i for player in player_dict.values()])

    # Run auction
    bids_u, final_price_u = auction.clock_auction(player_dict, descend=True, uorv='u')
    bids_v, final_price_v = auction.clock_auction(player_dict, descend=True, uorv='v', ineq='gini')

    print(final_price_u)
    print(bids2str(bids_u))
    print(final_price_v)
    print(bids2str(bids_v))

    # Calculate inequality at the end
    # ginicoeff_u = inequality.total_ineq(player_dict, bids_u)
    # ginicoeff_v = inequality.total_ineq(player_dict, bids_v)
    # print(ginicoeff_u)
    # print(ginicoeff_v)

    # Plot utility of all the players in the auction with and without inequality aversion
    utils.plotutilities(player_dict, final_price_u, final_price_v)
    utils.ploteffectc(player_dict, price_include_sw=True, auct='asc')


if __name__ == "__main__":
    main()

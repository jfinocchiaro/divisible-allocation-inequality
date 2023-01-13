import warnings

import numpy as np

from utils import bids2str


def clock_auction(player_dict, descend=True, uorv='u', ineq='gini',
                  reserve_price=0.1, max_price=10.,
                  alpha=1, eps=1e-3, max_iter=100):

    former_price = np.inf

    # Set initial price
    if descend:
        price = max_price
    else:
        price = reserve_price

    # Iterate
    n_iter = 0
    while np.abs(price - former_price) > eps:
        # Run a single suction
        bids = {}
        for key, player in player_dict.items():
            if uorv == 'u':
                bids[key] = player.bid_u(price)
            else:
                bids[key] = player.bid_v(price, len(player_dict), ineq)

        # Update the price
        total_bids = np.sum(list(bids.values()))

        former_price = price
        price = former_price + alpha*(total_bids - 1)

        # Check for max/min price
        if price > max_price:
            raise Exception('price exceeded max price! price=%.2f, max_price=%.2f\nbids=\n%s'
                            % (price, max_price, bids2str(bids)))
        if price < reserve_price:
            raise Exception('price is below reserve price! price=%.2f, max_price=%.2f\nbids=\n%s'
                            % (price, max_price, bids2str(bids)))
        # Check for max num. of iterations
        if n_iter > max_iter:
            warnings.warn('maximum number of iterations reached w/o convergence!')
            break
        n_iter += 1

    return bids, price

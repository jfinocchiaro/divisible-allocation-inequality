import numpy as np
import cvxpy as cp
import mechs
from players import Player

def FehrSchmidtIneq(player_dict, alloc):
    total = 0
    n = len(player_dict.values())
    for i, player_i in player_dict.items():
        player_tot = 0
        for j, player_j in player_dict.items():
            player_tot += np.abs(player_i.u(alloc[i,:]) - player_j.u(alloc[j,:]))
        player_tot /= (n-1.)
        total += player_tot
    return total

def total_ineq(player_dict, bids, good_or_util='u', iq_metric='gini'):
    if iq_metric == 'gini':
        if good_or_util == 'u':
            x = [player.u(bids[key]) for key, player in player_dict.items()]
        else:
            x = list(bids.keys())
        
        # Mean absolute difference
        mad = np.abs(np.subtract.outer(x, x)).mean()
        # Relative mean absolute difference
        rmad = mad/np.mean(x)
        # Gini coefficient
        g = 0.5 * rmad
        return g
    return 0

#generalized entropy of utilities
def gen_ent(player_dict, alloc, alpha):
    mu = np.mean([player.u(alloc[key]) for (key, player) in player_dict.items()])
    dev = [(player.u(alloc[key]) / mu) ** alpha - 1 for (key, player) in player_dict.items()]
    return 1 / (len(player_dict.keys()) *alpha * (alpha - 1)) * p.sum(dev)

#variance of utilities as opposed to variance of allocation
def variance(player_dict, alloc):
    mu = np.mean([player.u(alloc[key]) for (key, player) in player_dict.items()])
    dev = [(player.u(alloc[key]) / mu) ** 2 for (key, player) in player_dict.items()]
    return np.sum(dev) / (len(player_dict.keys()) - 1.)

# Gini coefficient of utilities
def total_gini(player_dict, alloc):
    x = [player.u(alloc[key]) for key, player in player_dict.items()]

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g
    
def loss(player_dict, alloc_u, alloc_v):
    tot = [player.u(alloc_u[i]) - player.u(alloc_v[i]) for i, player in enumerate(player_dict.values())]
    return np.sum(tot)

def gain(player_dict, alloc_u, alloc_v, alphas = None):
    if alphas is not None:
        for i, (key, player) in enumerate(player_dict.items()):
            player.setc(alphas[i])
            player_dict[key] = player
    
    
    #print('valuation x-alpha: ' + str(player.v_np(alloc_v)) + ' valuation x-star: ' + str(player.v_np(alloc_u)))
    #print('alpha: ' + str(alphas[0]))
    #print('I(x-star): ' + str(alphas[0] * FehrSchmidtIneq(player_dict, alloc_u)))
    #print('f(u(x-star)): ' + str(mechs.compute_usw(player_dict, alloc_u)))
    #print('I(x-alpha): ' + str(alphas[0] * FehrSchmidtIneq(player_dict, alloc_v)))
    #print('f(u(x-alpha)): ' + str(mechs.compute_usw(player_dict, alloc_v)))
    tot_v = [player.v_np(alloc_v) for i, player in enumerate(player_dict.values())]
    tot_u = [player.v_np(alloc_u) for i, player in enumerate(player_dict.values())]
    #print('tot v: ' + str(np.sum(tot_v)))
    #print('tot u: ' + str(np.sum(tot_u)))
    return np.sum(tot_v) - np.sum(tot_u)


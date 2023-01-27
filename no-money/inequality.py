import numpy as np
import cvxpy as cp
import mechs
from players import Player

# given x and preferences, returns I(x)
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

#given preferences, x* and x^\alpha, returns loss := f(u(x*)) - f(u(x^\alpha))
def loss(player_dict, alloc_u, alloc_v):
    tot = [player.u(alloc_u[i]) - player.u(alloc_v[i]) for i, player in enumerate(player_dict.values())]
    return np.sum(tot)

#given preferences, x* and x^\alpha, returns gain := f(v(x^\alpha)) - f(v(x*))
def gain(player_dict, alloc_u, alloc_v, alphas = None):
    if alphas is not None:
        for i, (key, player) in enumerate(player_dict.items()):
            player.setc(alphas[i])
            player_dict[key] = player
    
    
    tot_v = [player.v_np(alloc_v) for i, player in enumerate(player_dict.values())]
    tot_u = [player.v_np(alloc_u) for i, player in enumerate(player_dict.values())]
    return np.sum(tot_v) - np.sum(tot_u)


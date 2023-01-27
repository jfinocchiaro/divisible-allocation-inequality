import numpy as np
import mechs
import players
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import inequality
import utilities
from sklearn.preprocessing import normalize
from tabulate import tabulate
from random import uniform
from scipy import stats
from itertools import chain
from math import dist




def effect_alpha(init_u, n,g, alphas, util_type = 'linear', save=True):

    data = {'alphas': [], 'inequality tolerance': [], 'utilities': [],\
                'usw alloc u': [], 'usw alloc v': [], \
                'usw u': [], 'usw v': [], \
                'usw PoI' : [], \
                'usw max util loss' : [], \
                'usw xstar FS inequality' : [],\
                'usw xalpha FS inequality' : [],\
                'distance' : [], 'loss': [], 'gain' : [], 'gain to loss' : []\
           }



    player_dict = {}
    #initialize players
    for player in range(n):
        alp = max(0.,np.random.normal(0.0, 0.05))
        player_dict[player] = players.Player(player, init_u[player], alp, util_type, g, n, init_u) #random u_ij in [0,1] parameterizing the utility	

    for alpha in alphas:
        data['utilities'].append([init_u])
        data['alphas'].append(alpha)

        data['inequality tolerance'].append(alphas)

        for player in range(n):
            alp = max(0.,np.random.normal(alpha, 0.01))
            player_dict[player].setc(alp)

        # standard allocation setting; compute x*
        alloc = mechs.usw(player_dict, g)
        data['usw alloc u'].append([alloc])
        data['usw xstar FS inequality'].append(inequality.FehrSchmidtIneq(player_dict, alloc))



        # allocations with social preferences; compute x^\alpha
        v_alloc = mechs.usw_v(player_dict, g)
        data['usw alloc v'].append([v_alloc])
        data['usw xalpha FS inequality'].append(inequality.FehrSchmidtIneq(player_dict, v_alloc))

        #compute individual tradeoff
        data['usw max util loss'] = utilities.max_util_loss(player_dict, alloc, v_alloc)


        #compute social welfare f(u(x^*))
        usw_u = utilities.utilitariansocialwelfare(player_dict, alloc)
        data['usw u'].append(usw_u)

        # compute f(u(x^\alpha))
        usw_v = utilities.utilitariansocialwelfare(player_dict, v_alloc)
        data['usw v'].append(usw_v)

        # compute old loss ratio (deprecated but afraid to erase)
        data['usw PoI'].append(float(usw_v / usw_u))
        
        # compute loss f(u(x*)) - f(u(x^alpha))
        loss = inequality.loss(player_dict, alloc, v_alloc)
        data['loss'].append(loss)
        
        #compute gain f(v(x^\alpha)) - f(v(x*))
        gain = inequality.gain(player_dict, alloc, v_alloc,alphas=None)
        data['gain'].append(gain)

        # compute distance d(x*, x^\alpha) in allocation space
        distance = dist(list(chain(*alloc)), list(chain(*v_alloc)))
        data['distance'].append(distance)
        
        #compute gain-to-loss ratio
        data['gain to loss'].append(max(gain / (1.0 * loss), 0.))




    df = pd.DataFrame(data=data)

    return df


def effect_prop_averse(init_u, n,g, alpha, util_type = 'linear', save=True):
    
    data = {'prop cares': [], 'inequality tolerance': [], 'utilities': [],\
                'usw alloc u': [], 'usw alloc v': [], \
                'usw u': [], 'usw v': [], \
                'usw PoI' : [], \
                'usw max util loss' : [], \
                'usw xstar FS inequality' : [], 'usw xalpha FS inequality' : [] , 'loss' : [], 'gain': [], 'gain to loss' : [], 'distance' : [] }


    player_dict = {}
    #initialize players
    for player in range(n):
        player_dict[player] = players.Player(player, init_u[player], 0., util_type, g, n, init_u) #random u_ij in [0,1] parameterizing the utility	


    for socialists in range(0,n+1):
        #the proporiton of agents who are inequality averse
        data['prop cares'].append(float(socialists) / n)
        alphas = [player.c for player in player_dict.values()]
        data['inequality tolerance'].append(alphas)
        data['utilities'].append([init_u])

        for player in range(n):
            player_dict[player].setc(0.)

        for player in range(socialists):
            player_dict[player].setc(alpha)

        # standard allocation setting; compute x*
        alloc = mechs.usw(player_dict, g)
        data['usw alloc u'].append([alloc])
        data['usw xstar FS inequality'].append(inequality.FehrSchmidtIneq(player_dict, alloc))


        # allocations with social preferences; compute x^\alpha
        v_alloc = mechs.usw_v(player_dict, g)
        data['usw alloc v'].append([v_alloc])
        data['usw xalpha FS inequality'].append(inequality.FehrSchmidtIneq(player_dict, v_alloc))

        #compute individual tradeoff 
        data['usw max util loss'] = utilities.max_util_loss(player_dict, alloc, v_alloc)


        #compute f(u(x*))
        usw_u = utilities.utilitariansocialwelfare(player_dict, alloc)
        data['usw u'].append(usw_u)
        
        #compute f(u(x^alpha))
        usw_v = utilities.utilitariansocialwelfare(player_dict, v_alloc)
        data['usw v'].append(usw_v)
        
        # compute a loss ratio (deprecated but afraid to remove)
        data['usw PoI'].append(float(usw_v / usw_u))
        
        #compute the loss and gain
        loss = inequality.loss(player_dict, alloc, v_alloc)
        data['loss'].append(loss)
        
        gain = inequality.gain(player_dict, alloc, v_alloc,alphas=None)
        data['gain'].append(gain)

        # compute gain-to-loss.  
        # only take the positive part as sometimes comes out negative because of floating point precision
        data['gain to loss'].append(max(gain / (1.0 * loss), 0))
        
        #compute distance d(x*, x^\alpha) in allocation space
        distance = dist(list(chain(*alloc)), list(chain(*v_alloc)))
        data['distance'].append(distance)
        




    df = pd.DataFrame(data=data)


    return df

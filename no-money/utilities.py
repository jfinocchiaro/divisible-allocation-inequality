import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import inequality
import mechs
import players

# given x, returns f(u(x))
def utilitariansocialwelfare(player_dict, alloc):
    utilities = [player.u(alloc[key]) for key, player in player_dict.items()]
    return np.sum(utilities)

# given x, returns I(x)
def FS_inequality(player_dict, alloc):
    n = len(list(player_dict.values()))
    FS = lambda x : (1. / n) * ((1 / (n - 1.)) * np.sum([np.sum([np.abs(np.dot(primary.u_i, x[identifier]) - np.dot(player.u_i, x[k])) for (k, player) in player_dict.items()]) for (identifier, primary) in player_dict.items()]))
    return FS(alloc)

#given preferences and alpha, returns IT(alpha)
def max_util_loss(player_dict, alloc_u, alloc_v):
    return max([float(player.u(alloc_u[i])) / player.u(alloc_v[i]) for i, player in enumerate(player_dict.values())])


#plots from experiments varying alpha
def plot_alpha_exp(alpha_exp, yplot, n_trials, ylabel,alphamax, save=True):
    ## Printing plots! :upside down smiling face:
    yplotvals = [np.mean([alpha_exp[i][yplot][j] for i in range(len(alpha_exp))]) for j in range(len(alpha_exp[i][yplot]))]
    cis = [1.96 * np.mean([alpha_exp[i][yplot][j] / np.sqrt(n_trials) for i in range(len(alpha_exp))]) for j in range(len(alpha_exp[i][yplot]))]
    plt.figure()
    plt.title('Effect of $\\alpha$ on ' + ylabel)
    plt.xlabel('$\\mu : \\alpha_i \\sim \\mathcal{N}(\\mu, 0.05)$')
    plt.ylabel(ylabel)
    plt.plot(alpha_exp[0]['alphas'], yplotvals, marker='o', ls='-', color='blue')
    plt.fill_between(alpha_exp[0]['alphas'], ([max(yplotvals[i] - cis[i] , 0) for i in range(len(yplotvals))]), ([yplotvals[i] + cis[i] for i in range(len(yplotvals))]),color='b', alpha = 0.1)
    
    if save:
        plt.savefig('figs/synthetic/effect-alpha-' + yplot +'_n' + str(n) + '_g' + str(g) + '_alphamax' + str(alphamax) +  '.png')
    plt.show()

## Printing plots! :upside down smiling face:
#plots from experiments varying proportion of inequality-averse agents
def plot_propaverse(prop_averse_exp, yplot, n_trials, ylabel, alphamax, loc='upper left', save=True):
    colors = ['blue', 'red', 'black', 'orange', 'purple']
    labels = ['$\\mu = $' + str(np.around(mu, 2)) for mu in prop_averse_exp.keys()]
    
    plt.figure()
    
    for wee, av in enumerate(prop_averse_exp.keys()): 
        ylst = np.array([list(prop_averse_exp[av][exp][yplot]) for exp in range(len(prop_averse_exp[av]))])
        yplotvals = np.mean(ylst, axis=0)
        cis = 1.96 * np.std(ylst, axis = 0) / np.sqrt(n_trials)
        plt.plot(list(prop_averse_exp[av][0]['prop cares']), yplotvals, marker='o', ls='-', color=colors[wee], label = labels[wee])
        plt.fill_between(prop_averse_exp[av][0]['prop cares'], (np.subtract(yplotvals, cis)), (np.add(yplotvals, cis)),color=colors[wee], alpha = 0.1)
    plt.title('Effect of $p$ on ' + ylabel)
    plt.xlabel('Proportion $p$ of inequality-averse agents')
    plt.ylabel(ylabel)
    plt.legend(loc=loc)
    if save:
        plt.savefig('figs/synthetic/effect-p-' + yplot +'_n' + str(n) + '_g' + str(g) + '_alphamax' + str(alphamax)+ '.png')
    plt.show()

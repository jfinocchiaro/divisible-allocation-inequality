import numpy as np
import players
import cvxpy as cp


# given x, returns f(u(x))
def compute_usw(player_dict, alloc):
    return np.sum([player_dict[key].u_cp(alloc[i,:]) for i, key in enumerate(list(player_dict.keys()))])

#maximizing Nash social welfare.  equivalent to CEEI and proportional fairness
# CP comes from page 162 of following book: https://www.cs.cmu.edu/~sandholm/cs15-892F13/algorithmic-game-theory.pdf#page=156
#maximizing Utilitarian social welfare. returns the x* allocation
def usw(player_dict, num_goods): 
	num_agents = len(list(player_dict.keys()))
	alloc = cp.Variable((num_agents, num_goods))
	maxprob = cp.Maximize(cp.sum([player_dict[key].u_cp(alloc[i,:]) for i, key in enumerate(list(player_dict.keys()))]))
	constraints = [alloc >= 0] + [cp.sum(alloc[:, j]) <= 1 for j in range(num_goods)]
	prob = cp.Problem(maxprob, constraints)
	prob.solve()
	return alloc.value

# maximizing f(v(.)); reurns x^\alpha
def usw_v(player_dict, num_goods): 
	num_agents = len(list(player_dict.keys()))
	alloc = cp.Variable((num_agents, num_goods))
	#c = np.min([player_dict[key].c for key in player_dict.keys()])
	maxprob = cp.Maximize(cp.sum([player_dict[key].v_cp(alloc) for i, key in enumerate(list(player_dict.keys()))]))
	constraints = [alloc >= 0] + [cp.sum(alloc[:, j]) <= 1 for j in range(num_goods)] 
	prob = cp.Problem(maxprob, constraints)
	prob.solve()
	return alloc.value

# maximizes f(u(.)) subject to tighter allocation restrictions (e.g., can't just allocate all of something to one agent)
def usw_u_classes(player_dict, num_goods, class_caps): 
    num_agents = len(list(player_dict.keys()))
    alloc = cp.Variable((num_agents, num_goods))
    maxprob = cp.Maximize(cp.sum([player_dict[key].u_cp(alloc[i,:]) for i, key in enumerate(list(player_dict.keys()))]))
    constraints = [alloc >= 0] + [cp.sum(alloc[:, j]) <= 1 for j in range(num_goods)] + [alloc <= class_caps]
    prob = cp.Problem(maxprob, constraints)
    prob.solve()
    return alloc.value

# maximizes f(v(.)) subject to tighter allocation restrictions (e.g., can't just allocate all of something to one agent)
def usw_v_classes(player_dict, num_goods, class_caps): 
    # player_dict is a dictionary of Player objects
    #num_goods is the number of goods to be allocated
    # class_caps is a n x m matrix of good capacities. each column should be a constant times the ones vector (e.g., columnwise constant) 
	num_agents = len(list(player_dict.keys()))
	alloc = cp.Variable((num_agents, num_goods))
	c = np.min([player_dict[key].c for key in player_dict.keys()])
	maxprob = cp.Maximize(cp.sum([player_dict[key].v_cp(alloc) for i, key in enumerate(list(player_dict.keys()))]))
	constraints = [alloc >= 0] + [cp.sum(alloc[:, j]) <= 1 for j in range(num_goods)] + [alloc <= class_caps]
	prob = cp.Problem(maxprob, constraints)
	prob.solve()
	return alloc.value

#not sure what this is, but at this point I'm afraid to delete it.
def usw_v_cm(player_dict, num_goods): 
	num_agents = len(list(player_dict.keys()))
	alloc = cp.Variable((num_agents, num_goods))
	c = np.min([player_dict[key].c for key in player_dict.keys()])
	maxprob = cp.Maximize(cp.sum([player_dict[key].v_cp(alloc) for i, key in enumerate(list(player_dict.keys()))]))
	constraints = [alloc >= 0] + [cp.sum(alloc[:, j]) <= 1 for j in range(num_goods)] 
	prob = cp.Problem(maxprob, constraints)
	prob.solve()
	return alloc.value

import numpy as np
import players
import cvxpy as cp
import inequality


#maximizing Nash social welfare.  equivalent to CEEI and proportional fairness
# CP comes from page 162 of following book: https://www.cs.cmu.edu/~sandholm/cs15-892F13/algorithmic-game-theory.pdf#page=156
def nsw_ceei(player_dict, num_goods): 
	num_agents = len(list(player_dict.keys()))
	alloc = cp.Variable((num_agents, num_goods))
	maxprob = cp.Maximize(cp.sum([cp.log(player_dict[key].u_cp(alloc[i,:])) for i, key in enumerate(list(player_dict.keys()))]))
	constraints = [alloc >= 0] + [cp.sum(alloc[:, j]) <= 1 for j in range(num_goods)]
	prob = cp.Problem(maxprob, constraints)
	prob.solve()
	#print('Dual variables')
	#print([constraint.dual_value for constraint in constraints])
	return alloc.value


def nsw_ceei_v(player_dict, num_goods): 
	num_agents = len(list(player_dict.keys()))
	alloc = cp.Variable((num_agents, num_goods))
	c = np.min([player_dict[key].c for key in player_dict.keys()])
	maxprob = cp.Maximize(cp.sum([cp.log(player_dict[key].v_cp(alloc)) for i, key in enumerate(list(player_dict.keys()))]))
	constraints = [alloc >= 0] + [cp.sum(alloc[:, j]) <= 1 for j in range(num_goods)] #+ [inequality.total_gini(player_dict, alloc) <= c]
	prob = cp.Problem(maxprob, constraints)
	prob.solve()
	#print('Dual variables')
	#print([constraint.dual_value for constraint in constraints])
	return alloc.value



def partial_allocation(player_dict, num_goods, b):
	pf_alloc = nsw_ceei(player_dict, num_goods)
	pf_minus = {}
	f = {}
	for key in player_dict.keys():
		holder = player_dict.copy()
		del holder[key]
		holder_key_list = list(holder.keys())

		pf_min = nsw_ceei(holder, num_goods)
		pf_minus[key] = pf_min
		f[key] = (np.prod([player_dict[j].u(pf_alloc) ** b[j] for j in holder_key_list]) / np.prod([player_dict[j].u(pf_min) ** b[j] for j in holder_key_list])) ** b[key]

	burned_alloc = [f[key] * pf_alloc[key,:] for key in player_dict.keys()]

	return burned_alloc


def v_partial_allocation(player_dict, num_goods, b):
	pf_alloc = nsw_ceei_v(player_dict, num_goods)
	pf_minus = {}
	f = {}
	for key in player_dict.keys():
		holder = player_dict.copy()
		del holder[key]
		holder_key_list = list(holder.keys())

		pf_min = nsw_ceei_v(holder, num_goods)
		pf_minus[key] = pf_min
		f[key] = (np.prod([player_dict[j].u(pf_alloc) ** b[j] for j in holder_key_list])\
			 / np.prod(([holder[j].u(pf_min) ** b[j] for j in holder_key_list])) ** b[key])

		#f[key] = (np.prod([(player_dict[j].u(pf_alloc) - player_dict[j].c * inequality.total_gini\
		#		(player_dict, pf_alloc)) ** b[j] for j in holder_key_list])\
		#	 / np.prod(([(holder[j].u(pf_min) - player_dict[j].c * \
		#	  inequality.total_gini(holder, pf_min)) ** b[j] for j in holder_key_list])) ** b[key])

	burned_alloc = [f[key] * pf_alloc[key,:] for key in player_dict.keys()]

	return burned_alloc



#maximizing Utilitarian social welfare.  equivalent to CEEI and proportional fairness
def usw(player_dict, num_goods): 
	num_agents = len(list(player_dict.keys()))
	alloc = cp.Variable((num_agents, num_goods))
	maxprob = cp.Maximize(cp.sum([player_dict[key].u_cp(alloc[i,:]) for i, key in enumerate(list(player_dict.keys()))]))
	constraints = [alloc >= 0] + [cp.sum(alloc[:, j]) <= 1 for j in range(num_goods)]
	prob = cp.Problem(maxprob, constraints)
	prob.solve()
	#print('Dual variables')
	#print([constraint.dual_value for constraint in constraints])
	return alloc.value


def usw_v(player_dict, num_goods): 
	num_agents = len(list(player_dict.keys()))
	alloc = cp.Variable((num_agents, num_goods))
	c = np.min([player_dict[key].c for key in player_dict.keys()])
	maxprob = cp.Maximize(cp.sum([player_dict[key].v_cp(alloc) for i, key in enumerate(list(player_dict.keys()))]))
	constraints = [alloc >= 0] + [cp.sum(alloc[:, j]) <= 1 for j in range(num_goods)] #+ [inequality.total_gini(player_dict, alloc) <= c]
	prob = cp.Problem(maxprob, constraints)
	prob.solve()
	#print('Dual variables')
	#print([constraint.dual_value for constraint in constraints])
	return alloc.value


#maximizing Rawlsian social welfare.  maxmiizing the minimum utility
def maxmin_u(player_dict, num_goods): 
	num_agents = len(list(player_dict.keys()))
	alloc = cp.Variable((num_agents, num_goods))
	maxprob = cp.Maximize(cp.min(cp.bmat([[player_dict[key].u_cp(alloc[i,:]) for i, key in enumerate(list(player_dict.keys()))]])))
	constraints = [alloc >= 0] + [cp.sum(alloc[:, j]) <= 1 for j in range(num_goods)]
	prob = cp.Problem(maxprob, constraints)
	prob.solve()
	return alloc.value


def maxmin_v(player_dict, num_goods): 
	num_agents = len(list(player_dict.keys()))
	alloc = cp.Variable((num_agents, num_goods))
	c = np.min([player_dict[key].c for key in player_dict.keys()])
	maxprob = cp.Maximize(cp.min(cp.bmat([[player_dict[key].v_cp(alloc) for i, key in enumerate(list(player_dict.keys()))]])))
	constraints = [alloc >= 0] + [cp.sum(alloc[:, j]) <= 1 for j in range(num_goods)] #+ [inequality.total_gini(player_dict, alloc) <= c]
	prob = cp.Problem(maxprob, constraints)
	prob.solve()
	#print('Dual variables')
	#print([constraint.dual_value for constraint in constraints])
	return alloc.value

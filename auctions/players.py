import numpy as np
import inequality
import cvxpy as cp


class Player:
    def __init__(self, u_i=1., c_i=0.):
        self.u_cp = lambda x: u_i * cp.log((x + 1))
        self.u_i = u_i
        self.u = lambda x: u_i * np.log(x + 1)
        self.c = c_i

    def setc(self, c_i):
        self.c = c_i

    def bid_u(self, price):
        bid = cp.Variable(1)
        prob = cp.Problem(cp.Maximize(self.u_cp(bid) - price * bid), [0 <= bid, bid <= 1])
        prob.solve()
        return bid.value

    def bid_v(self, price, n, ineq='genent'):
        if ineq == 'genent':
            iq = inequality.gen_ent_bc(n)
        elif ineq == 'variance':
            iq = inequality.variance_bc(n)
        elif ineq == 'gini':
            iq = inequality.gini_bc(n)
        else:
            iq = lambda x: 0
        bid = cp.Variable(1)
        prob = cp.Problem(cp.Maximize(self.u_cp(bid) - self.c * iq(bid) - price * bid), [0 <= bid, bid <= 1])
        prob.solve()
        return bid.value

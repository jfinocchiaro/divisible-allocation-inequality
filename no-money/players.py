import numpy as np
import cvxpy as cp

#player object has inequality aversion c (alpha_i in paper), 
#preferences u_i is agent i's preferences (a_i in paper), 
#g = number of goods being allocated (m in paper), 
#n = number of agents, 
#U = total preference matrix (a in paper)

# NOTE: not fully functional for other utility types besides linear, but leaving in, in case anyone wants to extend
class Player:
    def __init__(self,identifier, u_i=[1.], c_i=0., util_type='linear', g=1, n= 2., U = [[1.] * 2]):
        self.c = c_i
        self.id = identifier
        if util_type == 'log':
            self.u_cp = lambda x : np.sum([u_i[i] * cp.log((x[i]+1)) for i in range(len(u_i))])
            self.u_i = u_i
            self.u = lambda x : np.sum([u_i[i] * np.log(x[i]+1) for i in range(len(u_i))])
        elif util_type == 'leontief':
            self.u_i = u_i
            self.u_cp = lambda x : cp.min(cp.vstack([x[i] / u_i[i] for i in range(len(u_i))]))
            self.u = lambda x : np.min([x[i] / u_i[i] for i in range(len(u_i))])
        elif util_type == 'linear':
            self.u_cp = lambda x : u_i @ x
            self.u_i = u_i
            self.u = lambda x : np.dot(x, u_i)
            self.v_cp = lambda x : np.array(u_i) @ x[self.id] - ((self.c / ((n - 1.))) * cp.sum([cp.abs(np.array(u_i) @ x[identifier] - np.array(U[k]) @ x[k]) for k in range(n)]))
            self.v_np = lambda x : np.dot(u_i, x[self.id]) - ((self.c / ((n - 1.))) * np.sum([np.abs(np.dot(u_i, x[self.id]) - np.dot(U[k], x[k])) for k in range(n)]))
        else:
            self.u_cp = lambda x : np.sum([u * cp.log((x+1)) for u in u_i])
            self.u_i = u_i
            self.u = lambda x : np.sum([u * np.log(x+1) for u in u_i])


    def setc(self, c_i):
        self.c = c_i


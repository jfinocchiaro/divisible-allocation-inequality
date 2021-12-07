import numpy as np
import cvxpy as cp

def gen_ent_bc(n, alpha = 2):
    return lambda x : (1. / (n * alpha * (alpha - 1.))) * ((n*x)**alpha + (n-1) * ((n * (1.-x)) / (n-1.))**alpha)
    
def variance_bc(n):
    return lambda x : (1./n)* (x - (1./n)** 2. + (n-1.) * (((1.-x) / (n-1.)) - (1./n)) ** 2)

def gini_bc(n):
    return lambda x : (n-1) * cp.abs( x - (1.-x) / (n - 1.)) / n
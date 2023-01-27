import numpy as np
import mechs
import players
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import inequality
import utils
from sklearn.preprocessing import normalize
from tabulate import tabulate
from random import uniform
from scipy import stats
import experiments

#really just used as testing code if you wanted to isolate one small example without running an entire jupyter notebook
def main():
    np.set_printoptions(suppress=True)

    #holder vars
    n = 10
    g = 5
    alpha = .3
    init_u = list(np.random.beta(2, 1, (n,g)))
    df = experiments.effect_prop_averse(init_u n,g, alpha, save=False, num_reps = 1)
    print(df['max tradeoff'])


if __name__ == "__main__":
	main()

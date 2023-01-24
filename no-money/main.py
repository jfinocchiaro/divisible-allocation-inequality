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


def main():
	np.set_printoptions(suppress=True)
	#holder vars
	n = 10
	g = 5
	#alphas = np.linspace(0,0.5, num=200)
	alpha = .3
	df = experiments.effect_prop_averse(n,g, alpha, save=False, num_reps = 1)
	print(df['max tradeoff'])


if __name__ == "__main__":
	main()

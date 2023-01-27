# Simulations for understanding allocations of divisible goods, possibly with preferences for mitigating inequality.
## File structure
* synthhetic-notebook.ipynb generates players with homogeneous utilities and run the allocation mechanism given by the Eisenberg Gale convex program. main.py might aslo be easier to use for small scale examples
* mechs.py contains the optimization problems returning x* and x^\alpha given a set of agents and their preferences.
* inequality.py has functions returning different inequality metrics given an allocation X (namely, computes Fehr Schmidt inequality, loss, and gain)
* players.py contains the Player class; each player is initialized with a utility type, parameters for said utility, inequality aversion constant c_i. 
* experiments.py has the code to run the two main sets of experiments: testing the effect of alpha and the proportion of inequality-averse agents on various metrics


## Dependencies
* numpy
* cvxpy
* pandas
* matplotlib
* sklearn
* random
* tabulate
* scipy
* pickle

## To run
install dependencies, clone the repo, etc. 
synthetic-notebook.ipynb is the best starting point

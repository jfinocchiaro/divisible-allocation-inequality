# Simulations for understanding allocations of divisible goods, possibly with preferences for mitigating inequality.
## File structure
* main.py generates players with homogeneous utilities and run the allocation mechanism given by the Eisenberg Gale convex program
* mechs.py contains the (1) EG convex optimizing Nash Social Welfare, (2) Partial Allocation mechanism given by Cole et al 2012 (3) a convex program maximizing utilitarian social welfare (gives all of an item to the person who values it the most), and (4) a maximin allocation convex program.
* inequality.py has functions returning different inequality metrics given an allocation X
* players.py contains the Player class; each player is initialized with a utility type, parameters for said utility, inequality aversion constant c_i. 


## Dependencies
* numpy
* cvxpy
* pandas
* matplotlib
* sklearn
* random
* tabulate
* scipy

## To run
install dependencies, clone the repo, etc. 
go to the no-money directory in your console and enter "python3 main.py"

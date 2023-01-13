# Simulations for understanding allocations of divisible goods, possibly with preferences for mitigating inequality.
## File structure
* main.py generates players with homogeneous utilities and run the allocation mechanism given by the Eisenberg Gale convex program
* mechs.py contains the EG convex program available to test and the Partial Allocation mechanism given by Cole et al 2012
* inequality.py has functions returning different inequality metrics in a best-case scenario for one player feedback as well as a few metrics that are a function of all utilities and allocation X
* players.py contains the Player class; each player is initialized with a utility type, parameters for said utility, inequality aversion constant c_i. 

## Future items to do
* Add inequality metrics that consider population level inequality, not just best case based on an individual player.
* is there a way to incorporate inequality preferences that preserves EG as a convex program?
* return social welfare, and compare when inequality averse vs neutral

## Dependencies
* numpy
* cvxpy
* pandas

## To run
go to the no/money directory in your console and enter "python3 main.py"

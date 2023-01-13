# Simulations for understanding the auctions of divisible goods, possibly with preferences for mitigating inequality.
## File structure
* main.py runs the allocation mechanism
* auction.py contains the aucitons available to test: currently ascending and descending simultaneous clock auctions
* inequality.py has functions returning different inequality metrics in a best-case scenario for one player feedback
* players.py contains the Player class; each player is initialized with a utility, inequality aversion, and potentially a budget. players bid according to a parameterized utility function, choosing their bid b maximizing u_i(b) - pb, where p is the charged price. 
* util.py has experiments and generates plots in various settings.

## Future items to do
* Add inequality metrics that consider population level inequality, not just best case based on an individual player.  How to implement this one is the least clear action step to me...
* Add more auctions (currently supports ascending and descending clock auctions)
* Would be interesting to see what happens when only a few agents care about inequality, and how their allocation, social welfare, and prices change as the number of agents caring about inequality increases

## Dependencies
* numpy
* cvxpy
* pandas

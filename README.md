# sim-anneal-net
A simulated annealing Monte Carlo optimization algorithm for cost functions defined on simple graphs of order N.
Specify the initial coordinates of the nodes (or randomize them) inside a square of arbitrary side length.
The number of simple graphs of order N>20 is an incomprehensibly large number.
The problem of finding the unique graph that optimizes a cost function is therefore computationally intractable if N is large.
Instead, we aspire merely to a near-optimal graph using a probabilistic search algorithm.
The algorithm, denoted perturb() in the code, tweaks a graph until it finds one of higher fitness, and then caches it.
It jumps randomly through the network state space in an attempt to avoid getting stuck in a local fitness maximum.

It does, however, have a nonzero chance (denoted K in the code) to return to the current optimum.
The average radius of the algorithm's orbit in state space is controlled by this parameter K:
- if K is too high, the algorithm will be constrained to a local fitness maximum.
- if K is too low, we could potentially be spending too much time searching an extraneous region of state space.

So there is a goldilocks range of K values for which the algorithm finds a good solution quickly.

The fitness function is defined node-wise: every node has a fitness, and we judge graphs by the sum of all node fitnesses.
Contributions to fitness include:

- Degree (total number of connections)
- Domination of neighbors (having more connections than the average of those connected to it)
- Minimizing the sum of edge costs, where cost is proportional to edge length

Parameters p1, p2, and p3 respectively control the relative weights of these considerations in the fitness function.

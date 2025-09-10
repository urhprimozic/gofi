from gofi.graphs.graph import random_adjacency_matrix, random_permutation_matrix
from gofi.graphs.fast_inversion_table.models import DistLinear
from gofi.graphs.fast_inversion_table.loss import loss, monte_carlo_loss

n = 5
# get first graph
M1 = random_adjacency_matrix(n)
Q = random_permutation_matrix(n)
# get second graph (isomorphic to the first)
M2 = Q @ M1 @ Q.T

# model
model = DistLinear(n, 4, n**2)


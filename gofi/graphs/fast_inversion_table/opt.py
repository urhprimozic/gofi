from gofi.graphs.graph import random_adjacency_matrix, random_permutation_matrix
from gofi.graphs.fast_inversion_table.models import DistLinear
from gofi.graphs.fast_inversion_table.loss import fast_loss
import torch

n = 5
# get first graph
M1 = random_adjacency_matrix(n)
Q = random_permutation_matrix(n)
# get second graph (isomorphic to the first)
M2 = Q @ M1 @ Q.T

# model
model = DistLinear(n, 4, n**2)

Pa = model()

loss = fast_loss(Pa, M1, M2)

# identity permutation should have zero loss
P_id  = torch.zeros((n,n))
P_id[:,0]=1
loss_id = fast_loss(P_id, M1, M1)

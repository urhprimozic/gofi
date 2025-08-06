from gofi.graphs.inversion_table.models import PermDistConnected, PermDistDissconnected
from gofi.graphs.inversion_table.probs import PermModel
from gofi.graphs.inversion_table.opt import training
from gofi.graphs.graph import random_adjacency_matrix

n = 5
# graph 
M = random_adjacency_matrix(n)
# model
model = PermDistConnected(n, 3,10,T=100)
dist = PermModel(model, n)

# train
training(dist, M, M, max_steps=1000, adam_parameters={"lr": 0.01}, verbose=10)

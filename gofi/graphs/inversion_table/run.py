from gofi.graphs.inversion_table.models import PermDistConnected, PermDistDissconnected
from gofi.graphs.inversion_table.probs import PermModel
from gofi.graphs.inversion_table.opt import training
from gofi.graphs.graph import random_adjacency_matrix
from gofi.graphs.inversion_table.loss import norm_loss_normalized, id_loss




n = 5
# graph 
M = random_adjacency_matrix(n)
# model
model = PermDistConnected(n, 3,10,T=100)
dist = PermModel(model, n)

def loss_function(dist, M1, M2):
    return norm_loss_normalized(dist, M1, M2) + id_loss(dist)

# train
training(dist, M, M,loss_function=loss_function, max_steps=1000, adam_parameters={"lr": 0.08}, verbose=10)

import argparse
from gofi.graphs.inversion_table.models import PermDistConnected, PermDistDissconnected
from gofi.graphs.inversion_table.probs import PermModel
from gofi.graphs.inversion_table.opt import training
from gofi.graphs.graph import random_adjacency_matrix, random_permutation_matrix
from gofi.graphs.inversion_table.loss import norm_loss_normalized, id_loss
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=str, help="Number of vertices in a graph")
    parser.add_argument("--layer_size", type=str, help="Size of each hidden layer", default="600")
    args = parser.parse_args()

    # collect args+
    n = int(args.n)
    layer_size = int(args.layer_size)

    # graph 
    M1 = random_adjacency_matrix(n)
    Q = random_permutation_matrix(n)
    M2 = Q.T @ M1 @ Q

    # model
    model = PermDistConnected(n, 4,layer_size, T=100)
    dist = PermModel(model, n)

    def loss_function(dist, M1, M2):
        return norm_loss_normalized(dist, M1, M2) + id_loss(dist)

    # train
    training(dist, M1, M2,loss_function=loss_function, max_steps=1000, adam_parameters={"lr": 0.08}, verbose=10)

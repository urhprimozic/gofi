import torch
import numpy as np
from gofi.models import RandomMap
from gofi.graphs.loss import BijectiveLoss, RelationLoss
from gofi.graphs.graph import random_adjacency_matrix, adjacency_matrix_cayley_Sn, random_permutation_matrix, permutation_to_permutation_matrix, permutation_matrix_to_permutation
from math import factorial
from gofi.graphs.opt import training
import matplotlib.pyplot as plt
from gofi.plot.colors import blueorange, blue, cmap_dakblue_blue
import networkx as nx
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="Number of vertices in a graph")
    parser.add_argument("--eps", type=float,default=0.001, help="Grad norm threshold")
    parser.add_argument("--max_steps", type=int,default=5000, help="Maximum numer of steps")
    parser.add_argument("--lr", type=float,default=0.1, help="Initial learning rate")
    args = parser.parse_args()     
    n = args.n 
    eps = args.eps
    max_steps = args.max_steps
    lr = args.lr

    M1 = random_adjacency_matrix(n)
    Q = random_permutation_matrix(n)
    M2 = Q.T @ M1 @ Q

    f = RandomMap(n).to(device)

    scheduler = ReduceLROnPlateau
    scheduler_parameters = {
        "mode": "min",
        "factor": 0.5,
        "patience": 300,
        "min_lr": 1e-5,
    }
    scheduler_input = "loss"

    losses = training(f, M1, M2, eps=eps, max_steps=max_steps, adam_parameters={"lr" : lr}, scheduler=scheduler, scheduler_parameters=scheduler_parameters, scheduler_input=scheduler_input, verbose=1, grad_clipping=5)

    with torch.no_grad():
        print("Final loss after training:", losses[-1])
        perm = [f.mode()[i] for i in range(1, n+1)]
        print("Most probable permutation:", perm)
        print("Target permutation:", (torch.argmax(Q, axis=1)+ 1).tolist())
        
        mpp = permutation_to_permutation_matrix(
            perm
        )
        print("|target(M1) - most_probable(M1)|: ", torch.norm(
            M2 - mpp.T @ M1 @ mpp ).item()
        )

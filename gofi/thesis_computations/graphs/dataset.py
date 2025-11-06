from gofi.graphs.graph import (
    random_adjacency_matrix,
    adjacency_matrix_cayley_Sn,
    random_permutation_matrix,
)
import torch
from math import factorial

def create_dataset(random_sizes, cayley_sizes):
    """
    Creates a dataset of graphs with sizes folowing the input.
    For a given n, each entry consists of a tuple (M, Q, Q.T @ M @ Q) of n*n matrices,
    where M is either a random adjecency matrix or an adjecency matrix of a cayley graph of Sn and Q is a random pemrutation matrix

    Returns
    ---------
    random_graphs : List[Tuple], cayley_graphs : List[Tuple]
        list of entries f randomly generated graphs and list of entries of cayley graphs
    """
    random_graphs = []
    cayley_graphs = []
    all_graphs = []

    for n in random_sizes:
        M1 = random_adjacency_matrix(n)
        Q = random_permutation_matrix(n)
        M2 = Q.T @ M1 @ Q
        random_graphs.append((M1, Q, M2))
        all_graphs.append(("random", M1, Q, M2))

    for n in cayley_sizes:
        M1 = adjacency_matrix_cayley_Sn(n)
        Q = random_permutation_matrix(factorial(n))
        M2 = Q.T @ M1 @ Q
        cayley_graphs.append((M1, Q, M2))
        all_graphs.append(("cayley", M1, Q, M2))

    return random_graphs, cayley_graphs, all_graphs

import torch
from functools import reduce
import operator
from gofi.graphs.loss import BijectiveLossMatrix
from gofi.actions.model import Group


def multiply_matrices(*tensors):
    return reduce(operator.matmul, tensors)

def relation_loss(group : Group, generator_to_P : dict,  eps=1e-10):
        """
        Return relation loss of the model, defined as

        1/|R| sum tr(log(P_r + eps)) for r in relations

        where P_r is the stohastic matrix of a random map, defined with r.

        Parameters
        --------------
        group : Group
            Group, which is being modeled.
        generator_to_rm : dict
            Dictionary between generators and their stochastic matrices.
        eps : float
            A small constant added to probabilities to avoid log(0) when computing the logarithm. Default is 1e-10.
        """
        ans = 0
        for r in group.relations:
            # get stochastic matrix 
            P_r = multiply_matrices(*[generator_to_P[s] for s in r])
            log = torch.log(P_r + eps)
            ans += torch.trace(log)
        ans *= -1/group.n_relations
        return ans 

def bijective_loss(group  :Group, generator_to_P  :dict, eps=1e-10):
    """
    Bijective loss of the model, defined as
    1/|G| sum |P P^T - I|^2 for s in generators

    where P is the stochastic matrix of a random map, defined with s.
    """
    ans = 0
    for s in group.generators:
        P = generator_to_P[s]
        ans += BijectiveLossMatrix(P)
    ans /= group.n_generators
    return ans
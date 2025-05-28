from gofi.models import RandomMap
import torch
import torch.nn as nn


def RelationLoss(f : RandomMap, M1 : torch.Tensor, M2 : torch.Tensor):
    """
    Returns Relation loss of random map f on graphs G1 and G2, given by adjacency matrices M1 and M2. 
    
    RelationLoss is equal to -tr(log(P_f M_2 P_f^T) M_1^T)

    Parameters
    ----------
    f : RandomMap
        Random map between graphs G1 and G2.           
    M1 : torch.Tensor
        Adjacency matrix of graph G1.
    M2 : torch.Tensor
        Adjacency matrix of graph G2.
    Returns
    -------
    torch.Tensor
        Relation loss of random map f on graphs G1 and G2.
    """
    return -torch.trace(torch.log(f.P() @ M2 @ f.P().T) @ M1.T)

def BijectiveLoss(f : RandomMap):
    """
    Returns bijective loss of a random map. Lower loss means bigger P(f is bijective).

    Bijective loss is defined as |f.P() f.P()^T - I|^2, where f.P() is the probability matrix of the random map f.
   
    Parameters
    ----------
    f : RandomMap
        Random map between graphs G1 and G2.
   
    Returns
    -------
    torch.Tensor
        Bijective loss of random map f.
    """
    if f.domain != f.codomain:
        raise ValueError("Bijective loss is defined only for maps between same-sized sets.")
    return torch.norm(f.P() @ f.P().T - torch.eye(f.domain)) ** 2


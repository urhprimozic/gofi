from gofi.models import RandomMap
import torch
import torch.nn as nn

def RelationLossMatrix(P, M1, M2, eps=0.1e-10):
    """
    Returns relation loss of random map, defined by shotachastic matrix P, on graphs G1 and G2, given by adjacency matrices M1 and M2.
    RelationLoss is equal to -tr(log(P M_2 P^T + epsI) M_1^T)
    Parameters
    ----------
    P : torch.Tensor
        Probability matrix of the random map f.
    M1 : torch.Tensor
        Adjacency matrix of graph G1.
    M2 : torch.Tensor
        Adjacency matrix of graph G2.
    eps : float
        Eps, which gets added to every element before applying logarithm. Avoids overflows.
    Returns
    -------
    torch.Tensor
        Relation loss of random map defined by P on graphs G1 and G2.
    """
    return -torch.trace(torch.log(P @ M2 @ P.T + eps) @ M1.T)

def BijectiveLossMatrix(P, eps=0.1e-10):
    """
    Returns bijective loss of a random map, defined by stochastic matrix P.
    Lower loss means bigger P(f is bijective).
    Bijective loss is defined as |P P^T - I|^2, where P is the probability matrix of the random map f.
    Parameters
    ----------
    P : torch.Tensor
        Stochastic matrix, which gives the probability distribution the map follows. (i,j)-th element of P equals to P(RandomMap(i)=j).
    Returns
    -------
    torch.Tensor
        Bijective loss of random map defined by P.      
    """
    return torch.norm(P @ P.T - torch.eye(P.shape[0])) ** 2


def RelationLoss(f : RandomMap, M1 : torch.Tensor, M2 : torch.Tensor, eps=0.1e-10):
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
    return RelationLossMatrix(f.P(), M1, M2)
    #return -torch.trace(torch.log(f.P() @ M2 @ f.P().T) @ M1.T)

def BijectiveLoss(f : RandomMap, eps=0.1e-10):
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
    return BijectiveLossMatrix(f.P())
    #return torch.norm(f.P() @ f.P().T - torch.eye(f.domain)) ** 2


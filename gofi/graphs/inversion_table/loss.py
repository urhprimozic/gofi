from gofi.graphs.inversion_table.probs import PermModel
import torch
from gofi.graphs.graph import adjacency_matrix_to_edges_list

def log_loss(dist : PermModel, M1 : torch.tensor, M2 : torch.tensor,  eps=10e-13, loops=False):
    raise NotImplementedError


def P_conected_images(dist : PermModel, i,j, edge_list2):
    '''
    Computes P(f(i) ~ f(j))
    '''
    ans = 0
    for (k, h) in edge_list2:
        ans += dist.prob(i,k,j,h)
    return ans


def norm_loss(dist : PermModel, M1 : torch.tensor, M2 : torch.tensor,  eps=10e-13, loops=False):
    '''
    Computes loss of a distribution over bijections for isomorphism search between M1 and M2.

    Parameters
    -----------
    dist : PermModel
        Distribution over bijections
    M1 : torch.tensor
        Adjecency matrix for graph 1
    M2 : torch.tensor
        Adjecency matrix for graph 2
    loops : boolean
        True, if graohs can include loops
    '''
    if loops:
        raise NotImplementedError("Graphs with loops are not yet implemented. ")
    ans = 0
    n = M1.shape[0]

    edge_list2 = adjacency_matrix_to_edges_list(M2) #indexed with 1, which is ok

    for i in range(n):
        for j in range(n):
            if j >= i:
                break
            # i and j are indexed by 0 --> +1
            ans += (P_conected_images(dist, i+1, j+1, edge_list2) - M1[i][j]) ** 2 
    return ans

def norm_loss_normalized(dist : PermModel, M1 : torch.tensor, M2 : torch.tensor,  eps=10e-13, loops=False):
    n = M1.shape[0]
    ans =  norm_loss(dist, M1, M2, eps, loops) / (n**2)
    return ans

def id_loss(dist : PermModel):
    """
    Returns P(f is identity)
    """
    ans = 1
    for i in range(dist.n -1):
        ans *= dist.model()[i][0]
    return ans
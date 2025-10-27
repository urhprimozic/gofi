import torch
from gofi.graphs.inversion_table.probs import PermModel
from gofi.graphs.graph import adjacency_matrix_to_edges_list
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def P_conected_images(dist: PermModel, i, j, edge_list2):
    ans = torch.tensor(0.0, device=device, dtype=torch.float32)
    # edge_list2 is 1-based unordered pairs; sum both orientations
    for (k, h) in edge_list2:
        ans = ans + dist.prob(i, k, j, h)
        ans = ans + dist.prob(i, h, j, k)
    return ans

def P_connected_preimages(dist: PermModel, i, j, edge_list1):
    """
    Probability that the preimages of (i,j) are connected in M1.
    Sums over edges (k,h) in M1 and uses f(k)=i, f(h)=j.
    """
    ans = torch.tensor(0.0, device=device, dtype=torch.float32)
    for (k, h) in edge_list1:
        ans = ans + dist.prob(k, i, h, j)
        ans = ans + dist.prob(k, j, h, i)
    return ans

def stochastic_norm_loss(dist: PermModel, M1: torch.Tensor, M2: torch.Tensor, sample_size: int, eps=1e-13, loops=False):
    """
    Unbiased estimator of normalized loss by sampling unordered pairs i>j.
    """
    n = M1.shape[0]
    edges2 = adjacency_matrix_to_edges_list(M2)
    total_pairs = n * (n - 1) // 2
    if sample_size is None or sample_size >= total_pairs:
        return norm_loss(dist, M1, M2, eps, loops) / total_pairs
    # sample unordered pairs (i>j) in 0-based; convert to 1-based for dist
    all_pairs = [(i, j) for i in range(n) for j in range(i)]
    sampled_pairs = random.sample(all_pairs, sample_size)
    ans = torch.tensor(0.0, device=device, dtype=torch.float32)
    for (i0, j0) in sampled_pairs:
        ans = ans + (P_conected_images(dist, i0 + 1, j0 + 1, edges2) - M1[i0, j0]) ** 2
    return ans / sample_size

def norm_loss(dist: PermModel, M1: torch.Tensor, M2: torch.Tensor, eps=1e-13, loops=False):
    n = M1.shape[0]
    edges2 = adjacency_matrix_to_edges_list(M2)
    ans = torch.tensor(0.0, device=device, dtype=torch.float32)
    for i in range(n):
        for j in range(i):
            ans = ans + (P_conected_images(dist, i + 1, j + 1, edges2) - M1[i, j]) ** 2
    return ans

def norm_loss_normalized(dist: PermModel, M1: torch.Tensor, M2: torch.Tensor, eps=1e-13, loops=False):
    n = M1.shape[0]
    denom = n * (n - 1) / 2
    return norm_loss(dist, M1, M2, eps, loops) / denom

def symetric_loss_normalized(dist: PermModel, M1: torch.Tensor, M2: torch.Tensor, eps=1e-13):
    """
    Symmetric, but both terms are normalized and use correct orientation.
    First term: edges in M2 vs images f(i),f(j).
    Second term: edges in M1 vs preimages f^{-1}(i),f^{-1}(j).
    """
    n = M1.shape[0]
    denom = n * (n - 1) / 2
    edges2 = adjacency_matrix_to_edges_list(M2)
    edges1 = adjacency_matrix_to_edges_list(M1)
    ans1 = torch.tensor(0.0, device=device, dtype=torch.float32)
    ans2 = torch.tensor(0.0, device=device, dtype=torch.float32)
    for i in range(n):
        for j in range(i):
            ans1 = ans1 + (P_conected_images(dist, i + 1, j + 1, edges2) - M1[i, j]) ** 2
            ans2 = ans2 + (P_connected_preimages(dist, i + 1, j + 1, edges1) - M2[i, j]) ** 2
    return (ans1 / denom + ans2 / denom) / 2.0

def stochastic_symetric_loss_normalized(dist: PermModel, M1: torch.Tensor, M2: torch.Tensor, sample_size: int, eps=1e-13):
    """
    Stochastic symmetric loss (normalized) using correct orientation for both terms.
    """
    n = M1.shape[0]
    total_pairs = n * (n - 1) // 2
    if sample_size is None or sample_size >= total_pairs:
        return symetric_loss_normalized(dist, M1, M2, eps)
    # sample unordered pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i)]
    sampled_pairs = random.sample(all_pairs, sample_size)
    edges2 = adjacency_matrix_to_edges_list(M2)
    edges1 = adjacency_matrix_to_edges_list(M1)
    ans = torch.tensor(0.0, device=device, dtype=torch.float32)
    for (i0, j0) in sampled_pairs:
        term1 = (P_conected_images(dist, i0 + 1, j0 + 1, edges2) - M1[i0, j0]) ** 2
        term2 = (P_connected_preimages(dist, i0 + 1, j0 + 1, edges1) - M2[i0, j0]) ** 2
        ans = ans + 0.5 * (term1 + term2)
    return ans / sample_size
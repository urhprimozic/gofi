from typing import Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from gofi.graphs.graph import permutation_matrix_to_permutation, permutation_to_permutation_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sinkhorn(a, b, C, reg=0.1, num_iters=50, tol=1e-9):
    """
    Sinkhorn algorithm to compute the regularized OT matrix.

    Args:
        a: torch tensor of shape (n,), source distribution (sum = 1)
        b: torch tensor of shape (m,), target distribution (sum = 1)
        C: torch tensor of shape (n, m), cost matrix
        reg: float, regularization coefficient
        num_iters: int, max iterations
        tol: float, convergence tolerance

    Returns:
        P: torch tensor of shape (n, m), transport plan
    """
    n, m = C.shape
    K = torch.exp(-C / reg).to(device)  # kernel matrix
    u = torch.ones(n, device=C.device).to(device) / n
    v = torch.ones(m, device=C.device).to(device) / m

    for _ in range(num_iters):
        u_prev = u.clone()
        u = a / (K @ v)
        v = b / (K.t() @ u)
        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    P = torch.diag(u).to(device) @ K @ torch.diag(v).to(device)
    return P


class NodeEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def cost_matrix(X, Y):
    # X: [n, d], Y: [n, d]
    return torch.cdist(X, Y, p=2) ** 2


def sinkhorn_matching(C, reg=0.1):
    n = C.shape[0]
    a = torch.ones(n).to(device) / n
    b = torch.ones(n).to(device) / n
    S = sinkhorn(a, b, C, reg)
    return S


def isomorphism_loss(M1, M2, S):
    return torch.norm(M1 - S @ M2 @ S.T, p="fro")


def adj_to_edge_index(M):
    row, col = M.nonzero(as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


class OTGraphMatcher(nn.Module):
    def __init__(self, node_dim, hidden_dim, emb_dim):
        super().__init__()
        self.encoder = NodeEncoder(node_dim, hidden_dim, emb_dim)

    def forward(self, M1, M2):
        edge_index1 = adj_to_edge_index(M1)
        edge_index2 = adj_to_edge_index(M2)

        n = M1.shape[0]

        x1 = torch.eye(n).to(device)
        x2 = torch.eye(n).to(device)

        Z1 = self.encoder(x1, edge_index1)
        Z2 = self.encoder(x2, edge_index2)

        C = cost_matrix(Z1, Z2)
        S = sinkhorn_matching(C)

        loss = isomorphism_loss(M1, M2, S)
        return loss, S

    def train(self, M1, M2, lr=0.001, epochs=1000, verbose=0, grad_eps=0.0001):
        """
        Trains the GNN and returns
         losses, loss, S
        """
        if grad_eps is None:
            grad_eps = -1.0  # disable early stopping based on grad norm
            
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        relation_losses = []

        rm = RandomMapGNN(self, M1, M2)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss, S = self.forward(M1, M2)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            if verbose > 0:
                if epoch % verbose == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item()}", end="\r")
            
            # get relation loss
            perm = rm.table()
            mpp = permutation_to_permutation_matrix(perm).to(device)
            relation_loss = isomorphism_loss(M1, M2, mpp).item()
            relation_losses.append(relation_loss)

            # get grad norm
            total_norm = 0.0
            for p in self.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            if total_norm < grad_eps:
                if verbose > 0:
                    print(f"Stopping early at epoch {epoch} due to small gradient norm: {total_norm} < {grad_eps} while achieving loss {loss.item()}.")
                break 
        return losses, relation_losses, S


def sinkhorn_matrix(log_alpha: torch.Tensor, n_iters=10, eps=1e-9):
    log_P = log_alpha
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True)
    return torch.exp(log_P).clamp_min(eps)


class RandomMapGNN(nn.Module):
    def __init__(self, graph_mathcer: OTGraphMatcher, M1, M2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_matcher = graph_mathcer
        self.M1 = M1
        self.M2 = M2
        self.domain = M1.shape[0]
        self.codomain = M2.shape[0]

    def P(self):
        _, S = self.graph_matcher.forward(self.M1, self.M2)
        return S

    def relation_loss(self, M1, M2):
        loss, _ = self.graph_matcher.forward(M1, M2)
        return loss

    def mode(self, programercic=False):
        """
        Returns the map between domain and codomain, which has the largest probability.

        Parameters
        ----------
        programercic : boolean
            False by default. If True, domain and codomain are indexed at 0 instead of 1.

        Returns
        ------------
        map : dict
            Dictionary of values {i : f(i)}, where i is in domain and f is the most probable map of the distribution, given by P
        """
        index_shift = int(not programercic)
        mode_indices = torch.argmax(self.P(), dim=1)
        return {
            i + index_shift: mode_indices[i].item() + index_shift
            for i in range(self.domain)
        }

    def table(self):
        d = self.mode()
        return [d[i] for i in range(1, self.domain + 1)]

import torch
import torch.nn as nn
from typing import Tuple
from scipy.optimize import linear_sum_assignment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch

def sinkhorn_matrix(log_alpha: torch.Tensor, n_iters=10, eps=1e-9):
    # log_alpha: (n,n) logits
    # log-domain Sinkhorn to reduce vanishing gradients
    log_P = log_alpha
    for _ in range(n_iters):
        # row normalize
        log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
        # column normalize
        log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True)
    return torch.exp(log_P).clamp_min(eps)


class ToMatrix(nn.Module):
    """
    A model that reshapes output of an inner model into a square matrix.
    """

    def __init__(self, model : nn.Module, n : int):
        super().__init__()
        self.model = model
        self.n = n

    def forward(self) -> torch.Tensor:
        return self.model().view(self.n, self.n)


def closest_permutation_matrix(M: torch.Tensor) -> torch.Tensor:
    # Convert to numpy for scipy
    M_np = M.cpu().detach().numpy()

    # Solve the linear sum assignment problem on the NEGATED matrix
    # (because we want to maximize the sum, but scipy minimizes)
    row_ind, col_ind = linear_sum_assignment(-M_np)

    # Create the permutation matrix
    n = M.size(0)
    P = torch.zeros_like(M)
    P[row_ind, col_ind] = 1.0
    return P

class Matrix(nn.Module):
    """
    A dummy model for a rows * columns matrix. Always returns the matrix itself.
    """

    def __init__(
        self,
        rows: int,
        columns: int,
        initial_params: None | torch.Tensor = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if initial_params is not None:
            self.matrix = nn.Parameter(initial_params)
        else:
            self.matrix = nn.Parameter(torch.rand((rows, columns)) * 2 - 1)

    def forward(self):
        return self.matrix


class RandomMap(nn.Module):
    def __init__(
        self,
        shape: int | Tuple[int, int],
        initial_params: None | torch.Tensor = None,
        inner_model: None | nn.Module = None,
        sinkhorn = False,sinkhorn_iters=50
    ):
        """
        Create a random map on {1, 2, ...., shape} (if shape is int) or a random map between {1, ..., a} and {1, ..., b} if shape is (a, b).

        Probabilities for the map are given in initial_params, and are selected at random if none are given.

        Parameters
        ----------
        shape : int or tuple
            Shape of the map, either a single integer for a map on {1, 2, ..., shape} or a tuple (a, b) for a map from {1, ..., a} to {1, ..., b}.
        initial_params : torch.tensor, optional
            Initial parameters for the map, representing the probabilities of mapping each element to each other element. If None, parameters are initialized randomly.
        inner_model : nn.Module, optional
            An inner model to be used for the map, if any. If None, no inner model is used.
        sinkhorn : bool, optional
            Whether to use Sinkhorn normalization for the probability matrix. Default is False.
        """
        super().__init__()
        # set domain and codomain
        if type(shape) == int:
            self.shape = (shape, shape)
        else:
            self.shape = shape
        self.domain = self.shape[0]
        self.codomain = self.shape[1]
        # set inner model
        if inner_model is None:
            self.overparameterized = False
            inner_model = Matrix(self.domain, self.codomain, initial_params)#.to(device)
        else:
            self.overparameterized = True
        self.inner_model = inner_model#.to(device)
        self.softmax = nn.Softmax(dim=1)#.to(device)
        self.sinkhorn = sinkhorn
        self.sinkhorn_iters = sinkhorn_iters

    @classmethod
    def from_probs(cls, probs: torch.Tensor, eps=1e-10):
        """
        Create a random map with given probabilities.

        Parameters
        ----------
        probs : torch.Tensor
            A tensor of shape (a, b) representing the probabilities of mapping each element from {1, ..., a} to {1, ..., b}.
            The tensor should be normalized such that each row sums to 1.

        eps : float, optional
            A small constant added to probabilities to avoid log(0) when computing the logarithm. Default is 1e-10.

        Returns
        -------
        RandomMap
            A RandomMap object with the given probabilities.
        """
        phi = torch.log(probs + eps)  # Adding a small constant to avoid log(0)
        return cls(shape=probs.shape, initial_params=phi)

    def phi(self):
        """
        Returns square matrix, which encodes the map. Probability matrix is given by applying softmax over phi.
        """
        return self.inner_model()#.to(device)

    def P(self):
        if self.sinkhorn:
            return sinkhorn_matrix(self.phi(), n_iters=self.sinkhorn_iters)
        else:
            return self.softmax(self.phi())#.to(device)

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

    def most_probable_map(self):
        return self.mode()

    def table(self):
        d = self.mode()
        return [d[i] for i in range(1, self.domain + 1)]
    
    def im_size(self):
        table = self.table()
        ans = 0
        for  i in range(1, self.domain +1):
            if i in table:
                ans += 1 
        return ans 
    def mode_matrix(self):
        return closest_permutation_matrix(self.P())






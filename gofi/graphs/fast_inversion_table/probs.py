import torch
import torch.nn as nn
from gofi.graphs.inversion_table.models import Cache
from functools import lru_cache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PermModel:
    def __init__(self, model, n, *args, **kwargs):
        """
        Parameters
        ------------
        model
            torch model. model() Should be a list of categorical distributions of lenghts n, n-1, ..., 1
        """
        self.model = Cache(model.to(device))
        self.n = n
        
        self.probabilities = None
       
    def clear_cache(self):
        self.p.cache_clear()
        self.q.cache_clear()
        self.probabilities = None
        self.model.clear()

    def model_value(self):
        return self.model()

    def P(self, m, s):
        """
        Returns the probability P(a_m = s) of placing m on the s-th empty file.
        """
        probs = self.model_value()[m - 1]  # we index with 0!
        return probs[s - 1]  # we index with zero!

    @lru_cache(maxsize=None)
    def P_sum(self, m, start, end):
        """
        Returns SUM_{s=start --> end} P(a_m = s). That equals to the probability of placing m anywhere between (and including ) start and end.
        """
        # get probabilities for placing m
        probs = self.model_value()[m - 1]  # index with zero
    
        # sum
        return torch.sum(
            probs[start - 1 : end]
        )  # index by zero, end -1 is the last one

    @lru_cache(maxsize=None)
    def q(self, j, h, m):
        """
        probability of placing h on the j-th empty file, if m-1 numbers were already placed.
        """
        # border cases
        if j <= 0 or j > self.n - m + 1 or m > self.n:
            return torch.tensor(0.0, device=device)

        # place m at j if m == h
        if m == h:
            return self.P(m, j)

        # recursion
        ans = torch.tensor(0.0, device=device)
        ans += self.q(j - 1, h, m + 1) * self.P_sum(m, 1, j - 1)
        ans += self.q(j, h, m + 1) * self.P_sum(m, j + 1, self.n - m + 1)
        return ans
    

    @lru_cache(maxsize=None)
    def p(self, i, k, j, h, m):
        """
        probability of placing k on the i-th empty file and h on the j-th empty file, if m-1 numbers were already placed.
        """
        # enforce i < j
        if i >= j:
            i, j = j, i
            k, h = h, k

        # out of scope
        if i <= 0 or j <= 0 or i > self.n - m + 1 or j > self.n - m + 1 or m > self.n:
            return torch.tensor(0.0, device=device)
        
        # handle placements when m matches one of the targets
        if m == k:
            return self.P(k, i) * self.q(j - 1, h, k + 1)
        if m == h:
            return self.P(h, j) * self.q(i, k, h + 1)

        # recursive case
        ans = torch.tensor(0.0, device=device)
        ans += self.p(i - 1, k, j - 1, h, m + 1) * self.P_sum(m, 1, i - 1)
        ans += self.p(i, k, j - 1, h, m + 1) * self.P_sum(m, i + 1, j - 1)
        ans += self.p(i, k, j, h, m + 1) * self.P_sum(m, j + 1, self.n - m + 1)
        return ans

    def prob(self, i, k, j, h):
        """
        Returns probabiity of generating a permutation, where k is palced on the i-th file and h is placed on the j-th file
        """
        return self.p(i, k, j, h, 1)

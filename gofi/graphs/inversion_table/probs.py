import torch
import torch.nn as nn
from gofi.graphs.inversion_table.models import Cache
from functools import lru_cache
from torch.amp import autocast

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
        self.P.cache_clear()
        self.P_sum.cache_clear()
        self.probabilities = None
        self.model.clear()

    def model_value(self):
        return self.model()
    
    def most_probable_permutation(self):
        """
        Returns the most probable permutation according to the model.

        TODO implement more efficiently using dynamic programming in nlogn time.
        Currently: O(n^2)
        """


        perm = [None] * self.n
        
        with autocast(device_type='cuda', enabled=False):
            probs = self.model_value()

            free_positions = list(range(1, self.n + 1))
            
            for m in range(1, self.n):
                prob_m = probs[m - 1]  
                it_index = torch.argmax(prob_m).item() + 1
                true_position = free_positions[it_index - 1]

                # remove this position from free positions
                free_positions.pop(it_index - 1)
                
                # put m in the true position
                perm[true_position - 1] = m        
                

            # place n in the last remaining position
            for i in range(self.n):
                if perm[i] is None:
                    perm[i] = self.n
                    break
            
            return perm

    @lru_cache(maxsize=None)
    def P(self, m, s):
        """
        Returns the probability P(a_m = s) of placing m on the s-th empty file.
        """
        with autocast(device_type='cuda', enabled=False):
            if m == self.n:
                return torch.tensor(1.0 if s == 1 else 0.0, device=device, dtype=torch.float32)
            probs = self.model_value()[m - 1].float()  # we index with 0!
            return probs[s - 1]  # we index with zero!

    @lru_cache(maxsize=None)
    def P_sum(self, m, start, end):
        """
        SUM_{s=start..end} P(a_m = s), inclusive.
        """
        with autocast(device_type='cuda', enabled=False):
            probs = self.model_value()[m - 1].float()
            max_pos = self.n - m + 1
            start = max(1, start)
            end = min(max_pos, end)
            if start > end:
                return torch.tensor(0.0, device=device, dtype=torch.float32)
            return torch.sum(probs[start - 1 : end])

    @lru_cache(maxsize=None)
    def q(self, j, h, m):
        """
        probability of placing h on the j-th empty file, if m-1 numbers were already placed.
        """
        with autocast(device_type='cuda', enabled=False):
            if j <= 0 or j > self.n - m + 1: 
                return torch.tensor(0.0, device=device, dtype=torch.float32)
            # terminal: only n remains and only slot j=1 is available for h=n
            if m == self.n + 1:
                print(f"Error in q({j}, {h}, {m}): m={m} > n={self.n}")
                raise ValueError("m cannot be greater than n+1")
            if m == h:
                return self.P(m, j)
            if m == self.n:
                raise ValueError(f"It should be m < n, but got m == n = {self.n}")
            ans = torch.tensor(0.0, device=device, dtype=torch.float32)
            ans = ans + self.q(j - 1, h, m + 1) * self.P_sum(m, 1, j - 1)
            ans = ans + self.q(j,     h, m + 1) * self.P_sum(m, j + 1, self.n - m + 1)
            return ans
     

    @lru_cache(maxsize=None)
    def p(self, i, k, j, h, m):
        """
        probability of placing k on the i-th empty file and h on the j-th empty file, if m-1 numbers were already placed.
        """
        if k == h or i == j:
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        # enforce i < j
        if i >= j:
            i, j = j, i
            k, h = h, k

        with autocast(device_type='cuda', enabled=False):
            if i <= 0 or j <= 0 or i > self.n - m + 1 or j > self.n - m + 1 or m > self.n:
                return torch.tensor(0.0, device=device, dtype=torch.float32)
            if m == k:
                return self.P(k, i) * self.q(j - 1, h, k + 1)
            if m == h:
                return self.P(h, j) * self.q(i, k, h + 1)
            if m == self.n:
                print(f"Error in p({i}, {k}, {j}, {h}, {m}): m={m} == n={self.n}, but m==k or m==h not satisfied")
                raise ValueError(f"It should be m < n, but got m == n = {self.n}")
                
            ans = torch.tensor(0.0, device=device, dtype=torch.float32)
            ans = ans + self.p(i - 1, k, j - 1, h, m + 1) * self.P_sum(m, 1, i - 1)
            ans = ans + self.p(i,     k, j - 1, h, m + 1) * self.P_sum(m, i + 1, j - 1)
            ans = ans + self.p(i,     k, j,     h, m + 1) * self.P_sum(m, j + 1, self.n - m + 1)
            return ans

    def prob(self, i, k, j, h):
        """
        Returns probabiity of generating a permutation, where k is palced on the i-th file and h is placed on the j-th file
        """
        return self.p(i, k, j, h, 1)

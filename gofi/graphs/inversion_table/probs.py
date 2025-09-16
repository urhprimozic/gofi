import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from gofi.graphs.inversion_table.models import Cache
from functools import lru_cache

torch.autograd.set_detect_anomaly(True)

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
        # memo for P
        #self.cache = {}

        # memo for model
        self.probabilities = None
        #self.q_cache = torch.full((n +1, n +1, n +1), torch.nan, device=device)  # q_cache[j,h,m] = q(j,h,m)
       # self.q_cache = {}

        # self.p_cache = torch.full((n + 1, n + 1, n + 1, n + 1, n + 1), torch.nan, device=device)  # p_cache[m,s] = P(m,s)
        self.PSums = torch.full((n + 1, n + 1, n + 1), torch.nan, device=device)  # PSums[m,s1,s2] = P_sum(m,s1,s2)

    def prepeare_cache(self):
        out = self.model_value()
        


        raise NotImplementedError("Not finished")

    def clear_cache(self):
        self.p.cache_clear()
        self.q.cache_clear()
       #  self.cahce = {}
       # self.q_cache = torch.full((self.n + 1, self.n + 1, self.n + 1), torch.nan, device=device)  # q_cache[j,h,m] = q(j,h,m)
        #self.p_cache = torch.full((self.n + 1, self.n + 1, self.n + 1,self.n + 1, self.n + 1), torch.nan, device=device)  # p_cache[m,s] = P(m,s)
        self.probabilities = None
        self.model.clear()

    def model_value(self):
        return self.model()
        if self.probabilities is None:
            self.probabilities = self.model()
        return self.probabilities
            

    def P(self, m, s):
        """
        Returns the probability P(a_m = s) of placing m on the s-th empty file.
        """
        probs = self.model_value()[m - 1]  # we index with 0!
        return probs[s - 1]  # we index with zero!

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
       # if not torch.isnan(self.q_cache)[j, h, m]:
          #  return self.q_cache[j, h, m]
        ### border cases ###
        # j outside of scope
        if j <= 0 or j > self.n - m + 1:
         #   self.q_cache[j, h, m] = 0
            return 0
        # j in scope but m is n --> just one option
        if m == self.n:
           # self.q_cache[j, h, m] = 1
            return 1

        ## place m on j-th place
        if m == h:
            #self.q_cache[j, h, m] = self.P(m, j)
            return self.P(m, j)

        # check cache
        #if self.cache.get((j, h, m)) is not None:
         #   return self.cache[(j, h, m)]
        ### recursive ###
        ans = 0
        ans += self.q(j - 1, h, m + 1) * self.P_sum(m, 1, j - 1)
        ans += self.q(j, h, m + 1) * self.P_sum(m, j + 1, self.n - m + 1)

        # save to cahce
       # self.cache[(j, h, m)] = ans
        #self.q_cache[j, h, m] = ans
        return ans
    

    @lru_cache(maxsize=None)
    def p(self, i, k, j, h, m):
        """
        probability of placing k on the i-th empty file and j on the j-th empty file, if m-1 numbers were already placed.

        TODO"""
        #if not torch.isnan(self.p_cache)[i, k, j, h, m]:
          #  return self.p_cache[i, k, j, h, m]
        # i < j
        if i >= j:
            # i should be SMALLER than j!
            i, j = j, i
            k, h = h, k

        ### border cases ###
        # i or j out of scope
        if i <= 0 or j <= 0 or i > self.n - m + 1 or j > self.n - m + 1:
            # outside of the region placed to many on one side
            #self.p_cache[i, k, j, h, m] = 0
            return 0
        # m maximal
        if m == self.n:
            # just one tile to place
           # self.p_cache[i, k, j, h, m] = 1
            return 1

        # check cache
       # if self.cache.get((i, k, j, h, m)) is not None:
          #  return self.cache[(i, k, j, h, m)]

        ### place m ###
        if m == k:
            ans = self.P(k, i) * self.q(j - 1, h, k + 1)
            
            # save to cache
            #self.cache[(i, k, j, h, m)] = ans
            #self.p_cache[i, k, j, h, m] = ans
            return ans
        if m == h:
            ans = self.P(h, j) * self.q(i, k, h + 1)

            # save to cache
           # self.cache[(i, k, j, h, m)] = ans
           # self.p_cache[i, k, j, h, m] = ans
            return ans

        ### recursive ###
        ans = 0
        ans += self.p(i - 1, k, j - 1, h, m + 1) * self.P_sum(m, 1, i - 1)
        ans += self.p(i, k, j - 1, h, m + 1) * self.P_sum(m, i + 1, j - 1)
        ans += self.p(i, k, j, h, m + 1) * self.P_sum(m, j + 1, self.n - m + 1)
        #ans = self.p_cache[i, k, j, h, m] 
        # save to cache
      #  self.cache[(i, k, j, h, m)] = ans

        return ans

    def prob(self, i, k, j, h):
        """
        Returns probabiity of generating a permutation, where k is palced on the i-th file and h is placed on the j-th file
        """
        return self.p(i, k, j, h, 1)

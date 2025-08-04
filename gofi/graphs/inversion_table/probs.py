import torch


class PermModel:
    def __init__(self, model, n):
        """
        Parameters
        ------------
        model
            torch model. model() Should be a list of categorical distributions of lenghts n, n-1, ..., 1
        """
        self.model = model
        self.n = n

    def P(self):
        return self.model()

    def p(self, i, k, j, h, m):
        """
        look in the README
        TODO"""

        
        if i >= j:
            return 0
        # border cases
        if i <= 0 or j <= 0:
            return 0
        if m == 1:
            return 1
    
        probs = self.model()[self.n - m]  # indexed by 0

                # check if we are at i or j:
        curr = self.n - m +1
        if curr == i:
            raise NotImplementedError("Če je i == curr, se more useen še j slikat v h. Rabu boš še prob za to i guess")
            return probs[k-1]
        if curr == j:
            raise NotImplementedError("Če je i == curr, se more useen še j slikat v h. Rabu boš še prob za to i guess")
            return probs[h-1]


        ans = 0
        ans += self.p(i - 1, k, j, h, m - 1) * torch.sum(probs[0 : i - 1]) 
        ans += self.p(i, k, j - 1, h, m - 1) * torch.sum(probs[i: j-1]) 
        ans += self.p(i , k, j, h, m - 1) * torch.sum(probs[j : m]) 

        return ans
    def prob(self, i, k, j, h):
        assert i < j
        return self.p(i,k,j,h,self.n)
    
    
from gofi.graphs.inversion_table.models import PermDistConnected
model = PermDistConnected(3, 2, 3)
p_model = PermModel(model, 3)
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
        assert i < j
        # border cases
        if i <= 0 or j <= 0:
            return 0
        if m == 1:
            return 1
        probs = self.model()[self.n - m]  # indexed by 0

        ans = 0
        ans += self.p(i - 1, k, j, h, m - 1) * torch.sum(probs[0 : i - 1]) 
        ans += self.p(i, k, j - 1, h, m - 1) * torch.sum(probs[i: j-1]) 
        ans += self.p(i - 1, k, j, h, m - 1) * torch.sum(probs[j : m]) 

        return ans
    
    

from gofi.graphs.inversion_table.probs import PermModel
import torch.nn as nn 
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConstantModel(nn.Module):
    def __init__(self, n, probs):
        """
        A model that always returns the same probabilities.

        Parameters
        ------------
        n : int
            number of elements in the permutation
        probs : list of torch.Tensor
            list of length n, where probs[i] is a tensor of shape (n-i,) representing 
            the categorical distribution for position i+1.
        """
        super(ConstantModel, self).__init__()
        self.n = n
        self.probs = probs

    def forward(self):
        return self.probs

class Id(PermModel):
    def __init__(self, n):
        """
        Identity permutation model.

        Parameters
        ------------
        n : int
            number of elements in the permutation
        """
        probs = []
        for i in range(n-1):
            prob = torch.zeros(n - i, device=device)
            prob[0] = 1.0  # always choose the first available position
            probs.append(prob)
        model = ConstantModel(n, probs)
        super().__init__(model, n)

class ConstantPermModel(PermModel):
    def __init__(self, n, probs):
        """
        Constant permutation model.

        Parameters
        ------------
        n : int
            number of elements in the permutation
        probs : list of torch.Tensor
            list of length n, where probs[i] is a tensor of shape (n-i,) representing 
            the categorical distribution for position i+1.
        """
        model = ConstantModel(n, probs)
        super().__init__(model, n)

class Uniform(PermModel):
    def __init__(self, n):
        """
        Uniform permutation model.

        Parameters
        ------------
        n : int
            number of elements in the permutation
        """
        probs = []
        for i in range(n-1):
            prob = torch.ones(n - i, device=device)
            prob = prob / prob.sum()  # uniform distribution over available positions
            probs.append(prob)
        model = ConstantModel(n, probs)
        super().__init__(model, n)
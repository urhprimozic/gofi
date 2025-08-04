import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNN(nn.Module):
    def __init__(self, *layer_sizes, clamp_value=10e8, T=None, softmax=True):
        super(LinearNN, self).__init__()
        assert len(layer_sizes) >= 2, "Need at least input and output layer size"

        self.softmax = softmax

        if T is None:
            self.T=1
        else:
            self.T = T
        self.clamp_value = clamp_value
        self.linears = nn.ModuleList()
        

        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.linears.append(nn.Linear(in_size, out_size))

        self.n_layers = len(self.linears)

    def forward(self, x):
        for i in range(self.n_layers - 1):  # all layers except last
            x = self.linears[i](x)
            if self.clamp_value is not None:
                x = torch.clamp(x, min=-self.clamp_value, max=self.clamp_value)
            x = F.relu(x)

        x = self.linears[-1](x)           # last layer
        if self.softmax:
            if x.dim() == 2:
                x = F.softmax(x / self.T, dim=1)           # apply softmax   
            if x.dim() == 1:  
                x = F.softmax(x / self.T, dim=0)           # apply softmax 
            else:
                raise ValueError("Dimensions of x should be 1 or 2.")
        return x

class PermDistDissconnected(nn.Module):
    def __init__(self,n, n_layers, layer_size, clamp_value=10e8, T=None, *args, **kwargs):
        """
        Creates a new model for distribution over permutations S_n, which consists of 
        n different linear models, each with n_layers layers and layer size of size layer_size. 

        >>> dist = PermDist(5, 4, 20)
        >>> dist()
        TODO
        """
        super().__init__(*args, **kwargs)
        args = [layer_size] * n_layers
        # za n ne rabiš svojega modelaa, itak vedno vrneš 1
        self.models = nn.ModuleList([LinearNN(*(args + [n-i]), clamp_value=clamp_value, T=T) for i in range(n-1)])
        self.n_layers = n_layers
        self.layer_size = layer_size
    def forward(self):
        x = torch.ones((self.layer_size,))
        return [model(x) for model in self.models]
    
class PermDistConnected(nn.Module):
    def __init__(self,n, n_layers, layer_size, clamp_value=10e8, T=None, *args, **kwargs):
        """
        Creates a new model for distribution over permutations S_n, which consists of 
        one linear model with n_layers layers and output size of n + n-1 + n-2 + ...

        n different linear models, each with n_layers layers and layer size of size layer_size. 

        >>> dist = PermDistConnecterd(5, 4, 20)
        >>> dist()
        TODO
        """
        if T is None:
            T = 1
        super().__init__(*args, **kwargs)
        args = (   [layer_size] * n_layers) + [int(0.5*n*(n+1)  - 1)]
        self.model = LinearNN(*args, clamp_value=clamp_value, T=T, softmax=False)# without softmax! - applied here!
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.n = n
        self.T = T 
    def forward(self):
        x = torch.ones((self.layer_size,))
        stacked =  self.model(x)
        ans = []
        for i in range(self.n-1):
            values = stacked[i*self.n : (i+1) * self.n - i] 
            # apply softmax to values- - get probs!
            if values.dim() == 2:
                values = F.softmax(values / self.T, dim=1)           # apply softmax   
            if values.dim() == 1:  
                values = F.softmax(values / self.T, dim=0)           # apply softmax 
            else:
                raise ValueError("Dimensions of x should be 1 or 2.")
            ans.append(values)
        return ans
    
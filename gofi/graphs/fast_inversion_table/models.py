import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class LinearNN(nn.Module):
    def __init__(self, *layer_sizes, clamp_value=10e8, T=None, softmax=True):
        super(LinearNN, self).__init__()
        assert len(layer_sizes) >= 2, "Need at least input and output layer size"

        self.softmax = softmax
        self.init = nn.Parameter(torch.randn(layer_sizes[0])).to(device)  # initial vector

        if T is None:
            self.T=1
        else:
            self.T = T
        self.clamp_value = clamp_value
        self.linears = nn.ModuleList()
        

        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.linears.append(nn.Linear(in_size, out_size).to(device))

        self.n_layers = len(self.linears)

    def forward(self):
        x = self.init
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

class DistLinear(nn.Module):
    '''
    Neural network, which outputs probability matrix 
    
    '''
    def __init__(self, n, n_layers, layer_size, clamp_value=1e8, T=None, *args, **kwargs):
        '''
        Creates a new NN with n_layers and a given layer size. Returns probability matrix.
        '''
        super().__init__(*args, **kwargs)
        if T is None:
            T = 1
        self.n = n
        self.T = T
        args = [layer_size] * n_layers + [(n-1) ** 2]   # direkt n*n output
        self.model = LinearNN(*args, clamp_value=clamp_value, T=T, softmax=False)

    def forward(self):
        stacked = self.model()   # shape: (batch, n*n) ali (n*n,)
        
        if stacked.dim() == 1:
            stacked = stacked.view(self.n, self.n)
        elif stacked.dim() == 2:
            stacked = stacked.view(-1, self.n, self.n)   # batch x n x n
        else:
            raise ValueError("stacked mora biti rank 1 ali 2 tensor")

        # Softmax po zadnji dimenziji (da dobiš porazdelitev po mestih)
        Pa = F.softmax(stacked / self.T, dim=-1)   # shape: (n, n) ali (batch, n, n)
        return Pa
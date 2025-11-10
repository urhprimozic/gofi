import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Cache(nn.Module):
    def __init__(self, model,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model 
        self.cache = None
    def forward(self):
        if self.cache is None:
            self.cache = self.model()
        return self.cache
    def clear(self):
        self.cache = None


class Linear_demo(nn.Module):
    def __init__(self, n, n_hidden_layers=2, softmax=True):
        super().__init__()
        hidden_dim = 10 * n
        output_dim = sum(n - i for i in range(n))  # n + n-1 + ... + 1
        self.init = nn.Parameter(torch.randn(n))  # ali konstanta
        self.net = nn.Sequential(*([
            nn.Linear(n, hidden_dim),
            nn.ReLU(),] + [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),] * n_hidden_layers + [ 
            nn.Linear(hidden_dim, output_dim),
            ])
        )
        self.softmax = softmax

    def forward(self):
        x = self.net(self.init)  # output je en velik vektor
        if self.softmax:
            if x.dim() == 2:
                x = F.softmax(x / self.T, dim=1)           # apply softmax   
            if x.dim() == 1:  
                x = F.softmax(x / self.T, dim=0)           # apply softmax 
            else:
                raise ValueError("Dimensions of x should be 1 or 2.")
        return x


class LinearNN(nn.Module):
    def __init__(self, *layer_sizes, clamp_value=10e8, T=None, softmax=True, batch_normalisation: bool = False):
        super(LinearNN, self).__init__()
        assert len(layer_sizes) >= 2, "Need at least input and output layer size"
        self.softmax = softmax
        self.batch_normalisation = batch_normalisation
        self.init = nn.Parameter(torch.randn(layer_sizes[0])).to(device)
        self.T = 1.0 if T is None else T
        self.clamp_value = clamp_value
        self.linears = nn.ModuleList()
        # both BN and LN; choose at runtime based on batch size
        self.bn_norms = nn.ModuleList() if self.batch_normalisation else None
        self.ln_norms = nn.ModuleList() if self.batch_normalisation else None
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.linears.append(nn.Linear(in_size, out_size))
            if self.batch_normalisation and out_size > 1:
                self.bn_norms.append(nn.BatchNorm1d(out_size))
                self.ln_norms.append(nn.LayerNorm(out_size))
        self.n_layers = len(self.linears)

    def forward(self):
        x = self.init
        for i in range(self.n_layers - 1):
            x = self.linears[i](x)
            if self.batch_normalisation:
                # Use LN when batch size == 1 or 1D vector; BN otherwise
                if x.dim() == 1 or (x.dim() == 2 and x.size(0) == 1):
                    x = self.ln_norms[i](x)
                elif x.dim() == 2:
                    x = self.bn_norms[i](x)
                else:
                    raise ValueError(f"Unexpected tensor dim for normalization: {x.dim()}")
            x = torch.relu(x)
            if self.clamp_value is not None:
                x = torch.clamp(x, -self.clamp_value, self.clamp_value)
        x = self.linears[-1](x)
        if self.softmax:
            if x.dim() == 2:
                x = torch.softmax(x / self.T, dim=1)
            elif x.dim() == 1:
                x = torch.softmax(x / self.T, dim=0)
            else:
                raise ValueError("Unexpected tensor dimension.")
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
        self.models = nn.ModuleList([LinearNN(*(args + [n-i]), clamp_value=clamp_value, T=T).to(device) for i in range(n-1)])
        self.n_layers = n_layers
        self.layer_size = layer_size
    def forward(self):
        return [model() for model in self.models]
    
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
        self.model = LinearNN(*args, clamp_value=clamp_value, T=T, softmax=False).to(device)# without softmax! - applied here!
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.n = n
        self.T = T 
    def forward(self):
        stacked =  self.model()
        ans = []
        for i in range(self.n-1):
            values = stacked[i * self.n - int((i-1)*(i)/2) : (i+1) * self.n - int((i)*(i+1)/2)] 
            # apply softmax to values- - get probs!
            if values.dim() == 2:
                values = F.softmax(values / self.T, dim=1)           # apply softmax   
            if values.dim() == 1:  
                values = F.softmax(values / self.T, dim=0)           # apply softmax 
            else:
                raise ValueError("Dimensions of x should be 1 or 2.")
            ans.append(values)
        return ans

class Destack(nn.Module):
    def __init__(self,n, model,T=50,  *args, **kwargs):
        '''
        Creates model, that destacts n(n-1)/2 - 1 dim input 
        '''
        super().__init__(*args, **kwargs)
        self.model = model 
        self.n=n 
        self.T=T
    def forward(self):
        stacked = self.model()
        ans = []
        for i in range(self.n-1):
            values = stacked[i * self.n - int((i-1)*(i)/2) : (i+1) * self.n - int((i)*(i+1)/2)] 
            # apply softmax to values- - get probs!
            if values.dim() == 2:
                values = F.softmax(values / self.T, dim=1)           # apply softmax   
            if values.dim() == 1:  
                values = F.softmax(values / self.T, dim=0)           # apply softmax 
            else:
                raise ValueError("Dimensions of x should be 1 or 2.")
            ans.append(values)
        return ans

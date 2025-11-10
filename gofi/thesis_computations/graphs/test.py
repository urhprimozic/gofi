from comparison import run_nn_it, run_vanilla, run_vanilla_it, run_nn
from dataset import create_dataset
from gofi.graphs.inversion_table.probs import PermModel
from gofi.models import RandomMap, ToMatrix
from gofi.graphs.inversion_table.models import PermDistConnected, LinearNN
import torch 
from gofi.graphs.loss import (
    BijectiveLoss,
    RelationLoss,
    BijectiveLossMatrix,
    RelationLossMatrix, LossGraphMatching, LossGraphMatchingRandomMap
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x, _, _ = create_dataset([5],[])
M1, Q, M2 = x[0]

n = M1.shape[0]
    
# prepare nn
layer_size = int(n**2)
n_layers = 4
args = (   [layer_size] * n_layers) + [n**2]
nn = LinearNN(*args, T=5).to(device)
inner_model = ToMatrix(nn, n).to(device)
f = RandomMap(n, inner_model=inner_model).to(device)
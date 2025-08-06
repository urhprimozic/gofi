from gofi.graphs.inversion_table.models import PermDistConnected, PermDistDissconnected
from gofi.graphs.inversion_table.probs import PermModel
from gofi.graphs.inversion_table.opt import training
from gofi.graphs.graph import random_adjacency_matrix
from gofi.graphs.inversion_table.loss import norm_loss_normalized, id_loss
from torch.optim.lr_scheduler import LambdaLR
import math

# setup:
# vsazga na 100 000 korakih, mal ožaš learning rate 
# za graf na n vozljiščih:
#  model je nevronska mreža z max(5, log(n)) skritimi layerji. Vsak layer je velikosti n**3.

MAX_STEPS = 100000
EPS = 0.001 # če pridemo pod en promil, smo gucci 

def lr_lambda(step):
    if step < 1000:
        return 1.0            # 0.001
    elif step < 10000:
        return 0.1            # 0.0001
    elif step < 50000:
        return 0.01           # 0.00001
    else:
        return 0.001          # 0.000001

def loss_function(dist, M1, M2):
        return norm_loss_normalized(dist, M1, M2) + id_loss(dist)

def run_on_random_graph(n, loss_function=loss_function, disconnected=False, verbose=1000):
    """
    Trains a network on a random graph. Returns tuple (M, dist, loss), where 
    M is adjecency matrix, dist is distribution model and loss is loss at the end.
    """
    # get random graph 
    M = random_adjacency_matrix(n)
    
    # model
    if disconnected:
         model = PermDistDissconnected(n, max(4, int(math.log(n))), n**3 ,T=100)
    else:
        model = PermDistConnected(n, max(4, int(math.log(n))), n**3 ,T=100)
    
    # distribution
    dist = PermModel(model, n)

    # train
    try:
        training(dist, M, M,loss_function=loss_function, max_steps=MAX_STEPS, eps=EPS, adam_parameters={"lr": 0.001}, verbose=verbose, scheduler=LambdaLR, scheduler_parameters={'lr_lambda':lr_lambda})

    except:
        print("error occured! Returning M, dist, last_loss ....")
        return M, dist, loss_function(dist, M, M).item()

    return M, dist, loss_function(dist, M, M).item()


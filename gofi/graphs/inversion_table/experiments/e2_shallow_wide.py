from gofi.graphs.inversion_table.models import Destack
from gofi.graphs.inversion_table.probs import PermModel
from gofi.graphs.inversion_table.opt import training
from gofi.graphs.graph import random_adjacency_matrix
from gofi.graphs.inversion_table.loss import norm_loss_normalized, id_loss
from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------
# MODEL
# ----------------------
class ShallowWide(nn.Module):
    def __init__(self, n, output_dim, hidden_dim_multiplier=20):
        super().__init__()

        self.n = n
        hidden_dim = n * hidden_dim_multiplier

        self.init = nn.Parameter(torch.randn(1)).to(device)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        ).to(device)

    def forward(self):
        return self.net(self.init)


# ----------------------
# OPTIMIZATOR + SCHEDULER
# ----------------------
epochs = 10

def lr_lambda_simple(step):
    if step < 50:
        return 100.0
    elif step < 500:
        return 10.0            # 0.001
    elif step < 1000:
        return 1.0
    elif step < 10000:
        return 0.1            # 0.0001
    elif step < 20000:
        return 0.01           # 0.00001
    else:
        return 0.001          # 0.000001
def get_lr_lambda(warmup_steps, max_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return 10*float(current_step) / float(max(1, warmup_steps))
        if current_step < 2 * warmup_steps:
            return 10 
        if current_step < 5 * warmup_steps:
            return 1
        if current_step < 10*warmup_steps:
            # Linear warmup from 0 to 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing from 1 down to 0 over remaining steps
            progress = float(current_step - warmup_steps) / float(
                max(1, max_steps - warmup_steps)
            )
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))

    return lr_lambda


def loss_function(dist, M1, M2):
    return norm_loss_normalized(dist, M1, M2) + id_loss(dist)


def run_on_random_graph(
    n,
    loss_function=loss_function,
    verbose=1000,
    max_steps=100000,
    eps=0.001,
    lr_max=1e-3,
    warmup_steps=500,
    lambda_lr_name=None,
    debug=False
):
    """
    Trains a network on a random graph. Returns tuple (M, dist, loss, dist.model()), where
    M is adjecency matrix, dist is distribution model and loss is loss at the end.
    """
    # get random graph
    M = random_adjacency_matrix(n)

    # model
    raw_model = ShallowWide(n, int(0.5 * n * (n + 1) - 1)).to(device)
    model = Destack(n, raw_model) 

    # distribution
    dist = PermModel(model, n)

    if lambda_lr_name is None:
        lambda_lr_name = "simple"
    if lambda_lr_name == "simple":
        lr_lambda = lr_lambda_simple
    elif lambda_lr_name == "cosine":
        lr_lambda = get_lr_lambda(warmup_steps, max_steps)
    else:
        raise NotImplementedError(f"Lambda lr name {lambda_lr_name} not yet implemented..")

    # train
    try:
        training(
            dist,
            M,
            M,
            loss_function=loss_function,
            max_steps=max_steps,
            eps=eps,
            adam_parameters={"lr": lr_max},
            verbose=verbose,
            scheduler=LambdaLR,
            scheduler_parameters={"lr_lambda": lr_lambda},
            debug=debug
        )

    except Exception as e:
        print("error occured! Returning M, dist, last_loss ....\nError:\n")
        print(e)
        return M, dist, loss_function(dist, M, M).item(), dist.model()

    return M, dist, loss_function(dist, M, M).item(), dist.model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=str, help="Number of vertices in a graph")
    parser.add_argument(
        "--max_steps",
        type=str,
        default="100000",
        help="Maximum number of steps in training.",
    )
    parser.add_argument(
        "--eps",
        type=str,
        default="0.001",
        help="Threshold for early stopping. If loss < eps, training stops.",
    )
    parser.add_argument(
        "--lr_max", default="0.0001", type=str, help="Initial learning rate."
    )
    parser.add_argument(
        "--warmup_steps", default="500", type=str, help="Number of warmup steps."
    )
    parser.add_argument(
        "--verbose",
        default="1000",
        type=str,
        help="Verbose level. Info will be printed every verbose steps. If 0, no info will be printed.",
    )
    parser.add_argument(
        "--lambda_lr",
        default="cosine",
        type=str,
        help="Either cosine or simple. more to come",
    )
    parser.add_argument(
        "--debug",
        default="0",
        type=str,
        help="Either 1 or 0",
    )
    args = parser.parse_args()

    # collect args
    n = int(args.n)
    max_steps = int(args.max_steps)
    eps = float(args.eps)
    lr_max = float(args.lr_max)
    warmup_steps = int(args.warmup_steps)
    verbose = int(args.verbose)
    lambda_lr_name = args.lambda_lr
    debug=bool(int(args.debug))
    # run
    M, dist, loss, model = run_on_random_graph(
        n,
        loss_function=loss_function,
        verbose=verbose,
        max_steps=max_steps,
        eps=eps,
        lr_max=lr_max,
        warmup_steps=warmup_steps,
        lambda_lr_name=lambda_lr_name,
        debug=debug
    )
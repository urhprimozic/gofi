import argparse
from gofi.graphs.inversion_table.models import PermDistConnected, PermDistDissconnected
from gofi.graphs.inversion_table.probs import PermModel
from gofi.graphs.inversion_table.opt import training
from gofi.graphs.graph import random_adjacency_matrix, random_permutation_matrix
from gofi.graphs.inversion_table.loss import (
    norm_loss_normalized,
    stochastic_norm_loss,
    symetric_loss_normalized,
    stochastic_symetric_loss_normalized,
)
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(0)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=str, help="Number of vertices in a graph")
    parser.add_argument("--layer_size", type=str, help="Size of each hidden layer", default="600")
    parser.add_argument("--lr", type=str, help="Initial learning rate of adam", default="0.001")
    parser.add_argument("--T", type=str, help="Temperature of softmax", default="1")
    parser.add_argument("--verbose", type=str, help="Number of steps taken between logs", default="100")
    parser.add_argument("--grad_clipping", type=float, help="Gradient clipping norm", default=2.0)
    parser.add_argument("--loss", type=str, help="Loss to use. stochastic/full", choices=["stochastic", "full"], default="full")
    parser.add_argument("--loss_sample_size", type=str, help="Sample size for stochastic loss", default="None")
    parser.add_argument("--scheduler", type=str, choices=["plateau", "cosine_wr"], default="plateau",
                        help="LR scheduler: plateau (ReduceLROnPlateau) or cosine_wr (CosineAnnealingWarmRestarts)")
    parser.add_argument("--cos_T0", type=int, default=1000, help="Cosine warm restart initial period T_0")
    parser.add_argument("--cos_Tmult", type=int, default=2, help="Cosine warm restart multiplier T_mult")
    parser.add_argument("--cos_eta_min", type=float, default=1e-5, help="Min LR for cosine schedule")
    parser.add_argument("--name", type=str, help="Run name", default=str(datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()

    # collect args+
    n = int(args.n)
    layer_size = int(args.layer_size)
    lr = float(args.lr)
    T = float(args.T)
    verbose = int(args.verbose)
    grad_clipping = float(args.grad_clipping)
    loss_function_name = args.loss
    sample_size = args.loss_sample_size
    if sample_size == "None":
        sample_size = int(n * (n - 1) // 4)
    else:
        sample_size = int(sample_size)
    # choose scheduler
    if args.scheduler == "cosine_wr":
        scheduler = CosineAnnealingWarmRestarts
        scheduler_parameters = {"T_0": args.cos_T0, "T_mult": args.cos_Tmult, "eta_min": args.cos_eta_min}
        scheduler_input = None  # step every iteration
        print(f"Using CosineAnnealingWarmRestarts: T_0={args.cos_T0}, T_mult={args.cos_Tmult}, eta_min={args.cos_eta_min}")
    else:
        scheduler = ReduceLROnPlateau
        scheduler_parameters = {"mode": "min", "factor": 0.5, "patience": 300, "min_lr": 1e-5}
        scheduler_input = "loss"
        print("Using ReduceLROnPlateau")

    # graph 
    M1 = random_adjacency_matrix(n)
    Q = random_permutation_matrix(n)
    # Use M2 = Q @ M1 @ Q.T so the natural mapping is f(i)=sigma(i)
    M2 = Q @ M1 @ Q.T

    # model
    model = PermDistConnected(n, 4,layer_size, T=T)
    dist = PermModel(model, n)

    # loss
    if loss_function_name == "stochastic":
        def stochastic_loss(dist, M1, M2):
            return stochastic_symetric_loss_normalized(dist, M1, M2, sample_size=sample_size)
        loss_function = stochastic_loss
        print(f"Using symmetric normalized stochastic loss with sample size {sample_size}")
    else :
        loss_function = symetric_loss_normalized
        print("Using symmetric normalized full loss")
    # train

    
    losses = training(
        dist, M1, M2,
        loss_function=loss_function,
        max_steps=15000 if n >= 10 else 5000,
        adam_parameters={"lr": lr},
        scheduler=scheduler,
        scheduler_parameters=scheduler_parameters,
        scheduler_input=scheduler_input,
        grad_clipping=grad_clipping,
        verbose=verbose
    )

    with torch.no_grad():
        print("Final loss after training:", loss_function(dist, M1, M2).item())
    # save losses
    with open(f"n_{n}_ls_{layer_size}_{args.name}.pkl", "wb") as f:
        pickle.dump(losses, f)

    plt.plot(losses)
    plt.savefig(f"n_{n}_ls_{layer_size}_{args.name}.pdf")

    with torch.no_grad():
        print("Final loss:", loss_function(dist, M1, M2).item())


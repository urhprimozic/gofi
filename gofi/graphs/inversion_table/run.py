import math
import argparse
from gofi.graphs.inversion_table.models import PermDistConnected, PermDistDissconnected
from gofi.graphs.inversion_table.probs import PermModel
import gofi.graphs.inversion_table.opt as opt
from gofi.graphs.graph import random_adjacency_matrix, random_permutation_matrix, permutation_matrix_to_permutation, permutation_to_permutation_matrix
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
from gofi.graphs.inversion_table.distributions import VanillaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=str, help="Number of vertices in a graph")
    parser.add_argument(
        "--layer_size", type=str, help="Size of each hidden layer", default="600"
    )
    parser.add_argument(
        "--lr", type=str, help="Initial learning rate of adam", default="0.001"
    )
    parser.add_argument("--T", type=str, help="Temperature of softmax", default="1")
    parser.add_argument(
        "--verbose", type=str, help="Number of steps taken between logs", default="1"
    )
    parser.add_argument(
        "--grad_clipping", type=float, help="Gradient clipping norm", default=2.0
    )
    parser.add_argument(
        "--loss",
        type=str,
        help="Loss to use. stochastic/full",
        choices=["stochastic", "stochastic_symetric", "full", "full_symetric"],
        default="full",
    )
    parser.add_argument(
        "--loss_sample_size",
        type=str,
        help="Sample size for stochastic loss",
        default="None",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["plateau", "cosine_wr", "none"],
        default="plateau",
        help="LR scheduler: plateau (ReduceLROnPlateau), cosine_wr (CosineAnnealingWarmRestarts) or none",
    )
    parser.add_argument(
        "--cos_T0",
        type=int,
        default=1000,
        help="Cosine warm restart initial period T_0",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.001,
        help="Stopping criterion for training based on loss value. Set to negative for None.",
    )
    parser.add_argument(
        "--grad_eps",
        type=float,
        default=-1.,
        help="Stopping criterion for training based on gradient norm. Default is none",
    )
    parser.add_argument(
        "--cos_Tmult", type=int, default=2, help="Cosine warm restart multiplier T_mult"
    )
    parser.add_argument(
        "--cos_eta_min", type=float, default=1e-5, help="Min LR for cosine schedule"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Run name",
        default=str(datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    parser.add_argument(
        "--amp",
        type=str,
        choices=["on", "of"],
        default="on",
        help="Enable/disable mixed precision (AMP) on CUDA",
    )
    parser.add_argument(
        "--opt",
        type=str,
        choices=["adam", "noise"],
        default="noise",
        help="Optimizer. Choose between Adam and Adam with noise.",
    )
    parser.add_argument(
        "--vanilla",
        type=str,
        choices=["on", "off"],
        default="off",
        help="if vanilla is on, there is no overparametrisation.",
    )
    args = parser.parse_args()     
    
     # Configure AMP (mixed precision) for opt.training via the module-global scaler
    if args.amp == "off":
        opt.scaler = None
        print("AMP disabled")
    else:
        if torch.cuda.is_available():
            from torch.amp import GradScaler
            opt.scaler = GradScaler('cuda')
            print("AMP enabled (fp16)")
        else:
            opt.scaler = None
            print("AMP disabled (no CUDA)")


    # collect args+
    n = int(args.n)
    layer_size = int(args.layer_size)
    lr = float(args.lr)
    T = float(args.T)
    verbose = int(args.verbose)
    grad_clipping = float(args.grad_clipping)
    loss_function_name = args.loss
    sample_size = args.loss_sample_size
    
    eps = args.eps
    if eps < 0:
        eps = None 
    
    grad_eps = args.grad_eps
    if grad_eps < 0:
        grad_eps = None 
    
    if sample_size == "None":
        sample_size = int((n ** (4 / 5)) * (n - 1) // 2)
    else:
        sample_size = int(sample_size)
    # choose scheduler
    if args.scheduler == "cosine_wr":
        scheduler = CosineAnnealingWarmRestarts
        scheduler_parameters = {
            "T_0": args.cos_T0,
            "T_mult": args.cos_Tmult,
            "eta_min": args.cos_eta_min,
        }
        scheduler_input = None  # step every iteration
        print(
            f"Using CosineAnnealingWarmRestarts: T_0={args.cos_T0}, T_mult={args.cos_Tmult}, eta_min={args.cos_eta_min}"
        )
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau
        scheduler_parameters = {
            "mode": "min",
            "factor": 0.5,
            "patience": 300,
            "min_lr": 1e-5,
        }
        scheduler_input = "loss"
        print("Using ReduceLROnPlateau")
    else:
        scheduler = None
        scheduler_parameters = {}
        scheduler_input = None
        print("Not using LR scheduler")

    # graph
    M1 = random_adjacency_matrix(n)
    Q = random_permutation_matrix(n)
    # Use M2 = Q @ M1 @ Q.T so the natural mapping is f(i)=sigma(i)
    #M2 = Q @ M1 @ Q.T
    M2 = Q.T @ M1 @ Q

    # model
    if args.vanilla == "off":
        print("Using NN overparametrisation.")
        model = PermDistConnected(n, 4, layer_size, T=T)
    else:
        print("Using vanilla model.")
        model = VanillaModel(n, T=T)
    dist = PermModel(model, n)

    # loss
    if loss_function_name == "stochastic_symetric":

        def stochastic_loss(dist, M1, M2):
            return stochastic_symetric_loss_normalized(
                dist, M1, M2, sample_size=sample_size
            )

        loss_function = stochastic_loss
        print(
            f"Using symmetric normalized stochastic loss with sample size {sample_size}"
        )
    elif loss_function_name == "full_symetric":
        loss_function = symetric_loss_normalized
        print("Using symmetric normalized full loss")
    elif loss_function_name == "full":
        loss_function = norm_loss_normalized
        print("Using  normalized full loss")
    elif loss_function_name == "stochastic":
        loss_function = stochastic_norm_loss
        print("Using stochastic loss")

    if args.opt == "adam":
        print("Using Adam optimizer")
    else:
        print("Using Adam with noise optimizer")
    # train

    losses = opt.training(
        dist,
        M1,
        M2,
        loss_function=loss_function,
        max_steps=15000 if n >= 10 else 5000,
        adam_parameters={"lr": lr},
        scheduler=scheduler,
        scheduler_parameters=scheduler_parameters,
        scheduler_input=scheduler_input,
        grad_clipping=grad_clipping,
        verbose=verbose,
        eps=eps, 
        grad_eps=grad_eps,
        adam_version=args.opt,
    )

    with torch.no_grad():
        print("Final loss after training:", loss_function(dist, M1, M2).item())
        print("Most probable permutation:", dist.most_probable_permutation())
        print("Target permutation:", (torch.argmax(Q, axis=1)+ 1).tolist())
        mpp = permutation_to_permutation_matrix(
            dist.most_probable_permutation()
        )

        print("|target(M1) - most_probable(M1)|: ", torch.norm(
            M2 - mpp.T @ M1 @ mpp ).item()
        )
    # save data
    results = {
        "n": n,
        "layer_size": layer_size,
        "lr": lr,
        "T": T,
        "verbose": verbose,
        "grad_clipping": grad_clipping,
        "loss_function": loss_function_name,
        "loss_sample_size": sample_size,
        "scheduler": args.scheduler,
        "scheduler_parameters": scheduler_parameters,
        "final_loss": loss_function(dist, M1, M2).item(),
        "most_probable_permutation": dist.most_probable_permutation(),
        "target_permutation": tuple(torch.argmax(Q, axis=1).cpu().numpy() + 1),
        "losses": losses,
        "vanilla" : args.vanilla,
    }

    #with open(f"results_n_{n}_ls_{layer_size}_{args.name}.pkl", "wb") as f:
    #    pickle.dump(results, f)
#
#
    #plt.plot(losses)
    #plt.savefig(f"loss_n_{n}_ls_{layer_size}_{args.name}.pdf")
#
    #with torch.no_grad():
    #    print("Final loss:", loss_function(dist, M1, M2).item())

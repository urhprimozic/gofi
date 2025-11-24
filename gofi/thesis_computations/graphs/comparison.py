from gofi.thesis_computations.graphs.dataset import create_dataset
import pickle
from gofi.graphs.inversion_table.distributions import VanillaModel
import gofi.graphs.inversion_table.opt as optit
from gofi.graphs.inversion_table.loss import norm_loss_normalized
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gofi.graphs.opt as optv
from gofi.models import RandomMap, ToMatrix
from gofi.graphs.inversion_table.models import PermDistConnected, LinearNN
from gofi.graphs.inversion_table.probs import PermModel
from gofi.graphs.graph import (
    random_adjacency_matrix,
    random_permutation_matrix,
    permutation_matrix_to_permutation,
    permutation_to_permutation_matrix,
)
from math import factorial
import torch
import tqdm
import argparse
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denumpy(x):
    """
    Changes all entries of shape np.int64(n) into n
    """
    x = list(x)
    for i, y in enumerate(x):
        if type(y) in [float, int]:
            continue
        x[i] = y.item()
    return x



def run_vanilla_it(M1, Q, M2, T=5, verbose=0,max_steps=1000, eps=0.009, grad_eps=None, **adam_params):
    '''
    Run vanilla inversion table model
    '''
    
    n = M1.shape[0]
    model = VanillaModel(n, T=T)
    dist = PermModel(model, n)

    if adam_params is None:
        adam_params = {"lr": 0.03}
    elif adam_params.get("lr") is None:
        adam_params["lr"] = 0.03

    scheduler = ReduceLROnPlateau
    scheduler_parameters = {
        "mode": "min",
        "factor": 0.5,
        "patience": 300,
        "min_lr": 1e-4,
    }
    scheduler_input = "loss"

    losses, relation_losses = optit.training(
        dist,
        M1,
        M2,
        loss_function=norm_loss_normalized,
        grad_eps=None,
        eps=eps,
        max_steps=max_steps,
        scheduler=ReduceLROnPlateau,
        scheduler_parameters=scheduler_parameters,
        scheduler_input=scheduler_input,
        verbose=verbose,
        adam_parameters=adam_params,
        store_relation_loss=True,
    )

    mpp = permutation_to_permutation_matrix(dist.most_probable_permutation())

    with torch.no_grad():
        results = {
            "losses": losses,
            "relation_losses": relation_losses,
            "final_loss": losses[-1],
            "most_probable_permutation": dist.most_probable_permutation(),
            "target_permutation": denumpy(
                list(torch.argmax(Q, axis=1).cpu().numpy() + 1)
            ),
            "diff_target_result": torch.norm(M2 - mpp.T @ M1 @ mpp).item(),
            "model": "vanilla_it",
        }
    return results


def run_nn_it(M1, Q, M2, T=5, verbose=0,adam_version="noise",max_steps=1000, eps=0.009, grad_eps=None, **adam_params):

    # TODO : different adam parameters for noise

    n = M1.shape[0]

    layer_size = int(n**2)

    model = PermDistConnected(n, 4, layer_size, T=T)
    dist = PermModel(model, n)

    scheduler = ReduceLROnPlateau
    scheduler_parameters = {
        "mode": "min",
        "factor": 0.5,
        "patience": 300,
        "min_lr": 1e-5,
    }
    scheduler_input = "loss"

    if adam_params is None:
        adam_params = {"lr": 0.01}
    elif adam_params.get("lr") is None:
        adam_params["lr"] = 0.01

    losses, relation_losses = optit.training(
        dist,
        M1,
        M2,
        loss_function=norm_loss_normalized,
        grad_eps=None,
        eps=eps,
        max_steps=max_steps,
        scheduler=ReduceLROnPlateau,
        scheduler_parameters=scheduler_parameters,
        scheduler_input=scheduler_input,
        verbose=verbose,
        adam_parameters=adam_params,
        adam_version="noise",
        store_relation_loss=True,
    )

    mpp = permutation_to_permutation_matrix(dist.most_probable_permutation())

    with torch.no_grad():
        results = {
            "losses": losses,
            "relation_losses": relation_losses,
            "final_loss": losses[-1],
            "most_probable_permutation": dist.most_probable_permutation(),
            "target_permutation": denumpy(
                list(torch.argmax(Q, axis=1).cpu().numpy() + 1)
            ),
            "diff_target_result": torch.norm(M2 - mpp.T @ M1 @ mpp).item(),
            "model": "nn_it",
        }
    return results


def run_vanilla(M1, Q, M2, verbose=0):
    n = M1.shape[0]
    f = RandomMap(n).to(device)
    losses, relation_losses = optv.training(
        f,
        M1,
        M2,
        max_steps=5000,
        eps=1e-4,
        adam_parameters={"lr": 0.01},
        store_relation_loss=True,
        verbose=verbose
    )  # eps je za grad_norm < eps tukaj

    with torch.no_grad():
        perm = [f.mode()[i] for i in range(1, n + 1)]
        mpp = permutation_to_permutation_matrix(perm)
        results = {
            "losses": losses,
            "relation_losses": relation_losses,
            "final_loss": losses[-1],
            "most_probable_permutation": perm,
            "target_permutation": denumpy(
                list(torch.argmax(Q, axis=1).cpu().numpy() + 1)
            ),
            "diff_target_result": torch.norm(M2 - mpp.T @ M1 @ mpp).item(),
            "model": "vanilla",
        }
    return results

def run_nn(M1, Q, M2, T=1, verbose=0):
    n = M1.shape[0]
    
    # prepare nn
    layer_size = int(n**2)
    n_layers = 4
    args = ([layer_size] * n_layers) + [n**2]
    nn = LinearNN(*args, T=1, softmax=False, batch_normalisation=True).to(device)
    inner_model = ToMatrix(nn, n).to(device)
    # use sinkhorn instead of softmax 
    f = RandomMap(n, inner_model=inner_model, sinkhorn=True, sinkhorn_iters=10).to(device)
    
    scheduler = ReduceLROnPlateau
    scheduler_parameters = {
        "mode": "min",
        "factor": 0.5,
        "patience": 300,
        "min_lr": 1e-5,
    }
    scheduler_input = "loss"

    losses, relation_losses = optv.training_stable(
        f,
        M1,
        M2,
        max_steps=5000,
        scheduler=ReduceLROnPlateau,
        scheduler_parameters=scheduler_parameters,
        scheduler_input=scheduler_input,
        eps=1e-4,
        adam_parameters={"lr": 0.02},
        store_relation_loss=True,
        verbose=verbose,
        use_relation_loss=False,
        grad_clipping=10,
        # weights after normalizing graph-matching by n^2
        A=0.1,
        B=1.0,
        # Anti-vanishing tweaks (opt-in; only for run_nn)
        anti_vanish=True,
        warmup_steps=max(100, n * 10),
        sinkhorn_warmup_disable=True,   # row-softmax during warmup
        sinkhorn_iters_warmup=2,
        sinkhorn_iters_post=10,
        entropy_weight=0.01,
        entropy_decay=0.995,
        grad_noise_std=5e-4,
        # scale LossGraphMatching by n^2 only for run_nn
        normalize_graph_matching=True,
    )  # eps is the grad_norm threshold here

    with torch.no_grad():
        perm = [f.mode()[i] for i in range(1, n + 1)]
        mpp = permutation_to_permutation_matrix(perm)
        results = {
            "losses": losses,
            "relation_losses": relation_losses,
            "final_loss": losses[-1],
            "most_probable_permutation": perm,
            "target_permutation": denumpy(
                list(torch.argmax(Q, axis=1).cpu().numpy() + 1)
            ),
            "diff_target_result": torch.norm(M2 - mpp.T @ M1 @ mpp).item(),
            "model": "nn",
        }
    return results

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="Run name")
    parser.add_argument("-rg", nargs="*"    , type=int, help="List of random graph sizes")
    parser.add_argument("-cg", nargs="*"    , type=int, help="List of cayley graph sizes for S_n")
    parser.add_argument("--timeless", type=str, choices=["yes", "no"], default="no" , help="If no, adds current time in run name")
    args = parser.parse_args()

    rg = args.rg
    cg = args.cg
    if rg is None:
        rg = [] 
    if cg is None:
        cg = []

    run_name = args.name 
    if args.timeless == "no":
        run_name += ( "_" + str(datetime.now()))

    print("Running comparisons")
    # create  and save dataset
    dataset = create_dataset(rg, cg)
    #dataset = create_dataset([4] , [3])

    with open(f"./results/dataset_{run_name}.pkl", "wb") as f:
        pickle.dump(dataset, f)

    random_graphs, cayley_graphs, all_graphs = dataset

    for index, (graph_type, M1, Q, M2) in tqdm.tqdm(enumerate(all_graphs), total=len(all_graphs)):
        # get results
        results_vanilla = run_vanilla(M1, Q, M2)
        results_vanilla_it = run_vanilla_it(M1, Q, M2)
        results_nn_it = run_nn_it(M1, Q, M2)
        results_nn = run_nn(M1, Q, M2)
        # save results 
        results = {"vanilla" : results_vanilla,
                   "vanilla_it" : results_vanilla_it,
                   "nn_it" : results_nn_it,
                   "nn" : results_nn,
                   "graph_tuple" : (M1, Q, M2)
                    }
        with open(f"./results/results_{run_name}_{index}_{graph_type}.pkl" , "wb") as f:
            pickle.dump(results, f)

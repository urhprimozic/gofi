from gofi.graphs.graph import random_adjacency_matrix, random_permutation_matrix
from gofi.thesis_computations.graphs.comparison import run_nn_it, run_vanilla_it
import pickle
from gofi.graphs.inversion_table.loss import norm_loss_normalized
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gofi.graphs.opt as opt
import torch
import tqdm
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vertices = [5, 7, 8, 9, 10, 10, 10]
noise_scales = [0.01, 0.001, 0.0001]
grad_thresholds = [0.01, 0.001]
cooldowns = [2, 5, 10]
decalys = [0.9, 0.99]
parameters_list = [
    {
        "lr": 0.005,
        "noise_scale": noise_scale,
        "grad_threshold": grad_threshold,
        "cooldown_steps": cooldown_steps,
        "decay": decay,
    }
    for noise_scale, grad_threshold, cooldown_steps, decay in itertools.product(
        noise_scales, grad_thresholds, cooldowns, decalys
    )
]

graph_tuples = []

if "__main__" == __name__:
    with tqdm.tqdm(total=len(vertices) * len(parameters_list)) as pbar:
        for n in vertices:
            # get new graph
            M1 = random_adjacency_matrix(n)
            Q = random_permutation_matrix(n)
            M2 = Q.T @ M1 @ Q
            graph_tuples.append((M1, Q, M2))
            
            # run different params for nn and vanilla
            for params in parameters_list:
                result_nn = run_nn_it(
                    M1,
                    Q,
                    M2,
                    T=5,
                    verbose=0,
                    max_steps=600,
                    eps=0.01,
                    **params,
                )
                # change lr
                params["lr"] = 0.03

                result_vanilla = run_vanilla_it(
                    M1,
                    Q,
                    M2,
                    T=5,
                    verbose=0,
                    max_steps=600,
                    eps=0.01,
                    **params,
                )
                result = {
                    "n": n,
                    "params": params,
                    "nn": result_nn,
                    "vanilla": result_vanilla,
                }
                with open(
                    f"./results/hyperparams_n{n}_ns{params['noise_scale']}_gt{params['grad_threshold']}_cd{params['cooldown_steps']}_decay{params['decay']}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(result, f)
                pbar.update(1)

    with open("./results/hyperparams_graphs.pkl", "wb") as f:
        pickle.dump(graph_tuples, f)

    print("Done all computations.")
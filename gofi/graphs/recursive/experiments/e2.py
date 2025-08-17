import pickle
import numpy as np
from tqdm import tqdm
import datetime
from gofi.graphs.graph import random_adjacency_matrix
import torch
import argparse
from gofi.graphs.recursive.opt import training
from gofi.graphs.recursive.model import PermutationGenerator, matrix_to_permutation
from gofi.graphs.recursive.loss import sample_losses_and_perms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    # defaults
    SAMPLE_SIZE=100
    MAX_STEPS = 50000
    EPS = 0.001
    VERBOSE = 1
    LEARNING_RATE = 0.001
    RUN_NAME = str(datetime.datetime.now())

    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=str, help="Number of vertices in a graph")
    parser.add_argument(
        "--sample_size",
        type=str,
        default=str(SAMPLE_SIZE),
        help="Number of diferent graphs tested",
    )
    parser.add_argument(
        "--max_steps",
        type=str,
        default=str(MAX_STEPS),
        help="Maximum number of steps in training.",
    )
    parser.add_argument(
        "--eps",
        type=str,
        default=str(EPS),
        help="Threshold for early stopping. If loss < eps, training stops.",
    )
    parser.add_argument(
        "--lr", default=str(LEARNING_RATE), type=str, help="Initial learning rate."
    )

    parser.add_argument(
        "--verbose",
        default=str(VERBOSE),
        type=str,
        help="Verbose level. Info will be printed every verbose steps. If 0, no info will be printed.",
    )
    parser.add_argument(
        "--run_name",
        default=RUN_NAME,
        type=str,
        help="Run name. Used to save results.",
    )
    args = parser.parse_args()

    # collect args
    n = int(args.n)
    sample_size = int(args.sample_size)
    max_steps = int(args.max_steps)
    eps = float(args.eps)
    lr = float(args.lr)
    verbose = int(args.verbose)
    run_name = str(args.run_name)

    graphs = []
    all_losses = []
    convergences = []

    for sample in tqdm(range(sample_size), total = sample_size):
        M = random_adjacency_matrix(n).to(device)
        generator = PermutationGenerator(n=n, hidden_size=n**2, num_layers=4).to(device)
        losses, converged = training(generator, M , M, eps = eps, max_steps=max_steps, verbose=-1, lr=lr, batch_size=min(50*n, 100))

        graphs.append(graphs)
        all_losses.append(losses)
        convergences.append(converged)

    print("Finished simulation.")
    x = np.array(convergences)
    print(f"{x.sum().item()} out of {len(x)} converged.")

    with open(f"n={n}_{run_name}.pkl", "wb") as f:
        pickle.dump(graphs, f)
        pickle.dump(all_losses, f)
        pickle.dump(convergences, f)

    # load
    # with open("myfile.pkl", "rb") as f:
    #     graphs = pickle.load(f)
    #     all_losses = pickle.load(f)
    #     convergences = pickle.load(f)


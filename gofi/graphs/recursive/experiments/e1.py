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
    MAX_STEPS = 100000
    EPS = 0.001
    VERBOSE = 1
    LEARNING_RATE = 0.001
    RUN_NAME = str(datetime.datetime.now())

    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=str, help="Number of vertices in a graph")
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
    max_steps = int(args.max_steps)
    eps = float(args.eps)
    lr = float(args.lr)
    verbose = int(args.verbose)
    run_name = str(args.run_name)

    M1 = random_adjacency_matrix(n)

    gen = PermutationGenerator(n=n, hidden_size=10*n**2, num_layers=4).to(device)
    losses, converged = training(gen, M1, M1, eps=eps, max_steps=max_steps, verbose=verbose, lr=lr, batch_size=max(50*n, 100))

    # Preveri en vzorec po treningu
    P, logp = gen(batch_size=1)
    try:
        print("Permutacijska matrika:\n", matrix_to_permutation(P[0]))
    except Exception as e:
        print("Error occured when trying to print the permutation.")
        print(e)
    print("probability:", torch.exp(logp).item())

    torch.save(gen, f"{run_name}_model.pt")
    torch.save(M1, f"{run_name}_M1.pt")

    print(sample_losses_and_perms(gen, M1, M1, batch_size=5))
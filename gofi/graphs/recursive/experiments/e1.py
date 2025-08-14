from gofi.graphs.graph import random_adjacency_matrix
import torch
import argparse
from gofi.graphs.recursive.opt import training
from gofi.graphs.recursive.model import PermutationGenerator, permutation_to_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_on_random_graph(n, eps=0.001, max_steps=100000, lr=0.01, verbose=1, safe_mode=False):
    generator = PermutationGenerator(n, hidden_size=50 * n, num_layers=5).to(device)
    M1 = random_adjacency_matrix(n)
    M2 = random_adjacency_matrix(n)
    if safe_mode:
        try:
            M1, M2, generator, loss = training(
                generator, M1, M2, eps=eps, max_steps=max_steps, lr=lr, verbose=verbose
            )
        except Exception as e:
            print("Error occured.")
            print(e)
            loss = None
        return M1, M2, generator, loss
    else:
        return training(
                generator, M1, M2, eps=eps, max_steps=max_steps, lr=lr, verbose=verbose
            )


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
        "--lr", default="0.0001", type=str, help="Initial learning rate."
    )

    parser.add_argument(
        "--verbose",
        default="1000",
        type=str,
        help="Verbose level. Info will be printed every verbose steps. If 0, no info will be printed.",
    )
    args = parser.parse_args()

    # collect args
    n = int(args.n)
    max_steps = int(args.max_steps)
    eps = float(args.eps)
    lr = float(args.lr)
    verbose = int(args.verbose)

    M1, M2, generator, loss = run_on_random_graph(
        n, eps=eps, max_steps=max_steps, lr=lr, verbose=verbose
    )

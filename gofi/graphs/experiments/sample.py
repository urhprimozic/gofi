import torch
from gofi.models import RandomMap
from gofi.graphs.loss import BijectiveLoss, RelationLoss
from gofi.graphs.graph import random_adjacency_matrix, adjacency_matrix_cayley_Sn
from math import factorial
from gofi.graphs.opt import training
import matplotlib.pyplot as plt
from gofi.plot.colors import blueorange
import networkx as nx

torch.set_printoptions(precision=2, sci_mode=False)


def save_graph_image(adj,  filename="graph.png",title=None, layout="spring", dpi=300):
    """
    Save a graph from an adjacency matrix as an image.

    Parameters
    ----------
    adj : np.ndarray or torch.Tensor
        Adjacency matrix of the graph (must be square).
    filename : str
        Output filename (should end in .png or .pdf).
    layout : str
        Layout algorithm: 'spring', 'circular', 'kamada_kawai', 'random'.
    dpi : int
        DPI for saved image.
    """
    # Convert to NumPy if it's a torch tensor
    if isinstance(adj, torch.Tensor):
        adj = adj.detach().cpu().numpy()

    # Ensure square matrix
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    # Create graph
    G = nx.from_numpy_array(adj)

    # Choose layout
    layout_funcs = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "random": nx.random_layout,
    }
    pos = layout_funcs.get(layout, nx.spring_layout)(G)

    # Draw and save
    plt.figure(figsize=(6, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        edge_color="gray",
        node_size=600,
        font_size=10,
    )
    plt.axis("off")
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    plt.savefig(filename, dpi=dpi)
    plt.close()


def sample_cayley(n_iterator, sample_size):
    for n in n_iterator:

        M = adjacency_matrix_cayley_Sn(n)
        for sample in range(sample_size):

            f = RandomMap(factorial(n))
            plt.imshow(f.P().clone().cpu().detach(), cmap=blueorange)
            plt.title(f"Initial values for Cayley$({n}) \\to $Cayley$({n})$")
            plt.axis("off")
            plt.colorbar()
            plt.savefig(f"Cayley_{n}_s{sample}_initial.pdf")
            plt.close()

            training(
                f, M, M, adam_parameters={"lr": 0.001}, max_steps=500000, eps=1e-12
            )

            plt.imshow(f.P().clone().cpu().detach(), cmap=blueorange)

            loss = RelationLoss(f, M, M) + BijectiveLoss(f)
            loss = round(float(loss.clone().cpu().detach().numpy()), 3)

            plt.title(
                f"Final values for Cayley$({n}) \\to $Cayley$({n})$\n" + f"Loss: {loss}"
            )
            plt.axis("off")
            plt.colorbar()
            plt.savefig(f"Cayley_{n}_s{sample}_final.pdf")
            plt.close()


def sample_cayley(n_iterator, sample_size):
    for index, n in enumerate(n_iterator):

        M = random_adjacency_matrix(n)
        save_graph_image(M, filename=f"graph_{index+1}.pdf")

        for sample in range(sample_size):

            f = RandomMap(n)
            plt.imshow(f.P().clone().cpu().detach(), cmap=blueorange)
            plt.title(f"Initial values for random graph$({n}) $")
            plt.axis("off")
            plt.colorbar()
            plt.savefig(f"Cayley_{n}_s{sample}_initial.pdf")
            plt.close()

            training(
                f, M, M, adam_parameters={"lr": 0.001}, max_steps=500000, eps=1e-12
            )

            plt.imshow(f.P().clone().cpu().detach(), cmap=blueorange)

            loss = RelationLoss(f, M, M) + BijectiveLoss(f)
            loss = round(float(loss.clone().cpu().detach().numpy()), 3)

            plt.title(f"Initial values for random graph$({n}) $" + f"Loss: {loss}")
            plt.axis("off")
            plt.colorbar()
            plt.savefig(f"Cayley_{n}_s{sample}_final.pdf")
            plt.close()
